"""
CEM planner / optimizer for the ball-in-cup JEPA world model.

Implements the Cross-Entropy Method inside a receding-horizon (MPC) loop:
each call to `plan_step` samples N candidate H-step action sequences,
rolls every one forward through P (jepa_evaluator.rollout), scores them
with `rollout_cost`, refits a per-timestep diagonal Gaussian to the
elites, and repeats for cfg.n_iters iterations -- returning only the
first action of the final mean, per the MPC/receding-horizon design (see
project_brief.md).

Design choices made here -- worth checking against project_brief.md /
the conversation before relying on this for anything real:

  - Diagonal, per-timestep-independent Gaussian sampling (not full
    covariance): a full covariance over H*ACTION_DIM dims would be badly
    underdetermined by any realistic population size.

  - Cost decodes position via the *position-only* probe (single z -> 4
    positions: cup_x, cup_z, ball_x, ball_z), not the K_CONTEXT-window
    position+velocity probe. v_rel (needed for the "don't drift upward
    away from the cup" penalty) comes instead from finite-differencing
    the *decoded* position sequence across the rollout (dt below).  This
    sidesteps needing the K_CONTEXT window at every predicted step (only
    the latest z is needed) at the cost of a noisier velocity estimate.
    state_probe.npz (saved by eval_linear_probe.py) has the window-based
    probe if the more accurate version is wanted instead -- swap it in
    by feeding it the actual predicted window at each tau, not z_preds
    alone.

  - No warm-starting: every call to `plan_step` initializes the sampling
    mean at zero by default -- consistent with the explicit "no
    warm-starting" decision. The *shift-previous-plan-forward* form of
    warm-starting (distinct from demo-seeding -- reusing last step's own
    solved plan as this step's init) is NOT implemented here; `prev_mean`
    is accepted so it can be wired in later without changing the call
    signature, but the default (prev_mean=None) is a from-scratch solve
    every call.

  - The time-tie-break term (mu * tau_star^2) sits outside the min, as
    discussed, so it only distinguishes between candidates that already
    roughly succeed. Defaults to mu=0 (off).

  - H, LAMBDA, MU, N_SAMPLES, ELITE_FRAC, N_ITERS in CEMConfig are all
    still placeholder values (H especially -- unset until the
    autoregressive drift diagnostic gives a real ceiling on how far P
    can be trusted). Treat every field as a guess to be revisited.

Run as a script for a smoke test against the real checkpoint + one real
eval episode's opening frames (no dm_control env needed -- only
jepa_evaluator + the persisted position probe). Run
eval_linear_probe.py first if position_probe.npz doesn't exist yet:
    python3 cem_planner.py
"""

import dataclasses
import pathlib

import numpy as np
import torch

import jepa_evaluator as je
import jepa_train as jt

POSITION_PROBE_PATH = "position_probe.npz"

# dm_control ball_in_cup's _CONTROL_TIMESTEP -- confirmed from source
# (dm_control/suite/ball_in_cup.py), not a guess.
CONTROL_TIMESTEP = 0.02

# Target offset in the same (cup_x, cup_z, ball_x, ball_z) qpos units the
# position probe predicts. Derived directly from ball_in_cup.xml: cup
# body pos z=.6, target site local offset z=-.05, ball body pos z=.2 =>
# target_z - ball_z(qpos-relative) = .6 - .05 - .2 = .35. x-offset is 0
# (target site has no x offset from the cup). Re-derive from the XML if
# it ever changes -- this was checked numerically against
# physics.ball_to_target(), not assumed.
CUP_TO_TARGET_Z_OFFSET = 0.35


@dataclasses.dataclass
class CEMConfig:
    H: int = 150            # horizon (steps) -- PLACEHOLDER, see module docstring
    n_samples: int = 500    # population size N
    elite_frac: float = 0.1
    n_iters: int = 6
    init_std: float = 0.5
    min_std: float = 0.05   # floor so sigma never collapses to exactly 0
    action_low: float = -1.0
    action_high: float = 1.0
    lam: float = 1.0        # weight on the "don't drift upward/away" term
    mu: float = 0.0         # weight on the time-to-success tie-break (off by default)
    dt: float = CONTROL_TIMESTEP


def load_position_probe(path=POSITION_PROBE_PATH):
    """W: (D+1, 4) affine map, [z | 1] @ W -> [cup_x, cup_z, ball_x, ball_z].
    Run eval_linear_probe.py first if this file doesn't exist yet."""
    data = np.load(path, allow_pickle=True)
    return torch.from_numpy(data["W"].astype(np.float32))


def load_position_probe_mlp(path):
    """Loads a nonlinear (MLP) position probe saved by
    eval_mlp_probe_strided.py -- returns an nn.Module, not a Tensor.
    decode_position() dispatches on type, so this is a drop-in
    replacement for load_position_probe()'s linear W anywhere W is used
    (rollout_cost, plan_step, etc. never need to know which kind they
    have). Imports eval_mlp_probe_strided lazily so callers who only
    ever use the linear probe don't need that module to exist."""
    import eval_mlp_probe_strided as embp

    ckpt = torch.load(path, map_location="cpu")
    latent_dim, hidden = ckpt["latent_dim"], ckpt["hidden"]
    model = embp.MLPProbe(
        latent_dim, hidden,
        z_mean=torch.zeros(latent_dim), z_std=torch.ones(latent_dim),
        y_mean=torch.zeros(4), y_std=torch.ones(4),
    )
    model.load_state_dict(ckpt["state_dict"])   # overwrites the dummy
                                                  # mean/std buffers above
                                                  # with the real fitted ones
    model.eval()
    return model


def decode_position(z, W):
    """z: (..., D) -> (..., 4) predicted [cup_x, cup_z, ball_x, ball_z].
    Works on any leading shape (B, H, D) or (B, D).

    W is either:
      - a Tensor (D+1, 4): the linear/affine probe (load_position_probe),
        decoded as [z | 1] @ W -- W already includes the bias row (see
        eval_linear_probe.py's fit_and_report, which fits against
        A = [X | 1]).
      - an nn.Module: a nonlinear probe (load_position_probe_mlp),
        called directly as W(z) -- it does its own input/output
        normalization internally (see eval_mlp_probe_strided.py).

    Callers (rollout_cost, plan_step) never need to know which kind W
    is -- this dispatch is the only place that cares."""
    if isinstance(W, torch.nn.Module):
        return W(z)
    ones = torch.ones(*z.shape[:-1], 1, dtype=z.dtype, device=z.device)
    z_aug = torch.cat([z, ones], dim=-1)
    return z_aug @ W


def rollout_cost(P, window, actions, W, cfg):
    """Scores a batch of candidate action sequences. Lower is better.

    window:  (1, K_CONTEXT, D) or (B, K_CONTEXT, D) -- current real state,
             as returned by jepa_evaluator.bootstrap_window.
    actions: (B, H, ACTION_DIM)
    Returns: (costs (B,), z_preds (B, H, D)) -- z_preds returned too, for
    inspection/debugging (e.g. checking predicted trajectories by eye).
    """
    B = actions.shape[0]

    # Decode the *current* real position too (from the last real z in the
    # window) -- purely from E/P/probe, no privileged ground truth -- so
    # tau=0's velocity has something to finite-difference against.
    pos_now = decode_position(window[:, -1, :], W)           # (B_win, 4)
    if pos_now.shape[0] == 1 and B > 1:
        pos_now = pos_now.expand(B, -1)

    z_preds = je.rollout(P, window, actions)                  # (B, H, D)
    pos_pred = decode_position(z_preds, W)                    # (B, H, 4)

    cup_x, cup_z = pos_pred[..., 0], pos_pred[..., 1]
    ball_x, ball_z = pos_pred[..., 2], pos_pred[..., 3]

    dx = cup_x - ball_x
    dz = (cup_z + CUP_TO_TARGET_Z_OFFSET) - ball_z
    dist_sq = dx**2 + dz**2                                   # (B, H)

    # v_rel = ball_z_vel - cup_z_vel, via finite differences of the
    # *decoded* position sequence -- see module docstring for why this
    # instead of the window-state probe.
    ball_z_full = torch.cat([pos_now[:, 3:4], ball_z], dim=1)    # (B, H+1)
    cup_z_full = torch.cat([pos_now[:, 1:2], cup_z], dim=1)      # (B, H+1)
    ball_z_vel = (ball_z_full[:, 1:] - ball_z_full[:, :-1]) / cfg.dt
    cup_z_vel = (cup_z_full[:, 1:] - cup_z_full[:, :-1]) / cfg.dt
    v_rel = ball_z_vel - cup_z_vel                             # (B, H)

    per_step_cost = dist_sq + cfg.lam * torch.clamp(v_rel, min=0.0) ** 2

    costs, tau_star = per_step_cost.min(dim=1)                 # (B,), (B,)
    if cfg.mu > 0:
        costs = costs + cfg.mu * (tau_star.float() ** 2)

    return costs, z_preds


def shift_warm_elites(elites):
    """Shifts a (M, H, ACTION_DIM) elite-sequence set forward by one
    P-call, for use as `warm_elites` at the NEXT real decision point:
    drops the first action (the one about to be/already executed) and
    pads the freed last slot by repeating the sequence's own last action
    (a "coast" assumption -- a placeholder default, not a claim it's
    optimal, same spirit as every other placeholder in this module)."""
    return torch.cat([elites[:, 1:, :], elites[:, -1:, :]], dim=1)


def plan_step(P, window, W, cfg, prev_mean=None, warm_elites=None):
    """One full CEM solve (all cfg.n_iters iterations) from a single real
    starting window. Returns (first_action, mean, std, info).

    first_action is the first action of the single BEST candidate seen
    across ALL iterations (info["best_actions"]/info["best_cost"]), not
    just whatever the final iteration's mean happens to be -- plain CEM
    (no running-best tracking) can regress between iterations purely from
    resampling variance, since each iteration draws a fresh population
    and nothing carries the best-found candidate forward (this is exactly
    what cost_history fluctuating non-monotonically was showing). Fixed
    two ways: (1) elitism -- the best candidate so far is injected as one
    guaranteed member of every iteration's population, so it can't be
    lost to bad luck in resampling; (2) the returned action comes from
    the tracked best, not the final population/mean, so even an unlucky
    last iteration doesn't corrupt the result.

    warm_elites: optional (M, H, ACTION_DIM) tensor, injected into
    iteration 0's population only (seeding the search, not force-fed
    every iteration -- after iteration 0, best_actions elitism takes
    over as usual). This is the shift-forward warm-start: a caller
    solving successive real decision points passes in the PREVIOUS
    decision point's info["pooled_elites"], shifted one P-call forward
    via shift_warm_elites(). `mean` still starts at zero (or prev_mean)
    and `std` always resets to cfg.init_std regardless of warm_elites --
    only these guaranteed-injected candidates carry real memory across
    decision points, so exploration is never at risk of collapsing.

    info["pooled_elites"] is built by keeping each iteration's OWN top
    slice (~n_elite/n_iters candidates), not by pooling every iteration
    together and re-ranking by cost. That distinction matters: a global
    rank across iterations would just re-select the final (most
    converged) iteration's tail again, since later iterations generally
    score better -- defeating the point. Keeping a slice from EACH
    iteration guarantees the carried-forward set spans the whole
    exploration-to-exploitation range of the solve (iteration 1, sampled
    at std=init_std, is still genuinely spread out even though its raw
    cost is worse) instead of handing the next decision point 50 near-
    identical copies of one converged point."""
    device = window.device
    B = cfg.n_samples

    mean = prev_mean.clone() if prev_mean is not None else torch.zeros(cfg.H, jt.ACTION_DIM, device=device)
    std = torch.full((cfg.H, jt.ACTION_DIM), cfg.init_std, device=device)

    n_elite = max(1, int(round(cfg.n_samples * cfg.elite_frac)))
    n_per_iter = max(1, n_elite // cfg.n_iters)
    info = {"cost_history": [], "best_cost_history": []}
    best_actions, best_cost = None, float("inf")
    pooled_elites = []   # one top-n_per_iter slice per iteration -- see docstring

    for it in range(cfg.n_iters):
        noise = torch.randn(B, cfg.H, jt.ACTION_DIM, device=device)
        actions = (mean.unsqueeze(0) + noise * std.unsqueeze(0)).clamp(cfg.action_low, cfg.action_high)
        if best_actions is not None:
            actions[0] = best_actions   # elitism -- guarantee survival, see docstring
        if it == 0 and warm_elites is not None:
            n_warm = min(warm_elites.shape[0], B - 1)
            actions[1:1 + n_warm] = warm_elites[:n_warm]   # seed iteration 0 only, see docstring

        costs, _ = rollout_cost(P, window, actions, W, cfg)
        info["cost_history"].append(costs.min().item())

        iter_best_idx = costs.argmin()
        if costs[iter_best_idx].item() < best_cost:
            best_cost = costs[iter_best_idx].item()
            best_actions = actions[iter_best_idx].clone()
        info["best_cost_history"].append(best_cost)

        elite_idx = torch.topk(costs, n_elite, largest=False).indices
        elites = actions[elite_idx]                           # (n_elite, H, ACTION_DIM)
        pooled_elites.append(elites[:n_per_iter])              # this iteration's own top slice, see docstring

        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp(min=cfg.min_std)

    info["best_actions"] = best_actions
    info["best_cost"] = best_cost
    info["pooled_elites"] = torch.cat(pooled_elites, dim=0)   # (n_iters * n_per_iter, H, ACTION_DIM)
    return best_actions[0], mean, std, info


def _smoke_test():
    """Runs one plan_step against the real checkpoint + one real eval
    episode's opening frames -- confirms shapes/sanity end-to-end, NOT a
    claim the resulting action is actually good (cost function is still
    a sketch, H is still a placeholder -- see module docstring)."""
    device = torch.device("cpu")
    E, P, k_context, latent_dim = je.load_world_model(device=device)
    W = load_position_probe()

    paths = sorted(pathlib.Path("jepa_eval_data").glob("episode_*.npz"))
    if not paths:
        print("No eval episodes found -- run collect_eval_data.py first.")
        return
    data = np.load(paths[0])
    frames = data["frames"]
    window = je.bootstrap_window(E, frames[:k_context], device)

    cfg = CEMConfig(H=30, n_samples=64, n_iters=3)  # small, just for a fast smoke test
    first_action, mean, std, info = plan_step(P, window, W, cfg)

    print(f"cost per iteration (best in population): {info['cost_history']}")
    print(f"best cost so far, per iteration (never regresses): {info['best_cost_history']}")
    print(f"first action: {first_action.tolist()}")
    print(f"mean shape: {tuple(mean.shape)}, std shape: {tuple(std.shape)}")


if __name__ == "__main__":
    _smoke_test()
