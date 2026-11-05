"""
CEM planner for the STRIDED world model -- same CEM/MPC algorithm as
cem_planner.py, but each of the H "steps" here is one call to the strided
P, spanning PREDICTOR_STRIDE real control-steps (action held constant
across them) instead of 1. This is what drift_diagnostic_strided.py
validated: ball position error stays flat (~0.08-0.11m, not growing)
across the full 30-call / 150-real-step / 3.0s horizon tested -- so H can
now actually be set to span a full pump-and-catch attempt, instead of
being a placeholder capped by untrustworthy drift.

Reuses cem_planner.py's decode_position and CUP_TO_TARGET_Z_OFFSET
directly (pure geometry/probe-application, independent of stride) and
jepa_evaluator's rollout/bootstrap_window/load_world_model (also
stride-agnostic -- rollout just calls P once per action in the sequence,
however much real time each one represents).

What's genuinely different from cem_planner.py:
  - dt for the v_rel finite-difference is PREDICTOR_STRIDE *
    CONTROL_TIMESTEP (0.1s), not 0.02s -- consecutive predicted steps are
    PREDICTOR_STRIDE real steps apart now.
  - Loads jepa_checkpoint_strided.pt and position_probe_strided.npz by
    default, not the fine-grained model/probe -- these are NOT
    interchangeable (different E weights).
  - H defaults to 30 (matching the exact span drift_diagnostic_strided.py
    validated -- 30 calls * stride 5 = 150 real steps = 3.0s), not an
    arbitrary placeholder. Still worth re-checking against a longer
    drift_diagnostic_strided.py run before raising it further.
  - Deploying a plan from this planner means holding first_action constant
    for PREDICTOR_STRIDE real env steps before observing and replanning,
    not for 1 -- MPC still applies, just at the coarser cadence.

Run as a script for a smoke test against the real strided checkpoint (no
env needed -- only jepa_evaluator + the strided probe):
    python3 cem_planner_strided.py
"""

import dataclasses
import pathlib

import numpy as np
import torch

import cem_planner as cp
import jepa_evaluator as je
import jepa_train as jt

CHECKPOINT_PATH = "jepa_checkpoint_strided.pt"
POSITION_PROBE_PATH = "position_probe_strided_mlp.pt"   # nonlinear (MLP)
                                                          # probe -- see
                                                          # eval_mlp_probe_strided.py.
                                                          # Was position_probe_strided.npz
                                                          # (linear); swapped after the MLP
                                                          # probe cut mean ball L2 error from
                                                          # 0.0827m to 0.0067m on the same
                                                          # single-frame z->position task.
PREDICTOR_STRIDE = 5      # must match jepa_train_strided.py / collect_data_strided.py


@dataclasses.dataclass
class CEMConfig:
    H: int = 30              # P-calls -- 30*PREDICTOR_STRIDE=150 real steps
                              # (3.0s), the exact span drift_diagnostic_strided.py
                              # validated as flat/trustworthy. Re-validate
                              # before raising further.
    n_samples: int = 500
    elite_frac: float = 0.1
    n_iters: int = 6
    init_std: float = 0.5
    min_std: float = 0.05
    action_low: float = -1.0
    action_high: float = 1.0
    lam: float = 1.0
    mu: float = 0.0
    dt: float = PREDICTOR_STRIDE * cp.CONTROL_TIMESTEP   # 0.1s -- real time
                                                            # spanned by ONE
                                                            # P call here, vs
                                                            # cem_planner.py's
                                                            # 0.02s


def rollout_cost(P, window, actions, W, cfg):
    """Identical logic to cem_planner.rollout_cost -- reuses
    cp.decode_position and cp.CUP_TO_TARGET_Z_OFFSET directly, since
    neither depends on stride. Only cfg.dt differs (via CEMConfig above)."""
    B = actions.shape[0]

    pos_now = cp.decode_position(window[:, -1, :], W)
    if pos_now.shape[0] == 1 and B > 1:
        pos_now = pos_now.expand(B, -1)

    z_preds = je.rollout(P, window, actions)
    pos_pred = cp.decode_position(z_preds, W)

    cup_x, cup_z = pos_pred[..., 0], pos_pred[..., 1]
    ball_x, ball_z = pos_pred[..., 2], pos_pred[..., 3]

    dx = cup_x - ball_x
    dz = (cup_z + cp.CUP_TO_TARGET_Z_OFFSET) - ball_z
    dist_sq = dx**2 + dz**2

    ball_z_full = torch.cat([pos_now[:, 3:4], ball_z], dim=1)
    cup_z_full = torch.cat([pos_now[:, 1:2], cup_z], dim=1)
    ball_z_vel = (ball_z_full[:, 1:] - ball_z_full[:, :-1]) / cfg.dt
    cup_z_vel = (cup_z_full[:, 1:] - cup_z_full[:, :-1]) / cfg.dt
    v_rel = ball_z_vel - cup_z_vel

    per_step_cost = dist_sq + cfg.lam * torch.clamp(v_rel, min=0.0) ** 2

    costs, tau_star = per_step_cost.min(dim=1)
    if cfg.mu > 0:
        costs = costs + cfg.mu * (tau_star.float() ** 2)

    return costs, z_preds


def plan_step(P, window, W, cfg, prev_mean=None, warm_elites=None):
    """Same running-best-candidate fix as cem_planner.plan_step (see that
    module for the full explanation): plain per-iteration-best tracking
    can regress between iterations from pure resampling variance, since
    nothing carries the best-found candidate forward. Fixed via elitism
    (best candidate injected into every population) plus returning the
    tracked best instead of the final population/mean.

    warm_elites / info["pooled_elites"]: shift-forward warm start across
    REAL decision points, identical mechanism and reasoning to
    cem_planner.plan_step -- see that module's docstring for the full
    explanation of why elites are kept per-iteration (stratified) rather
    than pooled-then-reranked. Use cem_planner.shift_warm_elites() to
    shift info["pooled_elites"] forward one P-call before passing it as
    warm_elites into the next real decision point's call."""
    device = window.device
    B = cfg.n_samples

    mean = prev_mean.clone() if prev_mean is not None else torch.zeros(cfg.H, jt.ACTION_DIM, device=device)
    std = torch.full((cfg.H, jt.ACTION_DIM), cfg.init_std, device=device)

    n_elite = max(1, int(round(cfg.n_samples * cfg.elite_frac)))
    n_per_iter = max(1, n_elite // cfg.n_iters)
    info = {"cost_history": [], "best_cost_history": []}
    best_actions, best_cost = None, float("inf")
    pooled_elites = []

    for it in range(cfg.n_iters):
        noise = torch.randn(B, cfg.H, jt.ACTION_DIM, device=device)
        actions = (mean.unsqueeze(0) + noise * std.unsqueeze(0)).clamp(cfg.action_low, cfg.action_high)
        if best_actions is not None:
            actions[0] = best_actions   # elitism -- guarantee survival
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
        elites = actions[elite_idx]
        pooled_elites.append(elites[:n_per_iter])

        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp(min=cfg.min_std)

    info["best_actions"] = best_actions
    info["best_cost"] = best_cost
    info["pooled_elites"] = torch.cat(pooled_elites, dim=0)
    return best_actions[0], mean, std, info


def _smoke_test():
    """Runs one FULL plan_step (H=30, the real validated horizon) against
    the real strided checkpoint + a real strided eval episode's opening
    frames. Confirms shapes/sanity end-to-end -- not a claim the result
    is a good ball-in-cup solution (cost function is still a sketch)."""
    device = torch.device("cpu")
    E, P, k_context, latent_dim = je.load_world_model(CHECKPOINT_PATH, device=device)
    W = cp.load_position_probe_mlp(POSITION_PROBE_PATH)

    paths = sorted(pathlib.Path("jepa_eval_data_strided").glob("episode_*.npz"))
    if not paths:
        print("No strided eval episodes found -- run collect_eval_data_strided.py first.")
        return
    data = np.load(paths[0])
    frames = data["frames"]
    window = je.bootstrap_window(E, frames[:k_context], device)

    cfg = CEMConfig()   # full H=30 -- no reason to shrink it for a smoke
                          # test anymore, that's the whole point of stride
    first_action, mean, std, info = plan_step(P, window, W, cfg)

    print(f"cost per iteration (best in population): {info['cost_history']}")
    print(f"best cost so far, per iteration (never regresses): {info['best_cost_history']}")
    print(f"first action: {first_action.tolist()}  (hold for {PREDICTOR_STRIDE} real steps)")
    print(f"mean shape: {tuple(mean.shape)}, std shape: {tuple(std.shape)}")


if __name__ == "__main__":
    _smoke_test()
