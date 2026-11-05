"""
Adaptive coarse-to-fine CEM planner for the JOINT multi-stride world model.

Same CEM/MPC algorithm as cem_planner.py / cem_planner_strided.py, but
instead of committing to one fixed stride for the whole solve, each
iteration picks ONE of the five jointly-trained predictors P_1..P_16
(jepa_train_joint.py) to evaluate the full horizon with -- starting at
the coarsest (stride 16, cheap: H_real/16 P-calls per candidate) and
halving toward stride 1 only when a cross-resolution check says the
coarse ranking can no longer be trusted.

Why this is possible without any resampling/interpolation logic: at
stride s, one P_s call consumes a BLOCK of s consecutive real actions
concatenated together (jt.Predictor(..., s*ACTION_DIM, ...) -- see
jepa_train_joint.py / drift_diagnostic_joint.py's build_rollout_batch,
which does exactly actions[t0+i*s : t0+(i+1)*s].reshape(-1)). Since
every stride here is a power of 2, the action-sequence parameterization
(mean, std, and every sampled candidate) is kept at FULL real per-step
resolution -- (H_real, ACTION_DIM) -- for the entire solve, and is only
ever reshaped to (H_real//s, s*ACTION_DIM) right before a P_s call.
Coarse "blocks" are literally contiguous concatenations of the true
per-step actions, so reshaping is exact and lossless in both directions
-- no resampling, no interpolation, nothing invented.

The halving trigger -- cross-resolution ranking-agreement check:
after each iteration at stride s>1, take the top RANK_CHECK_K of that
iteration's own elites (already known, no extra P_s calls needed to get
them), re-evaluate that SAME small subset at stride s//2 (a small number
of genuinely fresh P_{s//2} calls -- not a recombination of already-
computed numbers), and compare the top RANK_CHECK_TOPN under s against
the top RANK_CHECK_TOPN under s//2. Full agreement (identical sets) ->
stride s is still trustworthy, keep iterating there. Any disagreement ->
stride s can no longer be trusted, halve to s//2 for the next iteration.

Explicitly NOT done here, after discussion:
  - No Richardson-style combination of the two resolutions' costs into
    a "better" estimate -- would require a known error-order relationship
    between two INDEPENDENTLY TRAINED predictors, which we have no basis
    for. When the check disagrees, the finer predictor's own cost is used
    as-is on the next iteration, not blended with the coarser one.
  - No hard cap on iterations spent at a given stride level, and no
    forced final stride-1 refinement pass -- cfg.n_iters is still just
    the total iteration budget for the whole solve (same role it plays
    in cem_planner.py/cem_planner_strided.py), and this is receding-
    horizon MPC (replanned every real control step), so an occasional
    plan that never made it down to stride 1, or that was fooled by a
    shared blind spot in E, gets corrected on the next replan rather
    than needing to be prevented here. Revisit both if real runs show
    it's actually a problem.

Uses the JOINT fine-tuned checkpoints (jepa_checkpoint_joint.pt,
predictor_stride{s}_joint.pt, position_probe_joint_mlp.pt) -- the same
ones drift_diagnostic_joint.py validated, NOT the frozen-encoder ones.

Deploys at the FINE (real-step) cadence, unlike cem_planner_strided.py:
mean/std/actions are real-resolution throughout, so first_action should
be held for exactly one real control step before replanning, same as
cem_planner.py.

Run as a script for a smoke test against the real joint checkpoints (no
env needed -- only jepa_evaluator + the joint MLP probe):
    python3 cem_planner_multistride.py
"""

import dataclasses
import pathlib

import numpy as np
import torch

import cem_planner as cp
import jepa_evaluator as je
import jepa_train as jt

ENCODER_CHECKPOINT = "jepa_checkpoint_joint.pt"
POSITION_PROBE_PATH = "position_probe_joint_mlp.pt"
PREDICTOR_CHECKPOINT_SUFFIX = "_joint"   # predictor_stride{s}_joint.pt

STRIDE_LEVELS = [1, 2, 4, 8, 16]   # must match jepa_train_joint.py


@dataclasses.dataclass
class CEMConfig:
    H_real: int = 480        # real control steps in the horizon -- must be
                              # divisible by 16 (the coarsest stride). 480 =
                              # 30 stride-16 P-calls = 9.6s, comfortably past
                              # the 150-step/3s span drift_diagnostic_joint.py
                              # has actually validated -- shrink toward that
                              # if real runs show it's untrustworthy further
                              # out. Same "still a placeholder" status as H
                              # in cem_planner.py/cem_planner_strided.py.
    n_samples: int = 500
    elite_frac: float = 0.1
    n_iters: int = 6          # total iteration budget for the WHOLE solve,
                              # same role as in cem_planner.py -- NOT a
                              # per-stride-level cap (see module docstring).
    init_std: float = 0.5
    min_std: float = 0.05
    action_low: float = -1.0
    action_high: float = 1.0
    lam: float = 1.0
    mu: float = 0.0
    rank_check_k: int = 10    # size of the subset re-evaluated at the next-
                              # finer stride for the halving check
    rank_check_topn: int = 5  # size of the top-N set compared for agreement


def load_joint_encoder(device):
    ckpt = torch.load(ENCODER_CHECKPOINT, map_location=device)
    E = jt.Encoder(3, jt.LATENT_DIM).to(device)
    E.load_state_dict(ckpt["E"])
    E.eval()
    return E


def load_joint_predictors(device):
    """Returns {stride: P}, one entry per STRIDE_LEVELS -- the jointly
    fine-tuned predictors, same checkpoints drift_diagnostic_joint.py uses."""
    predictors = {}
    for s in STRIDE_LEVELS:
        ckpt = torch.load(f"predictor_stride{s}{PREDICTOR_CHECKPOINT_SUFFIX}.pt", map_location=device)
        P = jt.Predictor(jt.LATENT_DIM, s * jt.ACTION_DIM, jt.PREDICTOR_HIDDEN, jt.K_CONTEXT).to(device)
        P.load_state_dict(ckpt["P"])
        P.eval()
        predictors[s] = P
    return predictors


def to_blocks(actions_real, stride):
    """(B, H_real, ACTION_DIM) -> (B, H_real//stride, stride*ACTION_DIM),
    the exact block-concatenation convention P_stride was trained on (see
    module docstring). Exact and lossless -- no interpolation, since coarse
    blocks ARE contiguous concatenations of the real per-step actions."""
    B, H_real, A = actions_real.shape
    assert H_real % stride == 0, f"H_real={H_real} not divisible by stride={stride}"
    return actions_real.reshape(B, H_real // stride, stride * A)


def rollout_cost(P, window, actions_real, stride, W, lam, mu):
    """Same dist_sq + v_rel cost as cem_planner_strided.rollout_cost,
    generalized to take the stride explicitly (dt = stride *
    CONTROL_TIMESTEP) instead of baking it into a fixed cfg. Reuses
    cp.decode_position / cp.CUP_TO_TARGET_Z_OFFSET, since neither depends
    on stride.

    actions_real: (B, H_real, ACTION_DIM) -- reshaped to blocks internally,
    so callers never juggle block shapes themselves."""
    dt = stride * cp.CONTROL_TIMESTEP
    actions_block = to_blocks(actions_real, stride)
    B = actions_block.shape[0]

    pos_now = cp.decode_position(window[:, -1, :], W)
    if pos_now.shape[0] == 1 and B > 1:
        pos_now = pos_now.expand(B, -1)

    z_preds = je.rollout(P, window, actions_block)
    pos_pred = cp.decode_position(z_preds, W)

    cup_x, cup_z = pos_pred[..., 0], pos_pred[..., 1]
    ball_x, ball_z = pos_pred[..., 2], pos_pred[..., 3]

    dx = cup_x - ball_x
    dz = (cup_z + cp.CUP_TO_TARGET_Z_OFFSET) - ball_z
    dist_sq = dx**2 + dz**2

    ball_z_full = torch.cat([pos_now[:, 3:4], ball_z], dim=1)
    cup_z_full = torch.cat([pos_now[:, 1:2], cup_z], dim=1)
    ball_z_vel = (ball_z_full[:, 1:] - ball_z_full[:, :-1]) / dt
    cup_z_vel = (cup_z_full[:, 1:] - cup_z_full[:, :-1]) / dt
    v_rel = ball_z_vel - cup_z_vel

    per_step_cost = dist_sq + lam * torch.clamp(v_rel, min=0.0) ** 2

    costs, tau_star = per_step_cost.min(dim=1)
    if mu > 0:
        costs = costs + mu * (tau_star.float() ** 2)

    return costs, z_preds


def ranking_agrees(P_fine, window, elites_real, stride_coarse, W, lam, mu, cfg):
    """The halving trigger. elites_real: (n_elite, H_real, ACTION_DIM),
    already ranked by stride_coarse cost (best first) from the caller --
    torch.topk(..., sorted=True) is the default, so this holds without
    re-sorting. Re-evaluates the top rank_check_k of them at
    stride_coarse//2 and checks whether the top rank_check_topn set
    matches.

    Returns True (agree, stay coarse) or False (disagree, halve)."""
    k = min(cfg.rank_check_k, elites_real.shape[0])
    subset = elites_real[:k]   # already coarse-best-first -- this IS the
                                # coarse top-k ranking, no re-ranking needed

    fine_costs, _ = rollout_cost(P_fine, window, subset, stride_coarse // 2, W, lam, mu)
    fine_order = torch.argsort(fine_costs)

    n = min(cfg.rank_check_topn, k)
    coarse_top = set(range(n))                       # indices 0..n-1 by construction
    fine_top = set(fine_order[:n].tolist())
    return coarse_top == fine_top


def plan_step(predictors, window, W, cfg, prev_mean=None, warm_elites=None):
    """One full adaptive CEM solve, starting at the coarsest stride and
    halving via ranking_agrees(). Returns (first_action, mean, std, info) --
    same contract as cem_planner.plan_step/cem_planner_strided.plan_step:
    first_action is the first REAL action of the single best candidate seen
    across the whole solve (elitism + running-best, same reasoning as those
    modules -- see cem_planner.plan_step's docstring for why that's needed
    over just trusting the final iteration/mean).

    warm_elites / info["pooled_elites"]: same shift-forward warm start as
    the other two planners, and reusable as-is (cp.shift_warm_elites) since
    everything here is already real-resolution -- see cem_planner.py's
    plan_step docstring for the full explanation of the mechanism."""
    device = window.device
    B = cfg.n_samples
    H = cfg.H_real
    A = jt.ACTION_DIM

    mean = prev_mean.clone() if prev_mean is not None else torch.zeros(H, A, device=device)
    std = torch.full((H, A), cfg.init_std, device=device)

    n_elite = max(1, int(round(cfg.n_samples * cfg.elite_frac)))
    n_per_iter = max(1, n_elite // cfg.n_iters)
    info = {"cost_history": [], "best_cost_history": [], "stride_history": []}
    best_actions, best_cost = None, float("inf")
    pooled_elites = []

    stride = STRIDE_LEVELS[-1]   # start coarsest (16)

    for it in range(cfg.n_iters):
        P = predictors[stride]
        info["stride_history"].append(stride)

        noise = torch.randn(B, H, A, device=device)
        actions = (mean.unsqueeze(0) + noise * std.unsqueeze(0)).clamp(cfg.action_low, cfg.action_high)
        if best_actions is not None:
            actions[0] = best_actions   # elitism, real-resolution -- no reshape needed
        if it == 0 and warm_elites is not None:
            n_warm = min(warm_elites.shape[0], B - 1)
            actions[1:1 + n_warm] = warm_elites[:n_warm]

        costs, _ = rollout_cost(P, window, actions, stride, W, cfg.lam, cfg.mu)
        info["cost_history"].append(costs.min().item())

        iter_best_idx = costs.argmin()
        if costs[iter_best_idx].item() < best_cost:
            best_cost = costs[iter_best_idx].item()
            best_actions = actions[iter_best_idx].clone()
        info["best_cost_history"].append(best_cost)

        elite_idx = torch.topk(costs, n_elite, largest=False).indices
        elites = actions[elite_idx]                # (n_elite, H, A), best-first
        pooled_elites.append(elites[:n_per_iter])

        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp(min=cfg.min_std)

        if stride > 1:
            P_fine = predictors[stride // 2]
            if not ranking_agrees(P_fine, window, elites, stride, W, cfg.lam, cfg.mu, cfg):
                stride = stride // 2

    info["best_actions"] = best_actions
    info["best_cost"] = best_cost
    info["pooled_elites"] = torch.cat(pooled_elites, dim=0)
    return best_actions[0], mean, std, info


def _smoke_test():
    """Runs one full adaptive plan_step against the real joint checkpoints +
    one real eval episode's opening frames. Confirms shapes/sanity end-to-
    end and prints the stride schedule the ranking check actually chose --
    NOT a claim the resulting action is good (cost function is still the
    same sketch as the other planners)."""
    device = torch.device("cpu")
    E = load_joint_encoder(device)
    predictors = load_joint_predictors(device)
    W = cp.load_position_probe_mlp(POSITION_PROBE_PATH)

    paths = sorted(pathlib.Path("jepa_eval_data").glob("episode_*.npz"))
    if not paths:
        print("No eval episodes found -- run collect_eval_data.py first.")
        return
    data = np.load(paths[0])
    frames = data["frames"]
    window = je.bootstrap_window(E, frames[:jt.K_CONTEXT], device)

    cfg = CEMConfig(H_real=160, n_samples=64, n_iters=5)  # div by 16, small pop for a fast smoke test
    first_action, mean, std, info = plan_step(predictors, window, W, cfg)

    print(f"stride per iteration: {info['stride_history']}")
    print(f"cost per iteration (best in population): {info['cost_history']}")
    print(f"best cost so far, per iteration (never regresses): {info['best_cost_history']}")
    print(f"first action: {first_action.tolist()}  (hold for 1 real step)")
    print(f"mean shape: {tuple(mean.shape)}, std shape: {tuple(std.shape)}")


if __name__ == "__main__":
    _smoke_test()
