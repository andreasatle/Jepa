"""
Checks whether the hyperbolic embedding's radius (distance from the
origin) already encodes something like predictive confidence -- the
"radial hierarchy" idea from the conversation: more uncertain / less
trustworthy predictions (deeper autoregressive rollouts, or a known-
weaker stride like s16) should sit CLOSER to the origin than confident
ones, as a free-riding consequence of exp0's geometry, if the
hypothesis holds. Proposed twice before, never built until now.

No explicit hyp.exp0/hyp.dist call is needed to check this: hyperbolic.py's
own self-check already proves dist(0, exp0(v)) == 2*|v| for any raw
vector v (see hyperbolic.py's `if __name__` block, checks 3/4). So the
RAW Euclidean norm of a predictor's output z (before exp0 is ever
applied -- P's output is never actually projected into the ball except
transiently inside the training loss; the context window and every
z_pred used elsewhere in this project is the raw vector) is already
exactly proportional to its hyperbolic distance from the origin. No
extra geometry required, just ||z||.

For each stride, rolls out P over the held-out jepa_eval_data/ episodes
(same rollouts drift_diagnostic_refrozen.py scores), and for every
predicted step records both ||z_pred|| (radius proxy) and the decoded
ball position error at that step. Reports:
  - mean radius vs. rollout depth per stride (does it shrink as
    autoregressive error should compound?)
  - the per-step correlation between radius and ball position error
    across EVERY individual (rollout, step) pair, not just the
    aggregated means -- the real bar for radius being a usable
    per-candidate CEM confidence signal, since CEM would need to trust
    it candidate-by-candidate, not just as a nice-looking trend on
    average
  - mean radius per stride (does s16, the known-weaker predictor from
    drift_diagnostic_refrozen.py, sit at systematically lower radius
    than s1?)

Uses the REFROZEN checkpoints (predictor_stride{s}_refrozen.pt) -- the
current best predictors -- via drift_diagnostic_joint's own loader
functions with PREDICTOR_CHECKPOINT_SUFFIX overridden, same reuse
pattern as drift_diagnostic_refrozen.py.

Run:
    uv run python hyperbolic_radius_diagnostic.py
"""

import numpy as np
import torch

import cem_planner as cp
import drift_diagnostic_joint as ddj
import jepa_evaluator as je

ddj.PREDICTOR_CHECKPOINT_SUFFIX = "_refrozen"

STRIDE_LEVELS = ddj.STRIDE_LEVELS
TARGET_HORIZON_STEPS = ddj.TARGET_HORIZON_STEPS


def radius_vs_error_for_stride(E, episodes, stride, device):
    max_p_calls = TARGET_HORIZON_STEPS // stride
    dt_per_call = stride * cp.CONTROL_TIMESTEP

    batch = ddj.build_rollout_batch(E, episodes, stride, max_p_calls, device)
    if batch is None:
        print(f"stride={stride}: no episode is long enough for {max_p_calls} calls -- skipping.")
        return None
    window, action_seq, true_pos = batch
    B = window.shape[0]

    P = ddj.load_stride_predictor(stride, device)
    W = cp.load_position_probe_mlp(ddj.POSITION_PROBE_PATH)

    with torch.no_grad():
        z_preds = je.rollout(P, window, action_seq)          # (B, H, D)
        pred_pos = cp.decode_position(z_preds, W)             # (B, H, 4)

    radius = z_preds.norm(dim=-1)                              # (B, H) -- raw ||z||,
                                                                  # proportional to
                                                                  # hyperbolic dist(0, exp0(z))
                                                                  # (see module docstring)
    err = (pred_pos - true_pos).abs()
    ball_err = (err[..., 2] ** 2 + err[..., 3] ** 2).sqrt()     # (B, H)

    print(f"\n=== stride={stride} ({B} rollouts, {max_p_calls} P-calls) ===")
    print(f"{'n_calls':>8s} {'real s':>8s} {'mean radius':>12s} {'ball L2 err (m)':>18s}")
    checkpoints = sorted(set([0, max_p_calls // 4, max_p_calls // 2, (3 * max_p_calls) // 4, max_p_calls - 1]))
    for idx in checkpoints:
        n_calls = idx + 1
        real_s = n_calls * dt_per_call
        print(f"{n_calls:8d} {real_s:8.2f} {radius[:, idx].mean().item():12.4f} {ball_err[:, idx].mean().item():18.4f}")

    # Per-step (not just per-mean) correlation across ALL (rollout, step)
    # pairs -- see module docstring for why this is the real bar, not the
    # aggregated table above.
    r_flat = radius.reshape(-1).numpy()
    e_flat = ball_err.reshape(-1).numpy()
    corr = float(np.corrcoef(r_flat, e_flat)[0, 1])
    print(
        f"stride={stride}: radius-vs-ball_err correlation across all {r_flat.size} "
        f"(rollout, step) pairs: {corr:.3f}  "
        "(negative = supports the hypothesis: lower radius, higher error)"
    )

    return radius.mean().item(), corr


def main():
    device = torch.device("cpu")
    E = ddj.load_encoder(device)

    paths = sorted(ddj.EVAL_DATA_DIR.glob("episode_*.npz"))
    if not paths:
        print(f"No eval episodes found in {ddj.EVAL_DATA_DIR}/.")
        return
    episodes = []
    for p in paths:
        d = np.load(p)
        episodes.append((d["frames"], d["actions"], d["positions"], d["velocities"]))
    print(f"Loaded {len(episodes)} held-out episodes from {ddj.EVAL_DATA_DIR}/.")

    summary = []
    for stride in STRIDE_LEVELS:
        result = radius_vs_error_for_stride(E, episodes, stride, device)
        if result is not None:
            summary.append((stride, *result))

    print("\n=== summary: mean radius per stride ===")
    print(f"{'stride':>8s} {'mean radius':>12s} {'corr(radius, err)':>20s}")
    for stride, mean_radius, corr in summary:
        print(f"{stride:8d} {mean_radius:12.4f} {corr:20.3f}")


if __name__ == "__main__":
    main()
