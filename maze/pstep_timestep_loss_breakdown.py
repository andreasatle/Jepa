"""
pstep_timestep_loss_breakdown.py

Maze JEPA -- diagnostic for a hypothesis raised while investigating why
the treeloss E fix (rho 0.14 -> 0.9988, see WORKPLAN.md) barely moved
branch_point_diagnostic.py's one-step decision accuracy (69.3% -> 70.8%
overall) and left P_step's own one-step error 2.3-2.5x the true edge
length (pstep_onestep_error_check.py) -- unaffected by LR or hidden-width
changes (both cleanly ruled out, see WORKPLAN.md).

Hypothesis: jepa_train_step.py's loss is an UNWEIGHTED mean over
UNROLL_STEPS=20 per-timestep squared-distance terms. Later timesteps in a
20-step unroll are naturally farther from the start (bigger true
distance -> bigger squared distance, even for a perfect predictor), so if
those terms are numerically much larger than the early ones, the
unweighted mean could be dominated by long-horizon terms, starving
short-horizon precision -- the same "unweighted average across wildly
different scales" pattern jepa_train_treeloss.py had before --dist-scale
fixed it, one level up (across unroll steps instead of across pairs).

This loads an already-trained P_step + E (no retraining) and reports,
per timestep t=0..19: the actual loss contribution at t, and the
"natural scale" at t (squared distance from the real start to the real
target at that unroll position, i.e. what a working predictor would need
to represent) -- if loss/natural_scale is roughly FLAT across t, the
unweighted average isn't the problem (each t contributes proportionally
to its own scale). If it climbs sharply with t, the hypothesis holds and
per-timestep weighting (or truncating UNROLL_STEPS) is the next thing to
try.

Run:
    uv run python pstep_timestep_loss_breakdown.py --geometry hyperbolic \\
        --encoder-checkpoint jepa_checkpoint_treeloss_hyperbolic.pt \\
        --step-checkpoint jepa_step_checkpoint_treeloss_hyperbolic.pt
"""
import argparse

import torch

from jepa_model import squared_dist, to_geometry
from jepa_train_step import (
    DAGGER_DATA_PATH,
    DATA_PATH,
    DIRECTED_DATA_PATH,
    UNROLL_STEPS,
    _load_pool,
)
from maze_planner import load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--encoder-checkpoint", default=None)
    ap.add_argument("--step-checkpoint", default=None)
    ap.add_argument("--sample-size", type=int, default=2000,
                     help="total examples to sample across pools for this breakdown "
                          "(no gradient, one pass -- larger just means a more stable "
                          "per-t average).")
    args = ap.parse_args()
    geometry = args.geometry

    E, P_step, size = load_models(geometry, args.encoder_checkpoint, args.step_checkpoint)

    pools = []
    pools.append(("random-walk", _load_pool(DATA_PATH, size, "random-walk (maze_data.json)")))
    pools.append(("directed", _load_pool(DIRECTED_DATA_PATH, size, "directed (maze_data_directed.json)")))
    import os
    dagger_path = DAGGER_DATA_PATH.format(geometry=geometry)
    if os.path.exists(dagger_path):
        pools.append(("dagger", _load_pool(dagger_path, size, f"dagger ({dagger_path})")))

    pool_sizes = [len(idx_tensors[0]) for _, idx_tensors in pools]
    n_sources = len(pools)
    per_source = args.sample_size // n_sources

    with torch.no_grad():
        start_parts, action_parts, target_parts = [], [], []
        for (label, (start_idx, action_idx, target_idx)), n in zip(pools, pool_sizes):
            b = min(per_source, n)
            batch = torch.randint(0, n, (b,))
            start_parts.append(start_idx[batch])
            action_parts.append(action_idx[batch])
            target_parts.append(target_idx[batch])
        d_start_idx = torch.cat(start_parts)
        d_action_idx = torch.cat(action_parts)
        d_target_idx = torch.cat(target_parts)
        d_n = d_start_idx.shape[0]

        raw = E(d_start_idx)
        start_mapped = to_geometry(E(d_start_idx), geometry)
        target_raw_full = E(d_target_idx.reshape(-1)).reshape(d_n, UNROLL_STEPS, -1)

        print(f"[{geometry}] {d_n} examples, per-timestep breakdown:")
        print(f"  {'t':>3}  {'loss':>10}  {'natural_scale':>14}  {'loss/scale':>10}")
        for t in range(UNROLL_STEPS):
            pred_raw = P_step(raw, d_action_idx[:, t])
            pred_mapped = to_geometry(pred_raw, geometry)
            target_mapped = to_geometry(target_raw_full[:, t], geometry)
            step_loss_t = squared_dist(pred_mapped, target_mapped, geometry).mean().item()
            natural_scale_t = squared_dist(start_mapped, target_mapped, geometry).mean().item()
            ratio = step_loss_t / natural_scale_t if natural_scale_t > 0 else float("nan")
            print(f"  {t:>3}  {step_loss_t:>10.4f}  {natural_scale_t:>14.4f}  {ratio:>10.3f}")
            raw = pred_raw


if __name__ == "__main__":
    main()
