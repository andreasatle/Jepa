"""
pstep_distribution_mismatch_check.py

Maze JEPA -- Phase 3 diagnostic, follow-up to pstep_unroll_position_loss_check.py's
result: that script's per-unroll-position loss, measured on 4000 REAL
transitions sampled from maze_data.json (collect_data.py's training data),
stayed smooth and modest across all 20 unroll positions (hyperbolic:
0.23 -> 0.94, a gentle ~4x rise, nothing like the 10-30x catastrophic
blowups step_drift_diagnostic.py and pstep_manifold_drift_check.py saw on
their 5 hand-picked long pairs). Same checkpoint, same unroll depth, wildly
different outcome -- so the difference must be WHAT KIND of action sequence
is being unrolled, not how long it is.

collect_data.py confirms the suspicion directly: every training episode is
env.random_rollout(...) -- actions sampled UNIFORMLY AMONG VALID MOVES,
not steered toward any target. A perfect maze is a tree, so backtracking
under a random policy is common (any move that isn't the unique next step
toward a far-off point is, from the tree's perspective, either sideways or
backward). MAX_HORIZON=20 in collect_data.py, exactly matching
jepa_train_step.py's UNROLL_STEPS=20, so P_step's unrolled training has
literally only ever seen 20-step RANDOM-walk windows.

The 5 pairs step_drift_diagnostic.py/pstep_manifold_drift_check.py use are
the opposite: bfs_path_cells's actual shortest path, i.e. the unique
NON-BACKTRACKING, monotonically-progressing action sequence toward a
specific far-off target -- exactly the kind of sequence the CEM planner
needs P_step to handle, and exactly the kind collect_data.py's random
rollouts essentially never produce over 15+ consecutive steps. If that
mismatch (not chain length, not geometry) is the real cause of the
blowup, then a RANDOM rollout of the SAME length, from the SAME start
cell, should behave like pstep_unroll_position_loss_check.py's smooth
result -- while the BFS-directed path to a specific target, same start,
same length, blows up. This script runs that direct comparison.

For each of the same 5 (start, target, bfs_distance=d) pairs used by
step_drift_diagnostic.py (same DIAG_SEED/N_PAIRS/MIN_PAIR_DISTANCE, so the
start cells and path lengths line up exactly), generates a RANDOM valid
rollout of length d from that same start cell (env.random_rollout, same
policy collect_data.py used, separate ROLLOUT_SEED) and runs the identical
teacher-forced-vs-autoregressive comparison against it. Directly
comparable, step for step, chain-length for chain-length, against that
script's already-reported numbers for the BFS-directed path from the same
start.

Run (after jepa_train.py and jepa_train_step.py have both been run for the
geometry being checked):
    uv run python pstep_distribution_mismatch_check.py --geometry hyperbolic
    uv run python pstep_distribution_mismatch_check.py --geometry euclidean
"""
import argparse
import random

import torch

from env import ACTIONS, ALGO, GRID_SIZE, MazeEnv
from maze_gen import GENERATORS, bfs_distance
from jepa_model import (
    cell_index,
    load_encoder_checkpoint,
    load_step_checkpoint,
    squared_dist,
    to_geometry,
)

MAZE_SEED = 0            # same fixed maze as everywhere else in this project
DIAG_SEED = 3             # same as step_drift_diagnostic.py -- same 5 start
N_PAIRS = 5               # cells and path lengths, so rows line up directly
MIN_PAIR_DISTANCE = 15    # against that script's BFS-directed numbers.
ROLLOUT_SEED = 11         # separate rng for the random-rollout actions

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic"], required=True)
    args = parser.parse_args()
    geometry = args.geometry

    E, e_ckpt = load_encoder_checkpoint(geometry)
    P_step, s_ckpt = load_step_checkpoint(geometry)
    assert e_ckpt["size"] == s_ckpt["size"], "E and P_step checkpoints are for different maze sizes"
    size = e_ckpt["size"]

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())
    rng = random.Random(DIAG_SEED)

    pairs = []
    while len(pairs) < N_PAIRS:
        start, target = rng.sample(cells, 2)
        d = bfs_distance(adj, start, target)
        if d >= MIN_PAIR_DISTANCE:
            pairs.append((start, target, d))

    print(f"[{geometry}] {len(pairs)} pairs, same start cells/lengths as step_drift_diagnostic.py, "
          f"BUT following a RANDOM valid rollout of the same length instead of the BFS-optimal path")

    env = MazeEnv(size=size, algo=ALGO, seed=None)
    env.adj = adj  # reuse the exact same maze instance, not a freshly generated one
    env.rng = random.Random(ROLLOUT_SEED)

    with torch.no_grad():
        for start, target, d in pairs:
            states, actions = env.random_rollout(d, start=start, valid_only=True)
            end = states[-1]
            end_bfs_from_target = bfs_distance(adj, end, target)

            true_idx = torch.tensor([cell_index(c, size) for c in states])
            true_raw = E(true_idx)

            auto_raw = true_raw[0:1]
            print(f"\n  start={start} original_target={target} bfs_distance={d} "
                  f"(random rollout instead lands at {end}, "
                  f"which is {end_bfs_from_target} BFS steps from the original target)")
            print(f"  {'step':>4}  {'teacher-forced':>15}  {'autoregressive':>15}")
            for t, a in enumerate(actions):
                action_idx = torch.tensor([ACTION_INDEX[a]])
                target_next = to_geometry(true_raw[t + 1:t + 2], geometry)

                tf_pred = P_step(true_raw[t:t + 1], action_idx)
                tf_err = squared_dist(to_geometry(tf_pred, geometry), target_next, geometry).item()

                auto_raw = P_step(auto_raw, action_idx)
                auto_err = squared_dist(to_geometry(auto_raw, geometry), target_next, geometry).item()

                print(f"  {t + 1:>4}  {tf_err:>15.4f}  {auto_err:>15.4f}")


if __name__ == "__main__":
    main()
