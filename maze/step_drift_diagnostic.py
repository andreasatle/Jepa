"""
step_drift_diagnostic.py

Maze JEPA -- Phase 3 diagnostic: does P_step's autoregressive rollout
error explode with chain length? Tests the exposure-bias hypothesis
documented in jepa_train_step.py's module docstring, before assuming
it and rebuilding P_step's training around multi-step unrolling.

For a handful of (start, target) pairs, takes the TRUE BFS-optimal
action sequence (via maze_gen.bfs_path_cells -- the actual correct
path, not anything the planner or CEM produces) and chains P_step
along it two ways, at every step:

  - "teacher-forced" (one-step) error: predict from the REAL E(cell)
    at that step -- this is exactly the training regime (see
    jepa_train_step.py), so this error should stay low and roughly
    flat as step count grows, if the one-step model itself is sound.
  - "autoregressive" (free-running) error: predict from P_step's OWN
    previous prediction, never touching the real E again after step 0
    -- this is what maze_planner.py's CEM actually does when it rolls
    candidates forward in latent space. If this error grows sharply
    with step count while the teacher-forced error stays flat, that's
    exposure bias: P_step was never trained on its own (increasingly
    off-distribution) predictions as input, so small per-step errors
    compound instead of correcting -- directly analogous to the
    failure ball_in_cup_hyperbolic's jepa_train.py already documented
    and fixed via multi-step unrolled training.

Both errors are squared_dist (geometry-appropriate) between the
predicted latent and E(true_cell_at_that_step), so they're directly
comparable to each other, across step count, and across geometries.

Run (in Andreas's terminal, after jepa_train.py AND jepa_train_step.py
have both been run for the geometry being checked):
    uv run python step_drift_diagnostic.py --geometry euclidean
    uv run python step_drift_diagnostic.py --geometry hyperbolic
"""
import argparse
import random

import torch

from env import ACTIONS, ACTION_DELTA, ALGO, GRID_SIZE
from maze_gen import GENERATORS, bfs_distance, bfs_path_cells
from jepa_model import (
    cell_index,
    load_encoder_checkpoint,
    load_step_checkpoint,
    squared_dist,
    to_geometry,
)

MAZE_SEED = 0            # same fixed maze as everywhere else in this project
DIAG_SEED = 3             # separate from PLANNER_SEED -- own fixed set of pairs
N_PAIRS = 5
MIN_PAIR_DISTANCE = 15   # want long chains -- this is specifically testing drift over length

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
DELTA_TO_ACTION = {v: k for k, v in ACTION_DELTA.items()}


def path_to_actions(path):
    """Cell sequence -> action sequence, via the inverse of env.py's
    ACTION_DELTA. path is a BFS tree path so every consecutive pair is
    a real edge; this recovers the unique action that edge corresponds
    to."""
    actions = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        actions.append(DELTA_TO_ACTION[(r1 - r0, c1 - c0)])
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic"], required=True)
    parser.add_argument("--encoder-checkpoint", default=None,
                         help="Override E's checkpoint path template, e.g. "
                              "jepa_checkpoint_treeloss_{geometry}.pt.")
    parser.add_argument("--step-checkpoint", default=None,
                         help="Override P_step's checkpoint path template, matching "
                              "whichever E it was trained against.")
    args = parser.parse_args()
    geometry = args.geometry

    e_kwargs = {"path": args.encoder_checkpoint} if args.encoder_checkpoint else {}
    s_kwargs = {"path": args.step_checkpoint} if args.step_checkpoint else {}
    E, e_ckpt = load_encoder_checkpoint(geometry, **e_kwargs)
    P_step, s_ckpt = load_step_checkpoint(geometry, **s_kwargs)
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

    print(f"[{geometry}] {len(pairs)} pairs, BFS distances: {[d for _, _, d in pairs]}")

    with torch.no_grad():
        for start, target, d in pairs:
            path = bfs_path_cells(adj, start, target)
            actions = path_to_actions(path)
            assert len(actions) == d

            true_idx = torch.tensor([cell_index(c, size) for c in path])
            true_raw = E(true_idx)  # (d+1, dim) raw embeddings along the true path

            auto_raw = true_raw[0:1]  # (1, dim) -- starts real, then free-running
            print(f"\n  start={start} target={target} bfs_distance={d}")
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
