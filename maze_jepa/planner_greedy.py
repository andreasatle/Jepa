"""
planner_greedy.py

The simplest possible planner: from the current cell, try every VALID
action (env.valid_actions(cell) -- see the DELIBERATE SIMPLIFICATION
note below), predict the resulting embedding for each, and move to
whichever is closest to the target cell's TRUE embedding. Repeat until
reaching the target or hitting --max-steps.

This tests exactly one thing: is "closer to target in embedding space"
(squared_dist to E(target)) a reliable one-step-at-a-time heuristic for
making real progress toward the target? That's a DIFFERENT question
from the one-step transition accuracy already verified in
checks/jepa_test.py (100% for both geometries, confirming the predicted
NEXT cell is identified correctly). Nothing in training ever asked E's
distances to reflect maze/graph distance -- 2 of its dims are literally
raw (row, col) grid position (see jepa_model.py's Encoder), which walls
can make a poor proxy for actual path distance. This is the first test
of that separate, previously-untested property.

DELIBERATE SIMPLIFICATION: this planner is handed env.valid_actions(cell)
to restrict its candidates to genuinely walkable moves, rather than
blindly trying all 4 raw actions like a fully "real" planner eventually
would. That's intentional: Predictor was trained ONLY on valid moves (no
wall-bump pairs at all -- see jepa_train.py's module docstring), so its
behavior on an actual wall direction is untested/undefined. Isolating
that already-known issue lets this test answer only the question it's
meant to answer, rather than conflating it with the blind-action problem
the old project's cem_candidate_diagnostic.py already diagnosed once.

Run (from inside maze_jepa/):
    uv run python planner_greedy.py --geometry hyperbolic --start 0,0 --target 9,9
    uv run python planner_greedy.py --geometry hyperbolic --trials 40
"""
import argparse
import random
from collections import deque

import torch

from env import MazeEnv
from jepa_model import (
    cell_index,
    load_encoder_checkpoint,
    load_step_checkpoint,
    squared_dist,
    to_geometry,
)


def greedy_plan(env, E, P, geometry, start, target, max_steps):
    size = env.size
    target_idx = torch.tensor([cell_index(target, size)])
    with torch.no_grad():
        target_raw = E(target_idx)
        target_geo = to_geometry(target_raw, geometry)

        current = start
        path = [current]
        actions_taken = []
        for _ in range(max_steps):
            if current == target:
                return path, actions_taken, True

            options = env.valid_actions(current)
            s_idx = torch.tensor([cell_index(current, size)])
            start_raw = E(s_idx)

            best_action = best_dist = best_next = None
            for a in options:
                a_delta = torch.tensor([a.value], dtype=torch.float32)
                pred_raw = P(start_raw, a_delta)
                pred_geo = to_geometry(pred_raw, geometry)
                d = squared_dist(pred_geo, target_geo, geometry).item()
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_action = a
                    r, c = current
                    dr, dc = a.value
                    best_next = (r + dr, c + dc)

            current = best_next
            actions_taken.append(best_action.name)
            path.append(current)

        return path, actions_taken, current == target


def bfs_dist(env, start):
    """True shortest-path distances from start, for comparison."""
    dist = {start: 0}
    q = deque([start])
    while q:
        cell = q.popleft()
        for n in env.adj[cell]:
            if n not in dist:
                dist[n] = dist[cell] + 1
                q.append(n)
    return dist


def parse_cell(s):
    r, c = s.split(",")
    return (int(r), int(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["euclidean", "hyperbolic"], default="hyperbolic")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--algo", default="prim")
    ap.add_argument("--maze-seed", type=int, default=0)
    ap.add_argument("--start", default=None, help="row,col -- single-trial mode")
    ap.add_argument("--target", default=None, help="row,col -- single-trial mode")
    ap.add_argument("--trials", type=int, default=40, help="random trials when --start/--target not given")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = MazeEnv(size=args.size, algo=args.algo, seed=args.maze_seed)
    E, _ = load_encoder_checkpoint(args.geometry)
    P, _ = load_step_checkpoint(args.geometry)
    cells = list(env.adj.keys())
    rng = random.Random(args.seed)

    if args.start and args.target:
        pairs = [(parse_cell(args.start), parse_cell(args.target))]
        verbose = True
    else:
        pairs = [(rng.choice(cells), rng.choice(cells)) for _ in range(args.trials)]
        pairs = [(s, t) for s, t in pairs if s != t]
        verbose = False

    successes = 0
    ratios = []
    for start, target in pairs:
        true_len = bfs_dist(env, start).get(target)
        path, actions, success = greedy_plan(env, E, P, args.geometry, start, target, args.max_steps)
        steps = len(actions)
        if success:
            successes += 1
            ratios.append(steps / true_len if true_len else 1.0)
        if verbose:
            print(f"start={start} target={target} geometry={args.geometry}")
            print(f"true shortest path length: {true_len}")
            print(f"success={success}, steps_taken={steps}")
            print(f"path: {path}")
            print(f"actions: {actions}")
        else:
            print(f"  {start}->{target}: true={true_len} steps={steps} success={success}")

    n = len(pairs)
    print(f"\nsuccess_rate: {successes}/{n} = {100 * successes / n:.1f}%")
    if ratios:
        mean_ratio = sum(ratios) / len(ratios)
        print(f"mean steps/optimal ratio (successful trials only): {mean_ratio:.3f}")


if __name__ == "__main__":
    main()
