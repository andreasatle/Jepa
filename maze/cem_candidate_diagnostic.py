"""
cem_candidate_diagnostic.py

Maze JEPA -- Phase 3, cheap diagnostic (no retraining) to distinguish
the two remaining candidate explanations for why the planner stays at
0/40 despite P_step's measurably improved accuracy (see WORKPLAN.md's
"Balanced-sampling fix result" section):

  (a) CEM's search itself can't concentrate on the one correct discrete
      path even when given a reasonable cost signal (already partly
      supported by the oracle-CEM ablation capping at 20% under a
      PERFECT cost signal -- but that was never checked against what
      CEM actually does with the REAL, imperfect P_step cost signal).
  (b) P_step's cost signal, even though its own diagnostic-pair
      accuracy improved, is being asked something different during real
      CEM search than what it was measured on: cem_plan chains P_step
      through many DIFFERENT candidate sequences every iteration, most
      of them wrong/partial, not just the one BFS-optimal or
      hand-authored-directed sequence the diagnostics test.

This script reconstructs the EXACT same 40 (start, target) trial pairs
maze_planner.py's run_trials evaluates (same MAZE_SEED, PLANNER_SEED,
N_TRIALS, MIN_TRIAL_DISTANCE, same per-trial trial_seed sequence), reuses
cem_plan directly (identical N_SAMPLES/ELITE_FRAC/N_ITERS/MIN_PROB, so
this is exactly what the real planner run already did, not an
approximation), and for the first N_INSPECT FAILING trials prints, step
by step along the chosen plan:

  - the chosen action, and whether it still matches the unique true
    BFS-optimal path from that start (a maze is a tree, so there is
    exactly one correct path -- once the chosen plan takes one wrong
    turn it can never rejoin it)
  - the REAL cell the agent is actually in after executing that many
    steps of the chosen plan, and the REAL BFS leftover distance from
    there to target
  - CEM's PREDICTED cost (squared latent distance to target) after that
    many steps of the SAME chosen sequence, rolled through P_step
    exactly as cem_plan itself computed it internally

Reading the table answers the (a)-vs-(b) question directly: if the
chosen plan diverges from the correct path almost immediately (common
prefix length ~0-1) on most trials, CEM's search isn't even finding the
right initial direction -- a search problem, independent of P_step's
remaining noise. If predicted cost keeps dropping (or reads low) well
after the real trajectory has already diverged onto a wrong branch,
that's P_step's cost signal actively misleading the search once off the
one correct path -- a representation/cost-signal problem, and evidence
for option (b) (P_step's accuracy under CEM's OWN candidate distribution,
not the hand-authored diagnostic distribution, is what actually matters).
If instead the plan tracks the correct path for a long common prefix and
predicted cost rises in step with real leftover distance once it
diverges, that points at (a): P_step's local signal is fine, CEM's
search mechanism itself just isn't able to correct once early elites
commit to the wrong branch (consistent with the oracle-CEM 20% ceiling).

Run (after jepa_train.py and jepa_train_step.py have both been run for
the geometry being checked):
    uv run python cem_candidate_diagnostic.py --geometry hyperbolic
    uv run python cem_candidate_diagnostic.py --geometry euclidean
"""
import argparse
import random

import torch

from env import ACTION_DELTA, ALGO, GRID_SIZE, MazeEnv
from maze_gen import GENERATORS, bfs_distance, bfs_path_cells
from maze_planner import (
    HORIZON_MARGIN,
    MAZE_SEED,
    MIN_TRIAL_DISTANCE,
    N_TRIALS,
    PLANNER_SEED,
    ACTION_INDEX,
    cem_plan,
    load_models,
)
from jepa_model import cell_index, squared_dist, to_geometry

N_INSPECT = 5  # how many FAILING trials to print in full detail

DELTA_TO_ACTION = {v: k for k, v in ACTION_DELTA.items()}


def path_to_actions(path):
    actions = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        actions.append(DELTA_TO_ACTION[(r1 - r0, c1 - c0)])
    return actions


def common_prefix_len(actions, optimal_actions):
    n = 0
    for a, b in zip(actions, optimal_actions):
        if a != b:
            break
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic"], required=True)
    args = parser.parse_args()
    geometry = args.geometry

    E, P_step, size = load_models(geometry)
    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())
    env = MazeEnv(size=size, algo=ALGO, seed=MAZE_SEED)

    # reconstruct the exact same 40 trial pairs and trial_seeds run_trials uses
    rng = random.Random(PLANNER_SEED)
    trials = []
    trial_seed = 0
    while len(trials) < N_TRIALS:
        start, target = rng.sample(cells, 2)
        d = bfs_distance(adj, start, target)
        if d < MIN_TRIAL_DISTANCE:
            continue
        trials.append((start, target, d, trial_seed))
        trial_seed += 1

    n_success = 0
    prefix_lens = []
    inspected = 0

    for start, target, d, tseed in trials:
        horizon = d + HORIZON_MARGIN
        actions, _ = cem_plan(E, P_step, geometry, start, target, size, horizon, tseed)

        # execute the FULL chosen plan for real, step by step, no early
        # break -- if it never reaches target, real_cells ends up with
        # exactly len(actions)+1 entries, one per step including start
        env.reset(start=start)
        real_cells = [env.state]
        reached = env.state == target
        for a in actions:
            if reached:
                break
            env.step(a)
            real_cells.append(env.state)
            reached = env.state == target

        if reached:
            n_success += 1
            continue  # only inspecting failures in detail

        optimal_path = bfs_path_cells(adj, start, target)
        optimal_actions = path_to_actions(optimal_path)
        prefix = common_prefix_len(actions, optimal_actions)
        prefix_lens.append((prefix, d))

        if inspected >= N_INSPECT:
            continue
        inspected += 1

        with torch.no_grad():
            start_idx = torch.tensor([cell_index(start, size)])
            target_idx = torch.tensor([cell_index(target, size)])
            raw = E(start_idx)
            target_mapped = to_geometry(E(target_idx), geometry)
            predicted_costs = []
            for a in actions:
                action_idx = torch.tensor([ACTION_INDEX[a]])
                raw = P_step(raw, action_idx)
                cost = squared_dist(to_geometry(raw, geometry), target_mapped, geometry).item()
                predicted_costs.append(cost)

        final_leftover = bfs_distance(adj, real_cells[-1], target)
        print(f"\n  start={start} target={target} bfs_distance={d} horizon={horizon} "
              f"common_prefix_with_optimal={prefix}/{d} final_real_leftover={final_leftover}")
        print(f"  {'step':>4}  {'action':>6}  {'on_optimal?':>11}  {'real_cell':>10}  "
              f"{'real_leftover':>13}  {'predicted_cost':>14}")
        for t, a in enumerate(actions):
            on_opt = "yes" if t < prefix else ""
            real_leftover = bfs_distance(adj, real_cells[t + 1], target)
            print(f"  {t + 1:>4}  {a:>6}  {on_opt:>11}  {str(real_cells[t + 1]):>10}  "
                  f"{real_leftover:>13}  {predicted_costs[t]:>14.4f}")

    n_fail = len(trials) - n_success
    print(f"\n[{geometry}] {n_success}/{len(trials)} succeeded, {n_fail} failed. "
          f"Detailed above: first {inspected} failures.")
    if prefix_lens:
        fracs = [p / d for p, d in prefix_lens]
        mean_frac = sum(fracs) / len(fracs)
        zero_or_one = sum(1 for p, _ in prefix_lens if p <= 1)
        print(f"[{geometry}] common-prefix-with-optimal, as a fraction of bfs_distance, "
              f"across all {len(prefix_lens)} failing trials: mean={mean_frac:.2f}  "
              f"{zero_or_one}/{len(prefix_lens)} diverged within the first 1 action.")


if __name__ == "__main__":
    main()
