"""
oracle_receding_horizon_check.py

Maze JEPA -- cheap diagnostic testing whether CEM's OPEN-LOOP planning is
itself responsible for the independently-measured 20% oracle-CEM ceiling
(see maze_planner.py's oracle-CEM ablation, "ORACLE ABLATION" section).

Finding that prompted this: maze_planner.py's run_trials/run_oracle_trials
call cem_plan/oracle_cem_plan exactly ONCE per trial to get a full
horizon-length (~20-50 step) action sequence, then execute_plan() runs
that ENTIRE sequence for real with zero replanning and zero use of the
real intermediate state -- despite this project (and the ball_in_cup_
hyperbolic project it's a sequel to) being framed as a "CEM/MPC" planner
throughout. Classic MPC always replans at every step using the real
resulting state; this codebase never did. In a strict tree maze with
zero tolerance for a wrong turn (see WORKPLAN.md's "Is the tree structure
too rigid" discussion -- no partial credit for wrong turns), a single bad
early branch in a long open-loop plan guarantees failure for the rest of
the trial, with no mechanism to recover -- a very plausible explanation
for why even a PERFECT oracle cost signal only reaches 20%.

This script tests the fix cheaply first (no torch/E/P_step at all, pure
graph search -- same "cheap diagnostic before committing to a bigger fix"
practice as cem_candidate_diagnostic.py): receding-horizon oracle-CEM.
At every real step: call maze_planner.py's OWN oracle_cem_plan unchanged
(reused directly, not reimplemented, so this is guaranteed identical CEM
behavior -- same N_SAMPLES/ELITE_FRAC/N_ITERS/MIN_PROB) from the CURRENT
real cell with horizon = current bfs_distance-to-target + HORIZON_MARGIN,
but only commit to and execute the FIRST action of the returned sequence
for real; then replan from the new real cell. A fresh rng_seed per replan
call (base trial seed + step count) keeps this fully reproducible without
literally repeating the same random draws every step. Repeats until the
target is reached or a per-trial step budget is exhausted (max_steps_for,
generous relative to bfs_distance, just a safety cap against pathological
looping -- not expected to bind if replanning is actually helping).

Same 40 trials as maze_planner.py (identical PLANNER_SEED/MAZE_SEED/
MIN_TRIAL_DISTANCE/N_TRIALS, regenerated here the same way run_oracle_
trials does it), so success_rate here is directly comparable to the
existing oracle-CEM's 20% baseline.

Run:
    uv run python oracle_receding_horizon_check.py
"""
import random
import time

from env import ALGO, GRID_SIZE
from maze_gen import GENERATORS, bfs_distance
from maze_planner import (
    HORIZON_MARGIN,
    MAZE_SEED,
    MIN_TRIAL_DISTANCE,
    N_TRIALS,
    PLANNER_SEED,
    oracle_cem_plan,
    oracle_step,
)


def max_steps_for(d):
    """Generous per-trial safety cap, scaled to bfs_distance rather than
    a flat constant -- keeps worst-case cost bounded across the wide
    range of d values in the 40 trials without prematurely truncating a
    genuinely-longer recovery."""
    return d * 3 + 20


def receding_horizon_oracle_plan(adj, start, target, base_seed):
    """Runs oracle-CEM in a closed loop: replan (via maze_planner.py's
    own oracle_cem_plan, unmodified) from the REAL cell at every step,
    execute only the first action, repeat. Returns (success, steps_taken).
    """
    cell = start
    d0 = None
    for step in range(10_000):  # outer safety net; real cap enforced below
        if cell == target:
            return True, step
        d = bfs_distance(adj, cell, target)
        if d0 is None:
            d0 = d
        if step >= max_steps_for(d0):
            return False, step
        horizon = d + HORIZON_MARGIN
        actions, _ = oracle_cem_plan(adj, cell, target, horizon, base_seed + step)
        cell = oracle_step(adj, cell, actions[0])
    return cell == target, step


def main():
    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())

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
    steps_over_bfs = []
    t0 = time.time()
    for i, (start, target, d, tseed) in enumerate(trials):
        success, steps = receding_horizon_oracle_plan(adj, start, target, tseed * 1000)
        if success:
            n_success += 1
            steps_over_bfs.append(steps / d)
        print(f"  trial {i}: d={d} success={success} steps={steps}  "
              f"[{time.time()-t0:.1f}s elapsed]")

    print(f"receding-horizon oracle-CEM: success_rate={n_success/len(trials):.2f} "
          f"({n_success}/{len(trials)})")
    if steps_over_bfs:
        print(f"  mean actual/bfs-optimal steps among successes="
              f"{sum(steps_over_bfs)/len(steps_over_bfs):.2f}")
    print(f"  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
