"""
collect_dagger_data.py

Maze JEPA -- Phase 3, DAgger-style on-policy data collection for P_step,
replacing the earlier hand-authored approximation (collect_directed_data.py's
bias_prob-weighted rollouts toward BFS-optimal) with the real thing.
Motivation: that earlier fix trained P_step on a STATIC guess at what
CEM's candidates look like mid-search. DAgger (Ross/Gordon/Bagnell 2011)
replaces the guess with reality: run the CURRENT planner for real, see
exactly which action sequences it actually commits to, get the TRUE
outcome of each from the real simulator (never from any model's own
prediction -- always ground truth, so this can't be circular the way
training on a model's own guesses would be), and add those
(state, action, true-next-state) transitions to P_step's training data.
Retrain, replan with the improved P_step, collect again -- each round's
on-policy data reflects whatever mistakes THAT round's planner actually
makes. This naturally rediscovers and corrects the wall-bump blind spot
cem_candidate_diagnostic.py found too, without any special-casing: CEM
really does sample invalid actions sometimes, and this script just
executes and records exactly what it chooses, wall bumps included.

This script is ONE ROUND: collect, using whichever jepa_step_checkpoint
is currently on disk for the given geometry. Workflow: run this, retrain
(jepa_train_step.py), evaluate (maze_planner.py), then run this again to
collect a fresh round against the newly-retrained checkpoint. Each run
APPENDS to the existing maze_data_dagger_<geometry>.json rather than
overwriting it, so rounds accumulate -- the "Aggregation" in DAgger.

Geometry-specific output, unlike maze_data.json / maze_data_directed.json
(pure maze physics, independent of geometry): DAgger data depends on
what THIS geometry's P_step actually gets wrong during real planning, so
hyperbolic and euclidean each get their own file and their own rounds.

DAGGER_SEED kept separate from every other seed in the project
(collect_data.py's DATA_SEED=1, collect_directed_data.py's
DIRECTED_DATA_SEED=21, the diagnostics' DIAG_SEED=3, maze_planner.py's
PLANNER_SEED=2) so a planner improvement after this still means
generalization, not memorization of anything evaluated on. Advances by
round number each run, so successive rounds sample different pairs
while staying fully reproducible.

DAGGER_MIN_DISTANCE=20 (matches jepa_train_step.py's UNROLL_STEPS, not
imported directly to avoid a needless import-time coupling -- checked,
both are 20): every collected episode needs to be at least this long to
contribute even one training window to build_transitions' sliding-window
cut. Deliberately NOT maze_planner.py's much smaller MIN_TRIAL_DISTANCE=5,
which is tuned for eval realism/diversity, not training-data yield.

Episodes execute the CHOSEN plan's FULL horizon length for real, even
past the point (if any) where the agent reaches target early -- same
choice collect_directed_data.py made: later windows in a long episode
are still valid, useful supervision regardless of whether they're
"before" or "after" the nominal target was reached.

Run (after jepa_train.py and jepa_train_step.py have both produced a
checkpoint for the geometry being collected against):
    uv run python collect_dagger_data.py --geometry hyperbolic
    uv run python collect_dagger_data.py --geometry euclidean
"""
import argparse
import json
import os
import random

from env import ALGO, GRID_SIZE, MazeEnv
from maze_gen import GENERATORS, bfs_distance
from maze_planner import HORIZON_MARGIN, MAZE_SEED, cem_plan, load_models

DAGGER_SEED = 37          # separate from every other seed in the project -- see
                           # module docstring's DAGGER_SEED section
DAGGER_MIN_DISTANCE = 20  # matches UNROLL_STEPS -- see module docstring
N_DAGGER_TRIALS = 300     # meaningfully more than the 40-trial eval set; cheap since
                           # this is pure inference (no gradient) through a small MLP


def output_path(geometry):
    return f"maze_data_dagger_{geometry}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic"], required=True)
    parser.add_argument("--encoder-checkpoint", default=None,
                         help="Override E's checkpoint path template, e.g. "
                              "jepa_checkpoint_treeloss_{geometry}.pt.")
    parser.add_argument("--step-checkpoint", default=None,
                         help="Override P_step's checkpoint path template, matching "
                              "whichever E it was trained against.")
    parser.add_argument("--output-path", default=None,
                         help="Override the output DAgger data file path template (must "
                              "contain {geometry}). Defaults to maze_data_dagger_"
                              "{geometry}.json. IMPORTANT: use a different template (e.g. "
                              "maze_data_dagger_treeloss_{geometry}.json) whenever "
                              "--encoder-checkpoint/--step-checkpoint point at a "
                              "non-standard pair -- rounds collected on-policy against one "
                              "E/P_step pair are NOT valid on-policy data for a different "
                              "one, and jepa_train_step.py's --use-dagger loads the whole "
                              "file with no way to select only some rounds.")
    args = parser.parse_args()
    geometry = args.geometry

    E, P_step, size = load_models(geometry, args.encoder_checkpoint, args.step_checkpoint)
    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())
    env = MazeEnv(size=size, algo=ALGO, seed=MAZE_SEED)

    path = (args.output_path or "maze_data_dagger_{geometry}.json").format(geometry=geometry)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        round_num = len(data["rounds"]) + 1
    else:
        data = {
            "grid_size": GRID_SIZE,
            "algo": ALGO,
            "maze_seed": MAZE_SEED,
            "geometry": geometry,
            "rounds": [],
            "episodes": [],
        }
        round_num = 1

    rng = random.Random(DAGGER_SEED + round_num)  # advances per round -- fresh pairs
                                                     # each round, still reproducible

    new_episodes = []
    trial_seed = 0
    attempts = 0
    while len(new_episodes) < N_DAGGER_TRIALS:
        attempts += 1
        start, target = rng.sample(cells, 2)
        d = bfs_distance(adj, start, target)
        if d < DAGGER_MIN_DISTANCE:
            continue
        horizon = d + HORIZON_MARGIN
        actions, _ = cem_plan(E, P_step, geometry, start, target, size, horizon, trial_seed)
        trial_seed += 1

        # execute the CHOSEN plan for real, full horizon, wall bumps and all --
        # this is exactly what makes it "on-policy": the real outcome of what
        # the current planner actually decided to do, not a hand-authored guess
        env.reset(start=start)
        states = [env.state]
        for a in actions:
            env.step(a)
            states.append(env.state)

        new_episodes.append({
            "start": list(start),
            "actions": actions,
            "states": [list(s) for s in states],
            "target": list(target),
            "bfs_distance": d,
            "dagger_round": round_num,
        })

    data["episodes"].extend(new_episodes)
    data["rounds"].append({"round": round_num, "n_episodes": len(new_episodes)})

    with open(path, "w") as f:
        json.dump(data, f)

    total_episodes = len(data["episodes"])
    print(f"[{geometry}] DAgger round {round_num}: collected {len(new_episodes)} new on-policy "
          f"episodes ({attempts} pairs sampled, {attempts - len(new_episodes)} skipped for "
          f"bfs_distance < {DAGGER_MIN_DISTANCE}), lengths "
          f"{min(len(e['actions']) for e in new_episodes)}-"
          f"{max(len(e['actions']) for e in new_episodes)}")
    print(f"[{geometry}] {path} now has {total_episodes} episodes across {round_num} round(s)")


if __name__ == "__main__":
    main()
