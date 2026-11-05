"""
collect_data.py

Maze JEPA -- Phase 1: collect a rollout dataset for training the Phase 2
encoder/predictor comparison. See WORKPLAN.md.

ONE FIXED MAZE for the whole dataset, not a new maze per rollout -- Phase
0's synthetic test measured "does a predictor represent a FIXED set of
k target modes well," so the maze-domain version needs a fixed, well-
defined target distribution per (start cell, horizon) too. Mixing many
different maze topologies together would also require the encoder to
somehow condition on "which maze," which is out of scope -- the geometry
question here is about branching within one environment, not across many.

STARTS ARE JUNCTION CELLS ONLY (degree >= 3), not uniformly random cells --
this is the "deliberately oversample junctions" goal from WORKPLAN.md's
Phase 1 description, taken to its cleanest form: every rollout begins
exactly where the immediate future is genuinely ambiguous, rather than
hoping enough junction-crossing rollouts turn up by chance from random
starts.

WHAT P WILL ACTUALLY BE TRAINED TO PREDICT (different from
ball_in_cup_hyperbolic's P, deliberately): ball-in-cup's P(z, actions) ->
z' is a genuinely conditional prediction -- given the exact action
sequence, and ball-in-cup's dynamics are deterministic given actions, so
there's no real ambiguity left once the actions are known. If Phase 2's
maze predictor were conditioned on the exact action sequence too, the
same would be true here (a deterministic maze, given the exact moves, has
an exactly determined endpoint) -- no genuine multimodality would be left
for it to grapple with, and this project would be testing nothing. Phase
0's synthetic setup never conditioned on which mode would occur either --
the input was uninformative by construction. So Phase 2's predictor is
meant to take (start state, horizon) WITHOUT the action sequence --
predicting the marginal spread of where a random policy ends up, not a
specific known trajectory's endpoint. This dataset supports exactly that:
many independent rollouts sharing the same start cell reveal that cell's
real target distribution over endpoints, the same way domain_fit_check.py
measured k.

Each episode is saved as {"start": [r,c], "actions": [...],
"states": [[r,c], ...]} (states has length horizon+1, states[0] == start).
Phase 2 can slice states[:H+1] for any H <= MAX_HORIZON without
recollecting data.

No torch, no numpy -- pure stdlib, output is plain JSON.

Run (from inside maze/):
    uv run python collect_data.py
"""
import argparse
import json
import random

from env import MazeEnv, GRID_SIZE, ALGO

MAZE_SEED = 0              # ONE fixed maze for the whole dataset
MAX_HORIZON = 20           # matches domain_fit_check.py's tested range
ROLLOUTS_PER_JUNCTION = 200  # bumped from 40: jepa_train.py's 80/20
                             # holdout split left only 8 eval rollouts
                             # per junction, which capped the OBSERVABLE
                             # distinct-endpoint count at 8 regardless of
                             # the maze's true branching -- see
                             # eval_k_diagnostic.py. 200 gives ~40 holdout
                             # rollouts/junction, matching domain_fit_check.py's
                             # sample size closely enough to actually see k>=8
                             # where it's really there.
DATA_SEED = 1              # separate from MAZE_SEED -- same maze, resample-able rollouts
OUTPUT_PATH = "maze_data.json"


def junction_cells(adj):
    return [cell for cell, nbs in adj.items() if len(nbs) >= 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-path", default=None,
                     help="override OUTPUT_PATH (default maze_data.json) -- use "
                          "a distinct name when GRID_SIZE differs from the "
                          "standard maze, so this doesn't overwrite that data.")
    ap.add_argument("--backtrack-weight", type=float, default=None,
                     help="passed through to MazeEnv.random_rollout -- downweights "
                          "(doesn't forbid) immediately reversing the previous action "
                          "when another valid option exists. Off by default (original "
                          "uniform policy). See env.py's random_rollout docstring and "
                          "WORKPLAN.md's trajectory-level-cycles section for why this "
                          "exists -- not yet proven to help, test via ablation. Confirmed "
                          "NOT to fix the two known-broken branch cells on its own (their "
                          "dead ends are literal leaves -- the bounce back is forced, not "
                          "a choice) -- see --deadend-weight instead.")
    ap.add_argument("--deadend-weight", type=float, default=None,
                     help="passed through to MazeEnv.random_rollout -- downweights "
                          "(0.0 = forbids) choosing to walk INTO a literal dead end "
                          "(degree-1 cell) when another valid option exists. Off by "
                          "default (original uniform policy). See env.py's "
                          "random_rollout docstring for why this targets the actual "
                          "forced-choice point (entering), unlike --backtrack-weight "
                          "(leaving, which isn't a real choice at a true dead end).")
    args = ap.parse_args()
    output_path = args.output_path or OUTPUT_PATH

    env = MazeEnv(size=GRID_SIZE, algo=ALGO, seed=MAZE_SEED)
    starts = junction_cells(env.adj)
    print(f"maze: size={GRID_SIZE} algo={ALGO} seed={MAZE_SEED}  "
          f"{len(env.adj)} cells, {len(starts)} junction cells")

    env.rng = random.Random(DATA_SEED)  # rollout randomness, decoupled from maze topology

    episodes = []
    for start in starts:
        for _ in range(ROLLOUTS_PER_JUNCTION):
            states, actions = env.random_rollout(MAX_HORIZON, start=start, valid_only=True,
                                                   backtrack_weight=args.backtrack_weight,
                                                   deadend_weight=args.deadend_weight)
            episodes.append({
                "start": list(start),
                "actions": actions,
                "states": [list(s) for s in states],
            })

    print(f"{len(episodes)} episodes, horizon={MAX_HORIZON} each "
          f"({len(starts)} junctions x {ROLLOUTS_PER_JUNCTION} rollouts)")

    with open(output_path, "w") as f:
        json.dump({
            "grid_size": GRID_SIZE,
            "algo": ALGO,
            "maze_seed": MAZE_SEED,
            "max_horizon": MAX_HORIZON,
            "episodes": episodes,
        }, f)
    print(f"written to {output_path}")

    print("\nsanity check (should roughly match domain_fit_check.py's k for H=12):")
    for start in starts[:3]:
        eps_from_start = [e for e in episodes if tuple(e["start"]) == start]
        endpoints = {tuple(e["states"][12]) for e in eps_from_start}
        print(f"  start={start}: {len(eps_from_start)} rollouts, "
              f"{len(endpoints)} distinct endpoints at H=12")


if __name__ == "__main__":
    main()
