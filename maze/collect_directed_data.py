"""
collect_directed_data.py

Maze JEPA -- Phase 3 fix for the training-distribution mismatch confirmed
in pstep_distribution_mismatch_check.py's result (see WORKPLAN.md's
"pstep_distribution_mismatch_check.py result: CONFIRMED" section):
collect_data.py's rollouts are pure random walks (uniformly among valid
moves, never steered toward any target), so P_step's unrolled training
has never once seen a long, non-backtracking, goal-directed trajectory --
exactly the kind of trajectory a BFS-optimal path is, and exactly the
kind the CEM planner actually needs P_step to handle. Random rollouts of
the same length as the failing pairs stayed bounded; BFS-directed paths
from the same start, same length, exploded. Not a hyperbolic artifact,
not capacity, not training duration -- a straightforward gap in what
P_step was ever unrolled through.

WHY THIS ISN'T "CHEATING" THE REPRESENTATION (Andreas's question,
answered in chat -- repeating the key point here for the record): this
does NOT touch E. E stays exactly as trained by jepa_train.py, on
collect_data.py's random rollouts alone -- the encoder's own geometry,
and the whole hyperbolic-vs-euclidean comparison built on it, is
untouched by anything in this file. P_step is a different kind of
object: a supervised approximator of a known, deterministic
ground-truth function (the maze's actual transition rule), not a
representation-learning objective with a "stay task-agnostic" purity
requirement the way E and Phase 2's P deliberately do (see collect_data.py's
docstring on why P is NOT action-conditioned). Training P_step on inputs
that resemble what it'll actually be asked to predict along is ordinary
train/test distribution matching, not smuggling in the planner's answer
-- CEM still has to find its own path.

ONE FUNCTION, TWO OF ANDREAS'S OPTIONS AS ONE PARAMETER: directed_rollout()
below takes a `bias_prob` -- at each step, with probability bias_prob it
takes the unique action that reduces BFS distance to the target (the
tree's one "correct" next hop, found via a precomputed bfs_distances_from
FROM the target); otherwise it picks uniformly among ALL valid moves from
the current cell. bias_prob=1.0 reproduces the pure BFS-optimal path
(Andreas's option 1) exactly. bias_prob<1 gives an imperfect,
mostly-progressing walk (option 2) -- closer to what CEM's own candidate
sequences actually look like mid-search, since CEM rarely samples the
exact optimal sequence, especially before its elite population has
converged. BIAS_PROBS below spans several values, i.e. "mixing 1 and 2"
(Andreas's follow-up) is just sampling episodes across that spread rather
than committing to one value.

Episode length = bfs_distance(start, target) for that pair (not a fixed
horizon like collect_data.py's MAX_HORIZON=20) -- this naturally produces
episodes as long as UNROLL_STEPS=20 needs and longer, covering the
autoregressive depths (>20 steps) the planner actually reaches, which
collect_data.py's fixed 20-step episodes structurally could never
produce regardless of UNROLL_STEPS. MIN_DISTANCE=20 filters out pairs
too close to yield a usable UNROLL_STEPS=20 window at all.

SEEDS KEPT SEPARATE FROM EVERY EVALUATION SEED IN THE PROJECT
(DIRECTED_DATA_SEED=21, distinct from collect_data.py's DATA_SEED=1,
step_drift_diagnostic.py/pstep_manifold_drift_check.py/
pstep_distribution_mismatch_check.py's DIAG_SEED=3, and
maze_planner.py's PLANNER_SEED=2) so a good planner result after this
still means genuine generalization to unseen start/target pairs, not
memorization of anything the diagnostics or planner evaluate on.

WALL-BUMP (INVALID-ACTION) EXPOSURE (added after cem_candidate_diagnostic.py
found the actual mechanism behind the 0/40 plateau -- see WORKPLAN.md's
"cem_candidate_diagnostic.py result" section): both this file's fallback
branch and collect_data.py's random_rollout(valid_only=True) ONLY EVER
sampled among VALID moves -- so P_step had literally never seen a single
(cell, invalid-action) -> same-cell (wall-bump/no-op) example in its
entire training history. cem_plan has no notion of maze structure at
all -- it samples candidate actions from a plain categorical distribution
over all 4 actions, valid or not -- so real CEM search constantly asks
P_step to extrapolate on exactly the input it was never trained on, and
the diagnostic showed it extrapolates badly: predicted cost kept sliding
toward 0 across a chosen plan even while the real agent sat frozen at
one cell for 6+ consecutive wall-bump steps in a row. CEM, which only
ever sees the latent cost, happily found and exploited this blind spot
in nearly every one of 40 trials (36/40 diverged from the true path
within the very first action). Fix: directed_rollout's fallback branch
(the "not heading toward target" case, already used for the bias_prob<1
"wobble" -- see ONE FUNCTION section above) now samples uniformly among
ALL 4 actions, not just the valid ones, so wall-bump examples appear
naturally inside otherwise-ordinary directed episodes whenever bias_prob
< 1.0. bias_prob=1.0 episodes are unaffected (that branch is never taken
there) and stay pure, valid-only BFS-optimal paths.

Output: maze_data_directed.json, same {"start","actions","states"}
episode schema as collect_data.py's maze_data.json (plus a "bias_prob"
field per episode, for debugging/provenance -- unused by
jepa_train_step.py's build_transitions, which only reads
"states"/"actions"). jepa_train_step.py loads and concatenates episodes
from BOTH files, so the existing random-walk coverage (broad, dense
single-transition coverage near junctions) is kept alongside this new
directed-trajectory coverage, not replaced by it.

Run (from inside maze/, no torch needed -- pure stdlib like collect_data.py):
    uv run python collect_directed_data.py
"""
import argparse
import json
import random

from env import MazeEnv, GRID_SIZE, ALGO, ACTIONS, ACTION_DELTA
from maze_gen import bfs_distance, bfs_distances_from
from jepa_train_step import UNROLL_STEPS

MAZE_SEED = 0                  # same fixed maze as everywhere else in this project
DIRECTED_DATA_SEED = 21        # separate from every other seed in the project -- see
                                # module docstring's SEEDS KEPT SEPARATE section
MIN_DISTANCE = UNROLL_STEPS    # need at least one full UNROLL_STEPS-length window per episode
BIAS_PROBS = [1.0, 0.85, 0.7, 0.5]  # spans Andreas's option 1 (1.0) through option 2
                                     # (<1.0) -- see module docstring
EPISODES_PER_BIAS = 1500       # bumped from 500 -- the first run (see WORKPLAN.md's
                                # "Directed-data result" section) fixed 4/5 diagnostic
                                # pairs but left one (a physically-adjacent, long-winding-
                                # tree-path pair) still exploding, and the planner's real
                                # CEM result (rho=0.486, 0/40) is still well below oracle-
                                # CEM's own 20% ceiling -- more episodes per bias level
                                # means broader coverage of the maze's harder/less-common
                                # regions, directly targeting that kind of gap

OUTPUT_PATH = "maze_data_directed.json"

ACTION_FOR_DELTA = {v: k for k, v in ACTION_DELTA.items()}


def directed_rollout(env, adj, dist_to_target, start, target, length, bias_prob, rng):
    """One rollout of exactly `length` steps from `start`. At each step,
    with probability bias_prob takes the unique valid action that reduces
    BFS distance to target (found via dist_to_target, precomputed once
    per target with maze_gen.bfs_distances_from); otherwise picks
    uniformly among ALL 4 ACTIONS -- valid or not, so wall bumps show up
    naturally too, see module docstring's WALL-BUMP section. If the walk
    reaches target before `length` steps are up, keeps going (using the
    same bias_prob) rather than stopping early -- later windows in a
    long episode are still valid, useful training signal regardless of
    whether they're "before" or "after" the nominal target."""
    env.reset(start)
    states = [env.state]
    actions = []
    for _ in range(length):
        valid = env.valid_actions()
        toward_target = None
        if env.state != target:
            best = min(valid, key=lambda a: dist_to_target[_step_target(env.state, a, adj)])
            toward_target = best
        if toward_target is not None and rng.random() < bias_prob:
            a = toward_target
        else:
            a = rng.choice(ACTIONS)  # ALL 4 actions, not just `valid` -- see
                                       # module docstring's WALL-BUMP section
        actions.append(a)
        states.append(env.step(a))
    return states, actions


def _step_target(state, action, adj):
    r, c = state
    dr, dc = ACTION_DELTA[action]
    nxt = (r + dr, c + dc)
    return nxt if nxt in adj[state] else state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-path", default=None,
                     help="override OUTPUT_PATH (default maze_data_directed.json) -- "
                          "use a distinct name when GRID_SIZE differs from the "
                          "standard maze, so this doesn't overwrite that data.")
    args = ap.parse_args()
    output_path = args.output_path or OUTPUT_PATH

    env = MazeEnv(size=GRID_SIZE, algo=ALGO, seed=MAZE_SEED)
    adj = env.adj
    cells = list(adj.keys())
    rng = random.Random(DIRECTED_DATA_SEED)
    env.rng = rng  # share the same rng so action choices are reproducible top to bottom

    episodes = []
    for bias_prob in BIAS_PROBS:
        collected = 0
        attempts = 0
        while collected < EPISODES_PER_BIAS:
            attempts += 1
            start, target = rng.sample(cells, 2)
            d = bfs_distance(adj, start, target)
            if d < MIN_DISTANCE:
                continue
            dist_to_target = bfs_distances_from(adj, target)
            states, actions = directed_rollout(env, adj, dist_to_target, start, target, d, bias_prob, rng)
            episodes.append({
                "start": list(start),
                "actions": actions,
                "states": [list(s) for s in states],
                "bias_prob": bias_prob,
                "target": list(target),
                "bfs_distance": d,
            })
            collected += 1
        print(f"bias_prob={bias_prob}: {collected} episodes ({attempts} pairs sampled, "
              f"{attempts - collected} skipped for bfs_distance < {MIN_DISTANCE})")

    print(f"{len(episodes)} directed episodes total, lengths range "
          f"{min(len(e['actions']) for e in episodes)}-{max(len(e['actions']) for e in episodes)}")

    with open(output_path, "w") as f:
        json.dump({
            "grid_size": GRID_SIZE,
            "algo": ALGO,
            "maze_seed": MAZE_SEED,
            "bias_probs": BIAS_PROBS,
            "episodes": episodes,
        }, f)
    print(f"written to {output_path}")


if __name__ == "__main__":
    main()
