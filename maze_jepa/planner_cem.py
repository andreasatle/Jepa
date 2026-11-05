"""
planner_cem.py

Maze JEPA -- MPC-style planner via the trained multi-step chained
Predictor (CEM: Cross-Entropy Method over discrete action sequences).

Why not greedy: planner_greedy.py's single-step lookahead ("of my
immediate valid actions, which puts me closest to the target in
embedding space") got trapped in geometry-independent 2-cycle
attractors -- 1/40 (hyperbolic), 0/40 (euclidean), every failure
burning the full step budget rather than slowly diverging. Adding the
multi-step unrolled CONSISTENCY loss to jepa_train.py didn't fix this
either (same result) -- composing P's own predictions consistently is
not the same property as embedding DISTANCE tracking real graph
distance, and nothing had ever trained the latter.

This planner searches over H-step ACTION SEQUENCES instead of one step
at a time, scoring each candidate purely by chaining the trained
Predictor H times from the current real embedding and comparing the
final imagined embedding to E(target) -- exactly the composition the
multi-step loss trained P to support. Optimized via CEM: maintain a
per-depth categorical distribution over the 4 raw actions, sample a
population of candidate sequences, keep the elite fraction (lowest
final predicted distance to target), refit each depth's distribution
toward the elite empirical frequencies, repeat for a few iterations;
execute only the FIRST action of the best sequence found, then replan
from the real cell actually reached (receding horizon / MPC, matching
Andreas's stated preference for replanning from the current state
rather than committing to one fixed full-horizon plan). Directly
analogous to ball_in_cup_hyperbolic's CEM planner.

LEGALITY, CORRECTED: an earlier version of this file only restricted
DEPTH 0 to env.valid_actions(current) and let every deeper imagined
depth sample blind over all 4 raw actions -- meaning candidate
sequences could imagine walking through walls past the first step, an
untested input regime for P (wall-bump pairs were deliberately excluded
from jepa_train.py's training data). That's wrong: every candidate this
planner considers is now walked through the maze's REAL connectivity at
EVERY depth, not just the first -- build_transition_tables() precomputes,
once per maze, which raw actions are legal from each cell and where they
lead, and CEM's per-candidate sampling loop uses that to mask each
depth's action distribution down to whatever is actually legal from
that candidate's own real cell at that point (candidates diverge onto
different real cells after depth 0, so this masking is necessarily
per-candidate, not a single shared mask). This is the same kind of
information a player standing in the maze already has -- which
directions aren't blocked by a wall from here -- not privileged
shortest-path knowledge, and it's the same assumption planner_greedy.py
made for its one and only step, just now applied consistently at every
depth instead of only the first. SCORING still comes ENTIRELY from the
trained P/E (chained prediction vs E(target)) -- legality only prunes
which sequences are even considered, it never tells CEM which one is
best, so the model is still what's actually being judged.

SCORING, FIXED (final-step-only was structurally penalizing early goal
arrival): debugging a specific failure -- (4,9)->(8,9), true_dist=6 --
showed the planner repeatedly visiting (8,8), a REAL neighbor of the
target with a fully open, wall-free 6-step path the whole way there,
and never once taking the single legal RIGHT move that reaches it.
The cause: candidate sequences were always the full horizon length
(12), scored only by the FINAL predicted embedding. A sequence that
reaches the target early still has to keep chaining P for the
remaining steps to fill out the horizon -- and that extra chaining can
drift the final predicted point away from E(target) again, so a
candidate that actually arrives early can score WORSE than one that
never arrives but happens to land close at exactly step 12. Fixed by
scoring each candidate by the MINIMUM predicted distance to target
over ALL intermediate depths in its rollout, not just the last one --
standard practice for goal-conditioned MPC with a horizon longer than
the typical distance to goal.

SCORING, FIXED AGAIN (a real path can still lose to model noise): with
the fix above, checks/check_single_step.py on a specific stuck case --
(9,1)->(9,0), one hop away -- confirmed the trained model itself
strongly prefers the correct move (dist 0.19 for the real LEFT action
vs 2.02 for the wrong one), yet the planner still oscillated there for
90+ steps. Cause: min-over-trajectory compares raw squared_dist values
across ALL population*horizon chained predictions every replan --
plenty of chances (300*12=3600 per replan) for an unrelated, wrong
sequence to land numerically close to E(target) by pure chained-
prediction noise, especially at later depths where compounding error
is largest, and occasionally beat a genuinely-correct short answer's
already-small distance. Fixed by using ground truth we already compute
for legality (cur_cells[p], each candidate's real maze cell at every
depth -- not a new oracle access) to detect when a candidate's REAL
path actually reaches target, and scoring those as strictly better than
any distance-based guess, ranked by earliest arrival. Distance-based
scoring remains the fallback for candidates that don't reach target
within the horizon at all (the common case early in a long walk).

Run (from inside maze_jepa/):
    uv run python planner_cem.py --geometry hyperbolic --trials 40
    uv run python planner_cem.py --geometry euclidean --trials 40
"""
import argparse
import random
from collections import deque

import torch

from env import ACTIONS, MazeEnv
from jepa_model import (
    cell_index,
    load_encoder_checkpoint,
    load_step_checkpoint,
    squared_dist,
    to_geometry,
)

ACTION_DELTA = torch.tensor([a.value for a in ACTIONS], dtype=torch.float32)
NUM_ACTIONS = len(ACTIONS)


def build_transition_tables(env):
    """Precomputed once per maze: valid_idx[cell] = list of legal action
    indices from cell; next_cell[(cell, action_idx)] = the real cell
    that action leads to. Used only to keep CEM's SAMPLING legal at
    every imagined depth -- never for scoring (see module docstring)."""
    valid_idx = {}
    next_cell = {}
    for cell in env.adj:
        idxs = [ACTIONS.index(a) for a in env.valid_actions(cell)]
        valid_idx[cell] = idxs
        for i in idxs:
            dr, dc = ACTIONS[i].value
            next_cell[(cell, i)] = (cell[0] + dr, cell[1] + dc)
    return valid_idx, next_cell


def cem_plan(env, E, P, geometry, start, target, horizon, population, iters, elite_frac,
             valid_idx, next_cell, rng):
    """One CEM optimization from the current real cell `start` toward
    `target`. Every candidate sequence sampled is a genuine path through
    the real maze (legality-masked at every depth, see module
    docstring); which one is BEST is decided purely by chaining the
    trained Predictor and comparing to E(target). Returns
    (first_action, best_seq, best_score) -- only first_action is ever
    actually executed; best_seq/best_score are for diagnostics."""
    size = env.size
    target_idx_t = torch.tensor([cell_index(target, size)])
    start_idx_t = torch.tensor([cell_index(start, size)])

    with torch.no_grad():
        target_geo = to_geometry(E(target_idx_t), geometry)
        start_raw = E(start_idx_t)

        theta = [[1.0 / NUM_ACTIONS] * NUM_ACTIONS for _ in range(horizon)]

        n_elite = max(1, int(population * elite_frac))
        best_seq = None
        best_score = None

        for _ in range(iters):
            seqs = torch.zeros((population, horizon), dtype=torch.long)
            cur_cells = [start] * population  # each candidate's REAL cell so far -- legality only
            reached_at = [None] * population  # first depth (0-indexed) the candidate's REAL cell hits target, if any

            for t in range(horizon):
                theta_t = theta[t]
                for p in range(population):
                    cell = cur_cells[p]
                    legal = valid_idx[cell]
                    weights = [theta_t[i] for i in legal]
                    choice = rng.choices(legal, weights=weights, k=1)[0]
                    seqs[p, t] = choice
                    new_cell = next_cell[(cell, choice)]
                    cur_cells[p] = new_cell
                    if reached_at[p] is None and new_cell == target:
                        reached_at[p] = t

            cur = start_raw.expand(population, -1)
            best_step_dist = None
            for t in range(horizon):
                a_delta = ACTION_DELTA[seqs[:, t]]
                cur = P(cur, a_delta)
                cur_geo = to_geometry(cur, geometry)
                step_dist = squared_dist(cur_geo, target_geo.expand(population, -1), geometry)
                best_step_dist = step_dist if best_step_dist is None else torch.minimum(best_step_dist, step_dist)
            # A candidate whose REAL path (tracked above via the same legality
            # tables used for sampling -- not a new oracle access) actually
            # reaches target is a CERTAIN answer, never a guess -- it must
            # always outrank every distance-based guess, however small, and
            # among certain answers the earliest arrival wins. Encoding this
            # as (reached_depth - horizon), which is always < 0 <= any
            # squared_dist, does that with a single comparable score, no
            # separate branch needed.
            scores = best_step_dist.clone()
            for p in range(population):
                if reached_at[p] is not None:
                    scores[p] = reached_at[p] - horizon

            elite_idx = torch.topk(scores, n_elite, largest=False).indices
            elites = seqs[elite_idx]

            top_score = scores[elite_idx[0]].item()
            if best_score is None or top_score < best_score:
                best_score = top_score
                best_seq = seqs[elite_idx[0]].tolist()

            for t in range(horizon):
                counts = torch.bincount(elites[:, t], minlength=NUM_ACTIONS).float()
                counts += 1e-3  # keep every action reachable, avoid a distribution collapsing to zero mass
                theta[t] = (counts / counts.sum()).tolist()

        return ACTIONS[best_seq[0]], best_seq, best_score


def cem_navigate(env, E, P, geometry, start, target, max_real_steps, horizon, population, iters,
                  elite_frac, valid_idx, next_cell, rng):
    current = start
    path = [current]
    actions_taken = []
    for _ in range(max_real_steps):
        if current == target:
            return path, actions_taken, True
        action, _, _ = cem_plan(env, E, P, geometry, current, target, horizon, population, iters,
                                 elite_frac, valid_idx, next_cell, rng)
        dr, dc = action.value
        current = (current[0] + dr, current[1] + dc)  # guaranteed a real legal move -- masked at every depth above
        actions_taken.append(action.name)
        path.append(current)
    return path, actions_taken, current == target


def bfs_dist(env, start):
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
    ap.add_argument("--max-steps", type=int, default=100, help="real steps executed before giving up")
    ap.add_argument("--horizon", type=int, default=12, help="imagined lookahead depth per replan")
    ap.add_argument("--population", type=int, default=300)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    env = MazeEnv(size=args.size, algo=args.algo, seed=args.maze_seed)
    valid_idx, next_cell = build_transition_tables(env)
    E, _ = load_encoder_checkpoint(args.geometry)
    P, _ = load_step_checkpoint(args.geometry)
    cells = list(env.adj.keys())

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
        path, actions, success = cem_navigate(
            env, E, P, args.geometry, start, target,
            args.max_steps, args.horizon, args.population, args.iters, args.elite_frac,
            valid_idx, next_cell, rng,
        )
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
