"""
jepa_train.py

Maze JEPA -- E + Predictor, trained JOINTLY, on a ONE-STEP action
prediction loss PLUS a multi-step unrolled prediction loss PLUS a
negative-sampling repulsion loss (no treeloss/distance-regression term
-- see the MULTI-STEP and NEGATIVE-SAMPLING REPULSION notes below for
why). E is NOT frozen and NOT pretrained separately -- both are learned
end to end.

TRAINING DATA (single-step): every cell's FULL 4-action space is used
(env's universal action set), a valid action maps to the real next
cell, an invalid one (a wall bump) maps back to the SAME cell (no real
movement) -- wall-bumps were reinstated so P has a real, learned answer
for illegal moves instead of undefined behavior; see planner_cem.py's
LEGALITY note for why that matters. enumerate_transitions() walks every
cell and its FULL 4-action space, num_cells*4 pairs total (400 for size
10). ALL of it is used for training -- no holdout split: the goal here
is to memorize this one specific maze's full transition table exactly,
not to generalize to unseen mazes.

MULTI-STEP (unrolled) LOSS: single-step training only ever asks P to be
right one hop at a time -- it never asks the embedding space to compose
sensibly over several hops. The fix, directly analogous to
ball_in_cup_hyperbolic's UNROLL_STEPS fix for its own exposure-bias
problem: sample a REAL random walk of k actions from a random start
cell (each action drawn from env.valid_actions at the cell actually
reached so far, so the walk is always a genuine path through the real
maze), chain P k times -- feeding each call's own output back in as the
next start, never re-encoding through E in between -- and compare only
the FINAL chained prediction against E(<the real cell the walk actually
ended at>). No externally-computed distance target anywhere -- just a
real experienced multi-step transition. k ~ Uniform[2, unroll_max],
resampled fresh every training step.

NEGATIVE-SAMPLING REPULSION (the newest piece): the multi-step loss
above fixes COMPOSITION consistency, but composition consistency isn't
the same thing as embedding DISTANCE tracking real graph distance --
planner_greedy.py's and planner_cem.py's failures showed this directly
(2-cycle attractors; CEM barely beating greedy). The reason: every
attractive force in this file (single-step, multi-step) only ever pulls
together cells that ARE connected by a real edge or a real sampled
walk. Nothing has ever pushed apart cells that AREN'T related. VICReg's
cov_penalty_term doesn't fill that gap either -- it's a population-level
constraint (every dimension near unit variance, decorrelated across the
whole cell population), not a pairwise, relationship-aware repulsion;
in a moderate-dimensional space it actually makes every cell's raw
embedding norm concentrate around the SAME value (concentration of
measure), which works against, not for, a hierarchical/radial layout.
This mirrors Nickel & Kiela's "Poincare Embeddings for Learning
Hierarchical Representations" (NeurIPS 2017, arXiv:1705.08039): trained
purely from graph edges plus NEGATIVE SAMPLES of random non-edges (pull
true edges together, push random pairs apart), a hierarchy -- embedding
norm correlating with depth -- emerges with no root ever specified.
Their negative-sampling term is the ingredient we were missing.
negative_repulsion_loss() adds it here: every training step, sample
random (cell_i, cell_j) pairs (sample_negative_pairs() -- no relation
check beyond i != j, same as standard word2vec/node2vec negative
sampling; true edges are a small fraction of all possible pairs so
plain uniform pairing works fine in aggregate) and push their
embeddings apart with a hinge loss, zero once they're already farther
than `--neg-margin` apart. Set --neg-weight 0 to disable and get back
the exact previous behavior (for ablation).

ANTI-COLLAPSE: the single-step objective, on its own, has a trivial
degenerate solution (collapse every cell's embedding to the same point
-- "predict no change" then trivially "works"). jepa_model.py's VICReg
cov_penalty_term is applied to E's raw output for the start and end
cells of every single-step batch to guard against that, same mechanism
(and same formula) as the old project's Phase 2. Neither the multi-step
loss nor the new negative-sampling loss need their own separate cov
penalty -- the negative-sampling term in particular is itself a form of
anti-collapse (it directly penalizes cells sitting on top of each
other), on top of what VICReg already provides.

    L = mean(squared_dist(Predictor(E(start), action_delta), E(next)))
        + kappa/2 * (cov_penalty(E(start)) + cov_penalty(E(next)))
        + unroll_weight * mean(squared_dist(P^k(E(walk_start)), E(walk_end)))
        + neg_weight * mean(relu(margin - squared_dist(E(cell_i), E(cell_j))))

Single fixed maze (one seed) for the whole run -- E's embedding table is
indexed by grid position, so it only makes sense against one fixed set
of walls.

Run (from inside maze_jepa/):
    uv run python jepa_train.py --geometry euclidean
    uv run python jepa_train.py --geometry hyperbolic
"""
import argparse
import random

import torch
import torch.optim as optim

from env import ACTIONS, MazeEnv
from jepa_model import (
    CHECKPOINT_PATH,
    HIDDEN_DIM,
    LATENT_DIM,
    STEP_CHECKPOINT_PATH,
    Encoder,
    Predictor,
    cell_index,
    cov_penalty_term,
    squared_dist,
    to_geometry,
)

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
# Row i = action i's raw (row, col) delta -- indexing this with an integer
# action-index tensor gives a batch of raw direction vectors, no learned
# per-action embedding involved. See jepa_model.py's Predictor docstring.
ACTION_DELTA = torch.tensor([a.value for a in ACTIONS], dtype=torch.float32)


def enumerate_transitions(env):
    """The COMPLETE transition function over every cell's FULL 4-action
    space -- num_cells*4 pairs total (400 for size 10). A valid action
    maps to the real next cell (env.valid_actions); an INVALID one (a
    wall bump) maps back to the SAME cell -- no real movement. See the
    module docstring's TRAINING DATA note for why wall-bumps are back
    and what target they're given."""
    transitions = []
    for cell in env.adj:
        r, c = cell
        valid = set(env.valid_actions(cell))
        for a in ACTIONS:
            if a in valid:
                dr, dc = a.value
                next_cell = (r + dr, c + dc)
            else:
                next_cell = cell  # wall bump: no real movement
            transitions.append((cell, a, next_cell))
    return transitions


def to_tensors(transitions, size):
    starts = torch.tensor([cell_index(s, size) for s, _, _ in transitions])
    acts = torch.tensor([ACTION_INDEX[a] for _, a, _ in transitions])
    nexts = torch.tensor([cell_index(n, size) for _, _, n in transitions])
    return starts, acts, nexts


def sample_walk_batch(env, rng, size, batch_size, k):
    """batch_size REAL random walks of exactly k actions each, every
    action drawn from env.valid_actions at the cell the walk has
    actually reached so far -- so every walk is a genuine path through
    the real maze, and its endpoint is the REAL cell that path ends at.
    Returns (start_idx, action_idx_steps, end_idx): start_idx and
    end_idx are (batch_size,) index tensors; action_idx_steps is a
    length-k list of (batch_size,) action-index tensors, one per
    chained step, in order."""
    cells = list(env.adj.keys())
    starts = []
    ends = []
    action_steps = [[] for _ in range(k)]
    for _ in range(batch_size):
        current = rng.choice(cells)
        starts.append(current)
        for step in range(k):
            a = rng.choice(env.valid_actions(current))
            action_steps[step].append(ACTION_INDEX[a])
            dr, dc = a.value
            current = (current[0] + dr, current[1] + dc)
        ends.append(current)

    start_idx = torch.tensor([cell_index(c, size) for c in starts])
    end_idx = torch.tensor([cell_index(c, size) for c in ends])
    action_idx_steps = [torch.tensor(a) for a in action_steps]
    return start_idx, action_idx_steps, end_idx


def multi_step_loss(E, P, start_idx, action_idx_steps, end_idx, geometry):
    """Chains P's own output k times -- never re-encoding through E in
    between -- and compares only the final result against the REAL
    embedding of the cell the walk actually ended at. See the module
    docstring's MULTI-STEP note."""
    current_raw = E(start_idx)
    for a_idx in action_idx_steps:
        a_delta = ACTION_DELTA[a_idx]
        current_raw = P(current_raw, a_delta)

    end_raw = E(end_idx)
    pred_geo = to_geometry(current_raw, geometry)
    end_geo = to_geometry(end_raw, geometry)
    return squared_dist(pred_geo, end_geo, geometry).mean()


def sample_negative_pairs(num_cells, batch_size):
    """batch_size random (cell_i, cell_j) index pairs, i != j
    guaranteed, no relation check beyond that -- same as standard
    word2vec/node2vec negative sampling; see the module docstring's
    NEGATIVE-SAMPLING REPULSION note for why this is a reasonable
    approximation (true edges are a small fraction of all possible
    pairs)."""
    idx_a = torch.randint(0, num_cells, (batch_size,))
    idx_b = torch.randint(0, num_cells, (batch_size,))
    collide = idx_a == idx_b
    idx_b = torch.where(collide, (idx_b + 1) % num_cells, idx_b)
    return idx_a, idx_b


def negative_repulsion_loss(E, geometry, idx_a, idx_b, margin):
    """Hinge repulsion: push apart cells with no reason (from any other
    loss in this file) to be close, up to `margin` -- zero once they're
    already farther apart than that. See the module docstring's
    NEGATIVE-SAMPLING REPULSION note."""
    a_geo = to_geometry(E(idx_a), geometry)
    b_geo = to_geometry(E(idx_b), geometry)
    d = squared_dist(a_geo, b_geo, geometry)
    return torch.clamp(margin - d, min=0.0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["euclidean", "hyperbolic"], default="hyperbolic")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--algo", default="prim")
    ap.add_argument("--maze-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kappa", type=float, default=0.5,
                     help="weight on the VICReg covariance/variance penalty -- halved from an "
                          "initial 1.0 as a test: cov_loss dominates total_loss magnitude in every "
                          "observed run, and now that neg_loss provides its own targeted pairwise "
                          "repulsion, VICReg's blanket population-level constraint may be adding "
                          "per-pair noise rather than just guarding against collapse (see "
                          "jepa_train.py history / WORKPLAN.md)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--unroll-max", type=int, default=20,
                     help="each step samples k ~ Uniform[2, unroll_max] real actions to chain "
                          "P over; 0 disables the multi-step loss entirely -- raised from an "
                          "initial 8, which left most pairs (maze diameter is 27) with only the "
                          "crude negative-sampling floor and no graded compositional signal at "
                          "all (see jepa_train.py history / WORKPLAN.md)")
    ap.add_argument("--unroll-weight", type=float, default=1.0)
    ap.add_argument("--neg-weight", type=float, default=1.0,
                     help="weight on the negative-sampling repulsion loss; 0 disables it")
    ap.add_argument("--neg-margin", type=float, default=2.0,
                     help="hinge margin (squared_dist units) below which random unrelated "
                          "pairs get pushed apart -- tried raising this to 10.0, which left "
                          "hyperbolic's distance correlation unchanged and made euclidean's "
                          "distinctly worse (non-monotonic at the tail) -- reverted to 2.0, "
                          "the tested-better value (see jepa_train.py history / WORKPLAN.md)")
    ap.add_argument("--dim", type=int, default=LATENT_DIM,
                     help="total embedding dimension (free + POS_DIMS positional) -- exposed "
                          "as a CLI arg to deliberately shrink it and find where each geometry's "
                          "planner success rate starts breaking down, testing whether hyperbolic's "
                          "exponential-volume capacity advantage over euclidean's polynomial one "
                          "shows up as holding a higher success rate at the SAME (smaller) "
                          "dimension, not just at the dimension already tuned for (see "
                          "jepa_train.py history / WORKPLAN.md)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    env = MazeEnv(size=args.size, algo=args.algo, seed=args.maze_seed)
    transitions = enumerate_transitions(env)
    starts, acts, nexts = to_tensors(transitions, args.size)
    n = starts.shape[0]
    n_wall_bump = sum(1 for s, _, nx in transitions if s == nx)
    print(f"{n} transitions (full 4-action space per cell, {n_wall_bump} wall-bumps -> same "
          f"cell, {n - n_wall_bump} real moves), all used for training")
    if args.unroll_max >= 2:
        print(f"multi-step unrolled loss enabled: k ~ Uniform[2, {args.unroll_max}], weight={args.unroll_weight}")
    else:
        print("multi-step unrolled loss disabled (--unroll-max < 2)")
    if args.neg_weight > 0:
        print(f"negative-sampling repulsion enabled: margin={args.neg_margin}, weight={args.neg_weight}")
    else:
        print("negative-sampling repulsion disabled (--neg-weight <= 0)")

    num_cells = args.size * args.size
    print(f'embedding dim={args.dim} (free_dim={args.dim - 2}, POS_DIMS=2)')
    E = Encoder(num_cells, args.size, dim=args.dim)
    P = Predictor(dim=args.dim, hidden=HIDDEN_DIM, residual=False)
    opt = optim.Adam(list(E.parameters()) + list(P.parameters()), lr=args.lr)

    batch_size = min(args.batch_size, n)
    for step in range(args.steps):
        batch = torch.randint(0, n, (batch_size,))
        s_idx, a_idx, n_idx = starts[batch], acts[batch], nexts[batch]
        a_delta = ACTION_DELTA[a_idx]

        start_raw = E(s_idx)
        next_raw = E(n_idx)
        pred_raw = P(start_raw, a_delta)

        pred_geo = to_geometry(pred_raw, args.geometry)
        next_geo = to_geometry(next_raw, args.geometry)
        pred_loss = squared_dist(pred_geo, next_geo, args.geometry).mean()

        cov_loss = cov_penalty_term(start_raw) + cov_penalty_term(next_raw)

        if args.unroll_max >= 2:
            k = rng.randint(2, args.unroll_max)
            walk_start, walk_actions, walk_end = sample_walk_batch(env, rng, args.size, batch_size, k)
            unroll_loss = multi_step_loss(E, P, walk_start, walk_actions, walk_end, args.geometry)
        else:
            unroll_loss = torch.tensor(0.0)

        if args.neg_weight > 0:
            neg_a, neg_b = sample_negative_pairs(num_cells, batch_size)
            neg_loss = negative_repulsion_loss(E, args.geometry, neg_a, neg_b, args.neg_margin)
        else:
            neg_loss = torch.tensor(0.0)

        loss = (pred_loss + args.kappa / 2 * cov_loss
                + args.unroll_weight * unroll_loss
                + args.neg_weight * neg_loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step}: pred_loss={pred_loss.item():.4f} cov_loss={cov_loss.item():.4f} "
                  f"unroll_loss={unroll_loss.item():.4f} neg_loss={neg_loss.item():.4f} "
                  f"total_loss={loss.item():.4f}")

    # Full-data eval: prediction accuracy across the ENTIRE transition
    # table (there's no held-out set anymore -- see the module docstring).
    with torch.no_grad():
        a_delta = ACTION_DELTA[acts]
        start_raw = E(starts)
        next_raw = E(nexts)
        pred_raw = P(start_raw, a_delta)
        pred_geo = to_geometry(pred_raw, args.geometry)
        next_geo = to_geometry(next_raw, args.geometry)
        full_loss = squared_dist(pred_geo, next_geo, args.geometry).mean()

        all_raw = E(torch.arange(num_cells))
        spread = all_raw.norm(dim=-1).mean()

    print(f"full-data pred_loss={full_loss.item():.4f}")
    print(f"mean embedding norm (collapse check, ~0 = collapsed): {spread.item():.4f}")

    ckpt_e = {"E": E.state_dict(), "num_cells": num_cells, "size": args.size, "dim": args.dim}
    torch.save(ckpt_e, CHECKPOINT_PATH.format(geometry=args.geometry))
    ckpt_p = {
        "P_step": P.state_dict(),
        "dim": args.dim,
        "hidden": HIDDEN_DIM,
        "residual": False,
    }
    torch.save(ckpt_p, STEP_CHECKPOINT_PATH.format(geometry=args.geometry))
    print(f"saved {CHECKPOINT_PATH.format(geometry=args.geometry)} and "
          f"{STEP_CHECKPOINT_PATH.format(geometry=args.geometry)}")


if __name__ == "__main__":
    main()
