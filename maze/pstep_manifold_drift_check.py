"""
pstep_manifold_drift_check.py

Maze JEPA -- Phase 3 diagnostic: WHY does autoregressive error explode at
a roughly fixed step count (~12-17), independent of total chain length?
step_drift_diagnostic.py established THAT it explodes; this asks the
mechanistic question underneath, specific to hyperbolic: is P_step's raw
(tangent-space) output drifting toward the Poincare ball's boundary under
repeated autoregressive application, into a region where to_geometry's
exp0 (tanh-based) map is saturated and hyp.dist's arcosh becomes
hypersensitive to tiny raw-space differences (see hyperbolic.py's own
docstring: dist(0,x) = 2*artanh(|x|), which diverges as |x| -> 1)?

If so, the ~12-17-step-independent-of-distance onset stops being
mysterious: it isn't a capacity ceiling (already ruled out at dim=8 and
dim=12 -- see WORKPLAN.md) or a horizon-dependent effect, it's a fixed
number of P_step applications needed for the raw-vector norm to drift
into the saturated/high-sensitivity zone, after which ANY residual
per-step error (however small) blows up in hyperbolic distance.

Two independent, cheap checks, no retrain needed -- both against
whatever checkpoint is already on disk:

PART A -- does the raw norm actually drift during a real autoregressive
rollout? Reuses the exact same 5 pairs as step_drift_diagnostic.py (same
DIAG_SEED/N_PAIRS/MIN_PAIR_DISTANCE) so the step-by-step rows here line
up directly against that script's already-reported error numbers. Prints
||raw|| for both the autoregressive chain and the true path at the same
step, so "typical/on-manifold norm" vs "where the rollout drifts to" is
directly comparable.

PART B -- is P_step LOCALLY more sensitive (expansive) near high raw
norm? Independent of any specific rollout: samples real cells, perturbs
E(cell) by a small random raw-space delta, applies P_step once (same
action to both perturbed and unperturbed input), and measures the
expansion factor = (post-P_step geometry distance) / (pre-P_step
geometry distance). Bucketed by the input's raw norm. If the expansion
factor climbs sharply with norm, that's the mechanism, independent of
whatever specific trajectory Part A happened to sample.

Run with --geometry hyperbolic for the actual diagnosis. --geometry
euclidean also works (to_geometry is identity, hyp.dist is replaced by
flat squared distance) and serves as a CONTROL, not as euclidean
debugging effort (out of scope per WORKPLAN.md's scope decision) --
euclidean has no manifold boundary, so if its expansion factor stays
flat across norm buckets while hyperbolic's climbs, that's confirming
evidence the effect is boundary-specific, not just "MLPs are sometimes
locally expansive."

Run (in Andreas's terminal, after jepa_train.py AND jepa_train_step.py
have both been run for the geometry being checked):
    uv run python pstep_manifold_drift_check.py --geometry hyperbolic
    uv run python pstep_manifold_drift_check.py --geometry euclidean
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
DIAG_SEED = 3             # same as step_drift_diagnostic.py -- same 5 pairs,
N_PAIRS = 5               # so Part A's rows line up against that script's
MIN_PAIR_DISTANCE = 15    # already-reported error numbers step for step.

PERT_SEED = 7
N_PERT_CELLS = 40         # random sample of real cells for Part B
N_PERT_DIRECTIONS = 8     # random unit directions tried per (cell, action)
PERT_EPSILON = 0.05       # small perturbation magnitude in raw L2 space --
                          # small relative to E's typical raw norm (printed
                          # at run time for context)
BUCKET_WIDTH = 1.0
BUCKET_CAP = 8.0

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
DELTA_TO_ACTION = {v: k for k, v in ACTION_DELTA.items()}


def path_to_actions(path):
    """Cell sequence -> action sequence, via the inverse of env.py's
    ACTION_DELTA. Duplicated from step_drift_diagnostic.py -- small
    per-script helper, same convention as that file's own duplicated
    spearman_correlation in embedding_metric_check.py."""
    actions = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        actions.append(DELTA_TO_ACTION[(r1 - r0, c1 - c0)])
    return actions


def norm_bucket(n):
    if n >= BUCKET_CAP:
        return f">={BUCKET_CAP:.0f}"
    lo = int(n // BUCKET_WIDTH) * BUCKET_WIDTH
    return f"[{lo:.0f}-{lo + BUCKET_WIDTH:.0f})"


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
    n_cells = size * size

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())

    with torch.no_grad():
        all_idx = torch.arange(n_cells)
        all_raw_norms = E(all_idx).norm(dim=-1)
    print(f"[{geometry}] E's raw embedding norm across all {n_cells} cells: "
          f"min={all_raw_norms.min().item():.3f} mean={all_raw_norms.mean().item():.3f} "
          f"max={all_raw_norms.max().item():.3f}")

    # ---- Part A: raw norm along real autoregressive rollouts ----
    rng = random.Random(DIAG_SEED)
    pairs = []
    while len(pairs) < N_PAIRS:
        start, target = rng.sample(cells, 2)
        d = bfs_distance(adj, start, target)
        if d >= MIN_PAIR_DISTANCE:
            pairs.append((start, target, d))

    print(f"\n[{geometry}] PART A -- raw norm drift along {len(pairs)} real autoregressive rollouts "
          f"(same pairs as step_drift_diagnostic.py)")

    with torch.no_grad():
        for start, target, d in pairs:
            path = bfs_path_cells(adj, start, target)
            actions = path_to_actions(path)
            assert len(actions) == d

            true_idx = torch.tensor([cell_index(c, size) for c in path])
            true_raw = E(true_idx)

            auto_raw = true_raw[0:1]
            print(f"\n  start={start} target={target} bfs_distance={d}")
            print(f"  {'step':>4}  {'||auto_raw||':>12}  {'||true_raw||':>12}  {'auto_err':>10}")
            for t, a in enumerate(actions):
                action_idx = torch.tensor([ACTION_INDEX[a]])
                auto_raw = P_step(auto_raw, action_idx)
                auto_err = squared_dist(
                    to_geometry(auto_raw, geometry),
                    to_geometry(true_raw[t + 1:t + 2], geometry),
                    geometry,
                ).item()
                print(f"  {t + 1:>4}  {auto_raw.norm().item():>12.4f}  "
                      f"{true_raw[t + 1].norm().item():>12.4f}  {auto_err:>10.4f}")

    # ---- Part B: local expansion factor, bucketed by input raw norm ----
    pert_rng = random.Random(PERT_SEED)
    torch.manual_seed(PERT_SEED)
    sample_cells = pert_rng.sample(cells, N_PERT_CELLS)

    buckets = {}  # norm bucket -> list of expansion factors
    with torch.no_grad():
        for cell in sample_cells:
            idx = torch.tensor([cell_index(cell, size)])
            raw = E(idx)  # (1, dim)
            base_norm = raw.norm().item()
            for action_name, a_idx in ACTION_INDEX.items():
                action_idx = torch.tensor([a_idx])
                base_out = P_step(raw, action_idx)
                for _ in range(N_PERT_DIRECTIONS):
                    delta = torch.randn_like(raw)
                    delta = delta / delta.norm() * PERT_EPSILON
                    pert_raw = raw + delta
                    pert_out = P_step(pert_raw, action_idx)

                    input_dist = squared_dist(
                        to_geometry(pert_raw, geometry), to_geometry(raw, geometry), geometry
                    ).item()
                    output_dist = squared_dist(
                        to_geometry(pert_out, geometry), to_geometry(base_out, geometry), geometry
                    ).item()
                    if input_dist <= 0:
                        continue
                    # sqrt: expansion factor in linear (not squared-distance) terms
                    factor = (output_dist / input_dist) ** 0.5
                    buckets.setdefault(norm_bucket(base_norm), []).append(factor)

    print(f"\n[{geometry}] PART B -- local expansion factor of one P_step call, "
          f"bucketed by input ||raw|| (n={N_PERT_CELLS} cells x 4 actions x {N_PERT_DIRECTIONS} directions):")
    print(f"  factor > 1 means P_step AMPLIFIES a small input perturbation; < 1 means it damps it.")
    for b in sorted(buckets, key=lambda k: (k.startswith(">="), k)):
        vals = buckets[b]
        mean_f = sum(vals) / len(vals)
        print(f"  ||raw||={b:>8}  n={len(vals):>5}  mean_expansion_factor={mean_f:.4f}")


if __name__ == "__main__":
    main()
