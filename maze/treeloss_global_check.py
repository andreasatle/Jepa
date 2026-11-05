"""
treeloss_global_check.py

Maze JEPA -- cheap sanity check on jepa_train_treeloss.py's output BEFORE
spending time training a P_step against it. That script's loss only
directly constrains EDGES (every real edge -> unit embedded distance).
It does NOT directly constrain non-adjacent pairs -- the hope, per
Sarkar's construction, is that unit edges + hyperbolic's exponential
volume growth automatically separate distant subtrees too, with no
explicit long-range term needed. But Sarkar's guarantee is for an
explicit combinatorial construction with full knowledge of the tree;
this E was fit by plain gradient descent on local edge constraints only,
so there's a real risk it satisfies every edge but still collapses two
distant, unrelated cells close together (see WORKPLAN.md's "Design
discussion" section for the fuller reasoning).

This checks exactly that, exhaustively (256 cells -> ~32,640 pairs, cheap
enough to do all of them, no sampling needed): for every pair of cells,
compare the TRUE bfs_distance against the embedded squared_dist, and
report both the overall rank correlation (Spearman rho, same convention
as maze_planner.py's summarize_cost_correlation) and specifically
whether any FAR-apart (in the true tree) pairs ended up embedded
suspiciously CLOSE -- that specific failure mode (distant cells
collapsed together) is what would indicate the edge-only loss isn't
enough and the wider (x,b,y) triple loss from WORKPLAN.md is needed
before bothering to train P_step at all.

Run:
    uv run python treeloss_global_check.py [--geometry hyperbolic]
"""
import argparse
import random
import statistics

import torch

from env import ALGO, GRID_SIZE
from jepa_model import cell_index, load_encoder_checkpoint, squared_dist, to_geometry
from maze_gen import GENERATORS, bfs_distances_from
from maze_planner import MAZE_SEED

CHECKPOINT_PATH = "jepa_checkpoint_treeloss_{geometry}.pt"


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0] * len(xs)
    for rank, i in enumerate(order):
        ranks[i] = rank
    return ranks


def spearman_correlation(xs, ys):
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - (6 * d2) / (n * (n ** 2 - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="hyperbolic")
    ap.add_argument("--encoder-checkpoint", default=None,
                     help="override CHECKPOINT_PATH template")
    args = ap.parse_args()
    geometry = args.geometry

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())

    E, ckpt = load_encoder_checkpoint(geometry, path=args.encoder_checkpoint or CHECKPOINT_PATH)
    size = ckpt["size"]
    dist_scale = ckpt.get("dist_scale", 1.0)  # old checkpoints (pre this field) trained at
                                                # effectively dist_scale=1

    with torch.no_grad():
        true_dists = []
        pairs = []
        for i, a in enumerate(cells):
            bfs_from_a = bfs_distances_from(adj, a)
            for j in range(i + 1, len(cells)):
                b = cells[j]
                true_dists.append(bfs_from_a[b])
                pairs.append((a, b))

        # Look up each pair's embedding directly by cell_index -- NOT via a
        # precomputed `mapped` tensor indexed by position in `cells`, which
        # is insertion order from the maze generator, not sorted by
        # cell_index. Indexing that tensor with raw cell_index values (as
        # an earlier version of this script did) silently grabs the wrong
        # cell's embedding for almost every pair -- this direct E() call
        # sidesteps that entirely by using the same idx convention E always
        # expects (cell_index straight into the embedding table).
        idx_i = torch.tensor([cell_index(a, size) for a, b in pairs])
        idx_j = torch.tensor([cell_index(b, size) for a, b in pairs])
        a_mapped = to_geometry(E(idx_i), geometry)
        b_mapped = to_geometry(E(idx_j), geometry)
        embed_costs = squared_dist(a_mapped, b_mapped, geometry)
        embed_dists = embed_costs.sqrt().tolist()  # sqrt to compare in the same "distance" units

    n_pairs = len(pairs)
    rho = spearman_correlation(true_dists, embed_dists)
    print(f"[{geometry}] {n_pairs} pairs, Spearman rho(true bfs_distance, embedded distance) = {rho:.4f}")

    # specific failure mode: far-apart-in-the-tree pairs that ended up
    # embedded suspiciously close -- report the worst offenders directly,
    # not just a summary correlation, since a few bad collapses can hide
    # inside an otherwise-good rho.
    FAR_THRESHOLD = 15   # "far" in true tree distance -- unaffected by dist_scale, this is a
                          # true-distance threshold, not an embedded-distance one
    CLOSE_THRESHOLD = 3 / dist_scale  # "close" in embedded distance -- suspicious if true
                                        # distance is also FAR_THRESHOLD+. Scaled: a checkpoint
                                        # trained with dist_scale=4 naturally produces embedded
                                        # distances ~4x smaller, so the absolute "suspicious"
                                        # cutoff needs to shrink by the same factor to mean the
                                        # same thing relative to that checkpoint's own scale.
    offenders = [
        (a, b, td, ed) for (a, b), td, ed in zip(pairs, true_dists, embed_dists)
        if td >= FAR_THRESHOLD and ed <= CLOSE_THRESHOLD
    ]
    print(f"  far-but-collapsed pairs (true_dist>={FAR_THRESHOLD}, embedded_dist<={CLOSE_THRESHOLD}): "
          f"{len(offenders)}/{n_pairs}")
    if offenders:
        offenders.sort(key=lambda x: x[3])  # worst (smallest embedded distance) first
        print("  worst 10 offenders (a, b, true_bfs_distance, embedded_distance):")
        for a, b, td, ed in offenders[:10]:
            print(f"    {a} <-> {b}: true={td}  embedded={ed:.3f}")

    # complementary failure mode: NEAR-in-the-tree pairs (true edges) that
    # ended up embedded suspiciously FAR -- this is the one the distance-
    # weighted loss is trying to fix, so name the actual offenders instead
    # of just watching the true_dist=1 bucket's mean/max drift.
    NEAR_THRESHOLD = 1     # true edges only -- unaffected by dist_scale (true distance)
    FAR_EMBED_THRESHOLD = 6 / dist_scale  # suspicious if a true edge embeds this far apart --
                                            # scaled for the same reason as CLOSE_THRESHOLD above.
                                            # Without this, a checkpoint trained at dist_scale=4
                                            # would need a ~15-20x-worse-than-expected edge before
                                            # this ever triggered, instead of the original ~6x
    near_offenders = [
        (a, b, td, ed) for (a, b), td, ed in zip(pairs, true_dists, embed_dists)
        if td <= NEAR_THRESHOLD and ed >= FAR_EMBED_THRESHOLD
    ]
    print(f"  near-but-stretched pairs (true_dist<={NEAR_THRESHOLD}, embedded_dist>={FAR_EMBED_THRESHOLD}): "
          f"{len(near_offenders)}/{n_pairs}")
    if near_offenders:
        near_offenders.sort(key=lambda x: -x[3])  # worst (largest embedded distance) first
        print("  worst 10 near-but-stretched offenders (a, b, true_bfs_distance, embedded_distance):")
        for a, b, td, ed in near_offenders[:10]:
            print(f"    {a} <-> {b}: true={td}  embedded={ed:.3f}")

    # also report the overall relationship between true and embedded distance
    # at a few true-distance buckets, same spirit as branch_point_diagnostic.py's
    # distance-bucketed accuracy table
    buckets = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 30), (31, 10**9)]
    print("  mean embedded distance by true bfs_distance bucket (should increase steadily):")
    for lo, hi in buckets:
        vals = [ed for (a, b), td, ed in zip(pairs, true_dists, embed_dists) if lo <= td <= hi]
        if not vals:
            continue
        label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(f"    true_dist={label:>6}: n={len(vals):>6}  mean_embedded={statistics.mean(vals):.3f}  "
              f"min={min(vals):.3f}  max={max(vals):.3f}")


if __name__ == "__main__":
    main()
