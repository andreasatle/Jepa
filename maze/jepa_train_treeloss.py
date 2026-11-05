"""
jepa_train_treeloss.py

Maze JEPA -- direct tree-metric training for E, instead of Phase 2's
indirect objective (predict where H random steps land). Written after
branch_point_diagnostic.py found E's one-step cost comparisons are only
~69% correct overall, worse near the target (~58%) -- see WORKPLAN.md's
"branch_point_diagnostic.py" and "Design discussion" sections for the
full reasoning this follows from.

Deliberately a SEPARATE checkpoint (jepa_checkpoint_treeloss_{geometry}.pt),
never overwriting the standard jepa_checkpoint_{geometry}.pt -- per the
"don't replace, add" decision made with Andreas: the existing E (trained
on the indirect, generic Phase 2 objective) stays as the "does hyperbolic's
advantage emerge naturally" baseline; this is a separate, explicitly-
labeled experiment answering "how much of the gap is fixable by directly
optimizing for the tree-metric property."

Trains against ALL PAIRS, not just edges: for every pair of cells (x, y),
however far apart, bfs_distance(adj, x, y) already gives the exact true
distance (the sum along the tree's one unique path -- a tree has exactly
one path between any two nodes, so no shortest-of-several-routes search
is needed). Loss: minimize (embedded_dist(x,y) - target(x,y))^2 over
every pair. (An edges-only version of this loss was tried first and
failed globally -- see WORKPLAN.md's "unit-edge-only loss" section.)

--dist-scale is the one setting that actually mattered: the loss target
is bfs_distance / dist_scale, not raw bfs_distance. Only relative order
matters for CEM's argmin, so dividing every target by the same constant
is functionally free -- but it was NECESSARY, not cosmetic: hyperbolic.py's
exp0 maps raw embedding vectors through tanh(|v|), which saturates hard
as |v| grows, capping how large a distance the Poincare ball can
numerically represent (well short of raw bfs_distance's full range on
this maze, up to 46) before points are forced up against the boundary --
see treeloss_norm_check.py and WORKPLAN.md's "radius saturation, not
capacity" section for the full diagnosis. dist_scale=4 (target range
0.25-11.5) fixed it completely: rho 0.83 -> 0.9988, all failure modes to
zero. Bumping LATENT_DIM (12->24) was tried first and did nothing --
the ceiling is radial (one number, |v|, per point), not a capacity-count
problem, so more dimensions don't help.

A per-pair distance-based loss weight (favoring short distances) was
also tried, on the theory that averaging over ~32,640 mostly-far pairs
was starving the rare short-range ones. Ablated once dist_scale was
in place (--weight-power 0 vs 1): identical result either way (rho
0.9988 both, mean|embed-true| if anything slightly BETTER unweighted:
0.026 vs 0.039). The weighting was only ever compensating for the
saturation problem, not fixing anything real -- removed, per Andreas's
point that a scaffolding fix left in place after the real fix lands
just becomes clutter nobody re-checks.

At this maze's scale (256 cells -> 32,640 unique pairs), all of them fit
in one batch easily -- full-batch gradient descent every step, no
sampling needed.

Run:
    uv run python jepa_train_treeloss.py --geometry hyperbolic --dist-scale 4

Then check the result globally (not just on the training pairs -- there
are none held out at this scale, but treeloss_global_check.py's
distance-bucket table and far-pair-collapse check are still the
meaningful read):
    uv run python treeloss_global_check.py --geometry hyperbolic
"""
import argparse
import random

import torch

from env import ALGO, GRID_SIZE
from jepa_model import LATENT_DIM, Encoder, cell_index, squared_dist, to_geometry
from maze_gen import GENERATORS, bfs_distances_from
from maze_planner import MAZE_SEED

CHECKPOINT_PATH = "jepa_checkpoint_treeloss_{geometry}.pt"
N_STEPS = 3000
LR = 0.01
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["euclidean", "hyperbolic"], default="hyperbolic")
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dim", type=int, default=LATENT_DIM)
    ap.add_argument("--dist-scale", type=float, default=4.0,
                     help="divide true bfs_distance by this before using it as the "
                          "embedded-distance regression target -- see module "
                          "docstring. 4.0 is the value that resolved the radius-"
                          "saturation problem on this maze (diameter 46) -- retune "
                          "for a different GRID_SIZE, its diameter will differ.")
    ap.add_argument("--checkpoint-out", default=None,
                     help="override CHECKPOINT_PATH template (e.g. for a non-standard "
                          "GRID_SIZE, so it doesn't overwrite the standard-size "
                          "checkpoint). Defaults to jepa_checkpoint_treeloss_{geometry}.pt")
    args = ap.parse_args()
    geometry = args.geometry

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())
    num_cells = len(cells)
    size = GRID_SIZE

    # Every unique unordered pair of cells, with its TRUE distance (exact
    # edge-count along the tree's one unique path -- bfs_distances_from
    # does one BFS per cell, reused across all its pairs, cheaper than
    # calling bfs_distance separately for each of the ~32,640 pairs).
    pairs = []
    true_dists = []
    for i, a in enumerate(cells):
        dist_from_a = bfs_distances_from(adj, a)
        for j in range(i + 1, num_cells):
            b = cells[j]
            pairs.append((a, b))
            true_dists.append(dist_from_a[b])

    idx_i = torch.tensor([cell_index(a, size) for a, b in pairs])
    idx_j = torch.tensor([cell_index(b, size) for a, b in pairs])
    true_dist_raw = torch.tensor(true_dists, dtype=torch.float32)
    target_dist = true_dist_raw / args.dist_scale  # what the loss actually regresses toward
    print(f"[{geometry}] {num_cells} cells, {len(pairs)} pairs "
          f"(true distance range: {true_dist_raw.min().item():.0f}-{true_dist_raw.max().item():.0f}, "
          f"scaled target range: {target_dist.min().item():.2f}-{target_dist.max().item():.2f})")

    torch.manual_seed(args.seed)
    E = Encoder(num_cells, dim=args.dim)
    opt = torch.optim.Adam(E.parameters(), lr=args.lr)

    for step in range(args.steps):
        a_mapped = to_geometry(E(idx_i), geometry)
        b_mapped = to_geometry(E(idx_j), geometry)
        embed_dist = squared_dist(a_mapped, b_mapped, geometry).sqrt()  # (n_pairs,) -- compare
                                                                          # in actual distance units,
                                                                          # not squared, so it's on
                                                                          # the same scale as target_dist
        loss = ((embed_dist - target_dist) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        # Numerical-stability precaution: pushing pairs toward their true
        # distance in hyperbolic space means pushing points toward the
        # Poincare ball boundary, exactly the region hyperbolic.py's own
        # docstring flags as numerically delicate (torch.acosh's
        # derivative blows up there). Cheap insurance even with
        # dist_scale keeping targets inside the reachable range.
        torch.nn.utils.clip_grad_norm_(E.parameters(), max_norm=1.0)
        opt.step()

        if step % 500 == 0 or step == args.steps - 1:
            print(f"  [{geometry}] step {step}: loss={loss.item():.5f}  "
                  f"mean|embed-true|={(embed_dist - target_dist).abs().mean().item():.3f}")

    checkpoint_path = (args.checkpoint_out or CHECKPOINT_PATH).format(geometry=geometry)
    torch.save(
        {
            "E": E.state_dict(),
            "geometry": geometry,
            "num_cells": num_cells,
            "size": size,
            "dim": args.dim,
            "dist_scale": args.dist_scale,  # so downstream diagnostics (treeloss_global_check.py)
                                             # can scale their own absolute thresholds to match,
                                             # instead of assuming whatever scale happened to be
                                             # used at training time
        },
        checkpoint_path,
    )
    print(f"[{geometry}] checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
