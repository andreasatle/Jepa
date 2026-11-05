"""
embedding_tree_metric_check.py

Maze JEPA -- cheap diagnostic (Andreas's question): does the trained
encoder E actually embed the tree so that a child is closer to its
parent than to a sibling -- i.e. |c1-p| < |c1-c2|? This is the specific
geometric signature the "hyperbolic space suits trees" claim rests on:
place a parent near the origin, fan its children apart angularly so
different children end up far from EACH OTHER while each stays close
to the parent. It's also just the real graph fact -- the only path
between two children of the same parent goes through that parent,
there is no shortcut, so a metric-faithful embedding should reflect
that ordering.

Important caveat, unlike classical tree embeddings (Sarkar's
construction, Poincare embeddings), which directly optimize embedding
distance to match graph distance for every pair: this project's E was
never given that objective. It comes out of Phase 2's P (predicting
the horizon-steps-later reachable cell) plus VICReg anti-collapse -- a
much more indirect signal. So the parent-closer-than-sibling property
is plausible and hoped for, not guaranteed by the training loss. This
script checks it directly against the already-trained checkpoint, no
retraining, for whichever geometry is requested -- run it for both to
see whether hyperbolic actually respects the tree metric better than
euclidean does here, or whether this property is incidental/absent in
both.

Roots the maze's undirected adjacency tree at an arbitrary fixed cell
(ROOT, via BFS) to get parent/child structure -- a cell's number of
children in this rooted tree is exactly its junction branching factor
(degree minus 1 for the edge back to its own parent, or degree for
ROOT itself), matching the project's own "genuine fork" definition
from Phase 1. For every such junction cell and every unordered pair of
its children {c1, c2}, checks both:
    d(c1, parent) < d(c1, c2)
    d(c2, parent) < d(c1, c2)
(squared_dist is monotonic in true distance, so no sqrt is needed for
pure ordering comparisons.) Reports, per geometry: what fraction of
individual child-side checks hold, and how many sibling PAIRS are
fully consistent (both sides hold), partially consistent (one side),
or inverted (neither -- the embedding says two siblings are closer to
each other than either is to their own parent, directly contradicting
the maze's actual tree structure).

Run (after jepa_train.py has been run for the geometry being checked):
    uv run python embedding_tree_metric_check.py --geometry hyperbolic
    uv run python embedding_tree_metric_check.py --geometry euclidean
"""
import argparse
import random
from collections import deque

import torch

from env import ALGO, GRID_SIZE
from maze_gen import GENERATORS
from jepa_model import cell_index, load_encoder_checkpoint, squared_dist, to_geometry

MAZE_SEED = 0  # same fixed maze as everywhere else in this project
ROOT = (0, 0)  # arbitrary fixed root cell, just to define parent/child structure


def build_tree(adj, root):
    """BFS from root over the undirected adjacency graph (already a tree,
    since every maze here is a perfect/spanning-tree maze -- see
    maze_gen.py) -- returns children: dict cell -> list of child cells
    (root's own "parent" is None, not represented)."""
    parent = {root: None}
    order = [root]
    queue = deque([root])
    while queue:
        cur = queue.popleft()
        for nb in adj[cur]:
            if nb not in parent:
                parent[nb] = cur
                order.append(nb)
                queue.append(nb)
    children = {c: [] for c in order}
    for c in order:
        if parent[c] is not None:
            children[parent[c]].append(c)
    return children


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic"], required=True)
    args = parser.parse_args()
    geometry = args.geometry

    E, ckpt = load_encoder_checkpoint(geometry)
    size = ckpt["size"]

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    children = build_tree(adj, ROOT)
    all_cells = list(adj.keys())
    idx = {c: i for i, c in enumerate(all_cells)}

    with torch.no_grad():
        cell_idx_tensor = torch.tensor([cell_index(c, size) for c in all_cells])
        raw = E(cell_idx_tensor)
        mapped = to_geometry(raw, geometry)  # (N, dim)

    def dist(c1, c2):
        return squared_dist(mapped[idx[c1]:idx[c1] + 1], mapped[idx[c2]:idx[c2] + 1], geometry).item()

    n_junctions = sum(1 for kids in children.values() if len(kids) >= 2)
    n_pairs = 0
    n_side_checks = 0
    n_side_holds = 0
    n_fully_consistent = 0
    n_partially_consistent = 0
    n_inverted = 0
    margins = []

    for p, kids in children.items():
        if len(kids) < 2:
            continue
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                c1, c2 = kids[i], kids[j]
                d_c1p = dist(c1, p)
                d_c2p = dist(c2, p)
                d_c1c2 = dist(c1, c2)

                n_pairs += 1
                side1 = d_c1p < d_c1c2
                side2 = d_c2p < d_c1c2
                n_side_checks += 2
                n_side_holds += int(side1) + int(side2)
                if side1 and side2:
                    n_fully_consistent += 1
                elif side1 or side2:
                    n_partially_consistent += 1
                else:
                    n_inverted += 1
                margins.append((d_c1c2 - max(d_c1p, d_c2p)) / d_c1c2 if d_c1c2 > 0 else 0.0)

    mean_margin = sum(margins) / len(margins)
    print(f"[{geometry}] {len(children)} cells rooted at {ROOT}, {n_junctions} junctions "
          f"(cells with >=2 children), {n_pairs} sibling pairs checked")
    print(f"[{geometry}] per-child-side checks: {n_side_holds}/{n_side_checks} "
          f"({n_side_holds / n_side_checks:.1%}) satisfy dist(child,parent) < dist(child,sibling)")
    print(f"[{geometry}] per-sibling-pair: fully_consistent(both sides)={n_fully_consistent}/{n_pairs} "
          f"partially_consistent(one side)={n_partially_consistent}/{n_pairs} "
          f"inverted(neither side)={n_inverted}/{n_pairs}")
    print(f"[{geometry}] mean margin (d(c1,c2) - max(d(c1,p),d(c2,p))) / d(c1,c2): {mean_margin:.3f} "
          f"(positive = going through the parent is shorter on average, by this fraction of d(c1,c2); "
          f"negative = the direct sibling-to-sibling embedding distance is shorter than either child's "
          f"distance to their own parent -- a direct contradiction of the tree structure)")


if __name__ == "__main__":
    main()
