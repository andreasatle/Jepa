"""
embedding_metric_check.py

Maze JEPA -- direct check of E's own local geometry, independent of
P_step and the planner entirely: do REAL-WORLD-close cells (nearby in
the maze graph) actually get embedded close together by E? Everything
tested so far this session (step_drift_diagnostic.py, maze_planner.py's
cost-correlation and oracle checks) tests P_step's predictions or the
planner's search -- this instead asks the more basic question directly,
about E alone. Andreas's framing: are we just assuming "nearby real
cells => nearby embeddings" is emergent from Phase 2's training
objective (predicting H-step random-walk endpoint distributions),
without ever having actually checked it?

For every pair of cells, computes the REAL BFS distance (exact -- a
perfect maze has exactly one path between any two cells, via
maze_gen.bfs_distances_from) and E's latent distance (squared_dist,
geometry-appropriate, on E's raw embeddings mapped onto the geometry).
Reports two things:

  - the GLOBAL Spearman rank correlation between real distance and
    latent distance across every pair (n = C(num_cells, 2)) -- does
    latent distance track real distance at all, at every scale?
  - mean latent distance broken out PER real-BFS-distance bucket
    (1, 2, 3, ...) -- specifically whether DIRECT NEIGHBORS (BFS
    distance 1, i.e. one action apart -- exactly the transitions
    P_step is trained on) are embedded meaningfully closer together
    than farther-apart cells, not just that the overall ranking is
    right. If bucket 1's mean is close to bucket 20's mean, "nearby
    real cells get nearby embeddings" is NOT actually true locally,
    even if the global rho looks fine -- and that would mean the
    local-smoothness property step_drift_diagnostic and the planner's
    leftover-distance checks have been implicitly relying on was never
    really there.

Run (in Andreas's terminal, after jepa_train.py has produced
jepa_checkpoint_<geometry>.pt for the geometry being checked):
    uv run python embedding_metric_check.py --geometry euclidean
    uv run python embedding_metric_check.py --geometry hyperbolic
"""
import argparse
import random

import torch

from env import GRID_SIZE, ALGO
from maze_gen import GENERATORS, bfs_distances_from
from jepa_model import load_encoder_checkpoint, squared_dist, to_geometry

MAZE_SEED = 0  # same fixed maze as everywhere else in this project


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0] * len(xs)
    for rank, i in enumerate(order):
        ranks[i] = rank
    return ranks


def spearman_correlation(xs, ys):
    n = len(xs)
    rx, ry = _ranks(xs), _ranks(ys)
    mean_rx, mean_ry = (n - 1) / 2, (n - 1) / 2
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / (var_x * var_y) ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic"], required=True)
    args = parser.parse_args()
    geometry = args.geometry

    E, ckpt = load_encoder_checkpoint(geometry)
    size = ckpt["size"]
    n_cells = size * size

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    assert len(adj) == n_cells

    with torch.no_grad():
        idx = torch.arange(n_cells)
        raw = E(idx)                          # (N, dim)
        mapped = to_geometry(raw, geometry)    # (N, dim)
        # broadcasting (N, 1, dim) against (1, N, dim) -- squared_dist /
        # hyp.dist are written to broadcast over leading dims, so this
        # gives the full (N, N) pairwise distance matrix in one call,
        # no explicit N^2 python loop needed
        latent_dist = squared_dist(mapped.unsqueeze(1), mapped.unsqueeze(0), geometry)  # (N, N)

    # real BFS distance for every pair, one BFS per source cell (cheap:
    # N=256 BFS passes over a 256-cell graph)
    real_dist = {}
    for r in range(size):
        for c in range(size):
            start = (r, c)
            start_idx = r * size + c
            distances = bfs_distances_from(adj, start)
            for (rr, cc), d in distances.items():
                target_idx = rr * size + cc
                real_dist[(start_idx, target_idx)] = d

    real_flat, latent_flat = [], []
    buckets = {}  # real bfs distance -> list of latent distances
    for (i, j), d in real_dist.items():
        if i >= j:
            continue  # unordered pairs, count each once
        l = latent_dist[i, j].item()
        real_flat.append(d)
        latent_flat.append(l)
        buckets.setdefault(d, []).append(l)

    rho = spearman_correlation(real_flat, latent_flat)
    print(f"[{geometry}] real BFS distance vs latent distance, ALL {len(real_flat)} pairs: "
          f"Spearman rho={rho:.3f}")

    print(f"[{geometry}] latent distance by real BFS-distance bucket (mean, n):")
    for d in sorted(buckets):
        vals = buckets[d]
        mean_v = sum(vals) / len(vals)
        print(f"  bfs_distance={d:>3}  n={len(vals):>5}  mean_latent_dist={mean_v:.4f}")


if __name__ == "__main__":
    main()
