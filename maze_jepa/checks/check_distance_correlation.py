"""
checks/check_distance_correlation.py

Does the trained (E, Predictor) pair for one geometry actually give
embedding distance a relationship to real maze distance -- the
property planner_greedy.py's and planner_cem.py's failures showed was
missing, and that the negative-sampling repulsion in jepa_train.py was
added to try to induce (see that module's NEGATIVE-SAMPLING REPULSION
note)?

For every pair of cells (i, j), compute squared_dist(E(i), E(j)) (post
to_geometry -- the same metric the planner uses) and the TRUE maze
distance bfs_dist(i, j) (shortest path over env.adj, computed once via
BFS from every cell -- this script's only use of graph knowledge, and
only for MEASURING the result afterward, never for training or for the
planner itself).

Reports Pearson AND Spearman correlation between the two. Spearman
(rank) correlation matters more here than Pearson -- a distance-based
planner only needs "farther in the maze -> farther in embedding space"
to hold MONOTONICALLY; the exact numeric relationship doesn't need to
be linear, and hyperbolic distances in particular aren't expected to
grow linearly with hop count. Also prints mean embedding-distance per
true-distance bucket, so a non-monotonic region (if any) is visible
directly rather than averaged away by a single correlation number.

Run (from inside maze_jepa/):
    uv run python checks/check_distance_correlation.py --geometry hyperbolic
    uv run python checks/check_distance_correlation.py --geometry euclidean
"""
import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from env import MazeEnv
from jepa_model import cell_index, load_encoder_checkpoint, squared_dist, to_geometry


def bfs_all_pairs(env):
    """dist[a][b] = shortest-path hop count -- BFS from every cell.
    Used ONLY to measure the trained embedding's distance correlation
    afterward, never as a training signal."""
    dist = {}
    for start in env.adj:
        d = {start: 0}
        q = deque([start])
        while q:
            cell = q.popleft()
            for n in env.adj[cell]:
                if n not in d:
                    d[n] = d[cell] + 1
                    q.append(n)
        dist[start] = d
    return dist


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx ** 0.5 * vy ** 0.5)


def _rank(vals):
    """Average (fractional) ranks, ties split evenly -- standard
    Spearman convention."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    return pearson(_rank(xs), _rank(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["euclidean", "hyperbolic"], default="hyperbolic")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--algo", default="prim")
    ap.add_argument("--maze-seed", type=int, default=0)
    args = ap.parse_args()

    env = MazeEnv(size=args.size, algo=args.algo, seed=args.maze_seed)
    E, _ = load_encoder_checkpoint(args.geometry)

    cells = list(env.adj.keys())
    bfs = bfs_all_pairs(env)

    num_cells = args.size * args.size
    with torch.no_grad():
        all_raw = E(torch.arange(num_cells))
        all_geo = to_geometry(all_raw, args.geometry)

        true_dists = []
        emb_dists = []
        by_true = defaultdict(list)
        for i, a in enumerate(cells):
            for b in cells[i + 1:]:
                td = bfs[a][b]
                ia = cell_index(a, args.size)
                ib = cell_index(b, args.size)
                ed = squared_dist(all_geo[ia:ia + 1], all_geo[ib:ib + 1], args.geometry).item()
                true_dists.append(td)
                emb_dists.append(ed)
                by_true[td].append(ed)

    r_pearson = pearson(true_dists, emb_dists)
    r_spearman = spearman(true_dists, emb_dists)
    print(f"pairs compared: {len(true_dists)}")
    print(f"Pearson correlation (true maze distance vs embedding squared_dist): {r_pearson:.3f}")
    print(f"Spearman (rank) correlation: {r_spearman:.3f}")

    print("mean +/- std embedding squared_dist by true maze distance:")
    for td in sorted(by_true):
        vals = by_true[td]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        print(f"  true_dist={td:2d} (n={len(vals):4d}): mean_emb_dist={mean:7.3f} +/- {std:6.3f}")

    # Overlap check: how much do adjacent true-distance buckets overlap in embedding-distance
    # range? A distance-based planner needs buckets to mostly separate, not just have
    # increasing means -- this is the direct look at whether within-bucket noise (not just
    # the mean trend) is what caps the correlation.
    sorted_tds = sorted(by_true)
    print("adjacent-bucket overlap (fraction of bucket td+1 values below bucket td's mean):")
    for td, td_next in zip(sorted_tds, sorted_tds[1:]):
        mean_td = sum(by_true[td]) / len(by_true[td])
        vals_next = by_true[td_next]
        frac_below = sum(1 for v in vals_next if v < mean_td) / len(vals_next)
        print(f"  td={td:2d}->td={td_next:2d}: {100 * frac_below:5.1f}% of td={td_next} pairs embed "
              f"closer than td={td}'s mean")


if __name__ == "__main__":
    main()
