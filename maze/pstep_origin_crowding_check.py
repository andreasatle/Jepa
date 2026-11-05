"""
pstep_origin_crowding_check.py

Maze JEPA -- Phase 3 diagnostic, follow-up to pstep_origin_coverage_check.py:
that script refuted the "sparse training data near the origin" hypothesis --
the 14 near-origin cells (raw norm < 1) are actually OVER-represented in
training data (1-5x uniform rate), not under. So the instability
pstep_manifold_drift_check.py found there (expansion factor 1.19) isn't a
data-count problem.

Alternative hypothesis: representational crowding. 14 of 100 cells (14%)
are packed into raw norm < 1, a small fraction of E's full range (max
3.486). If these 14 cells are packed unusually close together in EMBEDDED
distance relative to what their TRUE bfs_distance from each other would
predict (i.e. embedded distance is smaller than the treeloss_global_check.py
bucket mean for their true distance), that's a resolution problem: P_step
has to make finer distinctions per unit raw-space movement there than
elsewhere, which could show up as exactly this kind of local instability,
independent of how much training data covers it.

Computes, for all pairs among the 14 near-origin cells: true bfs_distance,
actual embedded (hyperbolic) distance, and the ratio of that to the bucket
mean embedded distance for pairs at the same true distance elsewhere in the
maze (from treeloss_global_check.py's own bucketing) -- a ratio well below
1.0 across the board means these cells really are unusually compressed
together, not just coincidentally central.

Run:
    uv run python pstep_origin_crowding_check.py --geometry hyperbolic \
        --encoder-checkpoint jepa_checkpoint_treeloss_size10_hyperbolic.pt
"""
import argparse
import itertools
import random

import torch

from env import ALGO, GRID_SIZE
from maze_gen import GENERATORS, bfs_distances_from
from jepa_model import load_encoder_checkpoint, cell_index, to_geometry, squared_dist

ORIGIN_THRESHOLD = 1.0
MAZE_SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="hyperbolic")
    ap.add_argument("--encoder-checkpoint", default=None)
    args = ap.parse_args()

    E, ckpt = load_encoder_checkpoint(
        args.geometry,
        path=args.encoder_checkpoint or "jepa_checkpoint_treeloss_{geometry}.pt",
    )
    size = ckpt["size"]
    dist_scale = ckpt.get("dist_scale", 1.0)
    raw = E.table.weight.detach()
    norms = raw.norm(dim=-1)

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = [(r, c) for r in range(size) for c in range(size)]

    near_origin = [cells[i] for i in range(len(cells)) if norms[i] < ORIGIN_THRESHOLD]
    print(f"[{args.geometry}] {len(near_origin)} near-origin cells (raw norm < {ORIGIN_THRESHOLD})")

    print("caching all-pairs bfs distances...")
    bfs_from_cache = {c: bfs_distances_from(adj, c) for c in cells}

    def bucket_mean(true_d):
        pairs = []
        attempts = 0
        rng = random.Random(0)
        while len(pairs) < 300 and attempts < 20000:
            attempts += 1
            a, b = rng.sample(cells, 2)
            d = bfs_from_cache[a][b]
            if d == true_d:
                pairs.append((a, b))
        if not pairs:
            return None
        idx_a = torch.tensor([cell_index(a, size) for a, b in pairs])
        idx_b = torch.tensor([cell_index(b, size) for a, b in pairs])
        emb_a = to_geometry(raw[idx_a], args.geometry)
        emb_b = to_geometry(raw[idx_b], args.geometry)
        d_emb = squared_dist(emb_a, emb_b, args.geometry).clamp_min(0).sqrt()
        return d_emb.mean().item()

    print(f"\n{'cell A':>10} {'cell B':>10} {'true_d':>7} {'embedded_d':>11} {'bucket_mean':>12} {'ratio':>7}")
    bucket_mean_cache = {}
    ratios = []
    for a, b in itertools.combinations(near_origin, 2):
        true_d = bfs_from_cache[a][b]
        ia, ib = cell_index(a, size), cell_index(b, size)
        ea = to_geometry(raw[ia:ia+1], args.geometry)
        eb = to_geometry(raw[ib:ib+1], args.geometry)
        d_emb = squared_dist(ea, eb, args.geometry).clamp_min(0).sqrt().item()
        if true_d not in bucket_mean_cache:
            bucket_mean_cache[true_d] = bucket_mean(true_d)
        bm = bucket_mean_cache[true_d]
        if bm is None or bm == 0:
            continue
        ratio = d_emb / bm
        ratios.append(ratio)
        print(f"{str(a):>10} {str(b):>10} {true_d:>7} {d_emb:>11.4f} {bm:>12.4f} {ratio:>7.3f}")

    print(f"\nmean ratio (embedded_d / same-true-distance bucket mean) across {len(ratios)} near-origin pairs: "
          f"{sum(ratios)/len(ratios):.3f}")
    print("(< 1.0 means these cells sit closer together in embedded space than a typical pair at the same "
          "true distance -- i.e. genuinely compressed/crowded, not just coincidentally central)")


if __name__ == "__main__":
    main()
