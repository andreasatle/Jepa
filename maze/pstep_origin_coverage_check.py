"""
pstep_origin_coverage_check.py

Maze JEPA -- Phase 3 diagnostic, follow-up to pstep_manifold_drift_check.py's
Part B result on the size-10/UNROLL_STEPS=8 pair: the [0-1) raw-norm bucket
came back with a mean expansion factor of 1.19 (P_step AMPLIFIES there),
a reversal from the size-16 residual-P_step config, which was contractive
(0.98) in that same bucket. That bucket is also small (only 3 of 40 sampled
cells fell in it), which suggests a real hypothesis: P_step is unstable
near the embedding origin specifically because it rarely sees training
transitions anchored there, not because of anything horizon-length-related.

This checks that hypothesis directly, no retrain needed:
  1. Which real maze cells actually have raw embedding norm < 1 (the
     unstable bucket)? Prints their coordinates, degree (junction-ness),
     and exact raw norm.
  2. How often does each near-origin cell actually appear as a real state
     in the training data (both random-walk and directed pools), compared
     to a uniform 1/num_cells base rate? A ratio well below 1.0 means
     these cells are structurally under-visited, not just coincidentally
     rare in a 40-cell sample.

Run:
    uv run python pstep_origin_coverage_check.py --geometry hyperbolic \
        --encoder-checkpoint jepa_checkpoint_treeloss_size10_hyperbolic.pt \
        --data-path maze_data_size10.json \
        --directed-data-path maze_data_directed_size10.json
"""
import argparse
import json
import random

from env import ALGO, GRID_SIZE
from maze_gen import GENERATORS
from jepa_model import load_encoder_checkpoint

ORIGIN_THRESHOLD = 1.0  # matches pstep_manifold_drift_check.py's BUCKET_WIDTH boundary
MAZE_SEED = 0


def visit_counts(path):
    with open(path) as f:
        data = json.load(f)
    counts = {}
    total_steps = 0
    for ep in data["episodes"]:
        for s in ep["states"]:
            key = tuple(s)
            counts[key] = counts.get(key, 0) + 1
            total_steps += 1
    return counts, total_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="hyperbolic")
    ap.add_argument("--encoder-checkpoint", default=None,
                     help="override checkpoint path template")
    ap.add_argument("--data-path", default="maze_data.json")
    ap.add_argument("--directed-data-path", default="maze_data_directed.json")
    args = ap.parse_args()

    E, ckpt = load_encoder_checkpoint(
        args.geometry,
        path=args.encoder_checkpoint or "jepa_checkpoint_treeloss_{geometry}.pt",
    )
    size = ckpt["size"]
    raw = E.table.weight.detach()
    norms = raw.norm(dim=-1)

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    # NOTE: adj's own key order (insertion order from the maze generator) does NOT
    # match cell_index()'s row-major convention -- build cells explicitly in that
    # order instead, and look degree up in adj by coordinate (dict, order-independent).
    cells = [(r, c) for r in range(size) for c in range(size)]
    assert len(cells) == len(norms) == size * size, "cell ordering mismatch -- check cell_index()"

    near_origin = [(cells[i], float(norms[i]), len(adj[cells[i]]))
                   for i in range(len(cells)) if norms[i] < ORIGIN_THRESHOLD]
    near_origin.sort(key=lambda t: t[1])

    print(f"[{args.geometry}] {len(cells)} cells, {len(near_origin)} with raw norm < {ORIGIN_THRESHOLD}:")
    for cell, n, degree in near_origin:
        print(f"  cell={cell}  raw_norm={n:.4f}  degree={degree}")

    degrees = [len(nbs) for nbs in adj.values()]
    print(f"  (for context: degree across all {len(cells)} cells -- "
          f"mean={sum(degrees)/len(degrees):.2f}  max={max(degrees)})")

    if not near_origin:
        print("no near-origin cells found -- nothing to check visit frequency for")
        return

    rw_counts, rw_total = visit_counts(args.data_path)
    dir_counts, dir_total = visit_counts(args.directed_data_path)
    uniform_rate = 1.0 / len(cells)

    print(f"\nvisit frequency vs uniform base rate ({uniform_rate:.5f}):")
    print(f"  random-walk pool ({args.data_path}): {rw_total} total state-steps")
    print(f"  directed pool ({args.directed_data_path}): {dir_total} total state-steps")
    for cell, n, degree in near_origin:
        key = tuple(cell)
        rw_rate = rw_counts.get(key, 0) / rw_total
        dir_rate = dir_counts.get(key, 0) / dir_total
        print(f"  cell={cell} (raw_norm={n:.4f}, degree={degree}): "
              f"random-walk rate={rw_rate:.5f} ({rw_rate/uniform_rate:.2f}x uniform)  "
              f"directed rate={dir_rate:.5f} ({dir_rate/uniform_rate:.2f}x uniform)")


if __name__ == "__main__":
    main()
