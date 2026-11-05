"""
checks/jepa_test.py

Sanity check for a trained (E, Predictor) pair: is what jepa_train.py
produced actually meaningful, or numeric nonsense that happens to
minimize squared error?

Raw regression loss (pred_loss) doesn't answer this on its own -- a low
mean squared distance could still leave the WRONG cell as the nearest
neighbor if cells are clustered close together. The real test: for
every (cell, action) pair, predict the next embedding, then find which
of the 100 cells' embeddings it's actually closest to (argmin over ALL
cells, not just a comparison against the true target). If that argmin
matches the true next cell, the prediction is *meaningful* -- it
actually identifies the right cell, not just lands numerically close to
something.

jepa_train.py's training data is back to the FULL 4-action space per
cell (wall-bumps reinstated, targeting the SAME cell -- see that
module's docstring for why), so the wall-bump-vs-real-move accuracy
split is meaningful again: a wall-bump pair is "correct" if the argmin
lands back on the cell itself.

DEGREE BREAKDOWN (real moves only): a cell's degree (its number of
valid actions) sets how many DIFFERENT correct targets the Predictor
must distinguish from that one starting embedding, using only the 2-dim
action delta to tell them apart. Junctions (degree 3-4) are the hardest
case for exactly that reason -- bucketing accuracy by degree shows
whether failures concentrate there rather than guessing from a handful
of examples. Wall-bump pairs aren't bucketed by degree -- they all share
the same "correct answer" (the same cell), so degree isn't the relevant
axis for them.

Run (from inside maze_jepa/):
    uv run python checks/jepa_test.py --geometry hyperbolic
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from env import MazeEnv
from jepa_model import (
    cell_index,
    load_encoder_checkpoint,
    load_step_checkpoint,
    squared_dist,
    to_geometry,
)
from jepa_train import enumerate_transitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["euclidean", "hyperbolic"], default="hyperbolic")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--algo", default="prim")
    ap.add_argument("--maze-seed", type=int, default=0)
    args = ap.parse_args()

    env = MazeEnv(size=args.size, algo=args.algo, seed=args.maze_seed)
    E, _ = load_encoder_checkpoint(args.geometry)
    P, _ = load_step_checkpoint(args.geometry)

    degree = {cell: len(env.valid_actions(cell)) for cell in env.adj}

    num_cells = args.size * args.size
    with torch.no_grad():
        all_raw = E(torch.arange(num_cells))
        all_geo = to_geometry(all_raw, args.geometry)

        transitions = enumerate_transitions(env)
        correct = 0
        wall_bump = {"correct": 0, "total": 0}
        real_move = {"correct": 0, "total": 0}
        by_degree = defaultdict(lambda: {"correct": 0, "total": 0})  # real moves only
        wrong_examples = []
        for cell, a, next_cell in transitions:
            is_wall_bump = next_cell == cell
            deg = degree[cell]

            s_idx = torch.tensor([cell_index(cell, args.size)])
            a_delta = torch.tensor([a.value], dtype=torch.float32)
            true_idx = cell_index(next_cell, args.size)

            start_raw = E(s_idx)
            pred_raw = P(start_raw, a_delta)
            pred_geo = to_geometry(pred_raw, args.geometry)

            dists = squared_dist(pred_geo.expand(num_cells, -1), all_geo, args.geometry)
            pred_idx = dists.argmin().item()

            bucket = wall_bump if is_wall_bump else real_move
            bucket["total"] += 1
            if not is_wall_bump:
                by_degree[deg]["total"] += 1

            if pred_idx == true_idx:
                correct += 1
                bucket["correct"] += 1
                if not is_wall_bump:
                    by_degree[deg]["correct"] += 1
            elif len(wrong_examples) < 10:
                pred_cell = (pred_idx // args.size, pred_idx % args.size)
                wrong_examples.append((cell, a.name, next_cell, pred_cell))

    n = len(transitions)
    print(f"nearest-cell accuracy: {correct}/{n} = {100 * correct / n:.1f}%")
    print(f"  wall-bump accuracy:  {wall_bump['correct']}/{wall_bump['total']} = "
          f"{100 * wall_bump['correct'] / wall_bump['total']:.1f}%")
    print(f"  real-move accuracy:  {real_move['correct']}/{real_move['total']} = "
          f"{100 * real_move['correct'] / real_move['total']:.1f}%")

    print("by cell degree (real moves only, number of valid actions from that cell):")
    for deg in sorted(by_degree):
        s = by_degree[deg]
        print(f"  degree {deg}: {s['correct']}/{s['total']} = {100 * s['correct'] / s['total']:.1f}%")

    if wrong_examples:
        print("first wrong predictions (cell, action -> true_next, predicted_cell):")
        for cell, aname, next_cell, pred_cell in wrong_examples:
            print(f"  {cell} {aname} -> true {next_cell}, predicted {pred_cell}")


if __name__ == "__main__":
    main()
