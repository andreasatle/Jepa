"""
checks/check_single_step.py

One-off diagnostic for a specific stuck case: does the trained
(E, Predictor) pair itself already prefer the correct final move, or
is the oscillation purely a CEM search artifact?

Prints squared_dist(P(E(cell), action), E(target)) for a few
hand-picked (cell, action) pairs -- bypasses CEM's population/sampling
entirely to isolate whether the model's own single-step preference
already points the right way.

Run (from inside maze_jepa/):
    uv run python checks/check_single_step.py --geometry hyperbolic
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from env import Action
from jepa_model import cell_index, load_encoder_checkpoint, load_step_checkpoint, squared_dist, to_geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["euclidean", "hyperbolic"], default="hyperbolic")
    ap.add_argument("--size", type=int, default=10)
    args = ap.parse_args()

    E, _ = load_encoder_checkpoint(args.geometry)
    P, _ = load_step_checkpoint(args.geometry)
    size = args.size

    target = (9, 0)
    cases = [
        ((9, 1), "LEFT"),
        ((9, 1), "UP"),
        ((8, 1), "DOWN"),
        ((8, 1), "UP"),
    ]

    with torch.no_grad():
        target_geo = to_geometry(E(torch.tensor([cell_index(target, size)])), args.geometry)
        for cell, action_name in cases:
            a = Action[action_name]
            start_raw = E(torch.tensor([cell_index(cell, size)]))
            a_delta = torch.tensor([a.value], dtype=torch.float32)
            pred_raw = P(start_raw, a_delta)
            pred_geo = to_geometry(pred_raw, args.geometry)
            d = squared_dist(pred_geo, target_geo, args.geometry).item()
            print(f"{cell} --{action_name}--> dist_to_E{target}={d:.4f}")

        print()
        for cell in [(9, 1), (8, 1), target]:
            cg = to_geometry(E(torch.tensor([cell_index(cell, size)])), args.geometry)
            d = squared_dist(cg, target_geo, args.geometry).item()
            print(f"E{cell} raw dist to E{target}: {d:.4f}")


if __name__ == "__main__":
    main()
