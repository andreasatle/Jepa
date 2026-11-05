"""
pstep_onestep_error_check.py

Maze JEPA -- isolates P_step's own one-step prediction error from E's
quality. branch_point_diagnostic.py's DECISION accuracy (which action
CEM's argmin would pick) barely moved after fixing E to near-perfect
(rho 0.14 -> 0.9988 via jepa_train_treeloss.py + --dist-scale): 69.3% ->
70.8% overall, ~58% -> 66.8% near the target. If E were the dominant
error source, that number should now be close to 100%. It isn't, so this
checks the more basic question directly: for every real edge (b, real
neighbor), how far apart are P_step(E(b), action) and E(real neighbor)
in actual geometry distance -- not through the argmin decision layer at
all, just P_step's raw one-step regression quality.

Compares that error to the tree's own local scale (this maze's edges
embed at ~0.3 in the current treeloss E, per treeloss_global_check.py)
to judge whether P_step's error is small or large relative to what
"correct" looks like.

Run:
    uv run python pstep_onestep_error_check.py --geometry hyperbolic \\
        --encoder-checkpoint jepa_checkpoint_treeloss_hyperbolic.pt \\
        --step-checkpoint jepa_step_checkpoint_treeloss_hyperbolic.pt
"""
import argparse
import random
import statistics

import torch

from env import ACTIONS, ACTION_DELTA, ALGO, GRID_SIZE
from jepa_model import cell_index, squared_dist, to_geometry
from maze_gen import GENERATORS
from maze_planner import MAZE_SEED, load_models


def action_to_neighbor(cell, adj):
    out = {}
    r, c = cell
    for a in ACTIONS:
        dr, dc = ACTION_DELTA[a]
        nb = (r + dr, c + dc)
        if nb in adj[cell]:
            out[a] = nb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="hyperbolic")
    ap.add_argument("--encoder-checkpoint", default=None)
    ap.add_argument("--step-checkpoint", default=None)
    args = ap.parse_args()
    geometry = args.geometry

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())
    action_idx_map = {a: i for i, a in enumerate(ACTIONS)}

    E, P_step, size = load_models(geometry, args.encoder_checkpoint, args.step_checkpoint)

    errors = []          # P_step(E(b),a) vs E(real_neighbor), per real edge traversal
    edge_lengths = []    # E(b) vs E(real_neighbor) directly, for scale reference

    with torch.no_grad():
        for b in cells:
            a2n = action_to_neighbor(b, adj)
            b_idx = torch.tensor([cell_index(b, size)])
            b_raw = E(b_idx)
            for a, nb in a2n.items():
                a_idx = torch.tensor([action_idx_map[a]])
                pred_raw = P_step(b_raw, a_idx)
                pred_mapped = to_geometry(pred_raw, geometry)

                nb_idx = torch.tensor([cell_index(nb, size)])
                nb_mapped = to_geometry(E(nb_idx), geometry)
                b_mapped = to_geometry(b_raw, geometry)

                err = squared_dist(pred_mapped, nb_mapped, geometry).sqrt().item()
                true_edge_len = squared_dist(b_mapped, nb_mapped, geometry).sqrt().item()
                errors.append(err)
                edge_lengths.append(true_edge_len)

    print(f"[{geometry}] {len(errors)} directed real-edge traversals")
    print(f"  true edge length (E(b) to E(real neighbor)): mean={statistics.mean(edge_lengths):.3f}  "
          f"median={statistics.median(edge_lengths):.3f}")
    print(f"  P_step one-step error (P_step(E(b),a) to E(real neighbor)): "
          f"mean={statistics.mean(errors):.3f}  median={statistics.median(errors):.3f}  "
          f"min={min(errors):.3f}  max={max(errors):.3f}")
    print(f"  error / true-edge-length ratio: mean={statistics.mean(errors)/statistics.mean(edge_lengths):.2f}x "
          f"(1.0x would mean P_step's error is as large as the step it's supposed to predict)")


if __name__ == "__main__":
    main()
