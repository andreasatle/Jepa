"""
attractor_geometry_check.py

Maze JEPA -- direct test of a specific question about the "attractor"
mechanism found by branch_cell_inspect.py at b=(9,6)/(3,0): is the
dead-end action's argmin win (it looks spuriously cheap against almost
every target) a HYPERBOLIC-CURVATURE-SPECIFIC artifact, the same family
of scale-sensitivity that made --residual break at size 10 (a
parameterization/scale bug that only showed up once removed via direct
ablation, not by assuming it away)? Or does the same win happen under a
plain Euclidean metric applied to the identical predicted vectors, which
would mean it's a generic centrality/target-imbalance bias independent
of curvature?

This does NOT retrain anything and does NOT change dist_scale. It takes
the SAME raw/tangent-space vectors (E(b), P_step(b,a) for every action,
E(t) for every target) from a single frozen checkpoint pair, and computes
the argmin win-count TWICE: once mapping through to_geometry(...,
'hyperbolic') + hyp.dist (the actual planner's metric), once skipping the
exp0 map entirely and using plain squared Euclidean distance on the raw
vectors. If the dead-end action still wins the same lopsided majority
under the Euclidean metric, the mechanism isn't a hyperbolic-curvature
artifact -- it's already present in the raw predicted geometry, and
hyperbolic distance is just inheriting it, not causing it.

Run:
    uv run python attractor_geometry_check.py \\
        --encoder-checkpoint jepa_checkpoint_treeloss_size10_hyperbolic.pt \\
        --step-checkpoint jepa_step_checkpoint_treeloss_size10_norresidual_hyperbolic.pt \\
        --cells "(9,6)" "(3,0)"
"""
import argparse
import ast
import random

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
    ap.add_argument("--encoder-checkpoint", default=None)
    ap.add_argument("--step-checkpoint", default=None)
    ap.add_argument("--cells", nargs="+", required=True,
                     help='cells to inspect, e.g. --cells "(9,6)" "(3,0)"')
    args = ap.parse_args()

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    # trained checkpoints are always hyperbolic in this project's current phase --
    # loading via 'hyperbolic' just gets us the raw/tangent vectors; the point of
    # this script is to then map those SAME raw vectors through both geometries.
    E, P_step, size = load_models("hyperbolic", args.encoder_checkpoint, args.step_checkpoint)
    action_idx_map = {a: i for i, a in enumerate(ACTIONS)}
    cells_ordered = [(r, c) for r in range(size) for c in range(size)]

    with torch.no_grad():
        for cell_str in args.cells:
            b = ast.literal_eval(cell_str)
            a2n = action_to_neighbor(b, adj)
            action_list = list(a2n.keys())

            b_idx = torch.tensor([cell_index(b, size)])
            b_raw = E(b_idx)
            next_raw = {}
            for a in action_list:
                a_idx = torch.tensor([action_idx_map[a]])
                next_raw[a] = P_step(b_raw, a_idx)

            targets = [c for c in cells_ordered if c != b]
            target_idx = torch.tensor([cell_index(t, size) for t in targets])
            target_raw = E(target_idx)

            print(f"\n{'='*70}\nb={b}  degree={len(action_list)}")
            for geom in ["hyperbolic", "euclidean"]:
                target_mapped = to_geometry(target_raw, geom)
                cost_rows = []
                for a in action_list:
                    nm = to_geometry(next_raw[a], geom)
                    cost_rows.append(squared_dist(nm.expand(len(targets), -1), target_mapped, geom))
                cost_stack = torch.stack(cost_rows, dim=0)
                pred_best = cost_stack.argmin(dim=0)
                win_counts = {a: 0 for a in action_list}
                for i in range(len(targets)):
                    win_counts[action_list[pred_best[i].item()]] += 1
                summary = ", ".join(
                    f"{a}={win_counts[a]}/{len(targets)} ({100.0*win_counts[a]/len(targets):.1f}%)"
                    for a in action_list
                )
                print(f"  metric={geom:>10}: {summary}")

    print("\n(if the SAME action wins the same lopsided majority under both metrics, "
          "the attractor is already present in the raw predicted vectors -- not something "
          "hyperbolic distance/curvature is causing or amplifying)")


if __name__ == "__main__":
    main()
