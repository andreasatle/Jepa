"""
branch_cell_inspect.py

Maze JEPA -- drill-down on branch_error_clustering_check.py's headline
finding: b=(9,6) and b=(3,0) are wrong on ~97-98% of their 99 possible
targets, confidently so, regardless of which target is asked. That
target-independence is the key clue -- if a SINGLE wrong action wins the
argmin for nearly every target, that's an "attractor" failure (P_step
mispredicts one action's outcome to land somewhere that happens to look
close to almost everything), not a general ranking problem.

For each inspected cell b: prints its real neighbors/actions, its own
raw norm, and for each action: P_step's predicted next-state raw norm,
its actual embedded distance to the TRUE neighbor's embedding (should be
small if P_step is accurate for that specific transition), and how often
(across all 99 targets) that action wins the argmin -- a lopsided win
count (one action winning almost always) is the "attractor" signature;
a roughly-even split across all deg(b) actions would point elsewhere.

Run:
    uv run python branch_cell_inspect.py --geometry hyperbolic \
        --encoder-checkpoint jepa_checkpoint_treeloss_size10_hyperbolic.pt \
        --step-checkpoint jepa_step_checkpoint_treeloss_size10_norresidual_hyperbolic.pt \
        --cells "(9,6)" "(3,0)"
"""
import argparse
import ast
import random
from collections import deque

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


def components_excluding(adj, b):
    comp_of = {}
    for start in adj[b]:
        if start in comp_of:
            continue
        seen = {start}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb == b or nb in seen:
                    continue
                seen.add(nb)
                q.append(nb)
        for cell in seen:
            comp_of[cell] = start
    return comp_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="hyperbolic")
    ap.add_argument("--encoder-checkpoint", default=None)
    ap.add_argument("--step-checkpoint", default=None)
    ap.add_argument("--cells", nargs="+", required=True,
                     help='cells to inspect, e.g. --cells "(9,6)" "(3,0)"')
    args = ap.parse_args()
    geometry = args.geometry

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())
    E, P_step, size = load_models(geometry, args.encoder_checkpoint, args.step_checkpoint)
    action_idx_map = {a: i for i, a in enumerate(ACTIONS)}

    with torch.no_grad():
        for cell_str in args.cells:
            b = ast.literal_eval(cell_str)
            if b not in adj:
                print(f"b={b}: not a valid cell, skipping")
                continue

            deg = len(adj[b])
            a2n = action_to_neighbor(b, adj)
            neighbor_to_action = {nb: a for a, nb in a2n.items()}
            comp_of = components_excluding(adj, b)

            b_idx = torch.tensor([cell_index(b, size)])
            b_raw = E(b_idx)
            b_norm = b_raw.norm().item()

            print(f"\n{'='*70}\nb={b}  degree={deg}  raw_norm={b_norm:.4f}")
            print(f"  real neighbors: { {a: nb for a, nb in a2n.items()} }")

            action_list = list(a2n.keys())
            next_mapped = {}
            next_raw_norm = {}
            for a in action_list:
                nb = a2n[a]
                a_idx = torch.tensor([action_idx_map[a]])
                next_raw = P_step(b_raw, a_idx)
                next_mapped[a] = to_geometry(next_raw, geometry)
                next_raw_norm[a] = next_raw.norm().item()

                nb_idx = torch.tensor([cell_index(nb, size)])
                true_nb_mapped = to_geometry(E(nb_idx), geometry)
                pred_to_true_dist = squared_dist(next_mapped[a], true_nb_mapped, geometry).sqrt().item()
                print(f"    action={a} -> true_neighbor={nb}: "
                      f"P_step raw_norm={next_raw_norm[a]:.4f}  "
                      f"dist(P_step(b,a), E(true_neighbor))={pred_to_true_dist:.4f}")

            targets = [c for c in cells if c != b]
            target_idx = torch.tensor([cell_index(t, size) for t in targets])
            target_mapped = to_geometry(E(target_idx), geometry)

            cost_rows = [
                squared_dist(next_mapped[a].expand(len(targets), -1), target_mapped, geometry)
                for a in action_list
            ]
            cost_stack = torch.stack(cost_rows, dim=0)
            pred_best = cost_stack.argmin(dim=0)

            win_counts = {a: 0 for a in action_list}
            for i in range(len(targets)):
                win_counts[action_list[pred_best[i].item()]] += 1

            print(f"  argmin win count across all {len(targets)} targets (lopsided = attractor signature):")
            for a in action_list:
                print(f"    action={a}: won {win_counts[a]}/{len(targets)} "
                      f"({100.0*win_counts[a]/len(targets):.1f}%)  "
                      f"[correct answer for {sum(1 for t in targets if neighbor_to_action[comp_of[t]] == a)} of them]")

            # mean cost of each action across all targets, to see if one action's predicted
            # next-state is just globally "cheap" (close to everything) vs targets specifically
            print(f"  mean squared_dist(P_step(b,a), target) across all {len(targets)} targets:")
            for i, a in enumerate(action_list):
                print(f"    action={a}: mean_cost={cost_stack[i].mean().item():.4f}  "
                      f"min={cost_stack[i].min().item():.4f}  max={cost_stack[i].max().item():.4f}")


if __name__ == "__main__":
    main()
