"""
branch_point_diagnostic.py

Maze JEPA -- direct mechanistic test of the hypothesis raised (and not
chased down) after embedding_tree_metric_check.py: hyperbolic's E only
passes 63.3% of "parent closer than sibling" checks, not 100%, and that
residual embedding/P_step error is what caps the real planner's success
under --replan (0-3/40), even though oracle-CEM+replan solves this
exact maze 100%.

This isolates the cost signal from CEM's stochastic search and from
multi-step unroll error entirely: for every cell b with degree >= 2 in
the maze's spanning tree, and every other cell t as a candidate target,
there is exactly one correct first action -- removing b from the tree
splits the rest of the maze into deg(b) disjoint components (no
cycles), t falls in exactly one, and the neighbor on that side is the
unique correct move toward t. The question: does

    argmin_a  squared_dist( to_geometry(P_step(E(b), a), geom),
                             to_geometry(E(t), geom), geom )

(restricted to b's deg(b) actually-valid actions) pick that correct
action? Exactly the per-candidate cost comparison CEM's own cost
function makes at every replan step, here at horizon=1 with zero
sampling noise. For an exact tree with a well-behaved embedding this
should be close to 100%.

First run (degree split only) found: degree==2 (corridor) 71.9%,
degree>=3 (branch) 65.5%, overall 69.3% -- branches ARE worse, but only
by ~6pts, too small a gap to explain a 30% base error rate on its own.
So the dominant story isn't "siblings get confused at forks" -- it's
that E fails to preserve tree-distance ordering almost everywhere, not
just at forks. This run adds two follow-ups to characterize THAT:

  1. Accuracy vs bfs_distance(b, t), bucketed -- does the ~30% error
     hold flat, or worsen with range? (Uses maze_gen.bfs_distances_from
     for one BFS per branch cell instead of N per-pair calls.)
  2. Margin/confidence analysis -- for each pair, margin =
     cost(best wrong action) - cost(correct action), normalized by
     squared_dist(current, target) as a "how big relative to the
     remaining distance" scale. Positive margin = correctly ranked;
     its magnitude among WRONG pairs distinguishes "confidently wrong"
     (large negative -- a real embedding defect) from "near-tie noise"
     (small negative -- expected in any continuous embedding, less
     concerning).

Run:
    uv run python branch_point_diagnostic.py [--geometry hyperbolic]
"""
import argparse
import random
import statistics
from collections import deque

import torch

from env import ACTIONS, ACTION_DELTA, ALGO, GRID_SIZE
from jepa_model import cell_index, squared_dist, to_geometry
from maze_gen import GENERATORS, bfs_distances_from
from maze_planner import MAZE_SEED, load_models


DIST_BUCKETS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 10**9)]


def action_to_neighbor(cell, adj):
    """cell -> {action: neighbor_cell} for cell's actually-valid moves."""
    out = {}
    r, c = cell
    for a in ACTIONS:
        dr, dc = ACTION_DELTA[a]
        nb = (r + dr, c + dc)
        if nb in adj[cell]:
            out[a] = nb
    return out


def components_excluding(adj, b):
    """Removes b from the graph (without mutating adj) and returns
    {other_cell: which_neighbor_of_b_its_reachable_through} -- one
    component per neighbor of b, guaranteed disjoint and covering
    every other cell in the maze, since adj is a tree (no cycles)."""
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


def dist_bucket(d):
    for lo, hi in DIST_BUCKETS:
        if lo <= d <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="hyperbolic")
    ap.add_argument("--encoder-checkpoint", default=None,
                     help="Override E's checkpoint path template, e.g. "
                          "jepa_checkpoint_treeloss_{geometry}.pt -- same convention "
                          "as maze_planner.py's own override flags.")
    ap.add_argument("--step-checkpoint", default=None,
                     help="Override P_step's checkpoint path template, matching "
                          "whichever E it was trained against.")
    args = ap.parse_args()
    geometry = args.geometry

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())

    E, P_step, size = load_models(geometry, args.encoder_checkpoint, args.step_checkpoint)
    action_idx_map = {a: i for i, a in enumerate(ACTIONS)}

    # degree bucket -> [correct, total]; 3 means "degree >= 3"
    degree_results = {2: [0, 0], 3: [0, 0]}
    # dist bucket label -> [correct, total]
    dist_results = {}
    correct_margins = []    # relative margin when the right action won
    wrong_margins = []      # relative margin when the wrong action won (positive number = how wrong)

    with torch.no_grad():
        for b in cells:
            deg = len(adj[b])
            if deg < 2:
                continue  # leaf: no choice to test

            degree_bucket = 2 if deg == 2 else 3
            a2n = action_to_neighbor(b, adj)
            neighbor_to_action = {nb: a for a, nb in a2n.items()}
            comp_of = components_excluding(adj, b)
            dist_from_b = bfs_distances_from(adj, b)  # one BFS per branch cell

            targets = [c for c in cells if c != b]
            if not targets:
                continue

            b_idx = torch.tensor([cell_index(b, size)])
            b_raw = E(b_idx)  # (1, dim)
            b_mapped = to_geometry(b_raw, geometry)  # (1, dim) -- "current cost" reference

            action_list = list(a2n.keys())  # len == deg
            next_mapped = {}
            for a in action_list:
                a_idx = torch.tensor([action_idx_map[a]])
                next_raw = P_step(b_raw, a_idx)  # (1, dim)
                next_mapped[a] = to_geometry(next_raw, geometry)  # (1, dim)

            target_idx = torch.tensor([cell_index(t, size) for t in targets])
            target_mapped = to_geometry(E(target_idx), geometry)  # (n_targets, dim)

            cost_rows = [
                squared_dist(next_mapped[a].expand(len(targets), -1), target_mapped, geometry)
                for a in action_list
            ]
            cost_stack = torch.stack(cost_rows, dim=0)  # (n_actions, n_targets)
            pred_best = cost_stack.argmin(dim=0)  # (n_targets,) index into action_list

            # "current cost" scale per target, for normalizing margins
            current_cost = squared_dist(b_mapped.expand(len(targets), -1), target_mapped, geometry)

            for i, t in enumerate(targets):
                correct_a = neighbor_to_action[comp_of[t]]
                correct_i = action_list.index(correct_a)
                is_correct = pred_best[i].item() == correct_i

                degree_results[degree_bucket][1] += 1
                if is_correct:
                    degree_results[degree_bucket][0] += 1

                d = dist_from_b[t]
                bucket = dist_bucket(d)
                if bucket not in dist_results:
                    dist_results[bucket] = [0, 0]
                dist_results[bucket][1] += 1
                if is_correct:
                    dist_results[bucket][0] += 1

                correct_cost = cost_stack[correct_i, i].item()
                other_costs = [cost_stack[j, i].item() for j in range(len(action_list)) if j != correct_i]
                best_wrong_cost = min(other_costs)
                scale = current_cost[i].item()
                if scale > 1e-9:
                    rel_margin = (best_wrong_cost - correct_cost) / scale
                    if is_correct:
                        correct_margins.append(rel_margin)
                    else:
                        wrong_margins.append(-rel_margin)  # positive = "how wrong", easier to read

    print("=== accuracy by degree ===")
    for bucket, (correct, total) in degree_results.items():
        label = "degree==2 (corridor)" if bucket == 2 else "degree>=3 (branch)"
        pct = 100.0 * correct / total if total else float("nan")
        print(f"  {label}: {correct}/{total} = {pct:.1f}%")
    overall_correct = sum(c for c, _ in degree_results.values())
    overall_total = sum(t for _, t in degree_results.values())
    print(f"  overall: {overall_correct}/{overall_total} = {100.0 * overall_correct / overall_total:.1f}%")

    print("\n=== accuracy by bfs_distance(b, t) ===")
    for lo, hi in DIST_BUCKETS:
        label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        if label not in dist_results:
            continue
        correct, total = dist_results[label]
        pct = 100.0 * correct / total if total else float("nan")
        print(f"  d={label:>6}: {correct}/{total} = {pct:.1f}%")

    print("\n=== margin analysis (relative to current squared_dist-to-target) ===")
    if correct_margins:
        print(f"  when RIGHT:  n={len(correct_margins)}  mean margin={statistics.mean(correct_margins):.3f}  "
              f"median={statistics.median(correct_margins):.3f}")
    if wrong_margins:
        print(f"  when WRONG:  n={len(wrong_margins)}  mean margin={statistics.mean(wrong_margins):.3f}  "
              f"median={statistics.median(wrong_margins):.3f}  "
              f"(higher = more confidently wrong, not a near-tie)")
        near_tie = sum(1 for m in wrong_margins if m < 0.05)
        print(f"  of the wrong ones, {near_tie}/{len(wrong_margins)} = "
              f"{100.0 * near_tie / len(wrong_margins):.1f}% are near-ties (margin < 5% of current distance)")


if __name__ == "__main__":
    main()
