"""
branch_split_imbalance_check.py

Maze JEPA -- generalizes branch_cell_inspect.py's mechanistic finding on
b=(9,6) and b=(3,0): at both, the WINNING wrong action was the one
leading to a tiny dead-end (correct for only 1 of 99 targets), while the
correct action led to nearly the whole rest of the maze (97-98 of 99).
P_step's prediction for the rare dead-end-leading action drifted toward
a smaller/more "central" raw norm, making it look spuriously cheap
against almost every target -- winning the argmin despite being the
less accurate individual prediction.

This checks whether that's a GENERAL rule, not just two isolated cases:
for every branch cell b, computes a split-imbalance score (the fraction
of b's 99 possible targets that fall on b's SMALLEST component -- 0 for
a perfectly balanced multi-way branch, close to 0 for a lopsided
near-leaf junction like the two found) and correlates it against that
cell's confidently-wrong rate (same definition as
branch_error_clustering_check.py: wrong AND margin >= 5% of current
distance). A strong negative correlation (more lopsided = more
confidently wrong) would confirm this is THE mechanism, not a
coincidence specific to two cells.

Run:
    uv run python branch_split_imbalance_check.py --geometry hyperbolic \
        --encoder-checkpoint jepa_checkpoint_treeloss_size10_hyperbolic.pt \
        --step-checkpoint jepa_step_checkpoint_treeloss_size10_norresidual_hyperbolic.pt
"""
import argparse
import random
from collections import deque

import torch

from env import ACTIONS, ACTION_DELTA, ALGO, GRID_SIZE
from jepa_model import cell_index, squared_dist, to_geometry
from maze_gen import GENERATORS
from maze_planner import MAZE_SEED, load_models

NEAR_TIE_THRESHOLD = 0.05


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


def spearman_correlation(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks
    rx, ry = rank(xs), rank(ys)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((r - mean_rx) ** 2 for r in rx)
    var_y = sum((r - mean_ry) ** 2 for r in ry)
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / (var_x * var_y) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="hyperbolic")
    ap.add_argument("--encoder-checkpoint", default=None)
    ap.add_argument("--step-checkpoint", default=None)
    args = ap.parse_args()
    geometry = args.geometry

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())
    E, P_step, size = load_models(geometry, args.encoder_checkpoint, args.step_checkpoint)
    action_idx_map = {a: i for i, a in enumerate(ACTIONS)}

    rows = []  # (b, imbalance_score, confidently_wrong_rate, degree)

    with torch.no_grad():
        for b in cells:
            deg = len(adj[b])
            if deg < 2:
                continue

            a2n = action_to_neighbor(b, adj)
            neighbor_to_action = {nb: a for a, nb in a2n.items()}
            comp_of = components_excluding(adj, b)
            targets = [c for c in cells if c != b]
            if not targets:
                continue

            comp_sizes = {}
            for t in targets:
                side = comp_of[t]
                comp_sizes[side] = comp_sizes.get(side, 0) + 1
            smallest_frac = min(comp_sizes.values()) / len(targets)

            b_idx = torch.tensor([cell_index(b, size)])
            b_raw = E(b_idx)
            b_mapped = to_geometry(b_raw, geometry)

            action_list = list(a2n.keys())
            next_mapped = {}
            for a in action_list:
                a_idx = torch.tensor([action_idx_map[a]])
                next_raw = P_step(b_raw, a_idx)
                next_mapped[a] = to_geometry(next_raw, geometry)

            target_idx = torch.tensor([cell_index(t, size) for t in targets])
            target_mapped = to_geometry(E(target_idx), geometry)
            cost_rows = [
                squared_dist(next_mapped[a].expand(len(targets), -1), target_mapped, geometry)
                for a in action_list
            ]
            cost_stack = torch.stack(cost_rows, dim=0)
            pred_best = cost_stack.argmin(dim=0)
            current_cost = squared_dist(b_mapped.expand(len(targets), -1), target_mapped, geometry)

            conf_wrong = 0
            for i, t in enumerate(targets):
                correct_a = neighbor_to_action[comp_of[t]]
                correct_i = action_list.index(correct_a)
                is_correct = pred_best[i].item() == correct_i
                if is_correct:
                    continue
                correct_cost = cost_stack[correct_i, i].item()
                other_costs = [cost_stack[j, i].item() for j in range(len(action_list)) if j != correct_i]
                best_wrong_cost = min(other_costs)
                scale = current_cost[i].item()
                if scale > 1e-9:
                    rel_margin = -((best_wrong_cost - correct_cost) / scale)
                    if rel_margin >= NEAR_TIE_THRESHOLD:
                        conf_wrong += 1

            rows.append((b, smallest_frac, conf_wrong / len(targets), deg))

    rows.sort(key=lambda r: r[1])
    print(f"{'b':>10} {'minority_frac':>14} {'confidently_wrong_rate':>24} {'degree':>7}")
    for b, imb, cw, deg in rows:
        print(f"{str(b):>10} {imb:>14.4f} {cw*100:>23.1f}% {deg:>7}")

    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    rho = spearman_correlation(xs, ys)
    print(f"\nSpearman correlation (minority_frac vs confidently_wrong_rate): rho={rho:.3f}")
    print("(strongly negative = more lopsided branch points ARE more confidently wrong -- "
          "confirms the mechanism generalizes, not just the 2 cells inspected directly)")


if __name__ == "__main__":
    main()
