"""
branch_error_clustering_check.py

Maze JEPA -- follow-up to branch_point_diagnostic.py: that script found
79.6% one-step decision accuracy, with 46.3% of the wrong decisions
"confidently wrong" (not near-ties). Open question: does that confidently-
wrong share cluster on specific structural patterns (particular branch
cells, particular correct-actions/directions, particular targets), or is
it spread roughly uniformly across the maze? Clustering would point at a
further fixable gap (e.g. specific bad regions of the embedding); uniform
spread would look more like an intrinsic ceiling for this architecture at
this accuracy level.

Reuses the exact same per-(b, t) decision loop as branch_point_diagnostic.py
(same cost comparison, same "confidently wrong" definition: wrong AND
margin >= 0.05), but aggregates by branch cell b, by correct-action
direction, and by target cell t, instead of by degree/distance bucket.
Also cross-references each branch cell's raw embedding norm against the
14-near-origin-cell cluster found by pstep_origin_coverage_check.py /
pstep_origin_crowding_check.py, since that's the most specific structural
lead so far.

Run:
    uv run python branch_error_clustering_check.py --geometry hyperbolic \
        --encoder-checkpoint jepa_checkpoint_treeloss_size10_hyperbolic.pt \
        --step-checkpoint jepa_step_checkpoint_treeloss_size10_norresidual_hyperbolic.pt
"""
import argparse
import random
from collections import deque

import torch

from env import ACTIONS, ACTION_DELTA, ALGO, GRID_SIZE
from jepa_model import cell_index, squared_dist, to_geometry
from maze_gen import GENERATORS, bfs_distances_from
from maze_planner import MAZE_SEED, load_models

NEAR_TIE_THRESHOLD = 0.05
ORIGIN_THRESHOLD = 1.0


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
    args = ap.parse_args()
    geometry = args.geometry

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))
    cells = list(adj.keys())

    E, P_step, size = load_models(geometry, args.encoder_checkpoint, args.step_checkpoint)
    action_idx_map = {a: i for i, a in enumerate(ACTIONS)}

    # per-b stats: [correct, total, confidently_wrong]
    per_b = {}
    # per-correct-action stats
    per_action = {a: [0, 0, 0] for a in ACTIONS}
    # per-target stats
    per_t = {}

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

            b_idx = torch.tensor([cell_index(b, size)])
            b_raw = E(b_idx)
            b_mapped = to_geometry(b_raw, geometry)
            b_norm = b_raw.norm().item()

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

            per_b.setdefault(b, [0, 0, 0, b_norm])

            for i, t in enumerate(targets):
                correct_a = neighbor_to_action[comp_of[t]]
                correct_i = action_list.index(correct_a)
                is_correct = pred_best[i].item() == correct_i

                correct_cost = cost_stack[correct_i, i].item()
                other_costs = [cost_stack[j, i].item() for j in range(len(action_list)) if j != correct_i]
                best_wrong_cost = min(other_costs)
                scale = current_cost[i].item()
                confidently_wrong = False
                if not is_correct and scale > 1e-9:
                    rel_margin = -((best_wrong_cost - correct_cost) / scale)
                    confidently_wrong = rel_margin >= NEAR_TIE_THRESHOLD

                per_b[b][1] += 1
                per_action[correct_a][1] += 1
                per_t.setdefault(t, [0, 0, 0])
                per_t[t][1] += 1
                if is_correct:
                    per_b[b][0] += 1
                    per_action[correct_a][0] += 1
                    per_t[t][0] += 1
                elif confidently_wrong:
                    per_b[b][2] += 1
                    per_action[correct_a][2] += 1
                    per_t[t][2] += 1

    near_origin_cells = {c for c in cells if E(torch.tensor([cell_index(c, size)])).norm().item() < ORIGIN_THRESHOLD}

    print("=== per-correct-action breakdown (is a particular direction disproportionately mispredicted?) ===")
    for a, (correct, total, conf_wrong) in per_action.items():
        pct = 100.0 * correct / total if total else float("nan")
        cw_pct = 100.0 * conf_wrong / total if total else float("nan")
        print(f"  action={a}: correct={correct}/{total} ({pct:.1f}%)  confidently_wrong={conf_wrong} ({cw_pct:.1f}%)")

    print(f"\n=== per-branch-cell (b) confidently-wrong rate -- {len(per_b)} branch cells, "
          f"sorted worst first ===")
    ranked = sorted(per_b.items(), key=lambda kv: -kv[1][2] / kv[1][1])
    for b, (correct, total, conf_wrong, b_norm) in ranked[:15]:
        pct_correct = 100.0 * correct / total
        pct_cw = 100.0 * conf_wrong / total
        origin_flag = " [NEAR-ORIGIN]" if b in near_origin_cells else ""
        print(f"  b={b}: correct={pct_correct:.1f}%  confidently_wrong={conf_wrong}/{total} "
              f"({pct_cw:.1f}%)  raw_norm={b_norm:.3f}{origin_flag}")

    cw_rates = [conf_wrong / total for _, (correct, total, conf_wrong, _) in per_b.items()]
    mean_rate = sum(cw_rates) / len(cw_rates)
    print(f"\n  mean confidently-wrong rate across all {len(per_b)} branch cells: {mean_rate*100:.1f}%")
    print(f"  (if the worst cells above are well above this mean, that's clustering; "
          f"if they're all close to it, errors are spread roughly uniformly)")

    print(f"\n=== near-origin branch cells specifically ({len(near_origin_cells)} total) ===")
    for b in sorted(near_origin_cells):
        if b not in per_b:
            continue
        correct, total, conf_wrong, b_norm = per_b[b]
        pct_correct = 100.0 * correct / total
        pct_cw = 100.0 * conf_wrong / total
        print(f"  b={b}: correct={pct_correct:.1f}%  confidently_wrong={pct_cw:.1f}%  raw_norm={b_norm:.3f}")

    print(f"\n=== worst target cells (t) -- top 10 by confidently-wrong rate as a target ===")
    ranked_t = sorted(per_t.items(), key=lambda kv: -kv[1][2] / kv[1][1])
    for t, (correct, total, conf_wrong) in ranked_t[:10]:
        pct_correct = 100.0 * correct / total
        pct_cw = 100.0 * conf_wrong / total
        print(f"  t={t}: correct={pct_correct:.1f}%  confidently_wrong={conf_wrong}/{total} ({pct_cw:.1f}%)")


if __name__ == "__main__":
    main()
