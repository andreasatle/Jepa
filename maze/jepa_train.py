"""
jepa_train.py

Maze JEPA -- Phase 2: trains the E/P model from jepa_model.py on the
rollout dataset from collect_data.py, comparing Euclidean vs hyperbolic
geometry. See WORKPLAN.md for the full design reasoning, and
jepa_model.py's module docstring for the model itself (architecture,
anti-collapse mechanism).

This script only covers: loading maze_data.json, the train/holdout split,
the training loop (with VICReg's anti-collapse term), and evaluation
(Phase 0's nearest-endpoint/mean-spacing ratio metric, paired t-test +
Wilcoxon). The model definitions used to live here too, but were split
into jepa_model.py so other scripts (a future planner, a diagnostic) can
import the model without pulling in training/CLI code.

EVALUATION: nearest true endpoint distance / mean pairwise endpoint
spacing, computed per (start, horizon) on HELD-OUT rollouts (an 80/20
episode split per start, not seen during training), then paired t-test +
Wilcoxon across junction starts, same metric as multimodal_sanity_check.py
(Phase 0).

Run (from inside maze/, in Andreas's terminal -- torch/scipy commands are
always handed off, never run by Claude, per project convention):
    uv run python jepa_train.py --geometry euclidean
    uv run python jepa_train.py --geometry hyperbolic
    uv run python jepa_train.py --geometry both   (trains both, prints comparison table)
"""
import argparse
import json
import random
from collections import defaultdict

import torch
from scipy import stats

from jepa_model import (
    HIDDEN_DIM,
    LATENT_DIM,
    Encoder,
    Predictor,
    cell_index,
    cov_penalty_term,
    squared_dist,
    to_geometry,
)

CHECKPOINT_PATH = "jepa_checkpoint_{geometry}.pt"  # {geometry} filled in per run

DATA_PATH = "maze_data.json"
EVAL_HORIZONS = [3, 5, 8, 12, 20]   # matches domain_fit_check.py's sweep
TRAIN_HORIZON_STRIDE = 1            # train on every horizon 1..max_horizon
HOLDOUT_FRACTION = 0.2
N_STEPS = 3000
LR = 0.01
BATCH_SIZE = 256
KAPPA = 1.0            # anti-collapse weight -- same default as
                       # ball_in_cup_hyperbolic/jepa_train.py, watch
                       # avg_cov_penalty in the printed log and adjust
                       # if it's not settling down.
SEED = 0


def load_data(path=DATA_PATH):
    with open(path) as f:
        return json.load(f)


def split_episodes(episodes, holdout_fraction, rng):
    """Splits per start cell, not globally -- guarantees every junction
    has both train and holdout rollouts."""
    by_start = defaultdict(list)
    for ep in episodes:
        by_start[tuple(ep["start"])].append(ep)
    train, holdout = [], []
    for start, eps in by_start.items():
        eps = eps[:]
        rng.shuffle(eps)
        n_hold = max(1, int(len(eps) * holdout_fraction))
        holdout.extend(eps[:n_hold])
        train.extend(eps[n_hold:])
    return train, holdout


def build_examples(episodes, max_horizon, stride):
    """(start, horizon, end) triples, one per (episode, horizon)."""
    examples = []
    for ep in episodes:
        start = tuple(ep["start"])
        for h in range(1, max_horizon + 1, stride):
            examples.append((start, h, tuple(ep["states"][h])))
    return examples


def train_one(geometry, train_examples, size, num_cells, max_horizon, seed):
    torch.manual_seed(seed)

    E = Encoder(num_cells)
    P = Predictor(max_horizon)
    opt = torch.optim.Adam(list(E.parameters()) + list(P.parameters()), lr=LR)

    starts = torch.tensor([cell_index(s, size) for s, h, e in train_examples])
    horizons = torch.tensor([h for s, h, e in train_examples])
    ends = torch.tensor([cell_index(e, size) for s, h, e in train_examples])

    n = len(train_examples)
    for step in range(N_STEPS):
        batch = torch.randint(0, n, (BATCH_SIZE,))
        start_raw = E(starts[batch])
        target_raw = E(ends[batch])
        pred_raw = P(start_raw, horizons[batch])

        pred_mapped = to_geometry(pred_raw, geometry)
        target_mapped = to_geometry(target_raw, geometry)
        inv_loss = squared_dist(pred_mapped, target_mapped, geometry).mean()

        cov_penalty = cov_penalty_term(start_raw) + cov_penalty_term(target_raw)
        loss = inv_loss + (KAPPA / 2) * cov_penalty

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == N_STEPS - 1:
            print(f"  [{geometry}] step {step}: loss={loss.item():.5f}  "
                  f"inv_loss={inv_loss.item():.5f}  cov_penalty={cov_penalty.item():.5f}")

    checkpoint_path = CHECKPOINT_PATH.format(geometry=geometry)
    torch.save(
        {
            "E": E.state_dict(),
            "P": P.state_dict(),
            "geometry": geometry,
            "num_cells": num_cells,
            "size": size,
            "max_horizon": max_horizon,
            "dim": LATENT_DIM,
            "hidden": HIDDEN_DIM,
        },
        checkpoint_path,
    )
    print(f"  [{geometry}] checkpoint saved to {checkpoint_path}")

    return E, P


def evaluate(E, P, geometry, holdout_episodes, size, eval_horizons):
    """Per (start, horizon): ratio = nearest true endpoint distance /
    mean pairwise endpoint spacing, using held-out rollouts. Same metric
    as multimodal_sanity_check.py."""
    E.eval()
    P.eval()
    by_start = defaultdict(list)
    for ep in holdout_episodes:
        by_start[tuple(ep["start"])].append(ep)

    ratios = []
    with torch.no_grad():
        for start, eps in by_start.items():
            start_idx = torch.tensor([cell_index(start, size)])
            start_raw = E(start_idx)
            for h in eval_horizons:
                if h > len(eps[0]["states"]) - 1:
                    continue
                end_cells = [tuple(ep["states"][h]) for ep in eps]
                if len(set(end_cells)) < 2:
                    continue  # no spread to measure at this (start,h)
                end_idx = torch.tensor([cell_index(c, size) for c in end_cells])
                end_raw = E(end_idx)
                end_mapped = to_geometry(end_raw, geometry)

                horizon_t = torch.tensor([h])
                pred_mapped = to_geometry(P(start_raw, horizon_t), geometry)

                dists = squared_dist(pred_mapped.expand(len(eps), -1), end_mapped, geometry).sqrt()
                nearest = dists.min().item()

                pairwise = squared_dist(
                    end_mapped.unsqueeze(1), end_mapped.unsqueeze(0), geometry
                ).sqrt()
                mask = pairwise > 0
                mean_spacing = pairwise[mask].mean().item() if mask.any() else 1.0

                ratios.append((start, h, nearest / max(mean_spacing, 1e-8)))
    return ratios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic", "both"], default="both")
    args = parser.parse_args()

    rng = random.Random(SEED)
    data = load_data()
    size = data["grid_size"]
    max_horizon = data["max_horizon"]
    num_cells = size * size

    train_eps, holdout_eps = split_episodes(data["episodes"], HOLDOUT_FRACTION, rng)
    train_examples = build_examples(train_eps, max_horizon, TRAIN_HORIZON_STRIDE)
    print(f"{len(train_eps)} train episodes, {len(holdout_eps)} holdout episodes, "
          f"{len(train_examples)} train (start,H,end) triples")

    geometries = ["euclidean", "hyperbolic"] if args.geometry == "both" else [args.geometry]
    results = {}
    for geometry in geometries:
        print(f"\ntraining {geometry}...")
        E, P = train_one(geometry, train_examples, size, num_cells, max_horizon, SEED)
        ratios = evaluate(E, P, geometry, holdout_eps, size, EVAL_HORIZONS)
        results[geometry] = ratios

    if args.geometry != "both":
        by_h = defaultdict(list)
        for start, h, r in results[args.geometry]:
            by_h[h].append(r)
        print(f"\n{args.geometry} ratio by horizon (lower = closer to true endpoints):")
        for h in EVAL_HORIZONS:
            if by_h[h]:
                vals = by_h[h]
                print(f"  H={h:3d}: mean={sum(vals) / len(vals):.4f}  n={len(vals)}")
        return

    eucl_by_key = {(s, h): r for s, h, r in results["euclidean"]}
    hyp_by_key = {(s, h): r for s, h, r in results["hyperbolic"]}
    common_keys = sorted(set(eucl_by_key) & set(hyp_by_key))

    print("\n=== Euclidean vs Hyperbolic (paired by start,horizon; lower ratio = better) ===")
    for h in EVAL_HORIZONS:
        keys_h = [k for k in common_keys if k[1] == h]
        if len(keys_h) < 3:
            continue
        eucl_vals = [eucl_by_key[k] for k in keys_h]
        hyp_vals = [hyp_by_key[k] for k in keys_h]
        diffs = [e - hv for e, hv in zip(eucl_vals, hyp_vals)]  # positive => hyperbolic better
        t_p = stats.ttest_rel(eucl_vals, hyp_vals).pvalue
        try:
            w_p = stats.wilcoxon(diffs).pvalue
        except ValueError:
            w_p = float("nan")
        mean_eucl = sum(eucl_vals) / len(eucl_vals)
        mean_hyp = sum(hyp_vals) / len(hyp_vals)
        flag = "*" if t_p < 0.05 else " "
        winner = "HYP" if mean_hyp < mean_eucl else "EUCL"
        n_hyp_better = sum(1 for d in diffs if d > 0)  # eucl - hyp > 0 => hyp closer
        n_eucl_better = sum(1 for d in diffs if d < 0)
        print(f"  H={h:3d}: eucl={mean_eucl:.4f}  hyp={mean_hyp:.4f}  "
              f"winner={winner}{flag}  t_p={t_p:.4f}  w_p={w_p:.4f}  n={len(keys_h)}  "
              f"(hyp-better={n_hyp_better} eucl-better={n_eucl_better})")


if __name__ == "__main__":
    main()
