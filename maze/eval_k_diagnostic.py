"""
eval_k_diagnostic.py

Quick check: for the exact (start, horizon) pairs jepa_train.py's
--geometry both run evaluates against, how many distinct true endpoints
are actually in the HELD-OUT split (24 junctions x ~8 holdout rollouts
each, from an 80/20 split of 40 rollouts/junction)?

Motivation: the first non-collapsed jepa_train.py run showed Euclidean
winning significantly at every tested horizon (H=3..20), contradicting
Phase 0's prediction that hyperbolic should start winning once branching
factor k >= 8. Before concluding anything about geometry, check whether
these specific evaluated (start, horizon) pairs are actually IN that
k >= 8 regime at all -- domain_fit_check.py measured k with 300 rollouts
per (maze, start, horizon); jepa_train.py's holdout only has ~8 rollouts
per junction, which could easily undersample the true k down to well
below 8 even where the true reachable set is larger.

Pure stdlib, mirrors jepa_train.py's split_episodes()/SEED/HOLDOUT_FRACTION
exactly so these numbers reflect what was actually evaluated.

Run: python3 eval_k_diagnostic.py
"""
import json
import random
from collections import defaultdict

DATA_PATH = "maze_data.json"
HOLDOUT_FRACTION = 0.2
SEED = 0
EVAL_HORIZONS = [3, 5, 8, 12, 20]


def split_episodes(episodes, holdout_fraction, rng):
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


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)
    rng = random.Random(SEED)
    _, holdout = split_episodes(data["episodes"], HOLDOUT_FRACTION, rng)

    by_start = defaultdict(list)
    for ep in holdout:
        by_start[tuple(ep["start"])].append(ep)

    sizes = [len(v) for v in by_start.values()]
    print(f"{len(by_start)} junctions, holdout rollouts/junction: min={min(sizes)} "
          f"max={max(sizes)} mean={sum(sizes)/len(sizes):.1f}")

    for h in EVAL_HORIZONS:
        ks = []
        for start, eps in by_start.items():
            endpoints = {tuple(ep["states"][h]) for ep in eps}
            ks.append(len(endpoints))
        mean_k = sum(ks) / len(ks)
        n_at_least_8 = sum(1 for k in ks if k >= 8)
        print(f"H={h:3d}: mean distinct endpoints (holdout) = {mean_k:.2f}  "
              f"min={min(ks)} max={max(ks)}  junctions with k>=8: {n_at_least_8}/{len(ks)}")


if __name__ == "__main__":
    main()
