#!/bin/sh
# Train and evaluate both regularisers for the given seeds: ./run_all.sh 0 1 2
set -e
cd "$(dirname "$0")"
for seed in "$@"; do
  for reg in sigreg vicreg; do
    PYTHONUNBUFFERED=1 uv run python train.py --reg $reg --seed $seed > runs/$reg-s$seed.train.log 2>&1
    PYTHONUNBUFFERED=1 uv run python evaluate.py runs/$reg-s$seed/model.pt > runs/$reg-s$seed.eval.log 2>&1
  done
done
