#!/bin/sh
# ball_in_cup_sigreg: train with SIGReg, then the same evaluation chain as ball_in_cup.
set -e
cd "$(dirname "$0")"
uv run python jepa_train_strided.py
uv run python recalibrate_bn.py
uv run python eval_linear_probe_strided.py
uv run python eval_mlp_probe_strided.py
uv run python drift_diagnostic_strided.py
N_TRIALS=100 SEEDED=1 uv run python solve_strided_batch_random.py > batch100.log 2>&1
