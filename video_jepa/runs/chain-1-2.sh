#!/bin/bash
# Items 1 and 2 of the 2026-11-14 follow-ups: V-JEPA token planning, then second training seeds and their probes.
set -e
cd "$(dirname "$0")/.."
echo "== 1. V-JEPA token planning $(date +%H:%M)"
uv run python plan_tokens.py embed --encoder runs/vjepa-s0/encoder.pt --name ac-tokens-vjepa
uv run python plan_tokens.py train --encoder runs/vjepa-s0/encoder.pt --name ac-tokens-vjepa
uv run python plan_tokens.py evaluate --encoder runs/vjepa-s0/encoder.pt --name ac-tokens-vjepa
echo "== 2a. V-JEPA seed 1 $(date +%H:%M)"
uv run python vjepa.py --name vjepa-s1 --steps 8000 --seed 1 > runs/vjepa-s1.log 2>&1
echo "== 2b. LeVJEPA seed 1 $(date +%H:%M)"
uv run python levjepa.py --name levjepa-s1 --steps 9000 --seed 1 > runs/levjepa-s1.log 2>&1
echo "== 2c. probes $(date +%H:%M)"
for run in vjepa-s1 levjepa-s1; do for ps in 0 1 2; do
  uv run python probes.py --encoder runs/$run/encoder.pt --probe-seed $ps 2>&1 | grep -v "fork\|pid\|warn" | tail -4
done; done
echo "== done $(date +%H:%M)"
