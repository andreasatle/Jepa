#!/bin/bash
# Item 3 of the 2026-11-14 follow-ups, after items 1-2: the image world model's recipe with 4-frame clips.
set -e
cd "$(dirname "$0")/.."
while pgrep -f chain-1-2.sh > /dev/null; do sleep 60; done
echo "== 3. clip world model, train $(date +%H:%M)"
PYTHONUNBUFFERED=1 uv run python clip_wm.py train --seed 0 > runs/clip-sigreg-s0.train.log 2>&1
echo "== 3. evaluate $(date +%H:%M)"
PYTHONUNBUFFERED=1 uv run python clip_wm.py evaluate runs/clip-sigreg-s0/model.pt > runs/clip-sigreg-s0.eval.log 2>&1
tail -1 runs/clip-sigreg-s0.eval.log
echo "== done $(date +%H:%M)"
