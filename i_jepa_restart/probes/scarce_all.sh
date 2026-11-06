#!/bin/sh
# The label-scarce test, final pass: every arm, three labelled subsets, val and test.
# Frozen-feature arms are cheap (cached features); each trained network takes a few minutes.
set -e
cd "$(dirname "$0")/.."
CKPT=runs/lejepa-nozoom/encoder_e010.pt
for s in 0 1 2; do
  uv run python probes/scarce.py probe $CKPT --seed $s --split test
  uv run python probes/scarce.py probe $CKPT --random --seed $s --split test
  uv run python probes/scarce.py imagenet --seed $s --split val
  uv run python probes/scarce.py imagenet --seed $s --split test
done
for arch in resnet18-imagenet resnet18 vit; do
  for s in 0 1 2; do
    uv run python probes/scarce.py scratch --arch $arch --seed $s
  done
done
