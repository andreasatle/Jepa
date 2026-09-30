# Ledger

Every component in the code and what justifies it. Anything not listed here gets deleted.
Changes are made one at a time and measured against the baseline with the same probes.

## Probes (`probes/probe.py`)

Frozen encoder, full image, mean of the 196 patch embeddings per image.
- **Linear:** logistic regression on the train split, accuracy on val.
- **k-NN:** k = 20, cosine similarity, majority vote.

Floor = the same probes on a randomly initialised encoder of the same config.

## Baseline (2026-09-30), run `lr3e-4`

`train.py --epochs 10 --batch-size 64 --lr 3e-4`. Encoder: width 192, depth 12, 3 heads. Predictor: width 96, depth 6, 3 heads.

| Encoder | Linear (val) | k-NN (val) |
|---|---|---|
| Random init (floor) | 0.664 | 0.633 |
| Epoch 5 | 0.803 | 0.866 |
| Epoch 10 | 0.807 | 0.870 |

## Components

| Component | Where | Justification |
|---|---|---|
| ViT encoder, no [CLS], fixed 2-D sinusoidal positional encoding | model.py | Official I-JEPA |
| Narrow predictor, shared mask token + position, one run per target block | model.py | Official I-JEPA (paper: "apply our predictor M times") |
| Official init (trunc-normal 0.02, residual layers scaled by 1/sqrt(2l)) | model.py | Official I-JEPA |
| 4 target blocks, 15-20% of grid, aspect 0.75-1.5, size shared per batch | masks.py | Official I-JEPA |
| Context = grid minus target union, padded + attention mask | masks.py, model.py | Decision: official truncation drops non-target patches (bottom rows) |
| Target: online full-image output, stop-gradient | train.py | Decision: VICReg replaces EMA; stop-grad = EMA with momentum 0 |
| Loss: smooth-L1 vs LayerNormed target | loss.py | Official I-JEPA |
| VICReg variance + covariance on full-image tokens, weights 0.5 / 0.5 | loss.py | Decision: replaces EMA against collapse; baseline above |
| RandomResizedCrop(0.3-1.0) only, ImageNet normalisation | data.py | Official I-JEPA |
| AdamW, warmup + cosine lr, weight decay 0.04 -> 0.4 cosine, none on 1-D params | train.py | Official I-JEPA |
| lr 3e-4 at batch 64 | train.py (CLI) | Linear scaling of the official lr gave 3e-5: `inv` barely moved in 2 epochs (run `short`) |

## Open

- Stop-gradient on the target vs pure VICReg (gradients through both sides).
- Per-image embeddings get more alike during training (`img_rank` ~10 -> ~5); may cap the linear probe.
- No ceiling (known-good encoder) measured in this setup yet.
