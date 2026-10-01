# Ledger

Every component in the code and what justifies it. Anything not listed here gets deleted.
Changes are made one at a time and measured against the baseline with the same probes.

## Probes (`probes/probe.py`)

Frozen encoder, full image, mean of the 196 patch embeddings per image.
- **Linear:** logistic regression on the train split, accuracy on val.
- **k-NN:** k = 20, cosine similarity, majority vote.

Floor = the same probes on a randomly initialised encoder of the same config.

## Results (val accuracy, linear / k-NN)

All runs: `--epochs 10 --batch-size 64 --lr 3e-4`. Encoder: width 192, depth 12, 3 heads. Predictor: width 96, depth 6, 3 heads. VICReg weights 0.5 / 0.5.

| Run | Target | Epoch 5 | Epoch 10 |
|---|---|---|---|
| Random init (floor) | — | — | 0.664 / 0.633 |
| `lr3e-4` | online output, stop-gradient | 0.803 / 0.866 | 0.807 / 0.870 |
| `target-grad` | online output, gradients flow ("pure VICReg") | — | 0.741 / 0.779 |
| `target-grad-pp` | as above + VICReg on per-position-centred tokens | — | 0.776 / 0.806 |
| **`ema-vicreg`** (current default) | **EMA teacher** | 0.797 / 0.870 | **0.809 / 0.874** |

Findings:
- **Pure VICReg** finds shortcuts that satisfy VICReg's statistics while making the target easy to predict. `inv` fell to 0.019 (vs 0.33), and `img_rank` fell to 3.
  - First shortcut: position. The per-position fix closes it and recovers about 3 points.
  - The remaining gap is plausibly image-global appearance, e.g. stain colour: shared by context and target, varies between images. Not measured.
- **EMA vs stop-gradient:** tied (+0.2 / +0.4, within noise). EMA is the default because it's the paper's recipe and no worse.

## Components

| Component | Where | Justification |
|---|---|---|
| ViT encoder, no [CLS], fixed 2-D sinusoidal positional encoding | model.py | Official I-JEPA |
| Narrow predictor, shared mask token + position, one run per target block | model.py | Official I-JEPA (paper: "apply our predictor M times") |
| Official init (trunc-normal 0.02, residual layers scaled by 1/sqrt(2l)) | model.py | Official I-JEPA |
| 4 target blocks, 15-20% of grid, aspect 0.75-1.5, size shared per batch | masks.py | Official I-JEPA |
| Context = grid minus target union, padded + attention mask | masks.py, model.py | Decision: official truncation drops non-target patches (bottom rows) |
| EMA target encoder, momentum 0.996 -> 1.0; probe the target encoder | train.py | Official I-JEPA; `ema-vicreg` ties stop-gradient |
| Loss: smooth-L1 vs LayerNormed target | loss.py | Official I-JEPA |
| VICReg variance + covariance on online full-image tokens, weights 0.5 / 0.5 | loss.py, train.py | Decision; not yet measured against EMA alone |
| RandomResizedCrop(0.3-1.0) only, ImageNet normalisation | data.py | Official I-JEPA |
| AdamW, warmup + cosine lr, weight decay 0.04 -> 0.4 cosine, none on 1-D params | train.py | Official I-JEPA |
| lr 3e-4 at batch 64 | train.py (CLI) | Linear scaling of the official lr gave 3e-5: `inv` barely moved in 2 epochs (run `short`) |
| Switches `--no-ema`, `--target-grad`, `--vicreg-per-position` | train.py, loss.py | Reproduce the runs above; all lose to or tie the default |

## Open

- **EMA alone vs EMA + VICReg:** does VICReg still contribute?
- **Per-image embeddings get more alike** during training (`img_rank` ~10 -> ~5 in all stop-gradient/EMA runs). May cap the linear probe.
- **Accuracy plateaus** between epoch 5 and 10.
- **No ceiling** (known-good encoder) measured in this setup yet.
- **Hyperparameters** (4 blocks, block scale, EMA momentum) were tuned by the paper for EMA, ViT-H, ImageNet and batch 2048. Not re-tuned here.
