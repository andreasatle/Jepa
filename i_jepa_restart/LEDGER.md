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
| `ema-vicreg` | EMA teacher + VICReg 0.5 / 0.5 | 0.797 / 0.870 | 0.809 / 0.874 |
| **`ema-only`** (best so far; `--var-weight 0 --cov-weight 0`) | **EMA teacher, no VICReg** (official I-JEPA) | — | **0.886 / 0.931** |
| `ema-only-30` | as above, `--epochs 30` (cosine over 30) | 0.871 / 0.895 | 0.906 / 0.951; e030 0.952 / 0.978 |
| `sigreg` | LeJEPA's SIGReg (λ 0.05, 1024 slices, on mean-pooled online tokens), gradients flow, no EMA, no VICReg | 0.715 / 0.592 | 0.704 / 0.546 |
| **`lejepa`** (lejepa.py, best so far) | **none: LeJEPA, no EMA, no stop-gradient, no predictor**; 2 global + 6 local crops with colour augmentation, SIGReg λ 0.05 on a 64-dim projector, batch 128 | **0.954 / 0.970** | **0.970 / 0.980** |

| `lejepa-nozoom` | as `lejepa`, `--no-zoom`: global = whole image, local = 96 px native windows, never resized | 0.948 / 0.953 | 0.959 / 0.966 |
| `lejepa-windows` | as `lejepa`, `--native-windows`: never resized; per batch one global size from 160-224 px and one local from 64-128 px | 0.950 / 0.957 | 0.959 / 0.968 |
| `lejepa-nozoom-s16` | as `lejepa-nozoom`, `--sigreg-slices 16` (16 random directions per view per step instead of 1024) | 0.908 / 0.912 | 0.923 / 0.927 |

`lejepa-nozoom-s16` on test: 0.909 / 0.873.
`lejepa` at epoch 10 on the **test** split: **0.935 / 0.913**; `lejepa-nozoom`: 0.928 / 0.894; `lejepa-windows`: 0.924 / 0.895.
`ema-only-30` at epoch 30 on the **test** split: 0.824 / 0.788 (random init on test: 0.681 / 0.567). The val-test gap is PathMNIST's: its test set comes from a different clinical centre.

Findings:
- **Pure VICReg** finds shortcuts that satisfy VICReg's statistics while making the target easy to predict. `inv` fell to 0.019 (vs 0.33), and `img_rank` fell to 3.
  - First shortcut: position. The per-position fix closes it and recovers about 3 points.
  - The remaining gap is plausibly image-global appearance, e.g. stain colour: shared by context and target, varies between images. Not measured.
- **EMA vs stop-gradient:** tied (+0.2 / +0.4, within noise) while VICReg was on.
- **VICReg hurts once EMA is there:** EMA alone beats EMA + VICReg by 7.7 / 5.7 points. train.py still defaults to 0.5 / 0.5; switching the default to 0 is the next cleanup.
- **LeJEPA, EMA-free, wins clearly** (2026-11-02): +8.4 / +4.9 on val over `ema-only` at 10 epochs, and on test after 10 epochs it beats `ema-only-30` after 30 by +11.1 / +12.5. Confounded: other views and augmentation (colour jitter, which plausibly carries over to the test centre's staining), batch 128 vs 64, a projector. So "the recipe wins", not yet "SIGReg beats EMA". `img_rank` 5-9 (vs 1.3-3.6 for `sigreg`); SIGReg falls 7.9 -> 1.7.
- **Zoom (resizing crops) is worth 1-2 points** (2026-11-02): without it, val -1.1 / -1.4 and test -0.7 / -1.9 (one seed each). Zoom bakes in scale invariance, which is not justified at PathMNIST's fixed magnification (the user: "to build in bias by zooming sounds really bad"). Varying native window sizes (`lejepa-windows`) don't win it back: equal to `lejepa-nozoom` within noise. So the gap isn't missing variety; zoom's scale invariance (or its resampling blur) itself helped these probes a little.
- **16 directions instead of 1024 costs 3.6 / 3.9 points on val, 1.9 / 2.1 on test** (2026-11-03, a breaking run): it still learns and doesn't collapse, but clearly worse; final SIGReg 1.72 vs 1.54, img_std 0.36 vs 0.43. Measured gradient noise from directions (on the trained no-zoom encoder, last projector layer) scales as 1/S: 12x the batch noise at S = 16, 0.2x at S = 1024, crossover ~200 per view (~25 with 8 independent views). The paper's "16 can be enough" does not carry over at batch 128 and 10 epochs.
- **SIGReg's integral is exact at 9 points** on [0, 5] (17 is the default); it detects clusters strongly (16.3 vs 0.51 for a Gaussian, 128 x 64) but heavy tails weakly (Student-t 3: 0.61).
- **SIGReg in the I-JEPA setup fails** (2026-11-02): k-NN falls below the random floor and gets worse from epoch 5 to 10; `img_rank` stays at 1.3-3.6 of 192. Like pure VICReg, the encoder satisfies the distribution penalty on the pooled embedding while making masked targets trivially predictable (`inv` 0.017). Not a test of LeJEPA itself: the paper predicts between multi-crop views, not masked patches, and its λ was tuned there. Untested: masked prediction with a larger λ, SIGReg on tokens, LeJEPA's own multi-crop objective.

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
| VICReg variance + covariance, weights 0.5 / 0.5 by default | loss.py, train.py | Reproduces the runs above; loses to EMA alone |
| SIGReg, `--sigreg-weight` (off by default) | loss.py, train.py | Reproduces run `sigreg`; checked in checks/check_loss.py |
| RandomResizedCrop(0.3-1.0) only, ImageNet normalisation | data.py | Official I-JEPA |
| AdamW, warmup + cosine lr, weight decay 0.04 -> 0.4 cosine, none on 1-D params | train.py | Official I-JEPA |
| lr 3e-4 at batch 64 | train.py (CLI) | Linear scaling of the official lr gave 3e-5: `inv` barely moved in 2 epochs (run `short`) |
| Switches `--no-ema`, `--target-grad`, `--vicreg-per-position` | train.py, loss.py | Reproduce the runs above; all lose to or tie the default |

## Open

- **Per-image embeddings get more alike** during training (`img_rank` ~10 -> ~5 in all stop-gradient/EMA runs). May cap the linear probe.
- **Accuracy plateaus** between epoch 5 and 10.
- **No ceiling** (known-good encoder) measured in this setup yet.
- **Which part of LeJEPA's recipe matters:** EMA + the same crops and augmentation would separate SIGReg from the views.
- **Hyperparameters** (4 blocks, block scale, EMA momentum) were tuned by the paper for EMA, ViT-H, ImageNet and batch 2048. Not re-tuned here.
