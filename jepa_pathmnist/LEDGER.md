# Ledger

The current method is LeJEPA (`lejepa.py`); I-JEPA (`ijepa.py`) is kept as the reference the comparisons below are made against. Only the LeJEPA runs and `ema-only-30` are kept in `runs/`; the other runs' numbers are recorded here.

Every component in the code and what justifies it. Anything not listed here gets deleted.
Changes are made one at a time and measured against the baseline with the same probes.

## Probes: the reference (`probes/probe.py`, `probes/scarce.py`)

A probe asks whether a label nobody showed the encoder can be read off its frozen features. Its truth comes from outside the latent space (the tissue class), so it is evidence; the training loss is not, because it only compares embeddings with embeddings.

**Features, shared by every probe.** The encoder is frozen and in eval mode. Each image is fed whole (224 px, no augmentation, ImageNet normalisation), and its 196 patch embeddings are averaged into one 192-d vector. The ImageNet arm uses ResNet-18's 512-d global-average-pooled vector instead. Features are cached in `probes/cache/`.

- **Linear probe** (`linear_probe`). Multinomial logistic regression, 9 classes, on features standardised with the train statistics only. Full batch, L-BFGS, 500 iterations, L2 weight 1e-4. *Mechanism:* one hyperplane per class, so it succeeds only if the classes already lie in separable half-spaces of the embedding. *Measures:* linearly decodable class information, the standard SSL number and the one comparable across papers. *Fundamental:* yes, it is the goal's metric (PLAN.md). The train accuracy is reported beside it: a large train/eval gap means overfitting, which matters with 901 labels.
- **k-NN probe** (`knn_probe`). No training at all: each eval image takes the majority label of its k = 20 most cosine-similar train images (k = min(20, n) in the scarce test). *Mechanism:* local neighbourhoods only, no learned boundary. *Measures:* the raw geometry, whether same-class images are near each other. *Fundamental:* no, a cross-check. It is free of optimisation choices, so a linear gain that k-NN doesn't share points to a probe artefact rather than better features.
- **MLP probe:** not used here. A nonlinear head can carve classes out of almost any rich feature, including random ones, so it says less about the embedding. (The ball-in-cup project uses one to decode position, a different question.)

**How to read the numbers.**
1. **Against the floor first:** the same probe on a random-init encoder of the same config (`--random`). A random ViT already gives 68% with all labels and 56–57% with 1%, so only the distance above the floor is what pretraining did.
2. **Val vs test:** val comes from the training centre, test from another clinical centre. Checkpoints and settings are chosen on val only. Test is scored once, at the end, and also measures how well the features travel (I-JEPA vs LeJEPA was close on val, 10 points apart on test).
3. **Noise:** seed-to-seed spread is about 1–3 points with 901 labels, and well under 1 point with all labels. Differences inside that spread are not results.
4. **In the label-scarce test** the probe is fitted on the subset only (901 images, the same subset for every arm), so it competes on equal labels with networks trained end to end on that subset.

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

| `lejepa-nozoom` (**the default since 2026-11-04**; `lejepa` is now `--zoom`) | as `lejepa`, no zoom: global = whole image, local = 96 px native windows, never resized | 0.948 / 0.953 | 0.959 / 0.966 |
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
- **VICReg hurts once EMA is there:** EMA alone beats EMA + VICReg by 7.7 / 5.7 points. ijepa.py still defaults to 0.5 / 0.5; switching the default to 0 is the next cleanup.
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
| EMA target encoder, momentum 0.996 -> 1.0; probe the target encoder | ijepa.py | Official I-JEPA; `ema-vicreg` ties stop-gradient |
| Loss: smooth-L1 vs LayerNormed target | loss.py | Official I-JEPA |
| VICReg variance + covariance, weights 0.5 / 0.5 by default | loss.py, ijepa.py | Reproduces the runs above; loses to EMA alone |
| SIGReg, `--sigreg-weight` (off by default) | loss.py, ijepa.py | Reproduces run `sigreg`; checked in checks/check_loss.py |
| RandomResizedCrop(0.3-1.0) only, ImageNet normalisation | data.py | Official I-JEPA |
| AdamW, warmup + cosine lr, weight decay 0.04 -> 0.4 cosine, none on 1-D params | ijepa.py | Official I-JEPA |
| lr 3e-4 at batch 64 | ijepa.py (CLI) | Linear scaling of the official lr gave 3e-5: `inv` barely moved in 2 epochs (run `short`) |
| Switches `--no-ema`, `--target-grad`, `--vicreg-per-position` | ijepa.py, loss.py | Reproduce the runs above; all lose to or tie the default |

## Open

- **Per-image embeddings get more alike** during training (`img_rank` ~10 -> ~5 in all stop-gradient/EMA runs). May cap the linear probe.
- **Accuracy plateaus** between epoch 5 and 10.
- **No ceiling** (known-good encoder) measured in this setup yet.
- **Which part of LeJEPA's recipe matters:** EMA + the same crops and augmentation would separate SIGReg from the views.
- **Hyperparameters** (4 blocks, block scale, EMA momentum) were tuned by the paper for EMA, ViT-H, ImageNet and batch 2048. Not re-tuned here.

## Label-scarce test (PLAN.md goal; `probes/scarce.py`, 2026-11-05)

1% of the train labels (901 images, stratified), the same subset for every arm, three subsets (seeds 0, 1, 2). Val accuracy. Encoder: `lejepa-nozoom` epoch 10. From-scratch arms: AdamW lr 1e-3, wd 0.05, batch 64, 100 epochs (5 warmup, cosine), LeJEPA's global-view augmentations, no early stopping.

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| Linear probe, frozen LeJEPA | 91.4 | 92.3 | 91.0 | **91.6** |
| k-NN, frozen LeJEPA | 88.1 | 88.5 | 88.0 | 88.2 |
| ResNet-18 from scratch | 77.9 | 77.5 | 80.3 | 78.6 |
| ViT-Tiny from scratch (same network) | 75.8 | 77.3 | 77.1 | 76.7 |
| Linear probe, random encoder (floor) | 58.1 | 55.6 | 55.5 | 56.4 |

- **The goal is met on val:** frozen LeJEPA features with 1% of the labels beat both from-scratch baselines by 13–15 points, far outside the seed spread (1–3 points). With all labels the same probe reaches 95.9, so 99% fewer labels cost about 4 points.
- **Caveats:** the baselines use one fixed, untuned recipe. ResNet-18 nearly fits its 901 images (train 97–98%); the ViT underfits (train 79–84%), as ViTs from scratch on small data do. Tuning could add a few points to them, unlikely 13. Test split: see the final pass below.

### Final pass: test split and the ImageNet baseline (2026-11-06, `probes/scarce_all.sh`)

Every arm on val and test, three subsets; mean (seeds 0/1/2). The test split comes from another clinical centre. Trained networks are scored on val and test at the end of their fixed recipe, nothing chosen from either; the from-scratch reruns reproduced their val numbers exactly. ImageNet arms: torchvision ResNet-18 `IMAGENET1K_V1`, frozen (512-d pooled features, the same probes) or fine-tuned end to end (AdamW lr 1e-4, the standard fine-tuning rate, otherwise the from-scratch recipe).

| Arm | Val | Test |
|---|---|---|
| ImageNet ResNet-18, fine-tuned | **94.2** (94.5/94.0/94.0) | **89.6** (91.0/89.3/88.5) |
| LeJEPA ViT-Tiny, fine-tuned (`--init`, same recipe as ImageNet's) | 92.1 (91.8/92.1/92.3) | 89.5 (89.4/89.7/89.5) |
| Linear probe, frozen LeJEPA | 91.6 | 89.0 (88.4/89.3/89.4) |
| Linear probe, frozen ImageNet ResNet-18 | 89.5 (89.3/90.0/89.4) | 86.1 (85.6/85.7/87.0) |
| k-NN, frozen LeJEPA | 88.2 | 88.5 (88.9/87.8/88.7) |
| k-NN, frozen ImageNet ResNet-18 | 85.5 | 81.3 |
| ResNet-18 from scratch | 78.6 | 81.6 (81.1/80.5/83.1) |
| ViT-Tiny from scratch | 76.7 | 78.9 (78.5/80.3/77.8) |
| Linear probe, random encoder (floor) | 56.4 | 56.7 (56.1/57.7/56.4) |

- **The stated goal holds on test:** frozen LeJEPA beats both from-scratch networks by 7–10 points (89.0 vs 81.6 and 78.9).
- **Frozen vs frozen, LeJEPA wins:** +2.1 on val and +2.9 on test over ImageNet features, and the k-NN gap is larger (+7 on test): LeJEPA's neighbourhoods are tissue, ImageNet's less so. LeJEPA saw 90k unlabelled tissue images for 10 epochs on a ViT-Tiny; ImageNet saw 1.28M labelled photos.
- **But fine-tuned ImageNet is the strongest arm:** 94.2 on val, 2.6 above frozen LeJEPA; on the other centre's images the lead shrinks to 0.6, inside the seed spread. Frozen vs fine-tuned is not like for like: fine-tuning LeJEPA on the same 901 labels is the missing arm.
- **Fine-tuning LeJEPA (the missing arm, run after):** 92.1 val, 89.5 test, +0.5 over its frozen probe on both. Against fine-tuned ImageNet: a tie on test (89.5 vs 89.6), 2.1 behind on val. So on the other centre's images, 10 epochs of LeJEPA on 90k unlabelled tissue images match ImageNet pretraining; on the training centre's own images ImageNet fine-tuning is ahead. Bigger LeJEPA pretraining (epochs, model) is the obvious lever; not tried.
- **Domain shift:** fine-tuned ImageNet loses 4.6 points from val to test, frozen LeJEPA 2.6, k-NN on LeJEPA 0 (88.2 → 88.5).
