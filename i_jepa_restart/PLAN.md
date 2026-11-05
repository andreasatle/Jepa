# I-JEPA restart — plan

Self-supervised learning on PathMNIST. Started as a clean restart of the I-JEPA experiment; since 2026-11-05 the method is **LeJEPA** (the user: "lets go all in on LeJepa"), because it beat I-JEPA with EMA by about 10 points on the test split (another clinical centre) in a third of the epochs, with a fixed objective and no EMA (LEDGER.md). I-JEPA (`ijepa.py`, renamed from `train.py` on 2026-11-05) stays only as the ledger's reference; no further work on it, and no I-JEPA tutorial. The old `../i_jepa/` (v1–v4) stays as reference only; nothing is copied from it without being justified here first.

## Goal (decide before any code)

Dataset: PathMNIST (9 tissue classes). Decided: keep the medical data.

Success is measured by a classifier on the frozen encoder's embedding (a linear probe). Two candidate goal statements, not yet chosen:

1. **Plain:** close the gap between the floor (linear probe on a random-init encoder, 74.3% in the old setup) and the ceiling (supervised ResNet-18 trained end-to-end on all labels, ~91% in the MedMNIST v2 benchmark).
2. **Label-scarce:** with only 1% of the labels, a linear classifier on frozen LeJEPA features beats the same network trained from scratch on that same 1%.

## Two stages (labels only in stage 2)

1. **Pretraining, no labels at all.** LeJEPA learns the embedding purely from images. Tissue type is never seen; if it shows up in the embedding, it emerged.
2. **Classifier on the frozen embedding.** The encoder is frozen; only a small classifier (linear probe) is trained, on all or a fraction of the labels. It tests whether the classes emerged as (linearly) separable structure.

Label hygiene: choosing checkpoints or hyperparameters by probe accuracy uses labels indirectly. Do that on the validation split only; the test split is touched once, for the final number.

Fine-tuning (unfreezing the encoder while training the classifier) is a different, weaker test of emergence, since the labels then reshape the embedding. Not part of the goal.

## Rules

- Start from the LeJEPA recipe (`lejepa.py`): 2 global + 6 local views, every view's projection pulled to the mean of the global views' projections, SIGReg (sliced Epps–Pulley, 1024 directions per view per step) with λ = 0.05, no EMA, no predictor, no stop-gradient. One decided deviation: views at native resolution, no zoom (the user: zooming builds in a bias; it costs about 1 point, LEDGER.md).
- One change at a time, measured against the same baseline with the same probes. A change stays only if it clearly helps; otherwise it comes out.
- `LEDGER.md` lists every component in the code and the measurement that justifies it. Anything not in the ledger gets deleted.
- Tests are few and fixed: the views, SIGReg's value on known samples (Gaussian ≈ 0.52, collapse ≈ 0.131·n), and the probe separating floor from ceiling. No test per change. (The existing `checks/` are I-JEPA's; the two LeJEPA checks are still to be written, see To-do.)
- Keep experiments fast (lower resolution / small ViT) so each one takes minutes, not overnight.

## Steps

1. [ ] Choose the goal statement above.
2. [ ] Validate the measuring stick: the probes must clearly separate a random encoder (floor) from a known-good encoder (ceiling) before any training.
3. [x] Minimal LeJEPA (paper recipe), trained and probed (I-JEPA before it, kept as reference).
4. [ ] Changes one at a time, each recorded in `LEDGER.md`.

## To-do

- [ ] At the probe step: write a full reference section on the probes (linear, k-NN, and MLP only if used) — how each works mechanically, what it measures, fundamental vs. convenience, and how to read the numbers against the random-init baseline. All probes in one place.
- [ ] Fix the I-JEPA page on atle.dev: it says "EMA-free", but v4 had the EMA teacher.
- [ ] LeJEPA checks in `checks/`: the multi-crop views (sizes, count, native resolution) and SIGReg on known samples (a Gaussian sample near 0.52, a collapsed one near 0.131·n).
- [ ] The folder name `i_jepa_restart` is the user's call (2026-11-05: `train.py` is now `ijepa.py`, and README, PLAN and LEDGER name LeJEPA as current); renaming the folder (e.g. `jepa_pathmnist`) breaks ledger paths and the probe cache.

## Terms

- **SSL (self-supervised learning):** training without human labels; the model builds its own training signal from the data (LeJEPA: views of the same image must embed close together, and the embeddings must look like an isotropic Gaussian).
- **Probe:** a small model trained on the *frozen* encoder's features to predict a real-world label (tissue class). Its "truth" comes from outside the latent space, which is why it's evidence. Not the same as the JEPA training loss, which compares predicted and target embeddings.
- **Positional encoding:** fixed 2-D sin-cos vectors for each patch's (row, col), added to patch content in the encoder; positions are relative to the view. (I-JEPA also adds them to its predictor's mask tokens, so it knows where it is predicting.)
