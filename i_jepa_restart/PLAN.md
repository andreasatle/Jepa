# I-JEPA restart — plan

A clean restart of the I-JEPA experiment on PathMNIST. The old `../i_jepa/` (v1–v4) stays as reference only; nothing is copied from it without being justified here first.

## Goal (decide before any code)

Dataset: PathMNIST (9 tissue classes). Decided: keep the medical data.

Success is measured by a classifier on the frozen encoder's embedding (a linear probe). Two candidate goal statements, not yet chosen:

1. **Plain:** close the gap between the floor (linear probe on a random-init encoder, 74.3% in the old setup) and the ceiling (supervised ResNet-18 trained end-to-end on all labels, ~91% in the MedMNIST v2 benchmark).
2. **Label-scarce:** with only 1% of the labels, a linear classifier on frozen I-JEPA features beats the same network trained from scratch on that same 1%.

## Two stages (labels only in stage 2)

1. **Pretraining, no labels at all.** I-JEPA learns the embedding purely from images. Tissue type is never seen; if it shows up in the embedding, it emerged.
2. **Classifier on the frozen embedding.** The encoder is frozen; only a small classifier (linear probe) is trained, on all or a fraction of the labels. It tests whether the classes emerged as (linearly) separable structure.

Label hygiene: choosing checkpoints or hyperparameters by probe accuracy uses labels indirectly. Do that on the validation split only; the test split is touched once, for the final number.

Fine-tuning (unfreezing the encoder while training the classifier) is a different, weaker test of emergence, since the labels then reshape the embedding. Not part of the goal.

## Rules

- Start from the official I-JEPA recipe (multi-block masks, target LayerNorm + smooth-L1, official init and optimizer schedules), including the EMA target encoder, with two decided deviations: (1) VICReg on the online encoder, in addition to EMA (VICReg without EMA was tested and lost, see LEDGER.md); (2) the context is exactly the 14x14 grid minus the target patches, padded per batch with an attention mask, instead of the official 85-100% block truncated to the shortest context in the batch. Nothing else extra.
- One change at a time, measured against the same baseline with the same probes. A change stays only if it clearly helps; otherwise it comes out.
- `LEDGER.md` lists every component in the code and the measurement that justifies it. Anything not in the ledger gets deleted.
- Tests are few and fixed: mask correctness, EMA update, and the probe separating floor from ceiling. No test per change.
- Keep experiments fast (lower resolution / small ViT) so each one takes minutes, not overnight.

## Steps

1. [ ] Choose the goal statement above.
2. [ ] Validate the measuring stick: the probes must clearly separate a random encoder (floor) from a known-good encoder (ceiling) before any I-JEPA training.
3. [ ] Minimal I-JEPA (paper recipe), trained and probed.
4. [ ] Changes one at a time, each recorded in `LEDGER.md`.

## To-do

- [ ] At the probe step: write a full reference section on the probes (linear, k-NN, and MLP only if used) — how each works mechanically, what it measures, fundamental vs. convenience, and how to read the numbers against the random-init baseline. All probes in one place.
- [ ] Fix the I-JEPA page on atle.dev: it says "EMA-free", but v4 had the EMA teacher.
- [ ] Naming after the switch to LeJEPA (2026-11-04; the user will decide later): the folder holds both methods, I-JEPA kept as the reference for the ledger's comparisons. Proposed: `train.py` -> `ijepa.py` (it is I-JEPA's trainer; `lejepa.py` imports its helpers from it), README and LEDGER headers saying LeJEPA is current and I-JEPA the reference. The folder name `i_jepa_restart` is the user's call; renaming it (e.g. `jepa_pathmnist`) breaks ledger paths and the probe cache. This plan's Goal and Rules still describe I-JEPA and need the same update.

## Terms

- **SSL (self-supervised learning):** training without human labels; the model builds its own training signal from the data (I-JEPA: predict embeddings of hidden patches from visible ones).
- **Probe:** a small model trained on the *frozen* encoder's features to predict a real-world label (tissue class). Its "truth" comes from outside the latent space, which is why it's evidence. Not the same as the JEPA training loss, which compares predicted and target embeddings.
- **Positional encoding:** fixed 2-D sin-cos vectors for each patch's (row, col), added to patch content in the encoder, and to the predictor's mask tokens so it knows where it is predicting.
