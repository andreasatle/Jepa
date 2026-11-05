"""
Refits the nonlinear (MLP) position probe against the JOINT fine-tuning
run's encoder (jepa_checkpoint_joint.pt from jepa_train_joint.py), instead
of the frozen-encoder-era jepa_checkpoint_strided.pt.

Why this is needed at all: E's weights changed during the joint fine-tune
(it's no longer frozen), so its latent space has shifted from what
position_probe_strided_mlp.pt was fit against. Decoding jepa_checkpoint_joint.pt's
z through that stale probe would silently mix "how good is the new
encoder/predictors" with "how stale is the probe" -- refitting removes
that confound, same reasoning as the v2 roadmap's Phase 6 note about
refitting after ANY encoder change.

Everything else is identical to eval_mlp_probe_strided.py -- same
MLPProbe architecture, same single-frame z->position task, same eval
data/split -- this only swaps which checkpoint E comes from and where the
result is saved, so it doesn't overwrite the frozen-era probe (still
needed for comparison).

Run (after jepa_train_joint.py has produced jepa_checkpoint_joint.pt):
    uv run python eval_mlp_probe_joint.py
"""

import eval_linear_probe_strided as elps
import eval_mlp_probe_strided as embp

elps.CHECKPOINT_PATH = "jepa_checkpoint_joint.pt"
embp.OUT_PATH = "position_probe_joint_mlp.pt"

if __name__ == "__main__":
    embp.main()
