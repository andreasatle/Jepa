"""
Same held-out drift diagnostic as drift_diagnostic_joint.py, pointed at
the REFROZEN checkpoints (jepa_train_multistride.py's freeze-E-retrain-
predictors phase, predictor_stride{s}_refrozen.pt) instead of
predictor_stride{s}_joint.pt -- the real before/after check for whether
freezing E at its joint-fine-tuned state and retraining the predictors
alone (with LR decay this time, see conversation) recovered s8/s16's
held-out generalization back toward the frozen-baseline numbers quoted
in drift_diagnostic_joint.py's own docstring.

E itself is unchanged throughout the refreeze phase (still
jepa_checkpoint_joint.pt, epoch 1557, frozen -- never touched by
jepa_train_multistride.py's refreeze run), so ENCODER_CHECKPOINT stays
the same as drift_diagnostic_joint.py's; only PREDICTOR_CHECKPOINT_SUFFIX
differs. Reuses drift_diagnostic_joint's functions directly rather than
duplicating them, same pattern cem_planner_strided.py uses for cem_planner.

Run (after jepa_train_multistride.py's refreeze phase has produced all
five predictor_stride{s}_refrozen.pt files):
    uv run python drift_diagnostic_refrozen.py
"""

import drift_diagnostic_joint as ddj

ddj.PREDICTOR_CHECKPOINT_SUFFIX = "_refrozen"

if __name__ == "__main__":
    ddj.main()
