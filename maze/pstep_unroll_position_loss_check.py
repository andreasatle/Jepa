"""
pstep_unroll_position_loss_check.py

Maze JEPA -- Phase 3 diagnostic, follow-up to pstep_manifold_drift_check.py's
result: that script's Part B REFUTED the edge-of-ball-sensitivity hypothesis
(P_step's local expansion factor is < 1 everywhere in BOTH geometries --
contractive, not expansive -- and gets MORE contractive at higher raw norm
in hyperbolic, the opposite of what boundary hypersensitivity would predict).
The euclidean control showed essentially the same autoregressive error
explosion pattern despite having no manifold boundary at all. So the
~12-17-step-independent-of-distance blowup isn't a hyperbolic geometry
artifact -- it's shared by both geometries, which points back to P_step's
own one-step accuracy/bias, not local numerical instability.

jepa_train_step.py's training loop only ever reports ONE averaged scalar
per logged step: `loss = torch.stack(step_losses).mean()` across all
UNROLL_STEPS=20 positions. That average could be hiding a real problem: if
loss at LATE unroll positions (t close to 20) stays persistently high
throughout training while EARLY positions (t close to 1) are easy and
near-zero, the reported ~0.5 average is consistent with both "training
worked evenly across all positions" and "training never fixed the late
positions, the average just got dragged down by the easy early ones."
Those look identical in the logged scalar but are very different diagnoses
-- and the real rollouts in pstep_manifold_drift_check.py's Part A already
blow up well INSIDE this trained 20-step window (around step 12-17), which
is exactly what the second scenario would predict.

This script re-runs the SAME unrolled forward pass jepa_train_step.py's
training loop uses (imports build_transitions directly from it, no
duplication), against the ALREADY-TRAINED, frozen checkpoint -- no
retraining -- but reports the mean loss AT EACH of the unroll_steps
positions separately, across a large batch, so the two scenarios above are
told apart directly.

Run (after jepa_train.py and jepa_train_step.py have both been run for the
geometry being checked):
    uv run python pstep_unroll_position_loss_check.py --geometry hyperbolic
    uv run python pstep_unroll_position_loss_check.py --geometry euclidean
"""
import argparse
import json

import torch

from jepa_model import (
    cell_index,
    load_encoder_checkpoint,
    load_step_checkpoint,
    squared_dist,
    to_geometry,
)
from jepa_train_step import build_transitions, DATA_PATH, UNROLL_STEPS

EVAL_BATCH = 4000  # a large fixed sample of transitions, not the whole
                    # dataset -- big enough for stable per-position means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic"], required=True)
    args = parser.parse_args()
    geometry = args.geometry

    E, e_ckpt = load_encoder_checkpoint(geometry)
    P_step, s_ckpt = load_step_checkpoint(geometry)
    size = e_ckpt["size"]
    unroll_steps = s_ckpt.get("unroll_steps", UNROLL_STEPS)

    with open(DATA_PATH) as f:
        data = json.load(f)
    starts, action_seqs, target_seqs = build_transitions(data["episodes"], unroll_steps)

    torch.manual_seed(0)
    n = len(starts)
    batch = torch.randperm(n)[:EVAL_BATCH]

    # build_transitions (jepa_train_step.py) already converts actions to
    # indices internally -- action_seqs[i] is already a list of ints, not
    # action-name strings, so no ACTION_INDEX lookup here.
    start_idx = torch.tensor([cell_index(starts[i], size) for i in batch])
    action_idx = torch.tensor([action_seqs[i] for i in batch])
    target_idx = torch.tensor(
        [[cell_index(c, size) for c in target_seqs[i]] for i in batch]
    )

    with torch.no_grad():
        raw = E(start_idx)
        target_raw = E(target_idx.reshape(-1)).reshape(len(batch), unroll_steps, -1)
        per_position_loss = []
        for t in range(unroll_steps):
            pred_raw = P_step(raw, action_idx[:, t])
            loss_t = squared_dist(
                to_geometry(pred_raw, geometry), to_geometry(target_raw[:, t], geometry), geometry
            )
            per_position_loss.append(loss_t.mean().item())
            raw = pred_raw  # autoregressive, same as training -- own prediction feeds forward

    print(f"[{geometry}] per-unroll-position loss (mean over {len(batch)} transitions, "
          f"reusing the exact autoregressive unroll training used), unroll_steps={unroll_steps}:")
    for t, l in enumerate(per_position_loss):
        print(f"  position {t + 1:>3}  mean_loss={l:.4f}")
    print(f"  overall mean (this is what training's logged scalar averages together): "
          f"{sum(per_position_loss) / len(per_position_loss):.4f}")


if __name__ == "__main__":
    main()
