"""
Autoregressive drift diagnostic for the v3 multi-stride predictors
(P_1, P_2, P_4, P_8, P_16 from jepa_train_multistride.py), run against
HELD-OUT episodes -- jepa_eval_data/, the i.i.d.-per-step-random-action
set, matching what these predictors were actually trained on (NOT
jepa_eval_data_strided/, which is held-constant-action data specific to
the old PREDICTOR_STRIDE=5 model).

This is the actual test for the two open questions from the roadmap's
"nearsighted encoder" note and the coarse-to-fine CEM sampling idea:

  1. Does chaining a coarse predictor enough times to span a REAL horizon
     (not just a handful of hops) stay flat, or does it blow up like the
     original v1 fine-grained rollout did? ("if we can get long samples
     to be stable")
  2. Do the training-loss floors from jepa_train_multistride.py (which
     dropped with stride, possibly just from the shrinking/overfit-prone
     training set) actually translate into good held-out rollout
     accuracy, or was that a training-set-size artifact?

All five levels are evaluated against the SAME real-time horizon
(TARGET_HORIZON_STEPS real steps) so the results are directly comparable
to each other and to v1/v2's own 150-real-step convention.

Uses the SAME encoder (jepa_checkpoint_strided.pt) and position probe
(position_probe_strided_mlp.pt) as drift_diagnostic_strided.py --
frozen E, one probe -- since v3 never touched either.

Run (after jepa_train_multistride.py has produced predictor_stride*.pt):
    uv run python drift_diagnostic_multistride.py
"""

import pathlib

import numpy as np
import torch

import cem_planner as cp
import jepa_train as jt
import jepa_evaluator as je

EVAL_DATA_DIR = pathlib.Path("jepa_eval_data")
ENCODER_CHECKPOINT = "jepa_checkpoint_strided.pt"
POSITION_PROBE_PATH = "position_probe_strided_mlp.pt"

STRIDE_LEVELS = [1, 2, 4, 8, 16]
TARGET_HORIZON_STEPS = 150   # same real-time span as v1/v2's drift diagnostics
START_SPACING = 10           # spacing (real steps) between rollout start points

SUCCESS_TOLERANCE_M = 0.025  # same task-defined tolerance as the other diagnostics


def load_frozen_encoder(device):
    ckpt = torch.load(ENCODER_CHECKPOINT, map_location=device)
    E = jt.Encoder(3, jt.LATENT_DIM).to(device)
    E.load_state_dict(ckpt["E"])
    E.eval()
    return E


def load_stride_predictor(stride, device):
    ckpt = torch.load(f"predictor_stride{stride}.pt", map_location=device)
    P = jt.Predictor(jt.LATENT_DIM, stride * jt.ACTION_DIM, jt.PREDICTOR_HIDDEN, jt.K_CONTEXT).to(device)
    P.load_state_dict(ckpt["P"])
    P.eval()
    return P


def build_rollout_batch(E, episodes, stride, max_p_calls, device):
    """Same idea as drift_diagnostic_strided.py's build_rollout_batch, but
    the action fed per P call is the full block of `stride` REAL actions
    (flattened), not one held-constant action -- matching how P_stride was
    trained."""
    k = jt.K_CONTEXT
    windows, action_seqs, true_pos = [], [], []
    span = max_p_calls * stride
    for frames, actions, positions, _velocities in episodes:
        T = actions.shape[0]
        for t0 in range(k - 1, T - span, START_SPACING):
            context_idx = [t0 - (k - 1 - j) * stride for j in range(k)]
            win = je.bootstrap_window(E, frames[context_idx], device)
            windows.append(win.squeeze(0))
            action_block_seq = np.stack(
                [actions[t0 + i * stride: t0 + (i + 1) * stride].reshape(-1) for i in range(max_p_calls)]
            )
            action_seqs.append(action_block_seq)
            target_idx = [t0 + (i + 1) * stride for i in range(max_p_calls)]
            true_pos.append(positions[target_idx])

    if not windows:
        return None
    window = torch.stack(windows).to(device)
    action_seq = torch.from_numpy(np.stack(action_seqs).astype(np.float32)).to(device)
    true_pos = torch.from_numpy(np.stack(true_pos).astype(np.float32)).to(device)
    return window, action_seq, true_pos


def evaluate_stride(E, episodes, stride, device):
    max_p_calls = TARGET_HORIZON_STEPS // stride
    dt_per_call = stride * cp.CONTROL_TIMESTEP

    batch = build_rollout_batch(E, episodes, stride, max_p_calls, device)
    if batch is None:
        print(f"stride={stride}: no episode is long enough for {max_p_calls} calls -- skipping.")
        return
    window, action_seq, true_pos = batch
    B = window.shape[0]

    P = load_stride_predictor(stride, device)
    W = cp.load_position_probe_mlp(POSITION_PROBE_PATH)

    with torch.no_grad():
        z_preds = je.rollout(P, window, action_seq)       # (B, max_p_calls, D)
        pred_pos = cp.decode_position(z_preds, W)          # (B, max_p_calls, 4)

    err = (pred_pos - true_pos).abs()
    cup_err = (err[..., 0] ** 2 + err[..., 1] ** 2).sqrt()
    ball_err = (err[..., 2] ** 2 + err[..., 3] ** 2).sqrt()
    cup_mean, ball_mean = cup_err.mean(dim=0), ball_err.mean(dim=0)

    real_span = max_p_calls * stride * cp.CONTROL_TIMESTEP
    print(
        f"\n=== stride={stride} ({B} rollouts, {max_p_calls} P-calls, "
        f"{max_p_calls * stride} real steps, {real_span:.2f}s) ==="
    )
    print(f"{'n_calls':>8s} {'real s':>8s} {'cup L2 err (m)':>16s} {'ball L2 err (m)':>18s}")
    # idx i in cup_mean/ball_mean is the result AFTER (i+1) completed P calls.
    checkpoints = sorted(set([0, max_p_calls // 4, max_p_calls // 2, (3 * max_p_calls) // 4, max_p_calls - 1]))
    for idx in checkpoints:
        n_calls = idx + 1
        real_s = n_calls * dt_per_call
        print(f"{n_calls:8d} {real_s:8.2f} {cup_mean[idx].item():16.4f} {ball_mean[idx].item():18.4f}")

    over = (ball_mean > SUCCESS_TOLERANCE_M).nonzero()
    if len(over) > 0:
        idx_break = over[0].item()
        n_calls_break = idx_break + 1
        print(
            f"Ball error first exceeds {SUCCESS_TOLERANCE_M} m at "
            f"{n_calls_break} calls ({n_calls_break * dt_per_call:.2f}s)."
        )
    else:
        print(f"Ball error never exceeds {SUCCESS_TOLERANCE_M} m within the tested {real_span:.2f}s horizon.")


def main():
    device = torch.device("cpu")
    E = load_frozen_encoder(device)

    paths = sorted(EVAL_DATA_DIR.glob("episode_*.npz"))
    if not paths:
        print(f"No eval episodes found in {EVAL_DATA_DIR}/.")
        return
    episodes = []
    for p in paths:
        d = np.load(p)
        episodes.append((d["frames"], d["actions"], d["positions"], d["velocities"]))
    print(f"Loaded {len(episodes)} held-out episodes from {EVAL_DATA_DIR}/.")

    for stride in STRIDE_LEVELS:
        evaluate_stride(E, episodes, stride, device)


if __name__ == "__main__":
    main()
