"""
Same drift diagnostic as drift_diagnostic_multistride.py, pointed at the
JOINT fine-tuning run's checkpoints instead of the frozen-encoder ones --
this is the actual before/after comparison for whether unfreezing E and
training it together with all five predictors (jepa_train_joint.py)
bought anything over the frozen baseline.

Uses:
  - jepa_checkpoint_joint.pt      (E, fine-tuned -- jepa_train_joint.py)
  - predictor_stride{s}_joint.pt  (P_1..P_16, fine-tuned alongside E)
  - position_probe_joint_mlp.pt   (refit against the joint E --
    eval_mlp_probe_joint.py -- NOT position_probe_strided_mlp.pt, which
    was fit against the old frozen E and is now stale)

Same held-out jepa_eval_data/, same 150-real-step/3s target horizon, same
per-stride reporting -- so the two runs' printed tables are directly
comparable line-for-line against drift_diagnostic_multistride.py's
frozen-encoder results:
  stride  1:  0.0856m  (frozen)
  stride  2:  0.0764m
  stride  4:  0.0552m
  stride  8:  0.0392m
  stride 16:  0.0514m

Run (after jepa_train_joint.py and eval_mlp_probe_joint.py have both been
run):
    uv run python drift_diagnostic_joint.py
"""

import pathlib

import numpy as np
import torch

import cem_planner as cp
import jepa_train as jt
import jepa_evaluator as je

EVAL_DATA_DIR = pathlib.Path("jepa_eval_data")
ENCODER_CHECKPOINT = "jepa_checkpoint_joint.pt"
POSITION_PROBE_PATH = "position_probe_joint_mlp.pt"
PREDICTOR_CHECKPOINT_SUFFIX = "_joint"   # predictor_stride{s}_joint.pt

STRIDE_LEVELS = [1, 2, 4, 8, 16]
TARGET_HORIZON_STEPS = 150
START_SPACING = 10

SUCCESS_TOLERANCE_M = 0.025


def load_encoder(device):
    ckpt = torch.load(ENCODER_CHECKPOINT, map_location=device)
    E = jt.Encoder(3, jt.LATENT_DIM).to(device)
    E.load_state_dict(ckpt["E"])
    E.eval()
    return E


def load_stride_predictor(stride, device):
    ckpt = torch.load(f"predictor_stride{stride}{PREDICTOR_CHECKPOINT_SUFFIX}.pt", map_location=device)
    P = jt.Predictor(jt.LATENT_DIM, stride * jt.ACTION_DIM, jt.PREDICTOR_HIDDEN, jt.K_CONTEXT).to(device)
    P.load_state_dict(ckpt["P"])
    P.eval()
    return P


def build_rollout_batch(E, episodes, stride, max_p_calls, device):
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
        z_preds = je.rollout(P, window, action_seq)
        pred_pos = cp.decode_position(z_preds, W)

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
    E = load_encoder(device)

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
