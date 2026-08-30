"""
Autoregressive drift diagnostic for the STRIDED predictor P -- same idea
as drift_diagnostic.py, but each "step" here is one P call spanning
PREDICTOR_STRIDE real control-steps (action held constant across them),
not one real step.

Uses:
  - jepa_checkpoint_strided.pt (trained by jepa_train_strided.py)
  - position_probe_strided.npz (fit by eval_linear_probe_strided.py --
    the strided model's OWN probe, not position_probe.npz)
  - jepa_eval_data_strided/ (real held-constant-action rollouts +
    ground truth, from collect_eval_data_strided.py -- NOT
    jepa_eval_data/, which was collected under i.i.d. random actions and
    so isn't a valid ground truth for what this model was trained to
    predict)

je.rollout itself needed no changes -- it just calls P once per entry in
the action sequence, and doesn't care what real time each entry spans.

MAX_P_CALLS=30 at PREDICTOR_STRIDE=5 covers 150 real steps -- the same
real-time span as drift_diagnostic.py's MAX_H=150, so the two are directly
comparable: same amount of real time, fewer (hopefully more trustworthy)
P calls to get there.

Run (after collect_eval_data_strided.py and eval_linear_probe_strided.py
have both been run):
    python3 drift_diagnostic_strided.py
"""

import pathlib

import numpy as np
import torch

import cem_planner as cp
import jepa_evaluator as je

EVAL_DATA_DIR = pathlib.Path("jepa_eval_data_strided")
CHECKPOINT_PATH = "jepa_checkpoint_strided.pt"
POSITION_PROBE_PATH = "position_probe_strided.npz"

PREDICTOR_STRIDE = 5     # must match collect_eval_data_strided.py
MAX_P_CALLS = 30         # 30 * PREDICTOR_STRIDE = 150 real steps -- same
                          # real-time span as drift_diagnostic.py's MAX_H
START_SPACING = 10       # spacing (in real steps) between rollout start
                          # points within an episode

DT_PER_CALL = PREDICTOR_STRIDE * cp.CONTROL_TIMESTEP   # real seconds spanned
                                                          # by one P call

# Same success tolerance as drift_diagnostic.py -- from ball_in_cup.xml,
# confirmed from source, not a guess.
SUCCESS_TOLERANCE_M = 0.025


def build_rollout_batch(E, k_context, episodes, device):
    """Returns:
      window:     (B, k_context, D)
      action_seq: (B, MAX_P_CALLS, action_dim) -- one held-constant real
                  action per P call, taken from real index t0 + i*stride
      true_pos:   (B, MAX_P_CALLS, 4)  -- ground truth at t0+(i+1)*stride
      true_vel:   (B, MAX_P_CALLS, 4)
    """
    windows, action_seqs, true_pos, true_vel = [], [], [], []
    span = MAX_P_CALLS * PREDICTOR_STRIDE
    for frames, actions, positions, velocities in episodes:
        T = actions.shape[0]
        for t0 in range(k_context - 1, T - span, START_SPACING):
            call_idx = [t0 + i * PREDICTOR_STRIDE for i in range(MAX_P_CALLS)]
            target_idx = [t0 + (i + 1) * PREDICTOR_STRIDE for i in range(MAX_P_CALLS)]
            win = je.bootstrap_window(E, frames[t0 - k_context + 1 : t0 + 1], device)
            windows.append(win.squeeze(0))
            action_seqs.append(actions[call_idx])
            true_pos.append(positions[target_idx])
            true_vel.append(velocities[target_idx])

    window = torch.stack(windows).to(device)
    action_seq = torch.from_numpy(np.stack(action_seqs).astype(np.float32)).to(device)
    true_pos = torch.from_numpy(np.stack(true_pos).astype(np.float32)).to(device)
    true_vel = torch.from_numpy(np.stack(true_vel).astype(np.float32)).to(device)
    return window, action_seq, true_pos, true_vel


def main():
    device = torch.device("cpu")
    E, P, k_context, latent_dim = je.load_world_model(CHECKPOINT_PATH, device=device)
    W = cp.load_position_probe(POSITION_PROBE_PATH)

    paths = sorted(EVAL_DATA_DIR.glob("episode_*.npz"))
    if not paths:
        print(f"No eval episodes found in {EVAL_DATA_DIR}/ -- run collect_eval_data_strided.py first.")
        return
    episodes = []
    for p in paths:
        d = np.load(p)
        episodes.append((d["frames"], d["actions"], d["positions"], d["velocities"]))

    window, action_seq, true_pos, true_vel = build_rollout_batch(E, k_context, episodes, device)
    B = window.shape[0]
    print(
        f"{B} (episode, start) rollouts, each {MAX_P_CALLS} P-calls "
        f"({MAX_P_CALLS * PREDICTOR_STRIDE} real steps, {MAX_P_CALLS * DT_PER_CALL:.2f}s)."
    )
    if B == 0:
        print("No episode is long enough for this MAX_P_CALLS -- shorten it or check episode length.")
        return

    with torch.no_grad():
        z_preds = je.rollout(P, window, action_seq)     # (B, MAX_P_CALLS, D)
        pred_pos = cp.decode_position(z_preds, W)         # (B, MAX_P_CALLS, 4)

    err = (pred_pos - true_pos).abs()
    cup_err = (err[..., 0] ** 2 + err[..., 1] ** 2).sqrt()
    ball_err = (err[..., 2] ** 2 + err[..., 3] ** 2).sqrt()
    cup_mean, ball_mean = cup_err.mean(dim=0), ball_err.mean(dim=0)

    pred_ball_z_vel = (pred_pos[:, 1:, 3] - pred_pos[:, :-1, 3]) / DT_PER_CALL
    true_ball_z_vel = true_vel[:, 1:, 3]
    vel_err = (pred_ball_z_vel - true_ball_z_vel).abs().mean(dim=0)

    print(f"\n{'tau (calls)':>11s} {'real s':>8s} {'cup L2 err (m)':>16s} {'ball L2 err (m)':>18s} {'ball z-vel err (m/s)':>22s}")
    for tau in range(0, MAX_P_CALLS, 5):
        vel_str = f"{vel_err[tau-1].item():22.4f}" if tau >= 1 else f"{'--':>22s}"
        print(f"{tau:11d} {tau*DT_PER_CALL:8.2f} {cup_mean[tau].item():16.4f} {ball_mean[tau].item():18.4f} {vel_str}")
    last = MAX_P_CALLS - 1
    print(f"{last:11d} {last*DT_PER_CALL:8.2f} {cup_mean[last].item():16.4f} {ball_mean[last].item():18.4f} {vel_err[last-1].item():22.4f}")

    over = (ball_mean > SUCCESS_TOLERANCE_M).nonzero()
    if len(over) > 0:
        tau_break = over[0].item()
        print(
            f"\nBall position error first exceeds the task's own success "
            f"tolerance ({SUCCESS_TOLERANCE_M} m) at tau={tau_break} calls "
            f"({tau_break * DT_PER_CALL:.2f}s of rollout)."
        )
    else:
        print(
            f"\nBall position error never exceeds {SUCCESS_TOLERANCE_M} m "
            f"within {MAX_P_CALLS} calls -- P looks trustworthy over the full tested horizon."
        )


if __name__ == "__main__":
    main()
