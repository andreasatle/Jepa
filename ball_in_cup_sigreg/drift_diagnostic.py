"""
Autoregressive drift diagnostic for the JEPA predictor P.

Rolls P forward on REAL recorded action sequences (not resampled/random),
compares the *decoded* predicted position (position_probe.npz, same
decode_position used by cem_planner.py) against REAL ground-truth
position/velocity from the eval data at every step, and reports how the
error grows with rollout length tau.

This is the concrete answer to "how far can P be trusted", which is what
cem_planner.py's horizon H has been a placeholder pending: the cost
function is only meaningful at steps where P's predictions are still
roughly right.

Reports, separately for cup and ball (cup is control-driven + fairly
linear-spring, expected more predictable than the ball's swinging):
  - mean L2 position error (meters) at each tau
  - mean ball z-velocity error (m/s), from finite-differencing the
    *predicted* position sequence -- the same method cem_planner.py's
    rollout_cost uses for v_rel, so this directly checks whether that
    design choice is trustworthy, not just position itself.
  - the tau at which mean ball position error first exceeds the task's
    own success tolerance (~0.025m, derived from ball_in_cup.xml:
    target site half-extent .05 - ball radius .025 -- confirmed from
    source, not a guess). Past that tau, P's predictions are no more
    precise than the task's own margin, so trusting the cost function
    there is meaningless regardless of what it says.

Run (after eval_linear_probe.py has produced position_probe.npz):
    python3 drift_diagnostic.py
"""

import pathlib

import numpy as np
import torch

import cem_planner as cp
import jepa_evaluator as je

EVAL_DATA_DIR = pathlib.Path("jepa_eval_data")
MAX_H = 150         # matches cem_planner.py's placeholder horizon
START_STRIDE = 10   # spacing between rollout start points within an episode

# From ball_in_cup.xml: target site size=".05 .006 .05", ball geom
# size=".025" -- in_target() tolerance = target_half_extent - ball_radius
# = .05 - .025 = .025m, in both x and z. Confirmed from source.
SUCCESS_TOLERANCE_M = 0.025


def build_rollout_batch(E, k_context, episodes, device):
    """episodes: list of (frames, actions, positions, velocities) tuples,
    one per loaded episode_*.npz. Returns:
      window:     (B, k_context, D)      -- real bootstrapped starting windows
      action_seq: (B, MAX_H, action_dim) -- REAL recorded actions
      true_pos:   (B, MAX_H, 4)          -- ground truth [cup_x,cup_z,ball_x,ball_z]
      true_vel:   (B, MAX_H, 4)          -- ground truth velocities, same order
    B = number of (episode, start) pairs that fit a full MAX_H window.
    """
    windows, action_seqs, true_pos, true_vel = [], [], [], []
    for frames, actions, positions, velocities in episodes:
        T = actions.shape[0]  # number of actions/transitions in this episode
        for t0 in range(k_context - 1, T - MAX_H, START_STRIDE):
            win = je.bootstrap_window(E, frames[t0 - k_context + 1 : t0 + 1], device)
            windows.append(win.squeeze(0))
            action_seqs.append(actions[t0 : t0 + MAX_H])
            true_pos.append(positions[t0 + 1 : t0 + 1 + MAX_H])
            true_vel.append(velocities[t0 + 1 : t0 + 1 + MAX_H])

    window = torch.stack(windows).to(device)
    action_seq = torch.from_numpy(np.stack(action_seqs).astype(np.float32)).to(device)
    true_pos = torch.from_numpy(np.stack(true_pos).astype(np.float32)).to(device)
    true_vel = torch.from_numpy(np.stack(true_vel).astype(np.float32)).to(device)
    return window, action_seq, true_pos, true_vel


def main():
    device = torch.device("cpu")
    E, P, k_context, latent_dim = je.load_world_model(device=device)
    W = cp.load_position_probe()

    paths = sorted(EVAL_DATA_DIR.glob("episode_*.npz"))
    if not paths:
        print("No eval episodes found -- run collect_eval_data.py first.")
        return
    episodes = []
    for p in paths:
        d = np.load(p)
        episodes.append((d["frames"], d["actions"], d["positions"], d["velocities"]))

    window, action_seq, true_pos, true_vel = build_rollout_batch(E, k_context, episodes, device)
    B = window.shape[0]
    print(f"{B} (episode, start) rollouts, each {MAX_H} steps of REAL recorded actions.")
    if B == 0:
        print(f"No episode is long enough for MAX_H={MAX_H} -- shorten MAX_H or check episode length.")
        return

    with torch.no_grad():
        z_preds = je.rollout(P, window, action_seq)     # (B, MAX_H, D)
        pred_pos = cp.decode_position(z_preds, W)         # (B, MAX_H, 4)

    err = (pred_pos - true_pos).abs()                                 # (B, MAX_H, 4)
    cup_err = (err[..., 0] ** 2 + err[..., 1] ** 2).sqrt()             # (B, MAX_H)
    ball_err = (err[..., 2] ** 2 + err[..., 3] ** 2).sqrt()
    cup_mean, ball_mean = cup_err.mean(dim=0), ball_err.mean(dim=0)    # (MAX_H,)

    # Ball z-velocity, via finite-differencing the *predicted* position --
    # exactly what cem_planner.py's rollout_cost does for v_rel.
    dt = cp.CONTROL_TIMESTEP
    pred_ball_z_vel = (pred_pos[:, 1:, 3] - pred_pos[:, :-1, 3]) / dt   # (B, MAX_H-1)
    true_ball_z_vel = true_vel[:, 1:, 3]                                # aligned to same tau
    vel_err = (pred_ball_z_vel - true_ball_z_vel).abs().mean(dim=0)     # (MAX_H-1,)

    print(f"\n{'tau':>5s} {'cup L2 err (m)':>16s} {'ball L2 err (m)':>18s} {'ball z-vel err (m/s)':>22s}")
    for tau in range(10, MAX_H, 10):
        print(f"{tau:5d} {cup_mean[tau].item():16.4f} {ball_mean[tau].item():18.4f} {vel_err[tau-1].item():22.4f}")
    print(f"{MAX_H-1:5d} {cup_mean[-1].item():16.4f} {ball_mean[-1].item():18.4f} {vel_err[-1].item():22.4f}")

    over = (ball_mean > SUCCESS_TOLERANCE_M).nonzero()
    if len(over) > 0:
        tau_break = over[0].item()
        print(
            f"\nBall position error first exceeds the task's own success "
            f"tolerance ({SUCCESS_TOLERANCE_M} m) at tau={tau_break} "
            f"({tau_break * dt:.2f}s of rollout)."
        )
    else:
        print(
            f"\nBall position error never exceeds {SUCCESS_TOLERANCE_M} m "
            f"within {MAX_H} steps -- P looks trustworthy over the full tested horizon."
        )


if __name__ == "__main__":
    main()
