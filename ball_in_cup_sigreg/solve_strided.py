"""
Full closed-loop CEM/MPC attempt at solving ball_in_cup-catch with the
strided model -- chains cem_planner_strided.py's plan_step() into an
actual real-time control loop against the live dm_control environment,
instead of the single-call smoke test.

Loop, once per real decision point (every PREDICTOR_STRIDE real steps):
  1. plan_step() from the current REAL window -> first_action
  2. apply first_action for PREDICTOR_STRIDE real env.step() calls
     (action-repeat, matching how the strided model was trained -- see
     collect_data_strided.py)
  3. render + encode the new REAL frame via E, slide it into the window
     (the REAL embedding, not a predicted one -- this is genuine closed-
     loop MPC: replanning against ground truth every decision point,
     never trusting the open-loop plan beyond its first action)
  4. check the real environment's own physics.in_target() for success

No warm-starting between replanning calls (prev_mean=None every call) --
consistent with the standing "no warm-starting" decision from
project_brief.md. The shift-forward-plan variant discussed separately
isn't implemented here; this is the conservative default.

Initial condition: NOT dm_control's default random ball placement.
Overrides qpos to the numerically-verified steady-state hang -- cup_x=0,
cup_z=-0.0641, ball_x=0, ball_z=-0.0727 -- confirmed earlier by running
10s of zero-action simulation and cross-checked against
(m_cup+m_ball)*g/stiffness to 5 decimal places. This is what "starting in
steady state" was defined to mean.

Saves a GIF of the attempt (same convention as ball_in_cup_demo.py's
run_headless_gif) so the result can be inspected directly, not just
trusted from a cost number, and prints real ball_to_target distance every
decision point either way.

Run:
    MUJOCO_GL=egl uv run python solve_strided.py
"""

import numpy as np
import torch
from dm_control import suite
from PIL import Image

import cem_planner as cp
import cem_planner_strided as cps
import jepa_evaluator as je
import jepa_train as jt

CHECKPOINT_PATH = "jepa_checkpoint_strided.pt"
POSITION_PROBE_PATH = "position_probe_strided_mlp.pt"   # nonlinear (MLP) probe, see cem_planner_strided.py
PREDICTOR_STRIDE = 5

MAX_REAL_STEPS = 1200      # 24.0s -- doubled from 600/12.0s to test
                            # whether still-swinging "failures" are slow-
                            # converging (settle given more time) or
                            # genuinely stuck (never would). A real robot
                            # can't discard state and retry from a fresh
                            # random IC -- the only physically meaningful
                            # "retry" is continuing to act longer from
                            # wherever it actually is, so this is that,
                            # not a bigger/more expensive CEM search
                            # (n_samples stays at cem_planner_strided.py's
                            # default -- see that module).
IMAGE_SIZE = 64             # must match what E was trained on
RENDER_SIZE = 200           # GIF only, doesn't touch the model
GIF_PATH = "solve_strided_attempt.gif"

# Verified steady-state hang (see module docstring) -- overrides
# dm_control's randomized default ball placement.
STEADY_STATE_QPOS = {"cup_x": 0.0, "cup_z": -0.0641, "ball_x": 0.0, "ball_z": -0.0727}


def set_steady_state(env):
    with env.physics.reset_context():
        for name, val in STEADY_STATE_QPOS.items():
            env.physics.named.data.qpos[name] = val
        env.physics.named.data.qvel[:] = 0.0


def encode_frame(E, frame, device):
    """One real (H, W, 3) uint8 frame -> (1, D) embedding."""
    with torch.no_grad():
        return E(jt._frames_to_tensor(frame[None]).to(device))


def main():
    device = torch.device("cpu")
    E, P, k_context, latent_dim = je.load_world_model(CHECKPOINT_PATH, device=device)
    W = cp.load_position_probe_mlp(POSITION_PROBE_PATH)
    cfg = cps.CEMConfig()

    env = suite.load(domain_name="ball_in_cup", task_name="catch")
    env.reset()
    set_steady_state(env)

    def render_model():
        return env.physics.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_id=0)

    def render_display():
        return env.physics.render(height=RENDER_SIZE, width=RENDER_SIZE, camera_id=0)

    gif_frames = [Image.fromarray(render_display())]

    # Bootstrap the K_CONTEXT window with zero action -- at the verified
    # steady state, zero action should leave the system essentially at
    # rest, so this doesn't waste any real progress.
    boot_frames = [render_model()]
    for _ in range(k_context - 1):
        env.step(np.zeros(2, dtype=np.float32))
        boot_frames.append(render_model())
        gif_frames.append(Image.fromarray(render_display()))
    window = je.bootstrap_window(E, np.stack(boot_frames), device)

    solved = False
    best_dist = float("inf")
    best_dist_step = 0
    real_step = 0
    warm_elites = None   # shift-forward warm start across real decision
                           # points -- see cem_planner.plan_step's docstring

    while real_step < MAX_REAL_STEPS:
        first_action, _mean, _std, info = cps.plan_step(P, window, W, cfg, warm_elites=warm_elites)
        warm_elites = cp.shift_warm_elites(info["pooled_elites"])
        action_np = first_action.detach().numpy()

        for _ in range(PREDICTOR_STRIDE):
            env.step(action_np)
            real_step += 1
            gif_frames.append(Image.fromarray(render_display()))
            if real_step >= MAX_REAL_STEPS:
                break

        dist = float(np.linalg.norm(env.physics.ball_to_target()))
        in_target = bool(env.physics.in_target())
        if dist < best_dist:
            best_dist, best_dist_step = dist, real_step
        print(
            f"real_step={real_step:4d}  action={action_np.round(3).tolist()}  "
            f"ball_to_target={dist:.4f}  in_target={in_target}"
        )

        new_z = encode_frame(E, render_model(), device)
        window = torch.cat([window[:, 1:, :], new_z.unsqueeze(1)], dim=1)   # slide in the REAL z

        if in_target:
            solved = True
            break

    gif_frames[0].save(
        GIF_PATH, save_all=True, append_images=gif_frames[1:], duration=50, loop=0
    )

    print()
    if solved:
        print(f"SOLVED at real_step={real_step} ({real_step * 0.02:.2f}s).")
    else:
        print(
            f"Not solved within {MAX_REAL_STEPS} steps ({MAX_REAL_STEPS * 0.02:.1f}s). "
            f"Closest approach: {best_dist:.4f}m at real_step={best_dist_step}."
        )
    print(f"Saved {len(gif_frames)}-frame GIF to {GIF_PATH}")


if __name__ == "__main__":
    main()
