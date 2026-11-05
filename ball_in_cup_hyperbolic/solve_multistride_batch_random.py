"""
Closed-loop CEM/MPC evaluation of the adaptive multi-stride planner
(cem_planner_multistride.py) against the real dm_control environment --
same batch-of-random-starts protocol as
~/Projects/Jepa/ball_in_cup/solve_strided_batch_random.py (N_TRIALS,
randomized initial condition via env.reset(), success rate / solve time /
closest approach reporting), adapted for this planner's two real
differences from that precedent:

  1. Deploys at the FINE (real-step) cadence, not every PREDICTOR_STRIDE
     steps -- mean/std/actions are real-resolution throughout (see
     cem_planner_multistride.py's module docstring), so every real
     env.step() is preceded by its own full plan_step() call, not an
     action-repeat block. That's 5x more replanning calls per trial than
     solve_strided_batch_random.py, at potentially much higher per-call
     cost if the search walks down to stride 1 -- see the conversation
     before assuming this finishes in comparable wall-clock time.
  2. Uses the JOINT fine-tuned checkpoints via
     cem_planner_multistride.load_joint_encoder/load_joint_predictors,
     not jepa_checkpoint_strided.pt.

Prints per-decision-point plan_time and the stride schedule the ranking
check actually chose (info["stride_history"]) -- exactly the two numbers
a timing pilot needs: how slow one real step is, and whether the
adaptive scheme is spending its time coarse or fine during real play.

N_TRIALS=1 here is a deliberate TIMING PILOT, not the real evaluation --
see the conversation. Bump it to 30 (matching
solve_strided_batch_random.py's precedent) once this run's wall-clock
time is known and judged acceptable.

Run:
    MUJOCO_GL=egl uv run python solve_multistride_batch_random.py
"""

import sys
import time

import numpy as np
import torch
from dm_control import suite

import cem_planner as cp
import cem_planner_multistride as cpm
import jepa_evaluator as je
import jepa_train as jt

N_TRIALS = 1   # timing pilot -- see module docstring

IMAGE_SIZE = 64             # must match what E was trained on
MAX_REAL_STEPS = 1200       # 24.0s at CONTROL_TIMESTEP=0.02s -- same real-
                            # time budget as solve_strided_batch_random.py,
                            # for a like-for-like solve-time comparison,
                            # even though this planner spends one full
                            # plan_step() per real step instead of one per
                            # PREDICTOR_STRIDE=5 steps.


def encode_frame(E, frame, device):
    """One real (H, W, 3) uint8 frame -> (1, D) embedding."""
    with torch.no_grad():
        return E(jt._frames_to_tensor(frame[None]).to(device))


def main():
    device = torch.device("cpu")
    E = cpm.load_joint_encoder(device)
    predictors = cpm.load_joint_predictors(device)
    probe_path = sys.argv[1] if len(sys.argv) > 1 else cpm.POSITION_PROBE_PATH
    W = cp.load_position_probe_mlp(probe_path)
    print(f"Using probe: {probe_path}")
    cfg = cpm.CEMConfig()   # full defaults -- H_real=480, n_samples=500, n_iters=6

    k_context = jt.K_CONTEXT
    results = []

    for trial in range(1, N_TRIALS + 1):
        env = suite.load(domain_name="ball_in_cup", task_name="catch")
        env.reset()   # task's own randomized IC -- no steady-state override

        def render_model():
            return env.physics.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_id=0)

        boot_frames = [render_model()]
        for _ in range(k_context - 1):
            env.step(np.zeros(2, dtype=np.float32))
            boot_frames.append(render_model())
        window = je.bootstrap_window(E, np.stack(boot_frames), device)

        solved = False
        best_dist = float("inf")
        best_dist_step = 0
        real_step = 0
        warm_elites = None   # shift-forward warm start across real decision
                               # points -- see cem_planner.plan_step's docstring

        while real_step < MAX_REAL_STEPS:
            t0 = time.time()
            first_action, _mean, _std, info = cpm.plan_step(predictors, window, W, cfg, warm_elites=warm_elites)
            plan_time = time.time() - t0
            warm_elites = cp.shift_warm_elites(info["pooled_elites"])
            action_np = first_action.detach().numpy()

            env.step(action_np)
            real_step += 1

            dist = float(np.linalg.norm(env.physics.ball_to_target()))
            in_target = bool(env.physics.in_target())
            if dist < best_dist:
                best_dist, best_dist_step = dist, real_step

            print(
                f"real_step={real_step:4d}  plan_time={plan_time:.2f}s  "
                f"strides={info['stride_history']}  "
                f"ball_to_target={dist:.4f}  in_target={in_target}"
            )

            new_z = encode_frame(E, render_model(), device)
            window = torch.cat([window[:, 1:, :], new_z.unsqueeze(1)], dim=1)   # slide in the REAL z

            if in_target:
                solved = True
                break

        results.append((solved, real_step, best_dist, best_dist_step))
        status = f"SOLVED at {real_step * 0.02:.2f}s" if solved else f"not solved, closest {best_dist:.4f}m @ step {best_dist_step}"
        print(f"trial {trial:2d}/{N_TRIALS}: {status}")

    n_solved = sum(1 for r in results if r[0])
    solve_times = [r[1] * 0.02 for r in results if r[0]]
    closest_all = [r[2] for r in results]
    print()
    print(f"success rate: {n_solved}/{N_TRIALS}")
    if solve_times:
        print(f"solve times (s): mean={np.mean(solve_times):.2f}  min={np.min(solve_times):.2f}  max={np.max(solve_times):.2f}")
    print(f"closest approach (m), all trials: mean={np.mean(closest_all):.4f}  min={np.min(closest_all):.4f}  max={np.max(closest_all):.4f}")


if __name__ == "__main__":
    main()
