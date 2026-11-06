"""
Same as solve_strided_batch.py but starts each episode from the base
task's own randomized initial condition (env.reset(), no steady-state
override) instead of the verified steady-state hang -- tests whether the
9/10 result generalizes beyond the one IC it was tuned/tested against.

ball_in_cup-catch's own initializer randomizes ball position (roughly
ball_x ~ U(-.2,.2), ball_z ~ U(.2,.5), verified from source earlier in
this project); cup starts however the task's initialize_episode sets it.

Run:
    MUJOCO_GL=egl uv run python solve_strided_batch_random.py
"""

import numpy as np
import torch
from dm_control import suite

import cem_planner as cp
import cem_planner_strided as cps
import jepa_evaluator as je
import jepa_train as jt
import solve_strided as ss
import sys

N_TRIALS = 30


def main():
    device = torch.device("cpu")
    E, P, k_context, latent_dim = je.load_world_model(ss.CHECKPOINT_PATH, device=device)
    probe_path = sys.argv[1] if len(sys.argv) > 1 else ss.POSITION_PROBE_PATH
    is_mlp = probe_path.endswith(".pt")
    W = cp.load_position_probe_mlp(probe_path) if is_mlp else cp.load_position_probe(probe_path)
    print(f"Using probe: {probe_path} ({'MLP' if is_mlp else 'linear'})")
    cfg = cps.CEMConfig()

    results = []
    for trial in range(1, N_TRIALS + 1):
        env = suite.load(domain_name="ball_in_cup", task_name="catch")
        env.reset()  # task's own randomized IC -- no steady-state override

        def render_model():
            return env.physics.render(height=ss.IMAGE_SIZE, width=ss.IMAGE_SIZE, camera_id=0)

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

        while real_step < ss.MAX_REAL_STEPS:
            first_action, _mean, _std, info = cps.plan_step(P, window, W, cfg, warm_elites=warm_elites)
            warm_elites = cp.shift_warm_elites(info["pooled_elites"])
            action_np = first_action.detach().numpy()

            for _ in range(ss.PREDICTOR_STRIDE):
                env.step(action_np)
                real_step += 1
                if real_step >= ss.MAX_REAL_STEPS:
                    break

            dist = float(np.linalg.norm(env.physics.ball_to_target()))
            in_target = bool(env.physics.in_target())
            if dist < best_dist:
                best_dist, best_dist_step = dist, real_step

            new_z = ss.encode_frame(E, render_model(), device)
            window = torch.cat([window[:, 1:, :], new_z.unsqueeze(1)], dim=1)

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
