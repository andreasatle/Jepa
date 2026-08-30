"""
Runs random-IC closed-loop episodes (same logic as solve_strided.py, but
IC not overridden to steady state -- matches solve_strided_batch_random.py)
until it has collected a few failures and a success, saving both a GIF
and a labeled filmstrip PNG for each -- so the actual failure mode can be
inspected visually instead of just from ball_to_target numbers.

Filmstrip: ~10 evenly spaced frames from the episode, each labeled with
its real_step, laid out left to right in one PNG -- easier to eyeball a
whole trajectory at once than scrubbing through a GIF.

Run:
    MUJOCO_GL=egl uv run python solve_strided_inspect.py
"""

import numpy as np
import torch
from dm_control import suite
from PIL import Image, ImageDraw

import cem_planner as cp
import cem_planner_strided as cps
import jepa_evaluator as je
import solve_strided as ss

FAILURES_TO_COLLECT = 8
SUCCESSES_TO_COLLECT = 4
MAX_TRIALS = 80
N_FILMSTRIP_FRAMES = 10
FRAME_SIZE = 150


def make_filmstrip(gif_frames, real_steps, out_path):
    """gif_frames[0] is the boot frame (no step number); gif_frames[i]
    for i>=1 corresponds to real_steps[i-1]. Picks N_FILMSTRIP_FRAMES
    evenly spaced frames (always including the first and last) and lays
    them out horizontally with a real_step label under each."""
    n = len(gif_frames)
    idx = sorted(set(np.linspace(0, n - 1, N_FILMSTRIP_FRAMES).round().astype(int).tolist()))
    label_h = 20
    strip = Image.new("RGB", (FRAME_SIZE * len(idx), FRAME_SIZE + label_h), "white")
    draw = ImageDraw.Draw(strip)
    for col, i in enumerate(idx):
        frame = gif_frames[i].resize((FRAME_SIZE, FRAME_SIZE))
        strip.paste(frame, (col * FRAME_SIZE, 0))
        step_label = "boot" if i == 0 else str(real_steps[i - 1])
        draw.text((col * FRAME_SIZE + 4, FRAME_SIZE + 2), step_label, fill="black")
    strip.save(out_path)


def main():
    device = torch.device("cpu")
    E, P, k_context, latent_dim = je.load_world_model(ss.CHECKPOINT_PATH, device=device)
    W = cp.load_position_probe_mlp(ss.POSITION_PROBE_PATH)
    cfg = cps.CEMConfig()

    n_failures, n_successes = 0, 0
    trial = 0

    while trial < MAX_TRIALS and (n_failures < FAILURES_TO_COLLECT or n_successes < SUCCESSES_TO_COLLECT):
        trial += 1
        env = suite.load(domain_name="ball_in_cup", task_name="catch")
        env.reset()

        def render_model():
            return env.physics.render(height=ss.IMAGE_SIZE, width=ss.IMAGE_SIZE, camera_id=0)

        def render_display():
            return env.physics.render(height=ss.RENDER_SIZE, width=ss.RENDER_SIZE, camera_id=0)

        gif_frames = [Image.fromarray(render_display())]
        real_steps = []

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

        while real_step < ss.MAX_REAL_STEPS:
            first_action, _mean, _std, info = cps.plan_step(P, window, W, cfg, warm_elites=warm_elites)
            warm_elites = cp.shift_warm_elites(info["pooled_elites"])
            action_np = first_action.detach().numpy()

            for _ in range(ss.PREDICTOR_STRIDE):
                env.step(action_np)
                real_step += 1
                gif_frames.append(Image.fromarray(render_display()))
                if real_step >= ss.MAX_REAL_STEPS:
                    break

            dist = float(np.linalg.norm(env.physics.ball_to_target()))
            in_target = bool(env.physics.in_target())
            if dist < best_dist:
                best_dist, best_dist_step = dist, real_step
            real_steps.extend([real_step] * (len(gif_frames) - 1 - len(real_steps)))

            new_z = ss.encode_frame(E, render_model(), device)
            window = torch.cat([window[:, 1:, :], new_z.unsqueeze(1)], dim=1)

            if in_target:
                solved = True
                break

        status = "SOLVED" if solved else "FAILED"
        keep = (not solved and n_failures < FAILURES_TO_COLLECT) or (solved and n_successes < SUCCESSES_TO_COLLECT)
        print(f"trial {trial:2d}: {status}  real_step={real_step}  best_dist={best_dist:.4f}m @ {best_dist_step}"
              f"  {'(kept)' if keep else '(skipped)'}")

        if keep:
            tag = f"{'solved' if solved else 'failed'}_{n_successes if solved else n_failures}"
            gif_path = f"inspect_{tag}.gif"
            strip_path = f"inspect_{tag}_filmstrip.png"
            gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=50, loop=0)
            make_filmstrip(gif_frames, real_steps, strip_path)
            print(f"  saved {gif_path} and {strip_path}")
            if solved:
                n_successes += 1
            else:
                n_failures += 1

    print(f"\nDone: {n_failures} failure(s), {n_successes} success(es) saved, {trial} trials run.")


if __name__ == "__main__":
    main()
