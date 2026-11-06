"""
Ground-truth-state data collection for JEPA evaluation.

collect_data.py (used for JEPA pretraining itself) only saves (frames,
actions) -- fine for VICReg training, which is self-supervised and never
needs to know the true physical state. But to check whether the learned
32-dim embedding z = E(x) actually captures the physics (rather than just
satisfying the training loss in some degenerate way that still spreads out
and decorrelates), we need ground truth to compare against: a linear probe
fit from z to the true state, checked on held-out episodes.

This script rolls out the same random-action policy as collect_data.py, but
additionally reads out the true state from the environment at every frame:
    position = [cup_x, cup_z, ball_x, ball_z]        (meters)
    velocity = [cup_x_vel, cup_z_vel, ball_x_vel, ball_z_vel]   (m/s)
Confirmed directly from the task's own observation spec (ball_in_cup.py's
get_observation) and cross-checked against physics.named.data.qpos/qvel --
both expose the four slide joints in this same order: cup_x, cup_z, ball_x,
ball_z. Together (position, velocity) is the complete physical state of
this system -- nothing else moves.

Each saved episode file contains:
    frames:    (T+1, H, W, 3) uint8    -- same as collect_data.py
    actions:   (T, action_dim) float32 -- same as collect_data.py
    positions: (T+1, 4) float32        -- state aligned 1:1 with frames
    velocities:(T+1, 4) float32        -- state aligned 1:1 with frames

A much smaller N_EPISODES than collect_data.py's 100 is enough here --
fitting and checking a linear probe on 32 -> 4 (or 32 -> 8) doesn't need
anywhere near 20,000 transitions, and this data never goes near JEPA
pretraining itself.

Run (same display caveat as collect_data.py -- use MUJOCO_GL=egl only on a
genuinely headless machine):
    python3 collect_eval_data.py
"""

import pathlib

import numpy as np
from dm_control import suite

N_EPISODES = 20
STEPS_PER_EPISODE = 200   # matches collect_data.py, for comparable episodes
IMAGE_SIZE = 64
OUT_DIR = pathlib.Path("jepa_eval_data")


def collect_episode(env, steps):
    action_spec = env.action_spec()

    timestep = env.reset()
    frames = [env.physics.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_id=0)]
    positions = [timestep.observation["position"].astype(np.float32)]
    velocities = [timestep.observation["velocity"].astype(np.float32)]
    actions = []

    for _ in range(steps):
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
        timestep = env.step(action)
        frames.append(env.physics.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_id=0))
        positions.append(timestep.observation["position"].astype(np.float32))
        velocities.append(timestep.observation["velocity"].astype(np.float32))
        actions.append(action)

    return (
        np.array(frames, dtype=np.uint8),
        np.array(actions, dtype=np.float32),
        np.array(positions, dtype=np.float32),
        np.array(velocities, dtype=np.float32),
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    env = suite.load(domain_name="ball_in_cup", task_name="catch")

    for ep in range(N_EPISODES):
        frames, actions, positions, velocities = collect_episode(env, STEPS_PER_EPISODE)
        np.savez_compressed(
            OUT_DIR / f"episode_{ep:04d}.npz",
            frames=frames,
            actions=actions,
            positions=positions,
            velocities=velocities,
        )
        if (ep + 1) % 5 == 0:
            print(f"{ep + 1}/{N_EPISODES} episodes collected")

    print(f"Done. {N_EPISODES} episodes, {N_EPISODES * STEPS_PER_EPISODE} transitions total.")
    print(f"Saved to {OUT_DIR}/ -- frames, actions, positions, velocities per episode.")