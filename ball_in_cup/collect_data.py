"""
Data collection for JEPA pretraining -- step 2 of the project.

Rolls out random-action episodes on ball_in_cup-catch, rendering pixel
frames, and saves (frames, actions) per episode to disk. JEPA pretraining
is self-supervised, so it doesn't need a trained policy -- random actions
give broad, diverse coverage of the state space, which is fine (arguably
better) for this purpose than a narrow, converged policy. That's what lets
us build JEPA now without waiting on the RL baseline.

Each saved episode file contains:
    frames:  (T+1, H, W, 3) uint8   -- one more frame than actions
    actions: (T, action_dim) float32

Transition n for JEPA training is (frames[n], actions[n], frames[n+1]).

Run:
    MUJOCO_GL=egl python3 collect_data.py
"""

import pathlib

import numpy as np
from dm_control import suite

N_EPISODES = 100
STEPS_PER_EPISODE = 200  # not the full 1000-step episode -- more, shorter
                          # episodes means more diverse starting conditions
                          # (the ball's start position is re-randomized
                          # every reset) for the same total transition count
IMAGE_SIZE = 64
OUT_DIR = pathlib.Path("jepa_data")


def collect_episode(env, steps):
    action_spec = env.action_spec()
    env.reset()
    frames = [env.physics.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_id=0)]
    actions = []
    for _ in range(steps):
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
        env.step(action)
        frames.append(env.physics.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_id=0))
        actions.append(action)
    return np.array(frames, dtype=np.uint8), np.array(actions, dtype=np.float32)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    env = suite.load(domain_name="ball_in_cup", task_name="catch")

    for ep in range(N_EPISODES):
        frames, actions = collect_episode(env, STEPS_PER_EPISODE)
        np.savez_compressed(OUT_DIR / f"episode_{ep:04d}.npz", frames=frames, actions=actions)
        if (ep + 1) % 10 == 0:
            print(f"{ep + 1}/{N_EPISODES} episodes collected")

    print(f"Done. {N_EPISODES} episodes, {N_EPISODES * STEPS_PER_EPISODE} transitions total.")