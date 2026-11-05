"""
Ground-truth-state data collection for the STRIDED model's evaluation --
same as collect_eval_data.py, except the action is held constant every
PREDICTOR_STRIDE steps (matching collect_data_strided.py's training data),
instead of a fresh random action every single step.

Needed because drift_diagnostic_strided.py has to compare the strided
model's rollout against REAL ground truth generated under the SAME
action-repeat scheme it was trained on -- the existing jepa_eval_data/
was generated under i.i.d.-random actions, a different (and for this
purpose, wrong) real trajectory distribution.

Each saved episode file: same format as collect_eval_data.py --
    frames:     (T+1, H, W, 3) uint8
    actions:    (T, action_dim) float32   -- repeated in blocks of
                                              PREDICTOR_STRIDE (see
                                              collect_data_strided.py)
    positions:  (T+1, 4) float32
    velocities: (T+1, 4) float32

Run:
    MUJOCO_GL=egl python3 collect_eval_data_strided.py
"""

import pathlib

import numpy as np
from dm_control import suite

N_EPISODES = 20
STEPS_PER_EPISODE = 200
IMAGE_SIZE = 64
PREDICTOR_STRIDE = 5      # must match collect_data_strided.py / jepa_train_strided.py
OUT_DIR = pathlib.Path("jepa_eval_data_strided")


def collect_episode(env, steps, stride):
    action_spec = env.action_spec()
    timestep = env.reset()
    frames = [env.physics.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_id=0)]
    positions = [timestep.observation["position"].astype(np.float32)]
    velocities = [timestep.observation["velocity"].astype(np.float32)]
    actions = []
    action = None
    for i in range(steps):
        if i % stride == 0:
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
        frames, actions, positions, velocities = collect_episode(env, STEPS_PER_EPISODE, PREDICTOR_STRIDE)
        np.savez_compressed(
            OUT_DIR / f"episode_{ep:04d}.npz",
            frames=frames, actions=actions, positions=positions, velocities=velocities,
        )
        if (ep + 1) % 5 == 0:
            print(f"{ep + 1}/{N_EPISODES} episodes collected")

    print(f"Done. {N_EPISODES} episodes, {N_EPISODES * STEPS_PER_EPISODE} transitions total.")
    print(f"Saved to {OUT_DIR}/ -- action held constant every {PREDICTOR_STRIDE} steps.")
