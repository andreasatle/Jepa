"""
Collect additional i.i.d. random-action episodes for the multi-stride /
joint training data pool, appending to jepa_data/ instead of overwriting
it -- continues the episode numbering from whatever is already there.

Same collection logic as ball_in_cup/collect_data.py (i.i.d. uniform
random actions within the action bounds, 200-step episodes, 64x64
renders), kept identical so the new episodes are drawn from the same
distribution as the existing 100 and mix into the same training set
with no format changes.

Run:
    uv run python collect_data_more.py
"""

import pathlib

import numpy as np
from dm_control import suite

N_NEW_EPISODES = 400  # -> 500 total once combined with the existing 100
STEPS_PER_EPISODE = 200
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
    existing = sorted(OUT_DIR.glob("episode_*.npz"))
    start_idx = len(existing)  # continue numbering, don't overwrite
    print(f"{start_idx} existing episodes found; collecting {N_NEW_EPISODES} more "
          f"(episode_{start_idx:04d} .. episode_{start_idx + N_NEW_EPISODES - 1:04d}).")

    env = suite.load(domain_name="ball_in_cup", task_name="catch")

    for i in range(N_NEW_EPISODES):
        ep = start_idx + i
        frames, actions = collect_episode(env, STEPS_PER_EPISODE)
        np.savez_compressed(OUT_DIR / f"episode_{ep:04d}.npz", frames=frames, actions=actions)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{N_NEW_EPISODES} new episodes collected")

    total = start_idx + N_NEW_EPISODES
    print(f"Done. {total} total episodes in {OUT_DIR}/ ({N_NEW_EPISODES} new).")
