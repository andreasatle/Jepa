"""
Data collection for the PREDICTOR_STRIDE variant of P -- same as
collect_data.py, except the random action is held constant for
PREDICTOR_STRIDE consecutive control-steps before a new one is drawn
(action-repeat), instead of a fresh random action every single step.

Why this needs to exist at all: cem_planner.py's planned "one P call spans
PREDICTOR_STRIDE real steps" idea only makes physical sense if the training
data actually held one action constant over that span -- collect_data.py's
existing episodes draw a new i.i.d. random action every single 0.02s step,
so "z at t+5 given the action at t" would be asking P to predict an
outcome that depended on 4 other, hidden, random actions too. This script
produces data where that assumption is actually true.

Why a separate script/output dir rather than a flag on collect_data.py:
this is a genuinely different action-sampling distribution, not a superset
of it -- mixing it into jepa_data/ would let a training script silently
combine examples with inconsistent "what does holding this action produce"
assumptions. Kept as its own directory (jepa_data_strided/) so the two
can't get mixed by accident.

This ONLY produces the raw (frames, actions) episodes -- it does not build
PREDICTOR_STRIDE-spaced training examples for P (that's a jepa_train.py
change, not yet made). Frames/actions are still saved at the full 0.02s
control-step resolution, same as collect_data.py -- only how often a NEW
random action gets drawn changes -- so this data can still support
fine-grained K_CONTEXT windowing if that's how the stride-P's context ends
up being built.

Each saved episode file: same format as collect_data.py --
    frames:  (T+1, H, W, 3) uint8
    actions: (T, action_dim) float32   -- repeated in blocks of
                                           PREDICTOR_STRIDE, not i.i.d.
                                           per step (see collect_episode)

Run:
    MUJOCO_GL=egl python3 collect_data_strided.py
"""

import pathlib

import numpy as np
from dm_control import suite

N_EPISODES = 100
STEPS_PER_EPISODE = 200
IMAGE_SIZE = 64
PREDICTOR_STRIDE = 5      # action held constant for this many consecutive
                          # control-steps before a new one is drawn -- see
                          # module docstring. Keep in sync with whatever
                          # jepa_train.py/cem_planner.py end up using once
                          # the stride-P training-loop change is made.
OUT_DIR = pathlib.Path("jepa_data_strided")


def collect_episode(env, steps, stride):
    action_spec = env.action_spec()
    env.reset()
    frames = [env.physics.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_id=0)]
    actions = []
    action = None
    for i in range(steps):
        if i % stride == 0:
            # Draw a new random action only every `stride` steps -- held
            # constant (repeated) for the steps in between, unlike
            # collect_data.py which draws a fresh one every single step.
            action = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
        env.step(action)
        frames.append(env.physics.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_id=0))
        actions.append(action)
    return np.array(frames, dtype=np.uint8), np.array(actions, dtype=np.float32)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    env = suite.load(domain_name="ball_in_cup", task_name="catch")

    for ep in range(N_EPISODES):
        frames, actions = collect_episode(env, STEPS_PER_EPISODE, PREDICTOR_STRIDE)
        np.savez_compressed(OUT_DIR / f"episode_{ep:04d}.npz", frames=frames, actions=actions)
        if (ep + 1) % 10 == 0:
            print(f"{ep + 1}/{N_EPISODES} episodes collected")

    print(
        f"Done. {N_EPISODES} episodes, {N_EPISODES * STEPS_PER_EPISODE} transitions total, "
        f"action held constant every {PREDICTOR_STRIDE} steps."
    )
