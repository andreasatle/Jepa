"""
Minimal install-and-run demo for dm_control's ball_in_cup-catch task.

Install:
    pip install dm_control

Two modes:

1. Headless GIF mode (works anywhere, no display needed -- e.g. servers,
   containers, this sandbox). Renders a rollout and saves it as a GIF.

       MUJOCO_GL=egl python3 ball_in_cup/ball_in_cup_demo.py

2. Interactive viewer mode (needs a real local display -- your own
   desktop/laptop, not a headless server/container). Pops up a live window
   you can drag objects in with the mouse and watch the physics run.
   Do NOT set MUJOCO_GL=egl for this mode -- the viewer needs a real
   on-screen rendering backend (GLFW), not the offscreen one.

       python3 ball_in_cup/ball_in_cup_demo.py --viewer

3. Teleop mode -- same as the viewer, but the arrow keys actually drive the
   cup's real actuators (the same 2D action space a trained policy will
   use), instead of watching a random policy flail. Needs `pynput`:

       uv add pynput
       python3 ball_in_cup/ball_in_cup_demo.py --teleop

   On macOS you'll likely need to grant your terminal (or whatever runs
   the process) Accessibility permission the first time -- System Settings
   -> Privacy & Security -> Accessibility -- otherwise pynput can't see
   keypresses.
"""

import argparse

import numpy as np
from dm_control import suite
from PIL import Image


def make_env():
    return suite.load(domain_name="ball_in_cup", task_name="catch")


def random_policy(time_step, action_spec):
    del time_step  # unused -- stand-in for your policy/planner later
    return np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)


def run_headless_gif(n_steps=60, out_path="ball_in_cup_rollout.gif"):
    env = make_env()
    action_spec = env.action_spec()
    print("Action spec:", action_spec.shape, action_spec.minimum, action_spec.maximum)

    time_step = env.reset()
    print("Observation keys:", list(time_step.observation.keys()))
    for k, v in time_step.observation.items():
        print(f"  {k}: shape {np.array(v).shape}")

    frames = []
    for i in range(n_steps):
        action = random_policy(time_step, action_spec)
        time_step = env.step(action)

        frame = env.physics.render(height=200, width=200, camera_id=0)
        frames.append(Image.fromarray(frame))

        if time_step.reward is not None and time_step.reward > 0:
            print(f"step {i}: reward={time_step.reward}")

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,  # ms per frame
        loop=0,
    )
    print(f"Saved {len(frames)}-frame GIF to {out_path}")


def run_viewer():
    # Imported lazily -- pulls in GLFW, which needs a real display, so we
    # don't want this at module load time in headless mode.
    from dm_control import viewer

    env = make_env()
    action_spec = env.action_spec()

    def policy(time_step):
        return random_policy(time_step, action_spec)

    # Opens a live window. Close it (or Ctrl+C in the terminal) to exit.
    # Swap `policy` for your trained policy/planner once you have one --
    # it's called once per simulation step and just needs to return an
    # action array of the right shape.
    viewer.launch(env, policy=policy)


def run_teleop():
    # Confirmed via env.physics.model.actuator(i).name: this task has two
    # actuators, named 'x' and 'z' -- action[0] drives the cup horizontally,
    # action[1] drives it vertically.
    #
    # NOTE: letter keys (WASD included) are a bad choice here -- every
    # letter A-Z is already bound in dm_control's viewer to a built-in
    # rendering/visualization toggle (confirmed via mujoco.mjVISSTRING /
    # mjRNDSTRING), e.g. D = toggle static-body rendering (hides the ground
    # plane), W = toggle wireframe, A = toggle auto-connect, S = toggle
    # shadows. That's why W/A/S/D visibly messed with the rendering instead
    # of moving the cup. Using arrow keys instead: Up/Down/Left are fully
    # unclaimed. Right arrow has one existing binding (single-step-advance,
    # meant for frame-by-frame stepping while paused) -- shouldn't matter
    # during normal play, but flagging it since I can't test it end-to-end
    # without a real display+keyboard here.
    from dm_control import viewer
    from pynput import keyboard

    pressed = set()

    def on_press(key):
        pressed.add(key)

    def on_release(key):
        pressed.discard(key)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    env = make_env()
    action_spec = env.action_spec()
    magnitude = action_spec.maximum  # full-strength bang-bang control

    def policy(time_step):
        del time_step
        action = np.zeros(action_spec.shape)
        if keyboard.Key.right in pressed:
            action[0] += magnitude[0]
        if keyboard.Key.left in pressed:
            action[0] -= magnitude[0]
        if keyboard.Key.up in pressed:
            action[1] += magnitude[1]
        if keyboard.Key.down in pressed:
            action[1] -= magnitude[1]
        return np.clip(action, action_spec.minimum, action_spec.maximum)

    print("Teleop controls: Up/Down = cup up/down, Left/Right = cup left/right")
    print("(viewer controls -- Space to pause, Backspace to restart -- still work)")
    try:
        viewer.launch(env, policy=policy)
    finally:
        listener.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the interactive viewer with a random policy. Requires a real display.",
    )
    group.add_argument(
        "--teleop",
        action="store_true",
        help="Launch the interactive viewer under WASD keyboard control. Requires a real display and `pynput`.",
    )
    args = parser.parse_args()

    if args.teleop:
        run_teleop()
    elif args.viewer:
        run_viewer()
    else:
        run_headless_gif()