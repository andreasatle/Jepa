"""
Push-T by hand: a window where the blue pusher follows the mouse. Push the grey T onto the green goal T,
then park the pusher in the red ring. Success is the simulator's test: pusher and T together within 20 px
of their goal positions (one distance over all four coordinates), and the T's angle within 20 degrees.

    uv run python play.py            a random start and goal from the held-out episodes (as in the evaluation)
    uv run python play.py --seed 7   another one; R in the window restarts, Q or closing the window quits

The simulator is the evaluation's (swm/PushT-v1). Its actions are relative: each step moves the pusher's
target by action * 100 px, so the mouse offset is sent as (mouse - pusher) / 100, clipped to the range the
expert's actions cover.
"""

import argparse
import os
import warnings

import gymnasium as gym
import numpy as np
import pygame

# stable_worldmodel imports OpenCV, which bundles its own copy of SDL next to pygame's; macOS then prints one
# "Class SDL... is implemented in both" line per duplicated class. Harmless here (pygame's copy draws the
# window), so the import runs with stderr silenced.
_err = os.dup(2)
os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
import stable_worldmodel  # noqa: E402, F401  (registers swm/PushT-v1)
from stable_worldmodel.envs.utils import get_mouse_pos  # noqa: E402

os.dup2(_err, 2)
warnings.filterwarnings("ignore", message=".*Casting input x to numpy array")

GOAL_OFFSET = 25   # as in evaluate.py (not imported: it loads OpenCV, whose SDL clashes with pygame's)


def show_goal(env):
    """The simulator draws only the goal T, but success also needs the pusher within 20 px of where the expert's
    pusher ended: draw that spot as a ring, and the three errors in the title bar."""
    goal, now = np.asarray(env.goal_state), np.asarray(state_of(env))
    pygame.draw.circle(env.window, (220, 40, 40), (int(goal[0]), int(goal[1])), 20, 2)
    pygame.display.update()
    angle = abs(goal[4] - now[4]) % (2 * np.pi)
    angle = min(angle, 2 * np.pi - angle)
    pygame.display.set_caption(f"pusher {np.linalg.norm(goal[:2] - now[:2]):.0f} px (<20)   "
                               f"T {np.linalg.norm(goal[2:4] - now[2:4]):.0f} px   "
                               f"angle {np.degrees(angle):.0f} deg (<20)")


def state_of(env):
    return [*env.agent.position, *env.block.position, env.block.angle % (2 * np.pi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = dict(np.load("data/meta.npz"))
    lo, hi = meta["action"].min(0), meta["action"].max(0)
    rows = np.concatenate([meta["ep_offset"][e] + np.arange(meta["ep_len"][e] - GOAL_OFFSET)
                           for e in meta["held_out_episodes"]])
    rng = np.random.default_rng(args.seed)
    env = gym.make("swm/PushT-v1", render_mode="human").unwrapped

    def reset():
        s = int(rng.choice(rows))
        g = s + GOAL_OFFSET
        env.reset(seed=args.seed, options={"state": meta["state"][s].astype(np.float64),
                                           "goal_state": meta["state"][g].astype(np.float64)})
        env.render()
        show_goal(env)
        print(f"start row {s}, goal row {g} (the expert needed {GOAL_OFFSET} steps)")
        return 0

    steps = reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                env.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                steps = reset()
        mouse = np.array(get_mouse_pos(env.window), dtype=np.float64)
        action = np.clip((mouse - np.array(env.agent.position)) / env.action_scale, lo, hi)
        _, _, done, _, _ = env.step(action)
        env.render()
        show_goal(env)
        steps += 1
        if done:
            print(f"solved in {steps} steps; R for another")
            while not any(e.type == pygame.KEYDOWN for e in pygame.event.get()):
                pygame.time.wait(50)
            steps = reset()


if __name__ == "__main__":
    main()
