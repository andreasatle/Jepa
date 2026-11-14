"""
A side-by-side video of one Push-T puzzle for the homepage, with the puzzle's goal beside it (the green T the
simulator draws is the game's usual target, not the puzzle's goal, which is where the expert's T was 25 steps
later): the frozen LeVJEPA encoder's token planner (step 8)
and the image world model (../pusht, SIGReg 0.09), from the same start to the same goal, with each one's own
code; only the simulator's frames are recorded.

    uv run python make_video.py record --planner tokens --trial 0   # runs/video/tokens-0.npy
    uv run python make_video.py record --planner world --trial 0    # runs/video/world-0.npy (pusht's code)
    uv run python make_video.py compose --trial 0 --out clip.mp4

Each planner runs in its own process: pusht and video_jepa both have a model.py.
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PUSHT = os.path.join(HERE, "..", "pusht")
OUT = os.path.join(HERE, "runs", "video")


class Recorder:
    """The simulator, with every frame after a reset or a step kept (rendered at the env's resolution)."""

    def __init__(self, env):
        self.env, self.frames = env, []

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self.frames.append(self.env.render())
        return out

    def step(self, action):
        out = self.env.step(action)
        self.frames.append(self.env.render())
        return out

    def render(self):
        return self.env.render()


def record(planner: str, trial: int) -> None:
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    os.makedirs(OUT, exist_ok=True)
    if planner == "tokens":
        from data import PushTVideo
        from plan_ac import FRAMESKIP, action_stats, run_episode, starts
        from plan_tokens import TokenPlanner

        video = PushTVideo()
        meta = dict(np.load(os.path.join(PUSHT, "data", "meta.npz")))
        a = meta["action"]
        amu, asd = action_stats(video, a.astype(np.float32))
        p = TokenPlanner(amu, asd, np.tile(a.min(0), FRAMESKIP), np.tile(a.max(0), FRAMESKIP))
        env = Recorder(gym.make("swm/PushT-v1", resolution=224).unwrapped)
        s, g = starts(meta)[trial]
        ok = run_episode(env, p, video, meta, s, g, trial)
    else:
        sys.path.insert(0, PUSHT)
        os.chdir(PUSHT)
        import evaluate as ev
        from model import FRAMESKIP
        from train import Windows

        meta = dict(np.load("data/meta.npz"))
        frames = np.load("data/frames64.npy", mmap_mode="r")
        mu, sd = Windows.action_stats(meta)
        a = meta["action"]
        p = ev.Planner("runs/sigreg-s0/model.pt", mu, sd, np.tile(a.min(0), FRAMESKIP), np.tile(a.max(0), FRAMESKIP))
        env = Recorder(gym.make("swm/PushT-v1", resolution=224).unwrapped)
        s, g = ev.starts(meta)[trial]
        ok = ev.run_episode(env, "model", s, g, meta, frames, p, trial)
    np.save(os.path.join(OUT, f"{planner}-{trial}.npy"), np.stack(env.frames))
    goal = meta["state"][g].astype(np.float64)  # the puzzle's goal: the expert's state 25 steps on
    env.env.reset(seed=trial, options={"state": goal, "goal_state": goal})
    np.save(os.path.join(OUT, f"goal-{trial}.npy"), env.env.render())
    with open(os.path.join(OUT, f"{planner}-{trial}.txt"), "w") as f:
        f.write("solved" if ok else "not solved")
    print(f"{planner}, trial {trial}: {'SOLVED' if ok else 'not solved'} in {len(env.frames) - 1} steps")


def label(frame: np.ndarray, title: str, note: str, bar_height: int = 64) -> np.ndarray:
    img = cv2.resize(frame, (360, 360), interpolation=cv2.INTER_AREA)
    bar = np.full((bar_height, 360, 3), 255, np.uint8)
    cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(bar, note, (10, bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def compose(trial: int, out: str, fps: int = 10, card: bool = False) -> None:
    """The three panels (goal, video encoder, world model); `card`: the two planners only, 16:9 (732 x 412)
    for the homepage's project card."""
    left, right = (np.load(os.path.join(OUT, f"{p}-{trial}.npy")) for p in ("tokens", "world"))
    goal = label(np.load(os.path.join(OUT, f"goal-{trial}.npy")), "The goal", "the expert's T, 25 steps on")
    lend, rend = (open(os.path.join(OUT, f"{p}-{trial}.txt")).read() for p in ("tokens", "world"))
    n = max(len(left), len(right)) + fps  # hold the last frame for a second
    tmp = os.path.join(OUT, f"frames-{trial}" + ("-card" if card else ""))
    os.makedirs(tmp, exist_ok=True)
    for i in range(n):
        l, r = left[min(i, len(left) - 1)], right[min(i, len(right) - 1)]
        lnote = f"step {min(i, len(left) - 1)}" + (f"  ({lend})" if i >= len(left) - 1 else "")
        rnote = f"step {min(i, len(right) - 1)}" + (f"  ({rend})" if i >= len(right) - 1 else "")
        if card:
            gap = np.full((412, 12, 3), 255, np.uint8)
            frame = np.concatenate([label(l, "Video encoder, frozen", lnote, 52), gap,
                                    label(r, "World model, trained with actions", rnote, 52)], 1)
        else:
            gap = np.full((424, 12, 3), 255, np.uint8)
            frame = np.concatenate([goal, gap, label(l, "Video encoder, frozen (LeVJEPA)", lnote), gap,
                                    label(r, "Image world model, trained with actions", rnote)], 1)
        cv2.imwrite(os.path.join(tmp, f"{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", out], check=True)
    cv2.imwrite(out.rsplit(".", 1)[0] + "-poster.jpg", cv2.imread(os.path.join(tmp, "0000.png")))
    print(f"{out}: {n} frames at {fps} fps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["record", "compose"])
    ap.add_argument("--planner", choices=["tokens", "world"])
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT, "pusht-side-by-side.mp4"))
    ap.add_argument("--card", action="store_true", help="the two planners only, 16:9, for a project card")
    args = ap.parse_args()
    if args.command == "record":
        record(args.planner, args.trial)
    else:
        compose(args.trial, args.out, card=args.card)


if __name__ == "__main__":
    main()
