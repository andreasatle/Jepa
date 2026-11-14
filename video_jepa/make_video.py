"""
A side-by-side video and GIF of one Push-T puzzle for the homepage: the frozen LeVJEPA encoder's token planner
(step 8) and the image world model (../pusht, SIGReg 0.09), from the same start to the same goal, each with its
own code; only the simulator's frames are recorded. The puzzle's goal (where the expert's T was 25 steps later)
is drawn as an outline; the green T the simulator draws, the game's usual target, is taken out. Puzzle 45: the
T travels about 160 px, the world model solves it and the token planner doesn't.

    uv run python make_video.py record --planner tokens --trial 45   # runs/video/tokens-45.npy
    uv run python make_video.py record --planner world --trial 45    # runs/video/world-45.npy (pusht's code)
    uv run python make_video.py compose --trial 45 --out runs/video/pusht-45.mp4   # and pusht-45.gif

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


def label(
    frame: np.ndarray, title: str, note: str, outline: list[np.ndarray], size: int = 320, bar_height: int = 46
) -> np.ndarray:
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    cv2.polylines(img, outline, True, OUTLINE, 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (215, 215, 215), 1)
    bar = np.full((bar_height, size, 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(bar, note, (6, bar_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


OUTLINE = (40, 40, 40)


def staged(frame: np.ndarray) -> np.ndarray:
    """The simulator's frame without its green target (the game's usual target, not the puzzle's goal)."""
    img = frame.copy()
    c = img.astype(int)
    green = ((c[..., 1] > c[..., 0] + 25) & (c[..., 1] > c[..., 2] + 25)).astype(np.uint8)
    near = cv2.dilate(green, np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    img[green.astype(bool) | (near & (c.min(-1) > 170))] = 255  # the green T, its pale edges; not the T
    return img


def goal_outline(goal: np.ndarray, box: tuple[int, int, int], size: int) -> list[np.ndarray]:
    """The puzzle's goal (the T where the expert had it 25 steps later) as a clean polygon in panel pixels:
    the T's pixels in the goal frame, cropped and enlarged as the panels are, then simplified, so the line
    is drawn smooth at the final size instead of traced from small, anti-aliased pixels."""
    c = staged(goal).astype(int)
    blue = c[..., 2] > c[..., 0] + 60
    mask = ((np.abs(c - 255).sum(-1) > 60) & ~blue).astype(np.float32)
    mask[:8], mask[-8:], mask[:, :8], mask[:, -8:] = 0, 0, 0, 0  # the frame's own border
    big = cv2.resize(crop(mask, box), (size, size), interpolation=cv2.INTER_LINEAR) > 0.5
    big = cv2.morphologyEx(big.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # joins the bars
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 200]
    return [cv2.approxPolyDP(c, 0.018 * cv2.arcLength(c, True), True) for c in contours]  # straight edges


def window(frames: list[np.ndarray], margin: int = 12) -> tuple[int, int, int]:
    """The square (top, left, side) that holds everything that moves or matters in the frames, the goal
    included: the panels show it enlarged instead of the whole, mostly empty table."""
    ink = np.any([np.abs(f.astype(int) - 255).sum(-1) > 60 for f in frames], axis=0)
    ink[:8], ink[-8:], ink[:, :8], ink[:, -8:] = False, False, False, False  # the frame's own border
    rows, cols = np.nonzero(ink)
    inner = frames[0].shape[0] - 16  # inside the simulator's border
    side = min(max(rows.max() - rows.min(), cols.max() - cols.min()) + 2 * margin, inner)
    top = int(np.clip((rows.min() + rows.max() - side) // 2, 8, 8 + inner - side))
    left = int(np.clip((cols.min() + cols.max() - side) // 2, 8, 8 + inner - side))
    return top, left, int(side)


def crop(frame: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    top, left, side = box
    return frame[top : top + side, left : left + side]


def compose(trial: int, out: str, fps: int = 10) -> None:
    """The two planners side by side, 16:9 (650 x 366), the goal outlined in both; a video, a poster and a
    GIF. Played at the simulator's 10 steps a second, the end held a second."""
    left, right = (np.load(os.path.join(OUT, f"{p}-{trial}.npy")) for p in ("tokens", "world"))
    goal = np.load(os.path.join(OUT, f"goal-{trial}.npy"))
    lend, rend = (open(os.path.join(OUT, f"{p}-{trial}.txt")).read() for p in ("tokens", "world"))
    n = max(len(left), len(right)) + fps
    tmp = os.path.join(OUT, f"frames-{trial}")
    os.makedirs(tmp, exist_ok=True)
    for old in os.listdir(tmp):
        os.remove(os.path.join(tmp, old))
    gap = np.full((366, 10, 3), 255, np.uint8)
    box = window([staged(f) for f in (*left, *right, goal)])
    outline = goal_outline(goal, box, 320)
    for i in range(n):
        l, r = left[min(i, len(left) - 1)], right[min(i, len(right) - 1)]
        lnote = f"step {min(i, len(left) - 1)}" + (f"  ({lend})" if i >= len(left) - 1 else "")
        rnote = f"step {min(i, len(right) - 1)}" + (f"  ({rend})" if i >= len(right) - 1 else "")
        frame = np.concatenate([label(crop(staged(l), box), "Video encoder, frozen", lnote, outline), gap,
                                label(crop(staged(r), box), "World model, trained with actions", rnote, outline)], 1)
        cv2.imwrite(os.path.join(tmp, f"{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", out], check=True)
    cv2.imwrite(out.rsplit(".", 1)[0] + "-poster.jpg", cv2.imread(os.path.join(tmp, "0000.png")))
    gif(tmp, out.rsplit(".", 1)[0] + ".gif", fps)
    print(f"{out}: {n} frames at {fps} fps, and a GIF")


def gif(frames: str, out: str, fps: int, colours: int = 8) -> None:
    """A looping GIF at half the frame rate, from the lossless frames, with a fixed palette of the frames'
    most frequent colours plus the pusher's blue (a generated palette merged the small blue dot into grey),
    and each frame storing only what changed."""
    files = sorted(os.listdir(frames))[::3]
    px = np.concatenate([cv2.cvtColor(cv2.imread(os.path.join(frames, f)), cv2.COLOR_BGR2RGB).reshape(-1, 3)
                         for f in files])
    cols, counts = np.unique(px, axis=0, return_counts=True)
    cols = cols[np.argsort(-counts)]
    blue = cols[(cols[:, 2].astype(int) - cols[:, 0].astype(int)) > 80][0]
    keep = [*cols[:colours - 2], blue, cols[np.argmin(cols.sum(1))]]  # plus the darkest, for the text
    palette = np.array(keep + [keep[0]] * (256 - len(keep)), dtype=np.uint8).reshape(16, 16, 3)
    pal = os.path.join(OUT, "palette.png")
    cv2.imwrite(pal, cv2.cvtColor(palette, cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frames, "%04d.png"),
                    "-i", pal, "-lavfi", f"fps={fps}[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle",
                    "-loop", "0", out], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["record", "compose"])
    ap.add_argument("--planner", choices=["tokens", "world"])
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT, "pusht-side-by-side.mp4"))
    args = ap.parse_args()
    if args.command == "record":
        record(args.planner, args.trial)
    else:
        compose(args.trial, args.out)


if __name__ == "__main__":
    main()
