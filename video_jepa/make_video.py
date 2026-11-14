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
    """The simulator, with every frame after a reset or a step kept (rendered at the env's resolution), and
    the exact state behind it: the pusher's centre and the T's 8 corners, in the simulator's 512-pixel frame,
    from which the panels are drawn smooth at any size."""

    def __init__(self, env):
        self.env, self.frames, self.states = env, [], []

    def keep(self) -> None:
        self.frames.append(self.env.render())
        agent, block = self.env.agent, self.env.block
        corners = [c for v in TEE for c in block.local_to_world(v)]  # exact, as the simulator places them
        self.states.append([*agent.position, *corners])

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self.keep()
        return out

    def step(self, action):
        out = self.env.step(action)
        self.keep()
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
    np.save(os.path.join(OUT, f"{planner}-{trial}-states.npy"), np.array(env.states))
    radius = next(s.radius for s in env.env.agent.shapes)
    np.save(os.path.join(OUT, "pusher-radius.npy"), np.array(radius))
    goal = meta["state"][g].astype(np.float64)  # the puzzle's goal: the expert's state 25 steps on
    env.env.reset(seed=trial, options={"state": goal, "goal_state": goal})
    np.save(os.path.join(OUT, f"goal-{trial}.npy"), env.env.render())
    with open(os.path.join(OUT, f"{planner}-{trial}.txt"), "w") as f:
        f.write("solved" if ok else "not solved")
    print(f"{planner}, trial {trial}: {'SOLVED' if ok else 'not solved'} in {len(env.frames) - 1} steps")


TEE = [(-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30)]  # the T's outline
# in its body's frame: the simulator's two boxes (add_tee, scale 30, length 4) as one shape


def goal_tee(trial: int) -> None:
    """The goal T's outline in the simulator's 512-pixel frame: its exact corners, put where the simulator
    puts them (the block's body at the goal state, local_to_world), so the outline is drawn straight."""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from plan_ac import starts

    meta = dict(np.load(os.path.join(PUSHT, "data", "meta.npz")))
    _, g = starts(meta)[trial]
    goal = meta["state"][g].astype(np.float64)
    env = gym.make("swm/PushT-v1", resolution=224).unwrapped
    env.reset(seed=trial, options={"state": goal, "goal_state": goal})
    corners = np.array([tuple(env.block.local_to_world(v)) for v in TEE])
    np.save(os.path.join(OUT, f"goal-tee-{trial}.npy"), corners)
    print(f"goal T of trial {trial}: {corners.round(1).tolist()}")


def label(
    frame: np.ndarray, title: str, note: str, outline: list[np.ndarray], size: int = 320, bar_height: int = 46
) -> np.ndarray:
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    cv2.polylines(img, outline, True, OUTLINE, 2, cv2.LINE_AA, shift=4)
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


def goal_outline(corners: np.ndarray, box: tuple[int, int, int], render: int, size: int) -> list[np.ndarray]:
    """The goal T's corners (simulator pixels, 512 a side) in panel pixels: scaled to the rendered frame,
    cropped as the panels are, enlarged to the panel's size."""
    top, left, side = box
    points = (corners * render / 512 - [left, top]) * size / side
    return [np.round(points * 16).astype(np.int32).reshape(-1, 1, 2)]  # 4 bits of subpixel precision


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
    outline = goal_outline(np.load(os.path.join(OUT, f"goal-tee-{trial}.npy")), box, goal.shape[0], 320)
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


def svg(trial: int, out: str, fps: int = 10, hold: float = 1.5) -> None:
    """The two planners side by side as an animated SVG (SMIL, no script, so it runs in an <img>): the T and
    the pusher drawn from each step's exact state, moving smoothly between steps, the goal T outlined; each
    panel shows its result at the end, held `hold` seconds, then it loops. A few kilobytes, sharp at any size."""
    runs = {p: np.load(os.path.join(OUT, f"{p}-{trial}-states.npy")) for p in ("tokens", "world")}
    ends = {p: open(os.path.join(OUT, f"{p}-{trial}.txt")).read() for p in runs}
    goal = np.load(os.path.join(OUT, f"goal-tee-{trial}.npy"))
    radius = float(np.load(os.path.join(OUT, "pusher-radius.npy")))
    steps = max(len(v) for v in runs.values())
    total = (steps - 1) / fps + hold

    # the square of the 512-pixel table that holds everything, the goal included
    xs = np.concatenate([goal[:, 0], *[v[:, 2::2].ravel() for v in runs.values()],
                         *[v[:, 0] + d for v in runs.values() for d in (-radius, radius)]])
    ys = np.concatenate([goal[:, 1], *[v[:, 3::2].ravel() for v in runs.values()],
                         *[v[:, 1] + d for v in runs.values() for d in (-radius, radius)]])
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) + 40
    x0, y0 = (xs.min() + xs.max() - side) / 2, (ys.min() + ys.max() - side) / 2

    size, bar, gap = 320, 46, 10
    times = [min(i / fps, (steps - 1) / fps) / total for i in range(steps)] + [1.0]
    key_times = ";".join(f"{t:.4f}" for t in times)
    titles = {"tokens": "Video encoder, frozen", "world": "World model, trained with actions"}

    def pts(row: np.ndarray) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in row.reshape(-1, 2))

    panels = []
    for k, (name, states) in enumerate(runs.items()):
        padded = np.concatenate([states, np.repeat(states[-1:], steps + 1 - len(states), 0)])
        t_values = ";".join(pts(r[2:]) for r in padded)
        cx = ";".join(f"{r[0]:.1f}" for r in padded)
        cy = ";".join(f"{r[1]:.1f}" for r in padded)
        done = (len(states) - 1) / fps / total
        n = len(states) - 1
        result = f"solved in {n} steps" if ends[name] == "solved" else f"not solved in {n} steps"
        colour = "#2f7d4f" if ends[name] == "solved" else "#a33a3a"
        scale = size / side
        panels.append(f"""
  <g transform="translate({k * (size + gap)},0)">
    <text x="6" y="18" class="title">{titles[name]}</text>
    <text x="6" y="38" class="note" fill="{colour}" opacity="0">{result}<animate attributeName="opacity"
      values="0;1;1" keyTimes="0;{done:.4f};1" dur="{total:.2f}s" calcMode="discrete" repeatCount="indefinite"/></text>
    <svg x="0" y="{bar}" width="{size}" height="{size}" viewBox="{x0:.1f} {y0:.1f} {side:.1f} {side:.1f}">
      <rect x="{x0:.1f}" y="{y0:.1f}" width="{side:.1f}" height="{side:.1f}" fill="#fff"/>
      <polygon points="{pts(padded[0][2:])}" fill="#8fa3b8" stroke="#6f8399" stroke-width="{1.2 / scale:.2f}"
        stroke-linejoin="round"><animate attributeName="points" values="{t_values}" keyTimes="{key_times}"
        dur="{total:.2f}s" repeatCount="indefinite"/></polygon>
      <circle cx="{padded[0][0]:.1f}" cy="{padded[0][1]:.1f}" r="{radius:.1f}" fill="#4e7eff">
        <animate attributeName="cx" values="{cx}" keyTimes="{key_times}" dur="{total:.2f}s" repeatCount="indefinite"/>
        <animate attributeName="cy" values="{cy}" keyTimes="{key_times}" dur="{total:.2f}s" repeatCount="indefinite"/>
      </circle>
      <polygon points="{pts(goal)}" fill="none" stroke="#222" stroke-width="{2 / scale:.2f}" stroke-linejoin="miter"/>
    </svg>
    <rect x="0.5" y="{bar + 0.5}" width="{size - 1}" height="{size - 1}" fill="none" stroke="#d7d7d7"/>
  </g>""")
    width, height = 2 * size + gap, bar + size
    doc = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <style>.title{{font:14px system-ui,-apple-system,sans-serif;fill:#282828}}.note{{font:12px system-ui,-apple-system,sans-serif}}</style>
  <rect width="{width}" height="{height}" fill="#fff"/>{"".join(panels)}
</svg>
"""
    with open(out, "w") as f:
        f.write(doc)
    print(f"{out}: {len(doc) / 1024:.1f} KB, {total:.1f} s a loop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["record", "goal", "compose", "svg"])
    ap.add_argument("--planner", choices=["tokens", "world"])
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT, "pusht-side-by-side.mp4"))
    args = ap.parse_args()
    if args.command == "record":
        record(args.planner, args.trial)
    elif args.command == "goal":
        goal_tee(args.trial)
    elif args.command == "svg":
        svg(args.trial, args.out.rsplit(".", 1)[0] + ".svg")
    else:
        compose(args.trial, args.out)


if __name__ == "__main__":
    main()
