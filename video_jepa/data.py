"""
Push-T video clips and the probe labels for their last frame.

Source: the Push-T data already prepared for ../pusht (LeWorldModel's expert set at 64 px): frames64.npy
(N, 64, 64, 3) uint8, memory-mapped, and meta.npz (state, episode offsets and lengths, the 5% held-out
episodes, seed 0). The split is pusht's own, so no held-out episode is ever trained on by either project.

A clip is FRAMES frames, STRIDE env steps apart (16 frames over 30 steps = 3 s at 10 Hz), all inside one
episode. It is named by the row of its last frame; the labels describe that frame:

    pusher x, y; T x, y; T angle as sin, cos           position: one frame shows it
    pusher vx, vy; T vx, vy; T angular velocity         velocity: needs several frames
    contact (1 if the pusher touches the T)             "is something happening"

Velocities are differences over one clip step (STRIDE env steps), in units per second, so they are what the
clip's own frames can show; the simulator's recorded pusher velocity (state[5:7]) is used only to check them.
Positions are in simulator units (the window is 512 wide), angles in radians.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pusht", "data")
FRAMES, STRIDE, HZ = 16, 2, 10
SPAN = (FRAMES - 1) * STRIDE                    # env steps from a clip's first frame to its last

TARGETS = ["pusher_x", "pusher_y", "t_x", "t_y", "t_sin", "t_cos",
           "pusher_vx", "pusher_vy", "t_vx", "t_vy", "t_omega", "contact"]
GROUPS = {"position": TARGETS[0:6], "velocity": TARGETS[6:11], "contact": TARGETS[11:12]}


class PushTGeometry:
    """The T and the pusher as the simulator builds them (stable_worldmodel's PushT-v1, add_tee, scale 30;
    the pusher is a circle of radius 0.375 * 40). The T's state position is its body origin; its two boxes,
    in the body frame, are the bar x in [-60, 60], y in [0, 30] and the stem x in [-15, 15], y in [30, 120]."""

    BOXES = np.array([[-60.0, 0.0, 60.0, 30.0], [-15.0, 30.0, 15.0, 120.0]])   # x0, y0, x1, y1
    PUSHER_RADIUS = 15.0
    TOLERANCE = 2.0                                                            # simulator units

    @classmethod
    def distance(cls, state: np.ndarray) -> np.ndarray:
        """Distance from the pusher's centre to the T's outline (0 inside), for rows of state."""
        d = state[..., 0:2] - state[..., 2:4]
        a = state[..., 4]
        c, s = np.cos(a), np.sin(a)
        local = np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)   # R(-a) d
        out = []
        for x0, y0, x1, y1 in cls.BOXES:
            gx = np.maximum(np.maximum(x0 - local[..., 0], 0), local[..., 0] - x1)
            gy = np.maximum(np.maximum(y0 - local[..., 1], 0), local[..., 1] - y1)
            out.append(np.hypot(gx, gy))
        return np.minimum(*out)

    @classmethod
    def contact(cls, state: np.ndarray) -> np.ndarray:
        return (cls.distance(state) <= cls.PUSHER_RADIUS + cls.TOLERANCE).astype(np.float32)


class PushTVideo:
    """The frames, the state and the episode split, loaded once and shared by every Clips."""

    def __init__(self, data_dir: str = DATA):
        self.frames = np.load(os.path.join(data_dir, "frames64.npy"), mmap_mode="r")
        meta = np.load(os.path.join(data_dir, "meta.npz"))
        self.state = meta["state"].astype(np.float64)
        self.ep_offset, self.ep_len = meta["ep_offset"], meta["ep_len"]
        self.episodes = {"train": meta["train_episodes"], "held_out": meta["held_out_episodes"]}

    def clip_ends(self, split: str) -> np.ndarray:
        """Every row that can end a clip: at least SPAN steps after its episode's first frame."""
        eps = self.episodes[split]
        return np.concatenate([self.ep_offset[e] + np.arange(SPAN, self.ep_len[e]) for e in eps])

    def episode_of(self, rows: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.ep_offset, rows, side="right") - 1

    @staticmethod
    def clip_rows(end: int) -> np.ndarray:
        return end - SPAN + STRIDE * np.arange(FRAMES)

    def labels(self, ends: np.ndarray) -> np.ndarray:
        """(len(ends), len(TARGETS)) float32, in the order of TARGETS."""
        s, p = self.state[ends], self.state[ends - STRIDE]
        dt = STRIDE / HZ
        omega = (np.angle(np.exp(1j * (s[:, 4] - p[:, 4])))) / dt              # wrapped angle difference
        out = np.column_stack([
            s[:, 0:4], np.sin(s[:, 4]), np.cos(s[:, 4]),
            (s[:, 0:2] - p[:, 0:2]) / dt, (s[:, 2:4] - p[:, 2:4]) / dt, omega,
            PushTGeometry.contact(s),
        ])
        return out.astype(np.float32)


class Clips(Dataset):
    """Yields (frames, labels): frames (FRAMES, 3, 64, 64) uint8, labels (len(TARGETS),) float32.

    split: "train" or "held_out". n: a fixed random subset of n clips (seeded), as the probes use;
    None for every clip, as pretraining uses."""

    def __init__(self, video: PushTVideo, split: str, n: int | None = None, seed: int = 0):
        self.video = video
        ends = video.clip_ends(split)
        if n is not None:
            ends = np.sort(np.random.default_rng(seed).choice(ends, size=n, replace=False))
        self.ends = ends
        self.targets = video.labels(ends)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = PushTVideo.clip_rows(int(self.ends[i]))
        x = torch.from_numpy(np.array(self.video.frames[rows])).permute(0, 3, 1, 2)   # T, C, H, W
        return x, torch.from_numpy(self.targets[i])
