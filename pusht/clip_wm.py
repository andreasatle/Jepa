"""
The image world model's recipe with short clips as input (video_jepa follow-up 3, 2026-11-14): does seeing
motion help once the encoder is trained together with the action-conditioned predictor?

Everything is train.py's and evaluate.py's (SIGReg at 0.09, the predictor, windows of 4 moments 5 steps apart,
30,000 steps at batch 128, precise BatchNorm, LeWM's CEM, the same 50 puzzles) except the encoder's input: a
moment is the last CLIP = 4 frames, 2 env steps apart (0.6 s, enough to see the pusher's and the T's motion),
stacked as 12 channels into the same CNN (the first layer widened, nothing else). Before an episode's first
frame, its first frame is repeated. So the only difference from runs/sigreg-s0 is that the encoder sees
motion.

    uv run python clip_wm.py train --seed 0          -> runs/clip-sigreg-s0/{model.pt, log.csv}
    uv run python clip_wm.py evaluate runs/clip-sigreg-s0/model.pt
"""

import argparse
import csv
import math
import os
import time

import cv2
import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401  (registers swm/PushT-v1)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import evaluate as ev
from model import FRAMESKIP, HISTORY, WEIGHTS, Encoder, Predictor, make_reg
from train import Windows

CLIP, STRIDE = 4, 2  # frames per moment, env steps between them


class ClipEncoder(Encoder):
    """train.py's CNN with CLIP frames stacked as channels: (N, CLIP, 3, 64, 64) -> (N, D)."""

    def __init__(self):
        super().__init__()
        self.net[0] = nn.Conv2d(3 * CLIP, 32, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.flatten(1, 2))


class ClipWindows(Windows):
    """train.py's windows, each moment its clip: frames (HISTORY + 1, CLIP, 64, 64, 3), actions as before."""

    def __init__(self, episodes_key: str = "train_episodes"):
        super().__init__(episodes_key)
        meta = np.load("data/meta.npz")
        self.offsets = meta["ep_offset"]

    def clip_rows(self, row: int) -> np.ndarray:
        start = self.offsets[np.searchsorted(self.offsets, row, side="right") - 1]
        return np.maximum(row - STRIDE * np.arange(CLIP - 1, -1, -1), start)

    def __getitem__(self, i):
        s = self.starts[i]
        idx = s + FRAMESKIP * np.arange(HISTORY + 1)
        clips = np.stack([np.stack([self.frames[j] for j in self.clip_rows(int(r))]) for r in idx])
        acts = np.stack([self.action[j:j + FRAMESKIP].reshape(-1) for j in idx[:HISTORY]])
        return torch.from_numpy(clips), torch.from_numpy(acts)


def to_input(clips: torch.Tensor, device) -> torch.Tensor:
    """(..., CLIP, 64, 64, 3) uint8 -> (..., CLIP, 3, 64, 64) float in [0, 1] on device."""
    return clips.to(device).movedim(-1, -3).float().div(255)


def encode_windows(E, clips):
    b, t = clips.shape[:2]
    return E(clips.reshape(b * t, *clips.shape[2:])).reshape(b, t, -1)


@torch.no_grad()
def recalibrate_bn(E, P, loader, device, n_batches=200):
    """train.recalibrate_bn for clips."""
    bns = [m for m in list(E.modules()) + list(P.modules()) if isinstance(m, nn.BatchNorm1d)]
    E.eval(), P.eval()
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    for i, (clips, acts) in enumerate(loader):
        if i == n_batches:
            break
        z = encode_windows(E, to_input(clips, device))
        P(z[:, :HISTORY], acts.to(device))
    for bn in bns:
        bn.momentum = 0.1
    E.eval(), P.eval()


def train(seed: int, steps: int = 30_000, batch_size: int = 128, lr: float = 3e-4) -> None:
    out = f"runs/clip-sigreg-s{seed}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ClipWindows(), batch_size=batch_size, shuffle=True, drop_last=True, generator=g,
                        num_workers=6, persistent_workers=True, multiprocessing_context="fork")
    it = iter(loader)  # fork the workers before MPS is touched
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    E, P, reg = ClipEncoder().to(device), Predictor().to(device), make_reg("sigreg").to(device)
    weight = WEIGHTS["sigreg"]
    params = list(E.parameters()) + list(P.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    warmup = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
    log = csv.writer(open(f"{out}/log.csv", "w", newline=""))
    log.writerow(["step", "loss", "mse", "reg", "z_std", "seconds"])
    t0, acc = time.time(), []
    E.train(), P.train()
    for step in range(steps):
        try:
            clips, acts = next(it)
        except StopIteration:
            it = iter(loader)
            clips, acts = next(it)
        z = encode_windows(E, to_input(clips, device))
        mse = (P(z[:, :HISTORY], acts.to(device)) - z[:, 1:]).pow(2).mean()
        r = reg(z.transpose(0, 1))
        loss = mse + weight * r
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        acc.append((loss.item(), mse.item(), r.item()))
        if (step + 1) % 200 == 0:
            m = np.mean(acc, 0)
            acc = []
            z_std = z.detach().reshape(-1, z.shape[-1]).std(0).mean().item()
            log.writerow([step + 1, *np.round(m, 5), round(z_std, 4), round(time.time() - t0)])
            print(f"step {step + 1}  loss {m[0]:.4f}  mse {m[1]:.4f}  sigreg {m[2]:.4f}  z_std {z_std:.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    recalibrate_bn(E, P, loader, device)
    torch.save({"E": E.state_dict(), "P": P.state_dict(), "clip": CLIP, "stride": STRIDE}, f"{out}/model.pt")
    print(f"saved {out}/model.pt")


class ClipPlanner(ev.Planner):
    """evaluate.py's planner (its CEM and cost), encoding clips."""

    def __init__(self, ckpt: str, mu, sd, lo, hi):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        ck = torch.load(ckpt, map_location="cpu")
        self.E, self.P = ClipEncoder(), Predictor()
        self.E.load_state_dict(ck["E"])
        self.P.load_state_dict(ck["P"])
        self.E.to(self.device).eval()
        self.P.to(self.device).eval()
        self.mu, self.sd = mu, sd
        self.lo, self.hi = (lo - mu) / sd, (hi - mu) / sd

    @torch.no_grad()
    def encode(self, clip: np.ndarray) -> torch.Tensor:
        return self.E(to_input(torch.from_numpy(np.ascontiguousarray(clip))[None], self.device))[0]


def frame64(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)


def live_clip(seen: list[np.ndarray]) -> np.ndarray:
    return np.stack([seen[max(len(seen) - 1 - STRIDE * k, 0)] for k in range(CLIP - 1, -1, -1)])


def run_episode(env, s_row, g_row, meta, frames, planner: ClipPlanner, windows: ClipWindows, trial) -> bool:
    """evaluate.run_episode for the model, each moment a clip: the recorded past before the start, then the
    simulator's frames."""
    state, goal = meta["state"][s_row].astype(np.float64), meta["state"][g_row].astype(np.float64)
    env.reset(seed=trial, options={"state": state, "goal_state": goal})
    gen = torch.Generator().manual_seed(trial)
    z_goal = planner.encode(np.stack([frames[r] for r in windows.clip_rows(g_row)]))
    start = int(windows.clip_rows(s_row)[0])
    seen = [np.array(frames[r]) for r in range(start, s_row)] + [frame64(env.render())]
    z_hist, a_hist, steps = [], [], 0
    while steps < ev.BUDGET:
        z_hist.append(planner.encode(live_clip(seen)))
        blocks = planner.plan(z_hist[-HISTORY:], a_hist[-(HISTORY - 1):] if len(z_hist) > 1 else [], z_goal, gen)
        for h, block in enumerate(blocks):
            for a in block:
                _, _, done, _, _ = env.step(a.astype(np.float64))
                seen.append(frame64(env.render()))
                steps += 1
                if done:
                    return True
                if steps >= ev.BUDGET:
                    return False
            if h < ev.HORIZON - 1:
                a_hist.append(torch.tensor((block.reshape(-1) - planner.mu) / planner.sd, dtype=torch.float32,
                                           device=planner.device))
                z_hist.append(planner.encode(live_clip(seen)))
        a_hist.append(torch.tensor((blocks[-1].reshape(-1) - planner.mu) / planner.sd, dtype=torch.float32,
                                   device=planner.device))
    return False


def evaluate(ckpt: str) -> None:
    meta = dict(np.load("data/meta.npz"))
    frames = np.load("data/frames64.npy", mmap_mode="r")
    mu, sd = Windows.action_stats(meta)
    a = meta["action"]
    planner = ClipPlanner(ckpt, mu, sd, np.tile(a.min(0), FRAMESKIP), np.tile(a.max(0), FRAMESKIP))
    windows = ClipWindows()
    env = gym.make("swm/PushT-v1", resolution=224).unwrapped
    wins = []
    for trial, (s, g) in enumerate(ev.starts(meta)):
        ok = run_episode(env, s, g, meta, frames, planner, windows, trial)
        wins.append(ok)
        print(f"trial {trial:2d}: {'SOLVED' if ok else 'not solved'}", flush=True)
    print(f"success rate: {sum(wins)}/{ev.N_EVAL}  ({ckpt})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "evaluate"])
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30_000)
    args = ap.parse_args()
    if args.command == "train":
        train(args.seed, args.steps)
    else:
        evaluate(args.ckpt)


if __name__ == "__main__":
    main()
