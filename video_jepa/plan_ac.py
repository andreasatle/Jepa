"""
Step 7 of PLAN.md: planning Push-T with the frozen LeVJEPA encoder and an action-conditioned predictor
(V-JEPA 2-AC's recipe at our scale), against the image world model of ../pusht (35-39 of 50).

A moment of an episode is the encoder's [cls] output for a 16-frame clip, standardised per dimension with
the training embeddings' mean and std. Two ways to make the clip (--arm):

    still   the frame repeated 16 times (V-JEPA 2-AC encodes frames one at a time)
    clip    the 16 frames 2 env steps apart ending at the moment (so the embedding carries motion); before
            an episode's first frame, its first frame repeated

The predictor is pusht's (a causal transformer over up to HISTORY = 3 moments, each plus the embedding of the
5-step action block taken after it; output i predicts moment i + 1) with a LayerNorm head instead of
BatchNorm. It is trained on frozen embeddings with teacher forcing and MSE, no regulariser: a frozen encoder
can't collapse. Planning is pusht's evaluate.py exactly: LeWM's CEM, cost the squared distance to the goal
moment's embedding, the same 50 puzzles (seed 42, start and goal 25 steps apart in held-out episodes), 50
steps allowed, success by the simulator's own test.

    uv run python plan_ac.py embed --arm still        # every 5th frame of the training episodes (~25 min)
    uv run python plan_ac.py train --arm still        # the predictor (runs/ac-still/predictor.pt)
    uv run python plan_ac.py evaluate --arm still     # 50 puzzles
"""

import argparse
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

from data import PushTVideo
from model import VideoEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
ENCODER = os.path.join(HERE, "runs", "levjepa-s0", "encoder.pt")
FRAMES, CLIP_STRIDE = 16, 2                   # the encoder's clips: 16 frames, 2 env steps apart
HISTORY, FRAMESKIP = 3, 5                     # as pusht: 3 moments of history, action blocks of 5 steps
ACTION_DIM = 2 * FRAMESKIP
N_EVAL, GOAL_OFFSET, BUDGET = 50, 25, 50      # pusht's protocol
HORIZON, SAMPLES, ITERS, TOPK = 5, 300, 30, 30


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Moments:
    """Which recorded frames make the clip of a moment, for either arm."""

    def __init__(self, video: PushTVideo, arm: str):
        self.video, self.arm = video, arm

    def rows(self, row: int) -> np.ndarray:
        if self.arm == "still":
            return np.full(FRAMES, row)
        start = self.video.ep_offset[self.video.episode_of(np.array([row]))[0]]
        return np.maximum(row - CLIP_STRIDE * np.arange(FRAMES - 1, -1, -1), start)

    @staticmethod
    def from_frames(frames: list[np.ndarray], arm: str) -> np.ndarray:
        """A live clip from the frames seen so far, one per env step, the newest last: (16, 64, 64, 3)."""
        if arm == "still":
            return np.stack([frames[-1]] * FRAMES)
        picks = [frames[max(len(frames) - 1 - CLIP_STRIDE * k, 0)] for k in range(FRAMES - 1, -1, -1)]
        return np.stack(picks)


class Encoder:
    """The frozen LeVJEPA encoder: clips (B, 16, 64, 64, 3) uint8 -> [cls] (B, 192)."""

    def __init__(self, dev: torch.device):
        ck = torch.load(ENCODER, map_location="cpu")
        self.net = VideoEncoder(**ck["config"])
        self.net.load_state_dict(ck["encoder"])
        self.net.eval().to(dev)
        self.dev = dev

    @torch.no_grad()
    def __call__(self, clips: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(clips).to(self.dev).permute(0, 1, 4, 2, 3)        # B, T, 3, H, W
        return self.net(x)[:, 0].float()


class ClipRows(Dataset):
    def __init__(self, video: PushTVideo, moments: Moments, rows: np.ndarray):
        self.video, self.moments, self.rows = video, moments, rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.video.frames[self.moments.rows(int(self.rows[i]))]))


def embed(arm: str) -> None:
    video, dev = PushTVideo(), device()
    out = os.path.join(HERE, "runs", f"ac-{arm}")
    os.makedirs(out, exist_ok=True)
    rows = np.concatenate([video.ep_offset[e] + np.arange(0, video.ep_len[e], FRAMESKIP)
                           for e in video.episodes["train"]])
    loader = DataLoader(ClipRows(video, Moments(video, arm), rows), batch_size=64, num_workers=6,
                        multiprocessing_context="fork")
    encoder, embs, t0 = Encoder(dev), [], time.time()
    for i, clips in enumerate(loader):
        embs.append(encoder(clips).cpu().numpy().astype(np.float16))
        if i % 500 == 0:
            print(f"{(i + 1) * 64} of {len(rows)} moments, {(i + 1) * 64 / (time.time() - t0):.0f}/s", flush=True)
    z = np.concatenate(embs)
    np.save(os.path.join(out, "embeddings.npy"), z)
    np.save(os.path.join(out, "rows.npy"), rows)
    print(f"done: {len(z)} moments in {(time.time() - t0) / 60:.0f} min, std per dim {z.astype(np.float32).std(0).mean():.3f}")


class Predictor(nn.Module):
    """pusht's predictor at the encoder's width, with a LayerNorm head instead of BatchNorm."""

    def __init__(self, dim: int = 192, depth: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.action_embed = nn.Sequential(nn.Linear(ACTION_DIM, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        self.pos = nn.Parameter(torch.zeros(1, HISTORY, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, 2 * dim), nn.LayerNorm(2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) moments, a: (B, T, ACTION_DIM) the block after each; output i predicts moment i + 1."""
        t = z.shape[1]
        x = z + self.action_embed(a) + self.pos[:, :t]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=z.device), diagonal=1)
        return self.head(self.norm(self.blocks(x, mask=mask, is_causal=True)))


def action_stats(video: PushTVideo, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pusht's: mean and std of the 10-d action blocks over the training episodes' rows."""
    rows = np.concatenate([video.ep_offset[e] + np.arange(video.ep_len[e]) for e in video.episodes["train"]])
    a = actions[rows]
    return np.tile(a.mean(0), FRAMESKIP), np.tile(a.std(0), FRAMESKIP)


def load_actions() -> np.ndarray:
    meta = np.load(os.path.join(HERE, "..", "pusht", "data", "meta.npz"))
    return meta["action"].astype(np.float32)


class Windows:
    """Every run of HISTORY + 1 embedded moments FRAMESKIP apart in one episode, with the blocks between."""

    def __init__(self, arm: str):
        out = os.path.join(HERE, "runs", f"ac-{arm}")
        self.video = PushTVideo()
        raw = np.load(os.path.join(out, "embeddings.npy")).astype(np.float32)
        self.mu, self.sd = raw.mean(0), raw.std(0) + 1e-6
        self.z = (raw - self.mu) / self.sd
        self.rows = np.load(os.path.join(out, "rows.npy"))
        actions = load_actions()
        amu, asd = action_stats(self.video, actions)
        self.blocks = ((np.stack([actions[r:r + FRAMESKIP].reshape(-1) if r + FRAMESKIP <= len(actions)
                                  else np.zeros(ACTION_DIM) for r in self.rows]) - amu) / asd).astype(np.float32)
        ep = self.video.episode_of(self.rows)
        same = np.ones(len(self.rows) - HISTORY, dtype=bool)
        for k in range(1, HISTORY + 1):
            same &= ep[k:len(self.rows) - HISTORY + k] == ep[:len(self.rows) - HISTORY]
        self.starts = np.nonzero(same)[0]

    def batch(self, idx: np.ndarray, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        span = idx[:, None] + np.arange(HISTORY + 1)
        return (torch.from_numpy(self.z[span]).to(dev), torch.from_numpy(self.blocks[span[:, :HISTORY]]).to(dev))


def train(arm: str, steps: int, batch_size: int = 256, lr: float = 3e-4, seed: int = 0) -> None:
    torch.manual_seed(seed)
    rng, dev = np.random.default_rng(seed), device()
    data = Windows(arm)
    model = Predictor().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warmup = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
    held = rng.choice(data.starts, 4096, replace=False)
    train_starts = np.setdiff1d(data.starts, held)
    z, _ = data.batch(held, dev)
    copy = (z[:, 1:] - z[:, :-1]).pow(2).mean().item()      # predicting "nothing changes"
    print(f"{len(data.starts)} windows; copying the last moment scores MSE {copy:.4f}", flush=True)
    t0 = time.time()
    for step in range(steps):
        z, a = data.batch(rng.choice(train_starts, batch_size), dev)
        loss = (model(z[:, :HISTORY], a) - z[:, 1:]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                zh, ah = data.batch(held, dev)
                val = (model(zh[:, :HISTORY], ah) - zh[:, 1:]).pow(2).mean().item()
            model.train()
            print(f"step {step + 1}  train {loss.item():.4f}  held-out {val:.4f}  (copy {copy:.4f})  "
                  f"{time.time() - t0:.0f}s", flush=True)
    out = os.path.join(HERE, "runs", f"ac-{arm}")
    torch.save({"predictor": model.state_dict(), "mu": data.mu, "sd": data.sd, "steps": steps,
                "held_out_mse": val, "copy_mse": copy}, os.path.join(out, "predictor.pt"))
    print(f"saved {out}/predictor.pt")


class Planner:
    """pusht's CEM planner over the frozen encoder's standardised moments."""

    def __init__(self, arm: str, amu: np.ndarray, asd: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        self.arm, self.dev = arm, device()
        ck = torch.load(os.path.join(HERE, "runs", f"ac-{arm}", "predictor.pt"), map_location="cpu",
                        weights_only=False)
        self.P = Predictor()
        self.P.load_state_dict(ck["predictor"])
        self.P.eval().to(self.dev)
        self.mu, self.sd = (torch.tensor(x, device=self.dev) for x in (ck["mu"], ck["sd"]))
        self.encoder = Encoder(self.dev)
        self.amu, self.asd = amu, asd
        self.lo, self.hi = (lo - amu) / asd, (hi - amu) / asd

    def moment(self, clip: np.ndarray) -> torch.Tensor:
        return (self.encoder(clip[None])[0] - self.mu) / self.sd

    @torch.no_grad()
    def plan(self, z_hist: list, a_hist: list, z_goal: torch.Tensor, gen: torch.Generator) -> np.ndarray:
        dev = self.dev
        lo, hi = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (self.lo, self.hi))
        mean = torch.zeros(HORIZON, ACTION_DIM, device=dev)
        std = torch.ones(HORIZON, ACTION_DIM, device=dev)
        z0 = torch.stack(z_hist)[None].expand(SAMPLES, -1, -1)
        a0 = (torch.stack(a_hist)[None].expand(SAMPLES, -1, -1) if a_hist
              else torch.zeros(SAMPLES, 0, ACTION_DIM, device=dev))
        for _ in range(ITERS):
            noise = torch.randn(SAMPLES, HORIZON, ACTION_DIM, generator=gen).to(dev)
            cand = torch.maximum(torch.minimum(mean + std * noise, hi), lo)
            z, a = z0, a0
            for h in range(HORIZON):
                a = torch.cat([a, cand[:, h:h + 1]], 1)
                z = torch.cat([z, self.P(z[:, -HISTORY:], a[:, -HISTORY:])[:, -1:]], 1)
            cost = (z[:, -1] - z_goal).pow(2).sum(-1)
            elite = cand[cost.topk(TOPK, largest=False).indices]
            mean, std = elite.mean(0), elite.std(0)
        return (mean.cpu().numpy() * self.asd + self.amu).reshape(HORIZON, FRAMESKIP, 2)


def starts(meta) -> list[tuple[int, int]]:
    """pusht's: (start row, goal row) for N_EVAL random starts in held-out episodes (seed 42)."""
    rows = np.concatenate([meta["ep_offset"][e] + np.arange(meta["ep_len"][e] - GOAL_OFFSET)
                           for e in meta["held_out_episodes"]])
    pick = np.sort(np.random.default_rng(42).choice(rows, N_EVAL, replace=False))
    return [(int(r), int(r) + GOAL_OFFSET) for r in pick]


def frame64(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)


def run_episode(env, planner: Planner, video: PushTVideo, meta, s_row: int, g_row: int, trial: int) -> bool:
    state, goal = meta["state"][s_row].astype(np.float64), meta["state"][g_row].astype(np.float64)
    env.reset(seed=trial, options={"state": state, "goal_state": goal})
    gen = torch.Generator().manual_seed(trial)
    moments = Moments(video, planner.arm)
    z_goal = planner.moment(np.array(video.frames[moments.rows(g_row)]))
    start = video.ep_offset[video.episode_of(np.array([s_row]))[0]]
    seen = [np.array(video.frames[r]) for r in range(max(start, s_row - CLIP_STRIDE * (FRAMES - 1)), s_row)]
    seen.append(frame64(env.render()))                       # the recorded past, then the simulator
    z_hist, a_hist, steps = [], [], 0
    std_block = lambda block: torch.tensor((block.reshape(-1) - planner.amu) / planner.asd,  # noqa: E731
                                           dtype=torch.float32, device=planner.dev)
    while steps < BUDGET:
        z_hist.append(planner.moment(Moments.from_frames(seen, planner.arm)))
        blocks = planner.plan(z_hist[-HISTORY:], a_hist[-(HISTORY - 1):] if len(z_hist) > 1 else [], z_goal, gen)
        for h, block in enumerate(blocks):
            for a in block:
                _, _, done, _, _ = env.step(a.astype(np.float64))
                seen.append(frame64(env.render()))
                steps += 1
                if done:
                    return True
                if steps >= BUDGET:
                    return False
            a_hist.append(std_block(block))
            if h < HORIZON - 1:                               # the moment after each block joins the history
                z_hist.append(planner.moment(Moments.from_frames(seen, planner.arm)))
    return False


def evaluate(arm: str) -> None:
    video = PushTVideo()
    meta = dict(np.load(os.path.join(HERE, "..", "pusht", "data", "meta.npz")))
    actions = meta["action"]
    amu, asd = action_stats(video, actions.astype(np.float32))
    planner = Planner(arm, amu, asd, np.tile(actions.min(0), FRAMESKIP), np.tile(actions.max(0), FRAMESKIP))
    env = gym.make("swm/PushT-v1", resolution=224).unwrapped
    wins, t0 = [], time.time()
    for trial, (s, g) in enumerate(starts(meta)):
        wins.append(run_episode(env, planner, video, meta, s, g, trial))
        print(f"trial {trial:2d}: {'SOLVED' if wins[-1] else 'not solved'}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"success rate: {sum(wins)}/{N_EVAL}  (LeVJEPA, {arm})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["embed", "train", "evaluate"])
    ap.add_argument("--arm", required=True, choices=["still", "clip"])
    ap.add_argument("--steps", type=int, default=20_000)
    args = ap.parse_args()
    if args.command == "embed":
        embed(args.arm)
    elif args.command == "train":
        train(args.arm, args.steps)
    else:
        evaluate(args.arm)


if __name__ == "__main__":
    main()
