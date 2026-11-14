"""
Step 8 of PLAN.md: planning Push-T over the frozen LeVJEPA encoder's patch tokens instead of its [cls]
(V-JEPA 2-AC plans over tokens; step 7's [cls] solved 4-5 of 50, the image world model 35).

A moment is the last frame's 8 x 8 patch tokens of the encoder's output for the still clip (the frame repeated
16 times, as V-JEPA 2-AC encodes frames one at a time), average-pooled 2 x 2 to a 4 x 4 grid of 16 tokens (so
450,850 moments fit in 2.8 GB as float16), standardised per channel. The predictor is V-JEPA 2-AC's shape at
our scale: a transformer over HISTORY moments x 16 tokens, each token plus its time and grid position and the
embedding of the action block taken after its moment, with block-causal attention (a token sees its own
moment and earlier ones); token (t, n) predicts token n of moment t + 1. Trained and planned with the L1
distance, as V-JEPA 2-AC does. Planning is step 7's exactly (pusht's CEM, the same 50 puzzles), with the cost
the L1 distance between the last predicted moment and the goal's.

    uv run python plan_tokens.py embed       # ~25 min
    uv run python plan_tokens.py train       # runs/ac-tokens/predictor.pt
    uv run python plan_tokens.py evaluate    # 50 puzzles
"""

import argparse
import math
import os
import time

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401  (registers swm/PushT-v1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import PushTVideo
from model import VideoEncoder
from plan_ac import (ACTION_DIM, FRAMESKIP, HISTORY, HORIZON, ITERS, N_EVAL,
                     SAMPLES, TOPK, ClipRows, Encoder, Moments, action_stats, device, load_actions,
                     run_episode, starts)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "ac-tokens")  # --name changes it: runs/<name>
ENCODER_PATH = os.path.join(HERE, "runs", "levjepa-s0", "encoder.pt")  # --encoder changes it
GRID, POOLED, DIM = 8, 4, 192
TOKENS = POOLED * POOLED


class TokenEncoder(Encoder):
    def __init__(self, dev: torch.device):
        ck = torch.load(ENCODER_PATH, map_location="cpu")
        self.net = VideoEncoder(**ck["config"])
        self.net.load_state_dict(ck["encoder"])  # V-JEPA's checkpoint holds its target (EMA) encoder here
        self.net.eval().to(dev)
        self.dev = dev

    """A frozen video encoder (LeVJEPA by default, or --encoder): still clips (B, 16, 64, 64, 3) uint8 -> the
    last time slot's tokens pooled to a 4 x 4 grid, (B, 16, 192). Tokens are numbered time-major, after the
    [cls] when there is one (model.py), so the last time slot's 64 are the last 64: LeVJEPA's last frame,
    V-JEPA's last tubelet of two frames (the same frame, in a still clip)."""

    @torch.no_grad()
    def __call__(self, clips: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(clips).to(self.dev).permute(0, 1, 4, 2, 3)
        last = self.net(x)[:, -GRID * GRID:].float()                          # (B, 64, D), row-major
        grid = last.transpose(1, 2).reshape(-1, DIM, GRID, GRID)
        return F.avg_pool2d(grid, GRID // POOLED).flatten(2).transpose(1, 2)  # (B, 16, D)


def embed() -> None:
    video, dev = PushTVideo(), device()
    os.makedirs(OUT, exist_ok=True)
    rows = np.concatenate([video.ep_offset[e] + np.arange(0, video.ep_len[e], FRAMESKIP)
                           for e in video.episodes["train"]])
    loader = DataLoader(ClipRows(video, Moments(video, "still"), rows), batch_size=64, num_workers=6,
                        multiprocessing_context="fork")
    encoder, t0 = TokenEncoder(dev), time.time()
    z = np.lib.format.open_memmap(os.path.join(OUT, "embeddings.npy"), mode="w+", dtype=np.float16,
                                  shape=(len(rows), TOKENS, DIM))
    at = 0
    for i, clips in enumerate(loader):
        out = encoder(clips).cpu().numpy().astype(np.float16)
        z[at:at + len(out)] = out
        at += len(out)
        if i % 500 == 0:
            print(f"{at} of {len(rows)} moments, {at / (time.time() - t0):.0f}/s", flush=True)
    z.flush()
    np.save(os.path.join(OUT, "rows.npy"), rows)
    print(f"done: {len(rows)} moments in {(time.time() - t0) / 60:.0f} min")


class TokenPredictor(nn.Module):
    """Over HISTORY moments x 16 tokens: token + time position + grid position + its moment's action block;
    block-causal attention; a LayerNorm and a linear head per token."""

    def __init__(self, dim: int = DIM, depth: int = 6, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.action_embed = nn.Sequential(nn.Linear(ACTION_DIM, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        self.time = nn.Parameter(torch.zeros(1, HISTORY, 1, dim))
        self.space = nn.Parameter(torch.zeros(1, 1, TOKENS, dim))
        nn.init.trunc_normal_(self.time, std=0.02)
        nn.init.trunc_normal_(self.space, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z: (B, T, 16, D), a: (B, T, ACTION_DIM); output (B, T, 16, D), moment t predicting moment t + 1."""
        b, t = z.shape[:2]
        x = z + self.time[:, :t] + self.space + self.action_embed(a)[:, :, None]
        step = torch.arange(t, device=z.device).repeat_interleave(TOKENS)
        mask = torch.where(step[None, :] > step[:, None], float("-inf"), 0.0)  # no attending to later moments
        y = self.blocks(x.reshape(b, t * TOKENS, -1), mask=mask)
        return self.head(self.norm(y)).reshape(b, t, TOKENS, -1)


class TokenWindows:
    """Every run of HISTORY + 1 embedded moments FRAMESKIP apart in one episode, with the blocks between; the
    embeddings stay on disk (memory-mapped) and are standardised per channel as batches are read."""

    def __init__(self):
        self.video = PushTVideo()
        self.z = np.load(os.path.join(OUT, "embeddings.npy"), mmap_mode="r")
        sample = self.z[np.random.default_rng(0).choice(len(self.z), 20_000, replace=False)].astype(np.float32)
        self.mu, self.sd = sample.mean((0, 1)), sample.std((0, 1)) + 1e-6
        self.rows = np.load(os.path.join(OUT, "rows.npy"))
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
        span = np.sort(idx)[:, None] + np.arange(HISTORY + 1)
        z = (self.z[span.reshape(-1)].astype(np.float32) - self.mu) / self.sd
        z = z.reshape(len(idx), HISTORY + 1, TOKENS, DIM)
        return torch.from_numpy(z).to(dev), torch.from_numpy(self.blocks[span[:, :HISTORY]]).to(dev)


def train(steps: int, batch_size: int = 128, lr: float = 3e-4, seed: int = 0) -> None:
    torch.manual_seed(seed)
    rng, dev = np.random.default_rng(seed), device()
    data = TokenWindows()
    model = TokenPredictor().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warmup = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
    held = rng.choice(data.starts, 2048, replace=False)
    train_starts = np.setdiff1d(data.starts, held)
    zh, ah = data.batch(held, dev)
    copy = (zh[:, 1:] - zh[:, :-1]).abs().mean().item()      # predicting "nothing changes"
    print(f"{len(data.starts)} windows; copying the last moment scores L1 {copy:.4f}", flush=True)
    t0, val = time.time(), float("nan")
    for step in range(steps):
        z, a = data.batch(rng.choice(train_starts, batch_size), dev)
        loss = (model(z[:, :HISTORY], a) - z[:, 1:]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                val = sum((model(zh[i:i + 256, :HISTORY], ah[i:i + 256]) - zh[i:i + 256, 1:]).abs().mean().item()
                          for i in range(0, len(zh), 256)) / math.ceil(len(zh) / 256)
            model.train()
            print(f"step {step + 1}  train {loss.item():.4f}  held-out {val:.4f}  (copy {copy:.4f})  "
                  f"{time.time() - t0:.0f}s", flush=True)
    torch.save({"predictor": model.state_dict(), "mu": data.mu, "sd": data.sd, "steps": steps,
                "held_out_l1": val, "copy_l1": copy}, os.path.join(OUT, "predictor.pt"))
    print(f"saved {OUT}/predictor.pt")


class TokenPlanner:
    """Step 7's CEM over standardised token moments, cost the L1 distance to the goal's tokens; played by step
    7's episode loop (plan_ac.run_episode), which needs only arm, moment, plan, amu, asd and dev."""

    arm = "still"

    def __init__(self, amu: np.ndarray, asd: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        self.dev = device()
        ck = torch.load(os.path.join(OUT, "predictor.pt"), map_location="cpu", weights_only=False)
        self.P = TokenPredictor()
        self.P.load_state_dict(ck["predictor"])
        self.P.eval().to(self.dev)
        self.mu, self.sd = (torch.tensor(x, device=self.dev) for x in (ck["mu"], ck["sd"]))
        self.encoder = TokenEncoder(self.dev)
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
        z0 = torch.stack(z_hist)[None].expand(SAMPLES, -1, -1, -1)
        a0 = (torch.stack(a_hist)[None].expand(SAMPLES, -1, -1) if a_hist
              else torch.zeros(SAMPLES, 0, ACTION_DIM, device=dev))
        for _ in range(ITERS):
            noise = torch.randn(SAMPLES, HORIZON, ACTION_DIM, generator=gen).to(dev)
            cand = torch.maximum(torch.minimum(mean + std * noise, hi), lo)
            z, a = z0, a0
            for h in range(HORIZON):
                a = torch.cat([a, cand[:, h:h + 1]], 1)
                z = torch.cat([z, self.P(z[:, -HISTORY:], a[:, -HISTORY:])[:, -1:]], 1)
            cost = (z[:, -1] - z_goal).abs().sum((-2, -1))
            elite = cand[cost.topk(TOPK, largest=False).indices]
            mean, std = elite.mean(0), elite.std(0)
        return (mean.cpu().numpy() * self.asd + self.amu).reshape(HORIZON, FRAMESKIP, 2)


def evaluate() -> None:
    video = PushTVideo()
    meta = dict(np.load(os.path.join(HERE, "..", "pusht", "data", "meta.npz")))
    actions = meta["action"]
    amu, asd = action_stats(video, actions.astype(np.float32))
    planner = TokenPlanner(amu, asd, np.tile(actions.min(0), FRAMESKIP), np.tile(actions.max(0), FRAMESKIP))
    env = gym.make("swm/PushT-v1", resolution=224).unwrapped
    wins, t0 = [], time.time()
    for trial, (s, g) in enumerate(starts(meta)):
        wins.append(run_episode(env, planner, video, meta, s, g, trial))
        print(f"trial {trial:2d}: {'SOLVED' if wins[-1] else 'not solved'}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"success rate: {sum(wins)}/{N_EVAL}  ({os.path.relpath(ENCODER_PATH, HERE)}, 4 x 4 tokens)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["embed", "train", "evaluate"])
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--encoder", help="a video encoder checkpoint (default: runs/levjepa-s0/encoder.pt)")
    ap.add_argument("--name", default="ac-tokens", help="the run's folder under runs/")
    args = ap.parse_args()
    global OUT, ENCODER_PATH
    OUT = os.path.join(HERE, "runs", args.name)
    if args.encoder:
        ENCODER_PATH = os.path.join(HERE, args.encoder) if not os.path.isabs(args.encoder) else args.encoder
    if args.command == "embed":
        embed()
    elif args.command == "train":
        train(args.steps)
    else:
        evaluate()


if __name__ == "__main__":
    main()
