"""
Trains the Push-T world model with one regulariser: --reg sigreg | vicreg, and a seed.

Loss (LeWM's lejepa_forward): windows of HISTORY + 1 = 4 frames, FRAMESKIP = 5 env steps apart, from the
training episodes. Every frame is encoded; the predictor reads frames 0..2 with their action blocks and
predicts frames 1..3 (teacher forcing, all three positions at once);

    loss = mean squared error + WEIGHTS[reg] * reg(embeddings of all 4 frames)

After training, BatchNorm's running statistics are recomputed over 200 batches with the weights frozen
(precise BatchNorm; ball_in_cup_sigreg run 2 showed stale EMA statistics can wreck eval-mode planning).

    uv run python train.py --reg sigreg --seed 0          -> runs/sigreg-s0/{model.pt, log.csv}
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import FRAMESKIP, HISTORY, WEIGHTS, Encoder, Predictor, make_reg


class Windows(Dataset):
    """Every (episode, start) with HISTORY + 1 frames FRAMESKIP apart inside a training episode."""

    def __init__(self, episodes_key: str = "train_episodes"):
        meta = np.load("data/meta.npz")
        self.frames = np.load("data/frames64.npy", mmap_mode="r")
        self.action = meta["action"]
        mu, sd = Windows.action_stats(meta)
        self.action = ((self.action - mu[:2]) / sd[:2]).astype(np.float32)   # per 2-d action; blocks tile it
        span = HISTORY * FRAMESKIP
        starts = [meta["ep_offset"][e] + np.arange(meta["ep_len"][e] - span) for e in meta[episodes_key]]
        self.starts = np.concatenate(starts)

    @staticmethod
    def action_stats(meta) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of the 10-d action blocks over the training episodes' rows."""
        rows = np.concatenate([meta["ep_offset"][e] + np.arange(meta["ep_len"][e])
                               for e in meta["train_episodes"]])
        a = meta["action"][rows]
        return np.tile(a.mean(0), FRAMESKIP), np.tile(a.std(0), FRAMESKIP)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        idx = s + FRAMESKIP * np.arange(HISTORY + 1)
        frames = torch.from_numpy(np.stack([self.frames[j] for j in idx]))            # (4, 64, 64, 3)
        acts = np.stack([self.action[j:j + FRAMESKIP].reshape(-1) for j in idx[:HISTORY]])  # (3, 10)
        return frames, torch.from_numpy(acts)


def to_input(frames: torch.Tensor, device) -> torch.Tensor:
    """(..., 64, 64, 3) uint8 -> (..., 3, 64, 64) float in [0, 1] on device."""
    return frames.to(device).permute(*range(frames.dim() - 3), -1, -3, -2).float().div(255)


def step_loss(E, P, reg, weight, frames, acts):
    b, t = frames.shape[:2]
    z = E(frames.reshape(b * t, *frames.shape[2:])).reshape(b, t, -1)
    pred = P(z[:, :HISTORY], acts)
    mse = (pred - z[:, 1:]).pow(2).mean()
    r = reg(z.transpose(0, 1))
    return mse + weight * r, mse, r


@torch.no_grad()
def recalibrate_bn(E, P, loader, device, n_batches=200):
    bns = [m for m in list(E.modules()) + list(P.modules()) if isinstance(m, torch.nn.BatchNorm1d)]
    E.eval(), P.eval()
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    for i, (frames, acts) in enumerate(loader):
        if i == n_batches:
            break
        frames, acts = to_input(frames, device), acts.to(device)
        b, t = frames.shape[:2]
        z = E(frames.reshape(b * t, *frames.shape[2:])).reshape(b, t, -1)
        P(z[:, :HISTORY], acts)
    for bn in bns:
        bn.momentum = 0.1
    E.eval(), P.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", required=True, choices=["sigreg", "vicreg"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight", type=float, default=None, help="regulariser weight (default: WEIGHTS[reg])")
    args = ap.parse_args()
    out = f"runs/{args.reg}-s{args.seed}" if args.weight is None else f"runs/{args.reg}-w{args.weight:g}-s{args.seed}"
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(Windows(), batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g,
                        num_workers=6, persistent_workers=True, multiprocessing_context="fork")
    it = iter(loader)                     # fork the workers before MPS is touched
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    E, P, reg = Encoder().to(device), Predictor().to(device), make_reg(args.reg).to(device)
    weight = WEIGHTS[args.reg] if args.weight is None else args.weight
    params = list(E.parameters()) + list(P.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)
    warmup = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / (args.steps - warmup))))
    log = csv.writer(open(f"{out}/log.csv", "w", newline=""))
    log.writerow(["step", "loss", "mse", "reg", "z_std", "seconds"])
    t0, acc = time.time(), []
    E.train(), P.train()
    for step in range(args.steps):
        try:
            frames, acts = next(it)
        except StopIteration:
            it = iter(loader)
            frames, acts = next(it)
        loss, mse, r = step_loss(E, P, reg, weight, to_input(frames, device), acts.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        acc.append((loss.item(), mse.item(), r.item()))
        if (step + 1) % 200 == 0:
            m = np.mean(acc, 0)
            acc = []
            with torch.no_grad():
                z_std = E(to_input(frames[:, 0], device)).std(0).mean().item()
            log.writerow([step + 1, *np.round(m, 5), round(z_std, 4), round(time.time() - t0)])
            print(f"step {step + 1}  loss {m[0]:.4f}  mse {m[1]:.4f}  {args.reg} {m[2]:.4f}  z_std {z_std:.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    recalibrate_bn(E, P, loader, device)
    torch.save({"E": E.state_dict(), "P": P.state_dict(), "args": vars(args)}, f"{out}/model.pt")
    print(f"saved {out}/model.pt")


if __name__ == "__main__":
    main()
