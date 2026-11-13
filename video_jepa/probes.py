"""
Frozen-encoder probes: can position, velocity and contact of a clip's last frame be read off its tokens?

  1. Freeze the encoder (eval mode). Embed every probe clip whole, no masking or dropping: (N, D) tokens each,
     cached in half precision.
  2. Linear probe: ridge regression on the mean of the tokens (standardised with train statistics). The mean
     forgets where and when each token sits, so it succeeds only if the encoder put the answer into the
     tokens' content. Closed form, no training choices beyond the ridge weight.
  3. Attentive probe, as in V-JEPA: one learned query cross-attends over all tokens (several heads, so it can
     look at different frames at once), then an MLP and a linear layer to all targets. Trained with MSE on
     standardised targets, one fixed recipe for every encoder.

Every target is scored as R^2 on held-out episodes (1 = perfect, 0 = predicting the mean); contact also as
accuracy. Read every number against the floor: the same probes on a random-init encoder (--random).

    uv run python probes.py --random                          # floor
    uv run python probes.py --encoder runs/<run>/encoder.pt   # a trained encoder

Probe sets: 8,000 training clips (seed 1) and 2,000 held-out clips (seed 2), fixed for every arm.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import GROUPS, TARGETS, Clips, PushTVideo
from model import VideoEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
N_TRAIN, N_EVAL = 8000, 2000
CONTACT = TARGETS.index("contact")


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def embed(encoder: nn.Module, clips: Clips, dev: torch.device, batch_size: int = 64) -> torch.Tensor:
    """(n, N, D) float16 tokens of every clip, on the CPU."""
    encoder.eval().to(dev)
    out = []
    for x, _ in DataLoader(clips, batch_size=batch_size):
        out.append(encoder(x.to(dev)).half().cpu())
    return torch.cat(out)


def r2(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-column R^2."""
    return 1 - ((pred - y) ** 2).mean(0) / y.var(0, unbiased=False)


def contact_accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return ((pred[:, CONTACT] > .5) == (y[:, CONTACT] > .5)).float().mean().item()


def linear_probe(xtr: torch.Tensor, ytr: torch.Tensor, xev: torch.Tensor, ridge: float = 1.0) -> torch.Tensor:
    """Ridge regression on standardised features; returns predictions for xev in target units."""
    xtr, xev = xtr.double(), xev.double()
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    a, b = (xtr - mu) / sd, (xev - mu) / sd
    a, b = torch.cat([a, torch.ones(len(a), 1, dtype=a.dtype)], 1), torch.cat([b, torch.ones(len(b), 1, dtype=b.dtype)], 1)
    reg = ridge * torch.eye(a.shape[1], dtype=a.dtype)
    reg[-1, -1] = 0                                                           # no penalty on the bias
    w = torch.linalg.solve(a.T @ a + reg, a.T @ ytr.double())
    return (b @ w).float()


class AttentiveProbe(nn.Module):
    """One learned query cross-attending over the tokens, then a residual MLP and a linear map to the targets."""

    def __init__(self, dim: int, n_out: int, heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_out))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        kv = self.norm_kv(tokens)
        x, _ = self.attn(self.query.expand(len(tokens), -1, -1), kv, kv)
        x = x + self.mlp(self.norm(x))
        return self.head(x[:, 0])


def attentive_probe(ttr: torch.Tensor, ytr: torch.Tensor, tev: torch.Tensor, dev: torch.device,
                    epochs: int = 30, batch_size: int = 256, lr: float = 1e-3, wd: float = .05,
                    seed: int = 0) -> torch.Tensor:
    """Trains the probe on (ttr, ytr) and returns predictions for tev in target units."""
    torch.manual_seed(seed)
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-6
    probe = AttentiveProbe(ttr.shape[-1], ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((len(ttr) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=.1)
    ttr_d, y_d = ttr.to(dev), ((ytr - mu) / sd).to(dev)
    g = torch.Generator().manual_seed(seed)
    probe.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(ttr), generator=g).split(batch_size):
            idx = idx.to(dev)
            loss = ((probe(ttr_d[idx].float()) - y_d[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    probe.eval()
    with torch.no_grad():
        pred = torch.cat([probe(t.to(dev).float()).cpu() for t in tev.split(512)])
    return pred * sd + mu


def summarise(name: str, pred: torch.Tensor, y: torch.Tensor) -> dict:
    s = r2(pred, y)
    row = {"probe": name, **{t: round(s[i].item(), 3) for i, t in enumerate(TARGETS)}}
    for g, ts in GROUPS.items():
        row[g] = round(float(np.mean([row[t] for t in ts])), 3)
    row["contact_acc"] = round(contact_accuracy(pred, y), 3)
    return row


def print_rows(rows: list[dict]) -> None:
    print(f"{'probe':<28} {'position':>9} {'velocity':>9} {'contact':>8} {'contact acc':>12}")
    for r in rows:
        print(f"{r['probe']:<28} {r['position']:9.3f} {r['velocity']:9.3f} {r['contact']:8.3f} {r['contact_acc']:12.3f}")
    print("per target (R^2): " + "; ".join(r["probe"] + ": " + ", ".join(f"{t} {r[t]:.2f}" for t in TARGETS)
                                          for r in rows))


def probe_sets(video: PushTVideo) -> tuple[Clips, Clips]:
    return Clips(video, "train", n=N_TRAIN, seed=1), Clips(video, "held_out", n=N_EVAL, seed=2)


def run(encoder: nn.Module, label: str, dev: torch.device, probe_seed: int = 0) -> list[dict]:
    tr, ev = probe_sets(PushTVideo())
    t0 = time.time()
    ttr, tev = embed(encoder, tr, dev), embed(encoder, ev, dev)
    print(f"embedded {len(tr)} + {len(ev)} clips in {time.time() - t0:.0f} s, tokens {tuple(ttr.shape[1:])}")
    ytr, yev = torch.from_numpy(tr.targets), torch.from_numpy(ev.targets)
    rows = [summarise(f"{label} linear (mean)", linear_probe(ttr.float().mean(1), ytr, tev.float().mean(1)), yev)]
    t0 = time.time()
    rows.append(summarise(f"{label} attentive" + (f" p{probe_seed}" if probe_seed else ""),
                         attentive_probe(ttr, ytr, tev, dev, seed=probe_seed), yev))
    print(f"attentive probe trained in {time.time() - t0:.0f} s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", help="a checkpoint with the encoder's state_dict under 'encoder'")
    ap.add_argument("--random", action="store_true", help="random-init encoder of the same config: the floor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-seed", type=int, default=0, help="the attentive probe's init and data order")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.random:
        encoder, label = VideoEncoder(), f"random-s{args.seed}"
    else:
        ck = torch.load(args.encoder, map_location="cpu")
        encoder = VideoEncoder(**ck.get("config", {}))
        encoder.load_state_dict(ck["encoder"])
        label = os.path.basename(os.path.dirname(args.encoder))
    rows = run(encoder, label, device(), args.probe_seed)
    print_rows(rows)
    path = os.path.join(HERE, "probes_results.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
