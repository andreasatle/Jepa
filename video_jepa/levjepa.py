"""
Arm B, LeVJEPA (Kuhn et al. 2026): recognise the same clip from a crop, with SIGReg instead of EMA, predictor
and stop-gradient.

Per step, for a batch of clips (16 frames, one token per 8 x 8 patch per frame: 1,024 tokens, tubelet 1):

    global view   the whole clip; 5% of its 1,024 tokens kept at random                    -> 51 tokens
    local views   V = 4 crops of the same 16 frames: an s x s patch window (s drawn per batch from 5, 6:
                  39% or 56% of the frame), at native resolution, the window the same in every frame;
                  5% of its tokens kept at random                                          -> 20 or 29 tokens
    z_v           = projector(encoder(view)[cls])          Linear 192->2048, BatchNorm, GELU, Linear ->256
    inv           = mean over local views of mean (z_local - z_global)^2
    sig           = mean over the 5 views of SIGReg(z_v)  (LeWM's: 17 knots on [0, 3], weight e^{-t^2/2},
                                                           1,024 random directions, times the batch size)
    loss          = (1 - 0.02) * inv + 0.02 * sig

Departures from the paper, all for our 64 px Push-T clips (PLAN.md):
  - Local views are patch-aligned windows kept at native resolution with their true positions (3-D sin-cos),
    not 96 px crops resized; the paper's RoPE gives crops their positions, our fixed sin-cos needs the window
    to sit on the patch grid. (PathMNIST's LeJEPA did the same: native-resolution windows.)
  - Crops only, no photometric augmentation: Push-T's colours carry the meaning (green goal, grey T, blue
    pusher).
  - Bidirectional attention, not block-causal (the paper: block-causal has no measurable cost or gain).
Optimisation (not given by the paper): our PathMNIST LeJEPA recipe, AdamW lr 5e-4, warmup 5% of steps from
lr/100, cosine to lr/1000, weight decay 0.05 (none on biases, norms and the [cls] token); batch 128 (the paper's
ViT-Tiny run).

The probes read the encoder on whole clips, every token kept: (B, 1 + 1,024, 192), the [cls] first.

    uv run python levjepa.py --name le-short --steps 1500     # the short run
    uv run python levjepa.py --name levjepa-s0 --steps N      # the full run, N from the short run's speed
Writes runs/<name>/{log.csv, encoder.pt}.
"""

import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from data import Clips, PushTVideo
from model import VideoEncoder
from vjepa import spread

HERE = os.path.dirname(os.path.abspath(__file__))
ENCODER = dict(frames=16, img_size=64, patch=8, tubelet=1, dim=192, depth=6, heads=3, cls=True)
LAMBDA, LOCAL_VIEWS, KEEP = 0.02, 4, 0.05


class Views:
    """Token indices of the global and local views, time-major over (frames, grid, grid) as in model.py."""

    def __init__(self, frames: int = 16, grid: int = 8, sizes=(5, 6), local_views: int = LOCAL_VIEWS,
                 keep: float = KEEP):
        self.frames, self.grid, self.sizes, self.local_views, self.keep = frames, grid, sizes, local_views, keep
        self.n = frames * grid * grid

    def kept(self, tokens: torch.Tensor, b: int, g: torch.Generator) -> torch.Tensor:
        """(b, k): for each clip, round(keep * len(tokens)) of `tokens` at random, sorted."""
        k = max(1, round(self.keep * len(tokens)))
        pick = torch.rand(b, len(tokens), generator=g).argsort(1)[:, :k]
        return tokens[pick].sort(1).values

    def window(self, top: int, left: int, s: int) -> torch.Tensor:
        """Every token of the s x s patch window at (top, left), in every frame."""
        t, r, c = torch.meshgrid(torch.arange(self.frames), torch.arange(top, top + s),
                                 torch.arange(left, left + s), indexing="ij")
        return ((t * self.grid + r) * self.grid + c).flatten()

    def batch(self, b: int, g: torch.Generator) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """global (b, 51) and local_views x (b, k_local). The window size is drawn once per batch (so every
        clip keeps the same number of tokens), its place per clip and view."""
        glob = self.kept(torch.arange(self.n), b, g)
        s = self.sizes[torch.randint(len(self.sizes), (1,), generator=g).item()]
        local = []
        for _ in range(self.local_views):
            rows = []
            for _ in range(b):
                top, left = torch.randint(0, self.grid - s + 1, (2,), generator=g).tolist()
                rows.append(self.kept(self.window(top, left, s), 1, g)[0])
            local.append(torch.stack(rows))
        return glob, local


class SIGReg(nn.Module):
    """LeWM's SIGReg (../pusht/model.py), for one view: z (B, D) -> B * Epps-Pulley against N(0, 1), averaged
    over 1,024 random directions."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a = torch.randn(z.size(-1), self.num_proj, device=z.device)
        a = a / a.norm(dim=0)
        xt = (z.float() @ a).unsqueeze(-1) * self.t                      # (B, P, knots)
        err = (xt.cos().mean(0) - self.phi).square() + xt.sin().mean(0).square()
        return ((err @ self.weights) * z.size(0)).mean()


class LeVJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VideoEncoder(**ENCODER)
        d = ENCODER["dim"]
        self.projector = nn.Sequential(nn.Linear(d, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Linear(2048, 256))
        self.sigreg = SIGReg()

    def embed(self, clips: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(cls output (B, 192), projection (B, 256)) of the view that keeps tokens idx."""
        h = self.encoder(clips, idx)[:, 0]
        return h, self.projector(h)

    def loss(self, clips: torch.Tensor, glob: torch.Tensor, local: list[torch.Tensor]):
        """Returns loss, (inv, sig), the global view's cls outputs (for the collapse log)."""
        h, zg = self.embed(clips, glob)
        zl = torch.stack([self.embed(clips, idx)[1] for idx in local])          # (V, B, 256)
        inv = (zl - zg).square().mean()
        sig = torch.stack([self.sigreg(z) for z in (zg, *zl)]).mean()
        return (1 - LAMBDA) * inv + LAMBDA * sig, (inv, sig), h


def lr_at(step: int, total: int, lr: float = 5e-4) -> float:
    warm = max(1, round(.05 * total))
    if step < warm:
        return lr * (.01 + .99 * step / warm)
    low = lr / 1000
    return low + (lr - low) * .5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def param_groups(model: nn.Module, wd: float = .05) -> list[dict]:
    decay, no = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no if p.ndim < 2 or n.endswith("cls") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no, "weight_decay": 0.0}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = os.path.join(HERE, "runs", args.name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)

    model = LeVJEPA().to(dev)
    opt = torch.optim.AdamW(param_groups(model), lr=5e-6)
    data = Clips(PushTVideo(), "train")
    sampler = RandomSampler(data, replacement=True, num_samples=args.steps * args.batch_size,
                            generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler, num_workers=6,
                        persistent_workers=True, multiprocessing_context="fork", drop_last=True)
    views, g = Views(), torch.Generator().manual_seed(args.seed)
    log = csv.writer(open(os.path.join(out, "log.csv"), "w", newline=""))
    log.writerow(["step", "loss", "inv", "sig", "feat_std", "eff_rank", "lr", "clips_per_s"])
    t0, seen = time.time(), 0
    for step, (x, _) in enumerate(loader):
        lr = lr_at(step, args.steps)
        for grp in opt.param_groups:
            grp["lr"] = lr
        x = x.to(dev)
        glob, local = views.batch(len(x), g)
        loss, (inv, sig), h = model.loss(x, glob.to(dev), [v.to(dev) for v in local])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        seen += len(x)
        if step % 50 == 0 or step == args.steps - 1:
            if not math.isfinite(loss.item()):
                raise SystemExit(f"loss is {loss.item()} at step {step}")
            sd, rank = spread(h.detach()[:, None])
            rate = seen / (time.time() - t0)
            log.writerow([step, f"{loss.item():.5f}", f"{inv.item():.5f}", f"{sig.item():.5f}", f"{sd:.4f}",
                          f"{rank:.1f}", f"{lr:.2e}", f"{rate:.0f}"])
            print(f"step {step:6d}  loss {loss.item():.4f} (inv {inv.item():.4f}, sig {sig.item():.3f})  "
                  f"cls std {sd:.3f}  rank {rank:.1f}  lr {lr:.1e}  {rate:.0f} clips/s", flush=True)
        if (step + 1) % 5000 == 0 or step == args.steps - 1:
            torch.save({"encoder": model.encoder.state_dict(), "projector": model.projector.state_dict(),
                        "config": ENCODER, "step": step + 1}, os.path.join(out, "encoder.pt"))
    print(f"done: {args.steps} steps in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
