"""
Arm A, V-JEPA (Bardes et al. 2024): hide most of a clip and predict the features of the hidden part.

Per step, for each of the two mask kinds (masks.py: short-range and long-range multi-block):

    h      = LN(target_encoder(clip))[target]          the hidden tokens' features, from the EMA encoder that
                                                       saw the whole clip; no gradient
    z      = encoder(clip, context)                    only the visible tokens
    p      = predictor(z, context, target)             one prediction per hidden token
    loss  += mean |p - h| / 2                          L1, as in the paper

The encoder and predictor get gradients; the target encoder follows the encoder by EMA, momentum ramped
linearly 0.998 -> 1.0 (marked unverified in PLAN.md). Collapse would show as the target features' spread
across clips falling to zero; it is logged.

Optimisation (not given for our size by the paper; the recipe of our earlier JEPA runs): AdamW, lr warmup to
3e-4 over 5% of steps then cosine to 1e-6, weight decay cosine 0.04 -> 0.4 (none on biases and norms), batch 64.

    uv run python vjepa.py --name short --steps 1500          # the short run (about 20 minutes)
    uv run python vjepa.py --name vjepa-s0 --steps 40000      # the full run
Writes runs/<name>/{log.csv, encoder.pt}; encoder.pt holds the target encoder (what the probes read).
"""

import argparse
import copy
import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from data import Clips, PushTVideo
from masks import LONG, SHORT
from model import Block, VideoEncoder, init_weights, sincos_3d

HERE = os.path.dirname(os.path.abspath(__file__))
ENCODER = dict(frames=16, img_size=64, patch=8, tubelet=2, dim=192, depth=6, heads=3)


class Predictor(nn.Module):
    """A narrow transformer: context tokens (at their positions) plus one mask token per hidden position."""

    def __init__(self, enc_dim: int = 192, dim: int = 96, depth: int = 4, heads: int = 3, t_slots: int = 8, grid: int = 8):
        super().__init__()
        self.embed = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos", sincos_3d(dim, t_slots, grid)[None], persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm, self.proj = nn.LayerNorm(dim), nn.Linear(dim, enc_dim)
        init_weights(self, self.blocks)
        nn.init.trunc_normal_(self.mask_token, std=.02)

    @staticmethod
    def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))

    def forward(self, z: torch.Tensor, ctx_idx: torch.Tensor, ctx_valid: torch.Tensor,
                tgt_idx: torch.Tensor, tgt_valid: torch.Tensor) -> torch.Tensor:
        pos = self.pos.expand(len(z), -1, -1)
        ctx = self.embed(z) + self.gather(pos, ctx_idx)
        qry = self.mask_token + self.gather(pos, tgt_idx)
        x, valid = torch.cat([ctx, qry], 1), torch.cat([ctx_valid, tgt_valid], 1)
        for blk in self.blocks:
            x = blk(x, valid)
        return self.proj(self.norm(x[:, ctx.shape[1]:]))


class VJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VideoEncoder(**ENCODER)
        self.target = copy.deepcopy(self.encoder).requires_grad_(False)
        self.predictor = Predictor()

    def loss(self, clips: torch.Tensor, masks: list[tuple]) -> tuple[torch.Tensor, list, torch.Tensor]:
        """masks: one (ctx, ctx_valid, tgt, tgt_valid) per mask kind; the loss averages real target tokens only."""
        with torch.no_grad():
            h_full = self.target(clips)
            h = F.layer_norm(h_full, h_full.shape[-1:])
        losses = []
        for ctx, cv, tgt, tv in masks:
            p = self.predictor(self.encoder(clips, ctx, cv), ctx, cv, tgt, tv)
            err = (p - Predictor.gather(h, tgt)).abs().mean(-1)                  # (B, K_tgt)
            losses.append((err * tv).sum() / tv.sum())
        return sum(losses) / len(losses), losses, h_full

    @torch.no_grad()
    def ema(self, m: float) -> None:
        for t, o in zip(self.target.parameters(), self.encoder.parameters()):
            t.mul_(m).add_(o.detach(), alpha=1 - m)


def schedules(step: int, total: int) -> tuple[float, float, float]:
    """(lr, weight decay, EMA momentum) at a step."""
    warm = max(1, round(.05 * total))
    if step < warm:
        lr = 1e-5 + (3e-4 - 1e-5) * step / warm
    else:
        lr = 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    t = step / total
    return lr, .04 + (.4 - .04) * .5 * (1 - math.cos(math.pi * t)), .998 + (1 - .998) * t


def spread(h_full: torch.Tensor) -> tuple[float, float]:
    """Collapse check on the clips' mean-pooled target features: mean per-dimension std, and effective rank."""
    z = h_full.mean(1).float()
    z = z - z.mean(0)
    s = torch.linalg.svdvals(z.cpu())
    p = s / s.sum()
    return z.std(0).mean().item(), math.exp(-(p * (p + 1e-12).log()).sum().item())


def param_groups(*models: nn.Module) -> list[dict]:
    decay, no = [], []
    for m in models:
        for n, p in m.named_parameters():
            if p.requires_grad:
                (no if p.ndim < 2 or n.endswith("token") else decay).append(p)
    return [{"params": decay, "decay": True}, {"params": no, "decay": False, "weight_decay": 0.0}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = os.path.join(HERE, "runs", args.name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)

    model = VJEPA().to(dev)
    opt = torch.optim.AdamW(param_groups(model.encoder, model.predictor), lr=1e-5)
    data = Clips(PushTVideo(), "train")
    sampler = RandomSampler(data, replacement=True, num_samples=args.steps * args.batch_size,
                            generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler, num_workers=6,
                        persistent_workers=True, multiprocessing_context="fork", drop_last=True)
    g = torch.Generator().manual_seed(args.seed)
    log = csv.writer(open(os.path.join(out, "log.csv"), "w", newline=""))
    log.writerow(["step", "loss", "loss_short", "loss_long", "feat_std", "eff_rank", "lr", "wd", "ema", "clips_per_s"])
    t0, seen = time.time(), 0
    for step, (x, _) in enumerate(loader):
        lr, wd, m = schedules(step, args.steps)
        for grp in opt.param_groups:
            grp["lr"] = lr
            if grp["decay"]:
                grp["weight_decay"] = wd
        x = x.to(dev)
        masks = [tuple(t.to(dev) for t in mk.batch(len(x), g)) for mk in (SHORT, LONG)]
        loss, parts, h_full = model.loss(x, masks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.ema(m)
        seen += len(x)
        if step % 50 == 0 or step == args.steps - 1:
            if not math.isfinite(loss.item()):
                raise SystemExit(f"loss is {loss.item()} at step {step}")
            sd, rank = spread(h_full)
            rate = seen / (time.time() - t0)
            log.writerow([step, f"{loss.item():.5f}", f"{parts[0].item():.5f}", f"{parts[1].item():.5f}",
                          f"{sd:.4f}", f"{rank:.1f}", f"{lr:.2e}", f"{wd:.3f}", f"{m:.5f}", f"{rate:.0f}"])
            print(f"step {step:6d}  loss {loss.item():.4f} (short {parts[0].item():.4f}, long {parts[1].item():.4f})  "
                  f"feat std {sd:.3f}  rank {rank:.1f}  lr {lr:.1e}  {rate:.0f} clips/s", flush=True)
        if (step + 1) % 5000 == 0 or step == args.steps - 1:
            torch.save({"encoder": model.target.state_dict(), "online": model.encoder.state_dict(),
                        "predictor": model.predictor.state_dict(), "config": ENCODER, "step": step + 1},
                       os.path.join(out, "encoder.pt"))
    print(f"done: {args.steps} steps in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
