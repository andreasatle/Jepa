"""
LeJEPA pretraining on PathMNIST (Balestriero & LeCun 2025, arXiv 2511.08544): no EMA,
no stop-gradient, no predictor, no masking. Collapse is prevented by the objective
itself: SIGReg scores how far the embeddings are from an isotropic Gaussian, and a
collapsed embedding scores badly.

One step, for a batch of B images, each cut into V = Vg global + Vl local views:

    h_v   = encoder(view_v).mean(1)          # (B, 192) per view: mean-pooled tokens
    z_v   = projector(h_v)                   # (B, 64)  small MLP head, BatchNorm
    c     = mean over the Vg global views of z_v          # the "center" of each image
    pred  = mean_v || c - z_v ||^2           # every view (global and local) -> its center
    sig   = mean_v SIGReg(z_v)               # each view's batch of embeddings ~ N(0, I)
    loss  = (1 - lambda) * pred + lambda * sig            # lambda = 0.05

Gradients flow through everything, the centers included. The probe uses the encoder's
mean-pooled tokens (h), not the projector, as for train.py's runs.

As in the authors' recipe (paper + repository): 2 global views at 224 px, crop scale
0.3-1.0, and 6 local views at 96 px, scale 0.05-0.3, with DINO's augmentations (see
data.MultiCrop); projector = 3-layer MLP with BatchNorm, hidden 2048; SIGReg with 1024
random directions resampled every step, 17 points; AdamW, lr 5e-4 with one warmup
epoch then cosine to lr/1000, constant weight decay 0.05. Scaled down: ViT-Tiny, batch
128 (the paper recommends >= 128), 10 epochs.

Outputs go to runs/<name>/, in the same format as train.py so probes/probe.py reads them.

Run:
    uv run python lejepa.py --name lejepa --epochs 10
    uv run python lejepa.py --smoke
"""

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import MultiCrop, NativeWindows, RawPathMNIST
from loss import sigreg_loss
from model import Encoder
from train import collapse_stats, param_groups, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    img_size: int = 224          # global views, and what the probe embeds
    local_size: int = 96         # 6 x 6 patches
    patch_size: int = 16
    global_views: int = 2
    local_views: int = 6
    zoom: bool = True            # False: crops at native resolution, never resized (data.MultiCrop)
    # Native windows of varying size, never resized (data.NativeWindows); overrides zoom.
    native_windows: bool = False
    global_sizes: tuple = (160, 176, 192, 208, 224)
    local_sizes: tuple = (64, 80, 96, 112, 128)

    embed_dim: int = 192         # ViT-Tiny, as train.py
    depth: int = 12
    num_heads: int = 3
    proj_hidden: int = 2048
    proj_dim: int = 64

    sigreg_weight: float = 0.05  # lambda
    sigreg_slices: int = 1024

    batch_size: int = 128
    epochs: int = 10
    warmup_epochs: float = 1
    lr: float = 5e-4
    weight_decay: float = 0.05

    num_workers: int = 6
    log_every: int = 20
    save_every: int = 5
    max_steps: int = 0
    seed: int = 0


SMOKE = replace(Config(), embed_dim=64, depth=2, num_heads=2, proj_hidden=128, proj_dim=16,
                batch_size=8, epochs=1, warmup_epochs=0, num_workers=0, log_every=1, max_steps=5)


class Projector(nn.Sequential):
    """Linear-BN-ReLU, Linear-BN-ReLU, Linear: the authors' head (torchvision MLP with BatchNorm)."""

    def __init__(self, dim: int, hidden: int, out: int):
        super().__init__(nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


def lejepa_loss(z_global: torch.Tensor, z_all: torch.Tensor, lam: float, slices: int):
    """
    z_global: (Vg, B, K) projected global views; z_all: (V, B, K) all views (globals first).
    Returns (loss, parts), parts detached for logging.
    """
    centers = z_global.mean(0)                                   # (B, K)
    pred = (centers - z_all).square().mean()
    sig = torch.stack([sigreg_loss(z, slices) for z in z_all]).mean()
    loss = (1 - lam) * pred + lam * sig
    return loss, {"loss": loss.detach(), "pred": pred.detach(), "sig": sig.detach()}


def lr_at(step: int, total: int, warmup: int, cfg: Config) -> float:
    """Linear warmup from lr/100, then cosine down to lr/1000 (the authors' schedule)."""
    if step < warmup:
        return cfg.lr * (0.01 + 0.99 * step / warmup)
    t = (step - warmup) / max(1, total - warmup)
    low = cfg.lr / 1000
    return low + (cfg.lr - low) * 0.5 * (1 + math.cos(math.pi * t))


def train(cfg: Config, name: str):
    out = os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    torch.manual_seed(cfg.seed)
    collate = None
    if cfg.native_windows:
        data = RawPathMNIST("train", cfg.img_size)
        collate = NativeWindows(list(cfg.global_sizes), list(cfg.local_sizes), cfg.global_views,
                                cfg.local_views)
    else:
        data = MultiCrop("train", cfg.img_size, cfg.local_size, cfg.global_views, cfg.local_views,
                         cfg.zoom)
    loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=collate,
                        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
                        multiprocessing_context="fork" if cfg.num_workers > 0 else None)
    if cfg.num_workers:
        iter(loader)              # fork the (persistent) workers before MPS is touched: forking a
                                  # process that has used MPS segfaults the workers
    device = pick_device()

    encoder = Encoder(cfg.img_size, cfg.patch_size, 3, cfg.embed_dim, cfg.depth, cfg.num_heads).to(device)
    projector = Projector(cfg.embed_dim, cfg.proj_hidden, cfg.proj_dim).to(device)
    opt = torch.optim.AdamW(param_groups(encoder, projector))
    for g in opt.param_groups:
        g["weight_decay"] = cfg.weight_decay if g["decay"] else 0.0

    total = cfg.epochs * len(loader)
    warmup = round(cfg.warmup_epochs * len(loader))
    print(f"device={device}  run={out}  steps={total}  "
          f"encoder {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M params")

    log = open(os.path.join(out, "log.csv"), "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "step", "loss", "pred", "sig", "tok_std", "img_std", "img_rank", "lr",
                     "img_per_s"])

    step, t0 = 0, time.time()
    for epoch in range(cfg.epochs):
        for views in loader:
            lr = lr_at(step, total, warmup, cfg)
            for g in opt.param_groups:
                g["lr"] = lr

            # Same-size views go through the encoder together: one call for the globals,
            # one for the locals; the projector sees all of them in one batch.
            B = views[0].shape[0]
            g_imgs = torch.cat(views[:cfg.global_views]).to(device, non_blocking=True)
            l_imgs = torch.cat(views[cfg.global_views:]).to(device, non_blocking=True)
            g_tok = encoder(g_imgs)                                  # (Vg*B, 196, D)
            h = torch.cat([g_tok.mean(1), encoder(l_imgs).mean(1)])  # (V*B, D)
            z = projector(h).reshape(-1, B, cfg.proj_dim)            # (V, B, K)
            loss, parts = lejepa_loss(z[:cfg.global_views], z, cfg.sigreg_weight, cfg.sigreg_slices)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if cfg.native_windows and device.type == "mps":
                torch.mps.empty_cache()   # MPS caches buffers per size; with changing sizes the cache
                                          # fills memory until the machine swaps (20-70 s per step)
            step += 1

            if step % cfg.log_every == 0:
                stats = collapse_stats(g_tok[:B].detach())          # first global view: full-ish image
                ips = cfg.log_every * B / (time.time() - t0)
                t0 = time.time()
                p = {k: v.item() for k, v in parts.items()}
                print(f"e{epoch} s{step}/{total}  loss {p['loss']:.4f}  pred {p['pred']:.4f}  "
                      f"sig {p['sig']:.3f}  img_std {stats['img_std']:.3f}  "
                      f"img_rank {stats['img_rank']:.1f}  lr {lr:.2e}  {ips:.0f} img/s", flush=True)
                writer.writerow([epoch, step, p["loss"], p["pred"], p["sig"], stats["tok_std"],
                                 stats["img_std"], stats["img_rank"], lr, ips])
                log.flush()

            if cfg.max_steps and step >= cfg.max_steps:
                break
        else:
            torch.save({"encoder": encoder.state_dict(), "projector": projector.state_dict(),
                        "opt": opt.state_dict(), "epoch": epoch + 1, "step": step,
                        "config": asdict(cfg)}, os.path.join(out, "ckpt.pt"))
            if (epoch + 1) % cfg.save_every == 0 or epoch + 1 == cfg.epochs:
                torch.save({"encoder": encoder.state_dict(), "epoch": epoch + 1, "config": asdict(cfg)},
                           os.path.join(out, f"encoder_e{epoch + 1:03d}.pt"))
            continue
        break
    log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lejepa")
    ap.add_argument("--smoke", action="store_true", help="tiny model, 5 steps")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--sigreg-weight", type=float, help="lambda (default 0.05)")
    ap.add_argument("--sigreg-slices", type=int, help="random directions per view per step (default 1024)")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--no-zoom", action="store_true", help="crops at native resolution, never resized")
    ap.add_argument("--native-windows", action="store_true", help="native windows of varying size")
    args = ap.parse_args()
    if args.smoke:
        train(SMOKE, "smoke-lejepa")
    else:
        cfg = Config()
        for field, value in [("epochs", args.epochs), ("batch_size", args.batch_size), ("lr", args.lr),
                             ("sigreg_weight", args.sigreg_weight), ("max_steps", args.max_steps),
                             ("num_workers", args.num_workers),
                             ("sigreg_slices", args.sigreg_slices)]:
            if value is not None:
                cfg = replace(cfg, **{field: value})
        if args.native_windows:
            cfg = replace(cfg, native_windows=True)
        if args.no_zoom:
            cfg = replace(cfg, zoom=False)
        train(cfg, args.name)
