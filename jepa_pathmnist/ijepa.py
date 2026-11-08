"""
I-JEPA pretraining on PathMNIST: the official recipe (EMA target encoder),
plus VICReg on the online encoder, plus a full context.

One step:

    ctx_mask, ctx_keep, tgt_masks = masks.sample(B)          # context = grid minus targets
    h_tgt = target_encoder(imgs)                             # EMA teacher, full image, no grad
    h     = encoder(imgs)                                    # online, full image (for VICReg)
    z_ctx = encoder(imgs, ctx_mask, ctx_keep)                # context patches only (padded)
    p     = predictor(z_ctx, ctx_mask, ctx_keep, tgt_masks)  # predicted target embeddings
    loss  = smooth_l1(p, LN(h_tgt[tgt]))                     # invariance
          + var_weight * variance(h) + cov_weight * covariance(h)    # VICReg
    step; target_encoder <- m * target_encoder + (1 - m) * encoder   # m: 0.996 -> 1

Deviations from the official recipe (see LEDGER.md for the measurements):
  - VICReg variance/covariance on the online full-image tokens (official: none).
  - The context is exactly the grid minus the target patches, padded per batch,
    instead of the official 85-100% block truncated to the shortest context in
    the batch (see masks.py).
Switches kept for comparison: --no-ema (target = online output, stop-gradient),
--target-grad (no stop-gradient: "pure VICReg"), --vicreg-per-position,
--sigreg-weight L (LeJEPA's SIGReg on the mean-pooled online image embeddings:
loss = (1 - L) * prediction + L * SIGReg; with --target-grad and VICReg off, the
only thing resisting collapse, as in LeJEPA).

Everything else follows the official code: target blocks (masks.py), ViT
encoder + narrow predictor with its init (model.py), target LayerNorm +
smooth-L1 (loss.py), RandomResizedCrop only (data.py), AdamW with linear warmup
+ cosine lr, cosine weight-decay ramp 0.04 -> 0.4 (no decay on biases / norms),
no gradient clipping. Scaled down: small encoder, small batch.

Outputs go to runs/<name>/: config.json, log.csv, ckpt.pt (latest, overwritten
each epoch) and encoder_eXXX.pt snapshots for probing (with EMA: the target
encoder, as in the paper's evaluation).

Run:
    uv run python ijepa.py --name NAME --epochs 10 --batch-size 64 --lr 3e-4
"""

import argparse
import copy
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace

import torch
from torch.utils.data import DataLoader

from data import PathMNIST
from loss import center_per_position, sigreg_loss, vicreg_loss
from masks import MultiBlockMasks
from model import Encoder, Predictor, apply_masks

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # data: 224px / 16px patches -> 14 x 14 grid, as in the paper
    img_size: int = 224
    patch_size: int = 16

    # encoder: ViT-Tiny
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    # predictor: half the encoder width, 6 layers
    pred_dim: int = 96
    pred_depth: int = 6
    pred_heads: int = 3

    # VICReg on the online full-image tokens (old code's kappa=1 -> 0.5 each)
    var_weight: float = 0.5
    cov_weight: float = 0.5
    # False: target = h.detach() (stop-gradient). True: gradients flow through the
    # target too ("pure VICReg"); only the variance/covariance terms resist collapse.
    target_grad: bool = False
    # True: VICReg on per-position-centred tokens (loss.center_per_position), so
    # differences between positions don't count as variance.
    vicreg_per_position: bool = False
    # LeJEPA's SIGReg on the mean-pooled online image embeddings (B, D), weight lambda:
    # loss = (1 - lambda) * (prediction + VICReg terms) + lambda * SIGReg. 0 = off.
    # Paper default 0.05, 1024 random directions resampled every step.
    sigreg_weight: float = 0.0
    sigreg_slices: int = 1024
    # EMA target encoder (official I-JEPA): target = EMA copy of the encoder, no grad,
    # updated after every step with momentum ramped linearly ema_start -> ema_end.
    # False: target = the online encoder's output, gradient-stopped.
    ema: bool = True
    ema_start: float = 0.996
    ema_end: float = 1.0

    # optimisation. Official at batch 2048: lr 2e-4 -> 1e-3 over 40 of 300
    # epochs, cosine to 1e-6. lr scaled by batch / 2048; epochs shortened with the
    # same warmup fraction.
    batch_size: int = 256
    epochs: int = 100
    warmup_epochs: float = 13
    lr: float = 1e-3 * 256 / 2048
    start_lr: float = 2e-4 * 256 / 2048
    final_lr: float = 1e-6
    wd_start: float = 0.04
    wd_end: float = 0.4

    num_workers: int = 4
    log_every: int = 20          # steps
    save_every: int = 5          # epochs between encoder snapshots
    max_steps: int = 0           # 0 = no cap
    seed: int = 0


SMOKE = replace(Config(), embed_dim=64, depth=2, num_heads=2, pred_dim=32, pred_depth=1,
                pred_heads=2, batch_size=8, epochs=1, warmup_epochs=0, num_workers=0,
                log_every=1, max_steps=5)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def schedules(step: int, total: int, warmup: int, cfg: Config):
    """(lr, weight decay) at `step`. lr: linear warmup, then cosine. wd: cosine ramp up."""
    if step < warmup:
        lr = cfg.start_lr + (cfg.lr - cfg.start_lr) * step / warmup
    else:
        t = (step - warmup) / max(1, total - warmup)
        lr = cfg.final_lr + (cfg.lr - cfg.final_lr) * 0.5 * (1 + math.cos(math.pi * t))
    t = step / max(1, total)
    wd = cfg.wd_start + (cfg.wd_end - cfg.wd_start) * 0.5 * (1 - math.cos(math.pi * t))
    return lr, wd


@torch.no_grad()
def ema_update(target: torch.nn.Module, online: torch.nn.Module, m: float) -> None:
    """target <- m * target + (1 - m) * online, parameter by parameter."""
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(m).add_(po, alpha=1 - m)


def param_groups(*models):
    """Weight decay on matrices only; biases and LayerNorm parameters (1-D) are exempt."""
    decay, no_decay = [], []
    for m in models:
        for p in m.parameters():
            (no_decay if p.ndim == 1 else decay).append(p)
    return [{"params": decay, "decay": True}, {"params": no_decay, "decay": False, "weight_decay": 0.0}]


@torch.no_grad()
def collapse_stats(h: torch.Tensor) -> dict:
    """
    Cheap collapse diagnostics on the full-image embeddings h (B, N, D).
      tok_std   mean per-dim std over all tokens           (what VICReg acts on)
      img_std   mean per-dim std of the per-image means    (what a probe sees)
      img_rank  participation ratio of the per-image means: (sum l)^2 / sum l^2
                over covariance eigenvalues l. D if all dims equally used, 1 if
                one direction carries everything. Capped by the batch size.
    """
    h = h.float().cpu()                                   # eigvalsh has no MPS kernel
    tok = h.reshape(-1, h.shape[-1])
    img = h.mean(1)
    eig = torch.linalg.eigvalsh(torch.cov(img.T)).clamp_min(0)
    return {"tok_std": tok.std(0).mean().item(), "img_std": img.std(0).mean().item(),
            "img_rank": (eig.sum() ** 2 / (eig ** 2).sum().clamp_min(1e-12)).item()}


def train(cfg: Config, name: str):
    out = os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    torch.manual_seed(cfg.seed)
    device = pick_device()

    data = PathMNIST("train", img_size=cfg.img_size)
    loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
                        # fork: workers share the memmap instead of each pickling a copy
                        multiprocessing_context="fork" if cfg.num_workers > 0 else None)
    grid = cfg.img_size // cfg.patch_size
    masks = MultiBlockMasks(grid, seed=cfg.seed)

    encoder = Encoder(cfg.img_size, cfg.patch_size, 3, cfg.embed_dim, cfg.depth,
                      cfg.num_heads).to(device)
    predictor = Predictor(grid * grid, cfg.embed_dim, cfg.pred_dim, cfg.pred_depth,
                          cfg.pred_heads).to(device)
    opt = torch.optim.AdamW(param_groups(encoder, predictor))
    assert not (cfg.ema and cfg.target_grad), "EMA target never gets gradients"
    if cfg.ema:
        target_encoder = copy.deepcopy(encoder)
        target_encoder.requires_grad_(False)
    use_vicreg = bool(cfg.var_weight or cfg.cov_weight)
    use_online = use_vicreg or bool(cfg.sigreg_weight)          # needs the online full image

    total = cfg.epochs * len(loader)
    warmup = round(cfg.warmup_epochs * len(loader))
    print(f"device={device}  run={out}  steps={total}  "
          f"encoder {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M params  "
          f"predictor {sum(p.numel() for p in predictor.parameters()) / 1e6:.1f}M")

    log = open(os.path.join(out, "log.csv"), "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "step", "loss", "inv", "var", "cov", "sig",
                     "tok_std", "img_std", "img_rank", "lr", "wd", "img_per_s"])

    step, t0 = 0, time.time()
    for epoch in range(cfg.epochs):
        for imgs in loader:
            lr, wd = schedules(step, total, warmup, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
                if g["decay"]:
                    g["weight_decay"] = wd

            imgs = imgs.to(device, non_blocking=True)
            ctx_mask, ctx_keep, tgt_masks = masks.sample(imgs.shape[0])
            ctx_mask, ctx_keep = ctx_mask.to(device), ctx_keep.to(device)
            tgt_masks = [m.to(device) for m in tgt_masks]

            h = None                                             # online full image, (B, N, D)
            if cfg.ema:
                with torch.no_grad():
                    h_tgt = target_encoder(imgs)                 # EMA teacher, full image
                if use_online:
                    h = encoder(imgs)
            else:
                h = encoder(imgs)
                h_tgt = h if cfg.target_grad else h.detach()
            target = apply_masks(h_tgt, tgt_masks)               # (n_pred*B, K_tgt, D)
            z_ctx = encoder(imgs, ctx_mask, ctx_keep)            # (B, L, D), padded
            pred = predictor(z_ctx, ctx_mask, ctx_keep, tgt_masks)   # (n_pred*B, K_tgt, D)
            reg = [] if h is None else [center_per_position(h) if cfg.vicreg_per_position else h]
            loss, parts = vicreg_loss(pred, target, reg, cfg.var_weight, cfg.cov_weight)
            if cfg.sigreg_weight:
                sig = sigreg_loss(h.mean(1), cfg.sigreg_slices)     # one embedding per image
                loss = (1 - cfg.sigreg_weight) * loss + cfg.sigreg_weight * sig
                parts["sig"], parts["loss"] = sig.detach(), loss.detach()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if cfg.ema:
                m = cfg.ema_start + (cfg.ema_end - cfg.ema_start) * step / total
                ema_update(target_encoder, encoder, m)
            step += 1

            if step % cfg.log_every == 0:
                stats = collapse_stats(h if h is not None else h_tgt)
                ips = cfg.log_every * cfg.batch_size / (time.time() - t0)
                t0 = time.time()
                p = {k: v.item() for k, v in parts.items()}
                print(f"e{epoch} s{step}/{total}  loss {p['loss']:.4f}  inv {p['inv']:.4f}  "
                      f"var {p.get('var', 0):.3f}  cov {p.get('cov', 0):.3f}  sig {p.get('sig', 0):.3f}  "
                      f"tok_std {stats['tok_std']:.3f}  img_std {stats['img_std']:.3f}  "
                      f"img_rank {stats['img_rank']:.1f}  lr {lr:.2e}  {ips:.0f} img/s")
                writer.writerow([epoch, step, p["loss"], p["inv"], p.get("var", 0),
                                 p.get("cov", 0), p.get("sig", 0), stats["tok_std"], stats["img_std"],
                                 stats["img_rank"], lr, wd, ips])
                log.flush()

            if cfg.max_steps and step >= cfg.max_steps:
                break
        else:
            # With EMA the probed "encoder" is the target encoder, as in the paper's
            # evaluation; the online encoder is kept alongside.
            probed = target_encoder if cfg.ema else encoder
            extra = {"online_encoder": encoder.state_dict()} if cfg.ema else {}
            torch.save({"encoder": probed.state_dict(), "predictor": predictor.state_dict(),
                        "opt": opt.state_dict(), "epoch": epoch + 1, "step": step,
                        "config": asdict(cfg), **extra}, os.path.join(out, "ckpt.pt"))
            if (epoch + 1) % cfg.save_every == 0 or epoch + 1 == cfg.epochs:
                torch.save({"encoder": probed.state_dict(), "epoch": epoch + 1,
                            "config": asdict(cfg), **extra},
                           os.path.join(out, f"encoder_e{epoch + 1:03d}.pt"))
            continue
        break                                                    # max_steps hit
    log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--smoke", action="store_true", help="tiny model, 5 steps")
    ap.add_argument("--epochs", type=int, help="default 100; warmup scales along (13%%)")
    ap.add_argument("--batch-size", type=int, help="default 256; lr scales along (linear)")
    ap.add_argument("--lr", type=float, help="peak lr, overrides the batch scaling; warmup starts at lr/5")
    ap.add_argument("--target-grad", action="store_true", help="no stop-gradient on the target (pure VICReg)")
    ap.add_argument("--vicreg-per-position", action="store_true",
                    help="VICReg on per-position-centred tokens (closes the position loophole)")
    ap.add_argument("--no-ema", action="store_true", help="target = online output, stop-gradient (no EMA)")
    ap.add_argument("--var-weight", type=float, help="VICReg variance weight (default 0.5; 0 = off)")
    ap.add_argument("--cov-weight", type=float, help="VICReg covariance weight (default 0.5; 0 = off)")
    ap.add_argument("--sigreg-weight", type=float, help="LeJEPA SIGReg weight lambda (default 0 = off; paper 0.05)")
    args = ap.parse_args()
    if args.smoke:
        train(SMOKE, "smoke")
    else:
        cfg = Config()
        if args.epochs:
            cfg = replace(cfg, epochs=args.epochs,
                          warmup_epochs=cfg.warmup_epochs * args.epochs / cfg.epochs)
        if args.batch_size:
            s = args.batch_size / cfg.batch_size
            cfg = replace(cfg, batch_size=args.batch_size, lr=cfg.lr * s, start_lr=cfg.start_lr * s)
        if args.target_grad:
            cfg = replace(cfg, target_grad=True, ema=False)
        if args.vicreg_per_position:
            cfg = replace(cfg, vicreg_per_position=True)
        if args.lr:
            cfg = replace(cfg, lr=args.lr, start_lr=args.lr / 5)
        if args.no_ema:
            cfg = replace(cfg, ema=False)
        if args.var_weight is not None:
            cfg = replace(cfg, var_weight=args.var_weight)
        if args.cov_weight is not None:
            cfg = replace(cfg, cov_weight=args.cov_weight)
        if args.sigreg_weight is not None:
            cfg = replace(cfg, sigreg_weight=args.sigreg_weight)
        train(cfg, args.name)
