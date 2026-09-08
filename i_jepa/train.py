"""
I-JEPA pretraining on PathMNIST-224 -- EMA-free, stop-grad target, VICReg
anti-collapse.

One optimizer step (single encoder, no momentum teacher -- see loss.py):

    z_ctx = encoder(imgs, masks=enc_masks)      # context block, gets grad
    h     = encoder(imgs)                        # full image, SAME weights, gets grad
    h_tgt = apply_masks(h.detach(), pred_masks)  # prediction target: STOP-GRAD
    p     = predictor(z_ctx, enc_masks, pred_masks)
    loss  = smooth_l1(p, LN(h_tgt))
            + kappa/2       ( pen(z_ctx) + pen(h) )       # token-level VICReg
            + kappa_pooled/2  pen(h.mean(patches))        # image-level VICReg

The stop-grad on the target is what SimSiam identified as the essential
anti-collapse ingredient: the predictor must move to meet a target it cannot
pull back, so there is no reward for shrinking both sides to a constant. VICReg
keeps the online embeddings spread + decorrelated (token-level) and protects
between-image variance (pooled). No EMA network, no momentum schedule.

v1 (non-detached target, token VICReg only) collapsed at the image level; v2
(+ pooled VICReg) fixed the rank but frozen-probe accuracy stalled below
random-init because the non-detached target let context/target co-adapt into a
predictable-but-not-semantic code. v3 adds the stop-grad.

The config is sized for a single Apple-silicon / one-GPU box, not paper scale.
Resumes from jepa_ckpt.pt if it exists.

Run:
    python i_jepa/train.py              # full run
    python i_jepa/train.py --smoke      # tiny model, 3 steps, wiring check only
    python i_jepa/train.py --diag       # fresh 4-epoch run, separate ckpt/log
"""

import csv
import math
import os
import sys
import time
from dataclasses import dataclass, replace

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from vit import ViT, apply_masks
from predictor import VisionTransformerPredictor
from masks import MultiBlockMaskCollator
from loss import ijepa_loss
from data import PathMNIST224

VERSION = "v3"      # v1 non-detached target; v2 +pooled VICReg; v3 +stop-grad target


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    img_size: int = 224
    patch: int = 16

    # encoder -- ViT-S/16
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6

    # predictor -- narrow, shallow
    pred_dim: int = 192
    pred_depth: int = 6
    pred_heads: int = 6

    # mask sampler -- 4 target rectangles; context = grid minus their union
    n_pred: int = 4
    pred_hw: tuple = None           # target h,w integer range; None -> ~0.2..0.55 * (patches/side)

    # optim
    batch_size: int = 24
    n_epochs: int = 30
    base_lr: float = 1.5e-4
    warmup_steps: int = 3000       # fixed step count -- doesn't drift when batch_size changes
    weight_decay: float = 0.04
    grad_clip: float = 1.0
    # anti-collapse weight. This is EMA-free: `target` is the same encoder with
    # no stop-grad, so the degenerate fix-point (pred and target both -> a
    # constant) drives the prediction loss to 0 and `pen` is the ONLY thing
    # resisting it -- it has to be weighted for gradient balance, not just so
    # the loss magnitudes match. ball_in_cup used 1.0 with this same penalty.
    # A first attempt at 2.5e-3 (~1/embed_dim) collapsed: z_std 0.36 -> 0.08
    # over 700 steps. Raise further if z_std still sags; lower toward ~0.3 if
    # early training is unstable (features pinned before prediction learns).
    kappa: float = 1.0
    # `kappa` regularizes the PATCH-TOKEN covariance only -- it stays satisfied
    # (z_std ~1) even as the mean-pooled per-image feature (what eval_probe uses)
    # collapses to ~1-2 effective dims. The v1 30-epoch run did exactly that:
    # frozen linear probe 0.62 vs 0.76 for random init, eff-rank 41 -> 23 over
    # epochs 11-28. `kappa_pooled` adds the same VICReg penalty on h.mean(1), so
    # the variance term protects between-image variance. 0.0 -> v1 behaviour.
    kappa_pooled: float = 1.0

    # 0 keeps the ~13.5 GB decompressed array in one process (see data.py). The
    # MPS ViT dominates, so extra workers buy little; if you raise this, a fork
    # context is used so the array is COW-shared, not pickled per worker.
    num_workers: int = 0
    log_every: int = 25          # steps between console/CSV lines
    max_steps: int = 0           # 0 = no cap (used by --smoke)
    # written next to this script, not the cwd -- so `rm i_jepa/jepa_ckpt.pt`
    # and running from any directory both behave predictably
    ckpt: str = os.path.join(_HERE, "jepa_ckpt.pt")
    logcsv: str = os.path.join(_HERE, "jepa_log.csv")
    # `ckpt` is overwritten every epoch; keep an encoder-only snapshot per epoch
    # here too so eval_probe.py / sweep_probe.py can chart probe acc vs step.
    # ~90 MB each (no predictor, no optimizer state). 0 disables.
    ckpt_archive_dir: str = os.path.join(_HERE, "ckpt_archive")
    archive_every: int = 1        # epochs between kept encoder snapshots


SMOKE = replace(
    Config(),
    embed_dim=64, depth=2, num_heads=4,
    pred_dim=32, pred_depth=2, pred_heads=4,
    batch_size=4, n_epochs=1, num_workers=0,
    log_every=1, max_steps=3,
    # own paths -- otherwise maybe_resume() tries to load the real ViT-S
    # jepa_ckpt.pt into this 64-dim toy and dies on a shape mismatch
    ckpt=os.path.join(_HERE, "jepa_smoke_ckpt.pt"),
    logcsv=os.path.join(_HERE, "jepa_smoke_log.csv"),
    archive_every=0,
)

# short fresh run to check the kappa_pooled fix: separate ckpt/log so it never
# touches the main run, no per-epoch archive. Watch img_effrank in jepa_diag_log.csv
# -- the v1 pooled eff-rank sat ~1.5 and the token z_std stayed ~1.0 throughout;
# if img_effrank now climbs and holds (tens, not ~2) the fix works -> full run.
# The trend is usually clear within the first epoch (~50 min), so 4 is plenty.
DIAG = replace(
    Config(),
    n_epochs=4,
    archive_every=0,
    ckpt=os.path.join(_HERE, "jepa_diag_ckpt.pt"),
    logcsv=os.path.join(_HERE, "jepa_diag_log.csv"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, steps_per_epoch: int, cfg: Config) -> float:
    """Linear warmup then cosine decay to zero, in units of optimizer steps."""
    warmup = cfg.warmup_steps
    total = cfg.n_epochs * steps_per_epoch
    if step < warmup:
        return cfg.base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * cfg.base_lr * (1.0 + math.cos(math.pi * min(1.0, prog)))


def build_models(cfg: Config, device: torch.device):
    encoder = ViT(
        img_size=cfg.img_size, patch_size=cfg.patch, in_chans=3,
        embed_dim=cfg.embed_dim, depth=cfg.depth, num_heads=cfg.num_heads,
    ).to(device)
    predictor = VisionTransformerPredictor(
        num_patches=encoder.patch_embed.num_patches, encoder_dim=cfg.embed_dim,
        predictor_dim=cfg.pred_dim, depth=cfg.pred_depth, num_heads=cfg.pred_heads,
    ).to(device)
    return encoder, predictor


def save_ckpt(path, encoder, predictor, opt, epoch, step):
    torch.save({
        "version": VERSION,
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch,
        "step": step,
    }, path)


def archive_encoder(cfg, encoder, epoch, step):
    """Encoder-only snapshot for the probe sweep. Same key layout eval_probe.py
    expects (encoder / epoch / step), minus the predictor and optimizer state."""
    if not cfg.archive_every or (epoch % cfg.archive_every):
        return
    os.makedirs(cfg.ckpt_archive_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_archive_dir, f"e{epoch:03d}_s{step}.pt")
    torch.save({"version": VERSION, "encoder": encoder.state_dict(),
                "epoch": epoch, "step": step}, path)
    print(f"  archived encoder -> {path}")


def maybe_resume(path, encoder, predictor, opt, device):
    if not os.path.exists(path):
        return 0, 0
    ck = torch.load(path, map_location=device)
    encoder.load_state_dict(ck["encoder"])
    predictor.load_state_dict(ck["predictor"])
    try:
        opt.load_state_dict(ck["opt"])
        print(f"resumed from {path}: epoch {ck['epoch']}, step {ck['step']}")
        return ck["epoch"], ck["step"]
    except (KeyError, ValueError):
        print(f"resumed weights from {path}, optimizer state skipped")
        return ck.get("epoch", 0), ck.get("step", 0)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(cfg: Config):
    device = pick_device()
    torch.manual_seed(0)
    print(f"I-JEPA pretrain {VERSION}  device={device}  "
          f"encoder=ViT(d{cfg.embed_dim}x{cfg.depth})  predictor=(d{cfg.pred_dim}x{cfg.pred_depth})  "
          f"batch={cfg.batch_size}  kappa={cfg.kappa}  kappa_pooled={cfg.kappa_pooled}")

    dataset = PathMNIST224("train", augment=True, img_size=cfg.img_size)
    collator = MultiBlockMaskCollator(
        input_size=cfg.img_size, patch_size=cfg.patch,
        pred_hw=cfg.pred_hw, n_pred=cfg.n_pred,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, collate_fn=collator,
        persistent_workers=cfg.num_workers > 0,
        # fork so workers COW-share the big in-RAM image array (macOS default
        # is spawn, which would pickle a full copy into every worker).
        multiprocessing_context="fork" if cfg.num_workers > 0 else None,
    )
    steps_per_epoch = len(loader)

    encoder, predictor = build_models(cfg, device)
    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)

    start_epoch, step = maybe_resume(cfg.ckpt, encoder, predictor, opt, device)

    new_log = not os.path.exists(cfg.logcsv)
    logf = open(cfg.logcsv, "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["epoch", "step", "loss", "inv", "pen", "pen_pool", "z_std",
                       "img_std", "img_effrank", "lr", "imgs_per_s"])

    encoder.train()
    predictor.train()
    win_t, win_imgs = time.time(), 0        # rolling throughput over one log window
    img_buf, img_buf_cap = [], 100          # ~100 * batch pooled features for img_effrank
    for epoch in range(start_epoch, cfg.n_epochs):
        for imgs, enc_idx, ctx_keep, pred_masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            enc_idx = enc_idx.to(device)
            ctx_keep = ctx_keep.to(device)
            pred_masks = [m.to(device) for m in pred_masks]

            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps_per_epoch, cfg)

            z_ctx = encoder(imgs, masks=enc_idx, key_pad=ctx_keep)  # (B, L, D)  -- context, grad
            h = encoder(imgs)                           # (B, N, D)  -- full image, grad (for VICReg)
            h_tgt = apply_masks(h.detach(), pred_masks)  # (n_pred*B, K_tgt, D)  -- target, STOP-GRAD
            p = predictor(z_ctx, enc_idx, pred_masks, ctx_keep=ctx_keep)  # (n_pred*B, K_tgt, D)

            h_img = h.mean(dim=1)                        # (B, D) per-image, what eval_probe uses
            loss, parts = ijepa_loss(
                p, h_tgt, reg_embeddings=[z_ctx[ctx_keep], h], kappa=cfg.kappa,
                pooled_embeddings=[h_img], kappa_pooled=cfg.kappa_pooled,
                predictor_loss="smooth_l1", target_layernorm=True)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            step += 1
            win_imgs += imgs.shape[0]

            # rolling buffer of per-image features -> a live read on the collapse
            # the v1 run only revealed after an overnight probe sweep
            img_buf.append(h_img.detach().float().cpu())
            if len(img_buf) > img_buf_cap:
                img_buf.pop(0)

            if step % cfg.log_every == 0:
                z_std = h.detach().reshape(-1, h.shape[-1]).std(dim=0).mean().item()
                pooled = torch.cat(img_buf)                       # (<=cap*B, D)
                img_std = pooled.std(dim=0).mean().item()
                sv = torch.linalg.svdvals(pooled - pooled.mean(dim=0))
                img_effrank = float((sv.sum() ** 2) / (sv ** 2).sum().clamp_min(1e-12))
                pen = float(parts.get("pen", 0.0))
                pen_pool = float(parts.get("pen_pool", 0.0))
                ips = win_imgs / max(1e-6, time.time() - win_t)  # avg over the window
                win_t, win_imgs = time.time(), 0
                lr = opt.param_groups[0]["lr"]
                print(f"e{epoch} s{step}  loss {parts['loss']:.3f}  "
                      f"inv {parts['inv']:.3f}  pen {pen:.3f}  "
                      f"pen_pool {pen_pool:.3f}  z_std {z_std:.4f}  "
                      f"img_std {img_std:.4f}  img_effrank {img_effrank:.1f}  "
                      f"lr {lr:.2e}  {ips:.0f} img/s")
                logw.writerow([epoch, step, f"{parts['loss']:.5f}", f"{parts['inv']:.5f}",
                               f"{pen:.5f}", f"{pen_pool:.5f}", f"{z_std:.5f}",
                               f"{img_std:.5f}", f"{img_effrank:.3f}", f"{lr:.3e}", f"{ips:.1f}"])
                logf.flush()

            if cfg.max_steps and step >= cfg.max_steps:
                print("max_steps reached (smoke) -- stopping")
                logf.close()
                return

        save_ckpt(cfg.ckpt, encoder, predictor, opt, epoch + 1, step)
        print(f"epoch {epoch} done -- checkpoint saved to {cfg.ckpt}")
        archive_encoder(cfg, encoder, epoch + 1, step)

    logf.close()


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        train(SMOKE)
    elif "--diag" in sys.argv:
        train(DIAG)
    else:
        train(Config())
