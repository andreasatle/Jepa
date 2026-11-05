"""
LeJEPA, one step at a time, with figures. Runs a few real training steps on a small batch
and stops after each stage of a step to show what it computed. Uses lejepa.py's own parts
(encoder, projector, loss, SIGReg), so what you see is what training does.

    uv run python tutorial_lejepa.py                                    # fresh encoder
    uv run python tutorial_lejepa.py --ckpt runs/lejepa-nozoom/ckpt.pt  # the trained one
    options: --steps N (default 3)  --batch B (default 32)  --save DIR (PNGs, no pauses)
             --every K: train K-1 steps silently between the shown ones (default 1: show
             every step), with lejepa.py's warmup, so a fresh encoder visibly learns.
             E.g. --steps 4 --every 100 shows steps 1, 101, 201, 301.

Each step has six stages:
  1. views      one image cut into 2 global + 6 local views (native resolution, no zoom)
  2. encoder    patches -> tokens -> one 192-number vector per view (mean of the tokens)
  3. matching   the projector's 64-number vectors, each image's center, the pull to it
  4. SIGReg     one random direction: the projected batch vs a bell curve, and the test
  5. SIGReg     all 1024 directions, and all 8 views
  6. update     the loss, one AdamW step, and the history so far
Close nothing: press Enter in the terminal to go on.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2

from data import IMAGENET_MEAN, IMAGENET_STD, MultiCrop, RawPathMNIST
from lejepa import Config, Projector, lejepa_loss
from loss import sigreg_loss
from model import Encoder
from probes.probe import LABELS
from ijepa import collapse_stats

CLASSES = ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth muscle",
           "normal mucosa", "stroma", "tumour epithelium"]


class LeJEPATutorial:
    def __init__(self, cfg: Config, batch: int, ckpt: str | None, save: str | None, seed: int):
        torch.manual_seed(seed)
        self.cfg, self.B, self.save = cfg, batch, save
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.encoder = Encoder(cfg.img_size, cfg.patch_size, 3, cfg.embed_dim, cfg.depth, cfg.num_heads)
        self.projector = Projector(cfg.embed_dim, cfg.proj_hidden, cfg.proj_dim)
        if ckpt:
            ck = torch.load(ckpt, map_location="cpu")
            self.encoder.load_state_dict(ck["encoder"])
            self.projector.load_state_dict(ck["projector"])
        self.encoder.to(self.device).train()
        self.projector.to(self.device).train()
        self.opt = torch.optim.AdamW([*self.encoder.parameters(), *self.projector.parameters()],
                                     lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.state = "trained (" + ckpt + ")" if ckpt else "fresh (random init)"

        data = RawPathMNIST("train", cfg.img_size)
        labels = np.load(LABELS)["train_labels"][:, 0]
        idx = torch.randperm(len(data))[:batch].tolist()
        tissue = next(k for k, i in enumerate(idx) if labels[i] in (3, 6, 7, 8))   # not empty-looking
        idx[0], idx[tissue] = idx[tissue], idx[0]
        self.images = [data[i] for i in idx]                       # (224, 224, 3) uint8
        self.labels = [int(labels[i]) for i in idx]
        self.augment = MultiCrop.view(v2.Identity())               # flips + colour, no resizing
        self.history: list[dict] = []
        self.figure_no = 0

    # -- helpers ---------------------------------------------------------------------------

    def show(self, fig, step: int, stage: str, notes: str):
        print(f"\n=== step {step}, {stage} ===\n{notes}")
        fig.tight_layout()
        if self.save:
            self.figure_no += 1
            fig.savefig(os.path.join(self.save, f"{self.figure_no:02d}_step{step}_{stage}.png"), dpi=90)
        else:
            fig.show()
            plt.pause(0.1)
            input("  [Enter] next")
        plt.close(fig)

    @staticmethod
    def picture(x: torch.Tensor) -> np.ndarray:
        """Normalised (3, H, W) back to a displayable (H, W, 3) image."""
        mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
        return (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    def make_views(self):
        """Exactly MultiCrop(zoom=False), but keeping where each local window came from."""
        cfg, L = self.cfg, self.cfg.local_size
        views = [[] for _ in range(cfg.global_views + cfg.local_views)]
        boxes = []
        for img in self.images:
            for v in range(cfg.global_views):
                views[v].append(self.augment(img))
            img_boxes = []
            for v in range(cfg.global_views, len(views)):
                y, x = (int(torch.randint(n - L + 1, ())) for n in img.shape[:2])
                views[v].append(self.augment(np.ascontiguousarray(img[y:y + L, x:x + L])))
                img_boxes.append((y, x))
            boxes.append(img_boxes)
        return [torch.stack(v) for v in views], boxes

    # -- one step --------------------------------------------------------------------------

    def step(self, step: int, show: bool, lr: float):
        cfg, B = self.cfg, self.B
        Vg = cfg.global_views

        # 1. views
        views, boxes = self.make_views()
        if not show:
            g_tok = self.encoder(torch.cat(views[:Vg]).to(self.device))
            h = torch.cat([g_tok.mean(1), self.encoder(torch.cat(views[Vg:]).to(self.device)).mean(1)])
            self.update(self.projector(h).reshape(-1, B, cfg.proj_dim), g_tok, step, lr)
            return
        fig, ax = plt.subplots(2, 5, figsize=(15, 6.4))
        ax[0, 0].imshow(self.images[0])
        for k, (y, x) in enumerate(boxes[0]):
            ax[0, 0].add_patch(plt.Rectangle((x, y), cfg.local_size, cfg.local_size, fill=False,
                                             ec=f"C{k}", lw=2))
        ax[0, 0].set_title(f"original: {CLASSES[self.labels[0]]}\nboxes = the 6 local windows")
        for v in range(len(views)):
            a = ax.flat[v + 1]
            a.imshow(self.picture(views[v][0]))
            kind = "global" if v < Vg else f"local {v - Vg + 1}"
            a.set_title(f"view {v}: {kind}, {views[v].shape[-1]} px")
            if v >= Vg:
                for s in a.spines.values():
                    s.set(ec=f"C{v - Vg}", lw=3)
        for a in ax.flat:
            a.set_xticks([]), a.set_yticks([])
        ax.flat[-1].set_visible(False)
        self.show(fig, step, "1-views",
                  "One image, 8 views. Globals: the whole image, flipped and colour-shifted.\n"
                  "Locals: 96 px windows at native resolution (a cell keeps its size; they only\n"
                  "look bigger here because every panel is drawn the same size). Some views are\n"
                  "grey, solarized or oddly coloured: colour must not decide 'same image'.\n"
                  "Training will ask all 8 to land on one point: 'same image'.")

        # 2. encoder
        g = torch.cat(views[:Vg]).to(self.device)
        l = torch.cat(views[Vg:]).to(self.device)
        g_tok, l_tok = self.encoder(g), self.encoder(l)            # (Vg*B, 196, D), (Vl*B, 36, D)
        h = torch.cat([g_tok.mean(1), l_tok.mean(1)])              # (V*B, D)
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1.4, 1, 1.4]})
        for a, img, tok, n in [(ax[0], views[0][0], g_tok[0], cfg.img_size), (ax[2], views[Vg][0], l_tok[0],
                                                                               cfg.local_size)]:
            a.imshow(self.picture(img))
            for p in range(0, n + 1, cfg.patch_size):
                a.axhline(p - 0.5, c="w", lw=0.4), a.axvline(p - 0.5, c="w", lw=0.4)
            grid = n // cfg.patch_size
            a.set_title(f"{n} px: {grid} x {grid} = {grid * grid} patches")
            a.set_xticks([]), a.set_yticks([])
        for a, tok, name in [(ax[1], g_tok[0], "global"), (ax[3], l_tok[0], "local")]:
            t = tok.detach().cpu()
            a.imshow(torch.cat([t, t.mean(0, keepdim=True).expand(4, -1)]), aspect="auto", cmap="RdBu_r",
                     vmin=-2, vmax=2)
            a.axhline(len(t) - 0.5, c="k", lw=2)
            a.set_title(f"{name}: {len(t)} tokens x 192 numbers\nbottom strip = their mean (what we keep)")
            a.set_xlabel("embedding dimension"), a.set_ylabel("token (patch)")
        self.show(fig, step, "2-encoder",
                  "The ViT turns each 16x16 patch into 192 numbers, letting every patch look at all\n"
                  "the others (12 attention layers). Global views give 196 tokens, local 36; the\n"
                  "same weights handle both. The mean over tokens is the view's embedding h: the\n"
                  "vector the probe uses later.")

        # 3. projector + matching
        z = self.projector(h).reshape(-1, B, cfg.proj_dim)         # (V, B, 64)
        centers = z[:Vg].mean(0)                                   # (B, 64)
        show = min(8, B)
        pts = torch.cat([z[:, :show].reshape(-1, cfg.proj_dim), centers[:show]]).detach().cpu()
        pts = pts - pts.mean(0)
        _, _, vt = torch.linalg.svd(pts, full_matrices=False)
        p2 = pts @ vt[:2].T                                        # PCA to 2-D
        pv, pc = p2[:-show].reshape(len(z), show, 2), p2[-show:]
        dist = (centers - z).square().mean(-1).detach().cpu()      # (V, B)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        for i in range(show):
            c = f"C{i}"
            for v in range(len(z)):
                ax[0].plot(*zip(pv[v, i], pc[i]), c=c, lw=0.6, alpha=0.5)
            ax[0].scatter(*pv[:Vg, i].T, c=c, marker="o", s=40)
            ax[0].scatter(*pv[Vg:, i].T, c=c, marker="x", s=30)
            ax[0].scatter(*pc[i], c=c, marker="*", s=250, ec="k", label=CLASSES[self.labels[i]])
        ax[0].legend(fontsize=7, loc="best")
        ax[0].set_title(f"projector output (64-D, PCA to 2-D), {show} images\n"
                        "o global, x local, * center (mean of the globals); lines = the pull")
        ax[1].hist([dist[:Vg].flatten(), dist[Vg:].flatten()], bins=20, label=["global", "local"])
        ax[1].set_title(f"squared distance to the center, all {B} images\n"
                        f"matching loss = their mean = {dist.mean():.4f}")
        ax[1].legend()
        self.show(fig, step, "3-matching",
                  "The projector squeezes h (192) to z (64); the loss only sees z. Each image's\n"
                  "center is the mean of its 2 global views. Matching = mean squared distance of\n"
                  "all 8 views to their center. Alone it would win by putting everything on one\n"
                  "point (collapse); SIGReg forbids that.")

        # 4. SIGReg, one direction
        zv = z[0]                                                  # view 0's batch: (B, 64)
        a = torch.randn(cfg.proj_dim, cfg.sigreg_slices, device=zv.device)
        a = a / a.norm(dim=0, keepdim=True)
        x = (zv @ a).detach().cpu()                                # (B, S)
        t = torch.linspace(0, 5, 17)
        tx = x[:, :, None] * t
        re = torch.cos(tx).mean(0) - torch.exp(-t ** 2 / 2)        # (S, T)
        im = torch.sin(tx).mean(0)
        integrand = (re ** 2 + im ** 2) * torch.exp(-t ** 2)
        ep = B * 2 * torch.trapezoid(integrand, t, dim=-1)         # (S,) one test per direction
        s0 = int(ep.argsort()[len(ep) // 2])                       # a typical (median) direction
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
        grid = torch.linspace(-4, 4, 200)
        ax[0].hist(x[:, s0], bins=15, density=True, alpha=0.6, label=f"{B} images, projected")
        ax[0].plot(grid, torch.exp(-grid ** 2 / 2) / (2 * np.pi) ** 0.5, "k", label="N(0, 1)")
        ax[0].set_title("view 0's batch on one random direction"), ax[0].legend()
        tt = torch.linspace(0, 5, 200)
        txx = x[:, s0, None] * tt
        ax[1].plot(tt, torch.cos(txx).mean(0), label="mean cos(t x)  (real part)")
        ax[1].plot(tt, torch.sin(txx).mean(0), label="mean sin(t x)  (imaginary part)")
        ax[1].plot(tt, torch.exp(-tt ** 2 / 2), "k--", label="Gaussian: exp(-t^2/2), and 0")
        ax[1].plot(tt, torch.exp(-tt ** 2), "k:", alpha=0.5, label="weight exp(-t^2)")
        ax[1].scatter(t, torch.zeros_like(t), marker="|", c="k", s=80, label="the 17 points")
        ax[1].set_title("characteristic function: sample vs bell curve"), ax[1].set_xlabel("t")
        ax[1].legend(fontsize=7)
        ax[2].fill_between(t, integrand[s0], alpha=0.5)
        ax[2].plot(t, integrand[s0], "o-")
        ax[2].set_title(f"weighted squared gap; area x 2 x {B} = test value {ep[s0]:.2f}")
        ax[2].set_xlabel("t")
        self.show(fig, step, "4-sigreg-one-direction",
                  f"Pick one random direction in the 64-D space and project the {B} images onto it:\n"
                  f"{B} plain numbers. Are they a sample from N(0, 1)? Epps-Pulley compares the\n"
                  "averages of cos(tx) and sin(tx) with a true Gaussian's, weighted to small t\n"
                  "(coarse shape); the shaded area is the test value for this direction.")

        # 5. SIGReg, all directions and views
        torch.manual_seed(1000 + step)
        per_view = torch.stack([sigreg_loss(zz, cfg.sigreg_slices) for zz in z])   # what training uses
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
        ax[0].hist(ep, bins=40)
        ax[0].axvline(ep.mean(), c="r", label=f"mean = SIGReg of view 0 = {ep.mean():.2f}")
        ax[0].axvline(ep[s0], c="k", ls="--", label="the direction shown before")
        ax[0].set_title(f"test value for each of the {cfg.sigreg_slices} random directions (view 0)")
        ax[0].legend()
        ax[1].bar(range(len(z)), per_view.detach().cpu(), color=["C0"] * Vg + ["C1"] * cfg.local_views)
        ax[1].axhline(per_view.mean().item(), c="r", label=f"mean = SIGReg = {per_view.mean():.2f}")
        ax[1].set_xticks(range(len(z)), ["g", "g"] + ["l"] * cfg.local_views)
        ax[1].set_title("SIGReg per view (fresh directions each), blue global, orange local")
        ax[1].legend()
        self.show(fig, step, "5-sigreg-all",
                  "Average the test over all directions: that is one view's SIGReg. Every view gets\n"
                  "its own fresh directions (redrawn every step, so no direction can be cheated).\n"
                  f"For scale, {B} truly Gaussian points score about 0.5; clusters or collapse, 8+.")

        # 6. loss and update
        parts, grads = self.update(z, g_tok, step, lr)
        lam = cfg.sigreg_weight
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
        ax[0].bar(["(1-λ) x matching", "λ x SIGReg"], [(1 - lam) * parts["pred"].item(),
                                                         lam * parts["sig"].item()], color=["C0", "C3"])
        ax[0].set_title(f"loss = {parts['loss']:.4f}   (λ = {lam})")
        ax[1].bar(list(grads), [float(v) for v in grads.values()])
        ax[1].set_title("gradient size, then one AdamW step\n(no EMA, no stop-gradient: all of it trains)")
        steps = [r["step"] for r in self.history]
        marker = "o-" if len(steps) < 30 else "-"
        ax[2].plot(steps, [r["pred"] for r in self.history], marker, label="matching")
        ax[2].plot(steps, [r["sig"] / 10 for r in self.history], marker, label="SIGReg / 10")
        ax2 = ax[2].twinx()
        ax2.plot(steps, [r["rank"] for r in self.history], "k--", label="effective rank of h")
        ax[2].xaxis.get_major_locator().set_params(integer=True)
        ax[2].set_xlabel("step"), ax[2].legend(loc="upper left"), ax2.legend(loc="upper right")
        ax[2].set_title(f"history, lr {lr:.1e}\n(10-epoch run: matching 0.25 -> 0.07, SIGReg 7.9 -> 1.5)")
        self.show(fig, step, "6-update",
                  "Total = 0.95 x matching + 0.05 x SIGReg, one backward pass through projector and\n"
                  "encoder, one AdamW step. The two terms pull against each other: matching wants\n"
                  "views together, SIGReg wants the batch spread like a bell curve. Only an embedding\n"
                  "that groups views of the same image AND spreads images apart satisfies both.\n"
                  "From scratch (--every 100): matching drops within ~20 steps (easy: everything close\n"
                  "together), then SIGReg slowly forces the spread up (rank rises) and matching gives\n"
                  "a little back. That tug of war is the training.")


    def update(self, z: torch.Tensor, g_tok: torch.Tensor, step: int, lr: float):
        """The loss, one backward pass, one AdamW step; returns the loss parts and gradient sizes."""
        loss, parts = lejepa_loss(z[:self.cfg.global_views], z, self.cfg.sigreg_weight, self.cfg.sigreg_slices)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = {"encoder": sum(p.grad.norm() ** 2 for p in self.encoder.parameters()) ** 0.5,
                 "projector": sum(p.grad.norm() ** 2 for p in self.projector.parameters()) ** 0.5}
        self.opt.step()
        stats = collapse_stats(g_tok[:self.B].detach())
        self.history.append({k: float(v) for k, v in parts.items()} | {"rank": stats["img_rank"], "step": step})
        return parts, grads


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="a lejepa.py ckpt.pt (encoder + projector); default: random init")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--save", help="write the figures to this folder instead of pausing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="show every K-th step, train the rest silently")
    args = ap.parse_args()
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        plt.switch_backend("agg")
    else:
        plt.ion()
    tutorial = LeJEPATutorial(Config(), args.batch, args.ckpt, args.save, args.seed)
    print(f"LeJEPA tutorial: {tutorial.state} encoder, batch {args.batch}, {args.steps} steps")
    total = (args.steps - 1) * args.every + 1
    warmup = 0 if args.ckpt else min(100, total // 3)       # a trained encoder needs no warmup
    for s in range(1, total + 1):
        lr = tutorial.cfg.lr * (0.01 + 0.99 * s / warmup) if s < warmup else tutorial.cfg.lr
        show = (s - 1) % args.every == 0
        if not show and s % 20 == 0:
            print(f"  ... step {s}: matching {tutorial.history[-1]['pred']:.3f}  SIGReg {tutorial.history[-1]['sig']:.2f}")
        tutorial.step(s, show, lr)
