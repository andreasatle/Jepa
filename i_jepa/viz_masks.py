"""
Visualize what the mask sampler hands the encoder: for a few PathMNIST
images, show the context block and the 4 target blocks on the 14x14 patch
grid.

    col 1  image
    col 2  context  -- patches the context encoder sees; the rest dimmed
    col 3  targets   -- the 4 blocks the predictor must reproduce, one color
                        each (they may overlap each other; they never overlap
                        the context)

Uses split='test' (~1 GB) so it's cheap to run alongside training. Masking
works identically on the augmented random-resized crop during training; aug
is off here just for a clean view.
"""

import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import PathMNIST224
from masks import MultiBlockMaskCollator

N = 6
PATCH = 16
GRID = 224 // PATCH
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask_examples.png")


def patch_grid(idx_row: torch.Tensor) -> np.ndarray:
    g = np.zeros(GRID * GRID, dtype=bool)
    g[idx_row.numpy()] = True
    return g.reshape(GRID, GRID)


def upsample(mask_14: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(mask_14, PATCH, axis=0), PATCH, axis=1)


def main():
    ds = PathMNIST224("test", augment=False)
    sel = torch.randperm(len(ds))[:N]
    imgs = torch.stack([ds[int(i)] for i in sel])                 # (N,3,224,224) in [-1,1]
    disp = (imgs * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    col = MultiBlockMaskCollator(input_size=224, patch_size=PATCH, n_pred=4)
    enc_idx, ctx_keep, pred_masks = col.sample(N, generator=torch.Generator().manual_seed(0))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:4, :3]

    k_tgt = pred_masks[0].shape[1]
    ctx_lens = ctx_keep.sum(1).tolist()
    print(f"context: {ctx_lens} / {GRID*GRID} patches (exact complement)   "
          f"targets: 4 blocks, {k_tgt} patches each")
    print(f"example 0 context idx: {sorted(enc_idx[0][ctx_keep[0]].tolist())}")
    for b, pm in enumerate(pred_masks):
        print(f"example 0 target {b} idx: {sorted(pm[0].tolist())}")

    fig, axes = plt.subplots(N, 3, figsize=(8.5, 2.8 * N))
    for r in range(N):
        base = disp[r]

        ctx = upsample(patch_grid(enc_idx[r][ctx_keep[r]]))
        ctx_img = base.copy()
        ctx_img[~ctx] *= 0.15

        tgt_img = base * 0.25
        for b, pm in enumerate(pred_masks):
            m = upsample(patch_grid(pm[r]))
            tgt_img[m] = base[m] * 0.55 + colors[b] * 0.45

        for c, (im, ttl) in enumerate((
            (base, "image"),
            (ctx_img, f"context  ({ctx_lens[r]} patches)"),
            (tgt_img.clip(0, 1), f"4 targets  ({k_tgt} each)"),
        )):
            axes[r, c].imshow(im)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(ttl, fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
