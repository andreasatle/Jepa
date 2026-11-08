"""
One I-JEPA training step on one real image, drawn step by step.

The masks come from masks.py and the shapes printed in the figure come from
running the actual (untrained) model.py networks on this image -- nothing in
the figure is typed in by hand.

    uv run python checks/viz_pipeline.py        -> checks/pipeline.png
"""

import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import PathMNIST
from masks import MultiBlockMasks
from model import Encoder, Predictor, apply_masks
from viz_masks import COLORS, GRID, IMG, N, PATCH, patch_slice, to_rgb, validate

SEED = 1


def style(ax, title):
    ax.set_xticks(np.arange(0, IMG + 1, PATCH) - 0.5, minor=True)
    ax.set_yticks(np.arange(0, IMG + 1, PATCH) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.6)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=8.5, loc="left")


def main():
    # Same sample as viz_masks.py (first image of seed 1): 103 context patches.
    ctx_mask, ctx_keep, tgt_masks = MultiBlockMasks(GRID, seed=SEED).sample(6)
    validate(ctx_mask, ctx_keep, tgt_masks)
    torch.manual_seed(SEED)
    ds = PathMNIST("test", img_size=IMG)
    img = ds[torch.randint(len(ds), (6,)).tolist()[0]]

    ctx = ctx_mask[0][ctx_keep[0]]                       # this image's real context, no padding
    tgts = [m[0] for m in tgt_masks]
    union = torch.unique(torch.cat(tgts))

    # Run the real networks (untrained) on this one image to get the true shapes.
    enc = Encoder(IMG, PATCH, 3, 192, 12, 3).eval()
    pred = Predictor(N, 192, 96, 6, 3).eval()
    with torch.no_grad():
        x = img[None]
        h = enc(x)                                                   # full image
        z_ctx = enc(x, ctx[None], torch.ones(1, len(ctx), dtype=torch.bool))
        p = pred(z_ctx, ctx[None], torch.ones(1, len(ctx), dtype=torch.bool),
                 [t[None] for t in tgts])
        truth = apply_masks(h, [t[None] for t in tgts])
    assert len(ctx) + len(union) == N
    shp = lambda t: "×".join(str(s) for s in t.shape)

    rgb = to_rgb(img)
    grey = rgb * 0.25 + 0.55

    fig, axes = plt.subplots(3, 4, figsize=(13, 10.4))

    # Row 1: the inputs
    ax = axes[0, 0]
    ax.imshow(rgb)
    style(ax, f"1. Image, cut into {GRID}×{GRID} = {N} patches\n    of {PATCH}×{PATCH} pixels")

    ax = axes[0, 1]
    v = rgb * 0.35 + 0.4
    for j, t in enumerate(tgts):
        for i in t:
            v[patch_slice(i)] = 0.45 * rgb[patch_slice(i)] + 0.55 * COLORS[j]
    ax.imshow(v)
    style(ax, f"2. Sample 4 target blocks of {len(tgts[0])} patches\n"
              f"    ({4 * len(tgts[0])} total, {len(union)} distinct: they overlap)")

    ax = axes[0, 2]
    v = grey.copy()
    for i in ctx:
        v[patch_slice(i)] = rgb[patch_slice(i)]
    ax.imshow(v)
    style(ax, f"3. Context = the other {N} − {len(union)} = {len(ctx)}\n"
              f"    patches → context encoder: {shp(z_ctx)}")

    ax = axes[0, 3]
    ax.imshow(rgb)
    style(ax, f"4. Full image, all {N} patches → encoder\n"
              f"    (no gradient to the target): {shp(h)}")

    # Row 2: the predictor, once per target block
    for j, t in enumerate(tgts):
        ax = axes[1, j]
        v = grey.copy()
        for i in ctx:
            v[patch_slice(i)] = rgb[patch_slice(i)]           # what the predictor has: context
        for i in t:
            v[patch_slice(i)] = 0.2 * grey[patch_slice(i)] + 0.8 * COLORS[j]   # asked: fill these in
        ax.imshow(v)
        style(ax, f"5.{j + 1} Predictor run {j + 1}: {len(ctx)} context\n"
                  f"    embeddings + {len(t)} '?' tokens → {shp(p[j])}")

    # Row 3: the loss, per block
    for j, t in enumerate(tgts):
        ax = axes[2, j]
        v = rgb * 0.3 + 0.45
        for i in t:
            v[patch_slice(i)] = rgb[patch_slice(i)]           # truth: these patches' embeddings in step 4
        ax.imshow(v)
        err = torch.nn.functional.smooth_l1_loss(
            p[j], torch.nn.functional.layer_norm(truth[j], (192,))).item()
        style(ax, f"6.{j + 1} Loss: prediction {shp(p[j])} vs\n"
                  f"    full-image embedding {shp(truth[j])}   ({err:.3f})")

    fig.suptitle(
        f"One I-JEPA step on one image: context {len(ctx)} + targets {len(union)} = {N} patches.   "
        f"Predictions: 4 runs × {len(tgts[0])} = {4 * len(tgts[0])}.   (Untrained model; loss values meaningless)",
        fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.png")
    fig.savefig(out, dpi=100)
    print(f"context {shp(z_ctx)}, full {shp(h)}, predictions {shp(p)}, truth {shp(truth)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
