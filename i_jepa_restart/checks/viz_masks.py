"""
Show multi-block masks on real PathMNIST images, and check that a patch index
means the same patch everywhere.

Index convention (masks.py / model.py): patch i sits at row i // grid, col i % grid.

Checks, all against pixels rather than against the code's own bookkeeping:
  1. Encoder patch order: an "index image" whose patch (r, c) is filled with the
     value r * grid + c goes through the encoder's patch embedding (weights set
     to average the patch); token i must come out as i.
  2. apply_masks on those tokens returns exactly the mask's indices, so the
     tokens the model gathers are the patches drawn in the picture.
  3. Each target mask is a filled h x w rectangle (read back from the index
     image), all 4 the same size.
  4. The real context patches (read back from the index image) are exactly the
     grid minus the union of the 4 targets.

Picture per sample: the image, the context the encoder sees (targets greyed
out -- nothing else), and the 4 target blocks in colour.

    uv run python checks/viz_masks.py        -> checks/masks.png
"""

import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import IMAGENET_MEAN, IMAGENET_STD, PathMNIST
from masks import MultiBlockMasks
from model import Encoder, apply_masks

IMG, PATCH = 224, 16
GRID = IMG // PATCH
N = GRID * GRID
COLORS = np.array([[230, 60, 60], [60, 160, 230], [240, 200, 40], [150, 80, 220]]) / 255


def index_image(b: int) -> torch.Tensor:
    """(b, 3, IMG, IMG) image: every pixel of patch (r, c) equals r * GRID + c."""
    idx = torch.arange(N, dtype=torch.float32).reshape(GRID, GRID)
    img = idx.repeat_interleave(PATCH, 0).repeat_interleave(PATCH, 1)
    return img.expand(b, 3, IMG, IMG).clone()


def validate(ctx_mask, ctx_keep, tgt_masks):
    B = ctx_mask.shape[0]
    enc = Encoder(IMG, PATCH, 3, embed_dim=4, depth=1, num_heads=1)
    with torch.no_grad():
        enc.patch_embed.weight.fill_(1.0 / (3 * PATCH * PATCH))   # every output = patch mean
        enc.patch_embed.bias.zero_()
        tokens = enc.patch_embed(index_image(B)).flatten(2).transpose(1, 2)   # as in Encoder.forward
    # 1. token i is patch i
    assert torch.allclose(tokens[..., 0], torch.arange(N).float().expand(B, -1)), \
        "encoder patch order != row * grid + col"
    # 2. gathered tokens are the patches the masks name; read indices back from pixels
    ctx_px = apply_masks(tokens, ctx_mask)[..., 0].round().long()
    tgt_px = [apply_masks(tokens, m)[..., 0].round().long() for m in tgt_masks]
    assert torch.equal(ctx_px[ctx_keep], ctx_mask[ctx_keep])
    assert all(torch.equal(p, m) for p, m in zip(tgt_px, tgt_masks))
    shapes = set()
    for b in range(B):
        # 3. targets: filled rectangles, same size
        tgt = set()
        for p in tgt_px:
            r, c = p[b] // GRID, p[b] % GRID
            h, w = int(r.max() - r.min() + 1), int(c.max() - c.min() + 1)
            assert h * w == len(p[b]) == len(set(p[b].tolist())), "target is not a filled rectangle"
            shapes.add((h, w))
            tgt |= set(p[b].tolist())
        # 4. context: exactly grid minus targets
        ctx = ctx_px[b][ctx_keep[b]].tolist()
        assert len(ctx) == len(set(ctx)) and set(ctx) == set(range(N)) - tgt, \
            "context != grid minus targets"
    assert len(shapes) == 1, f"target sizes differ within the batch: {shapes}"
    return shapes.pop()


def to_rgb(x: torch.Tensor) -> np.ndarray:
    mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
    return (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def patch_slice(i: int):
    r, c = divmod(int(i), GRID)
    return slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH)


def main(n: int = 6, seed: int = 1):
    ctx_mask, ctx_keep, tgt_masks = MultiBlockMasks(GRID, seed=seed).sample(n)
    h, w = validate(ctx_mask, ctx_keep, tgt_masks)
    n_ctx = ctx_keep.sum(1).tolist()
    print(f"target blocks {h}x{w}, context sizes {n_ctx} of {N} -- all checks OK")

    torch.manual_seed(seed)
    ds = PathMNIST("test", img_size=IMG)
    imgs = [ds[i] for i in torch.randint(len(ds), (n,)).tolist()]

    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.6 * n))
    for b in range(n):
        rgb = to_rgb(imgs[b])
        ctx_view = rgb * 0.25 + 0.55                              # grey out everything...
        for i in ctx_mask[b][ctx_keep[b]]:
            ctx_view[patch_slice(i)] = rgb[patch_slice(i)]        # ...except the context
        tgt_view = rgb * 0.35 + 0.4
        for j, m in enumerate(tgt_masks):
            for i in m[b]:
                s = patch_slice(i)
                tgt_view[s] = 0.45 * rgb[s] + 0.55 * COLORS[j]
        for ax, im, title in zip(axes[b], [rgb, ctx_view, tgt_view],
                                 ["image", f"context ({n_ctx[b]} patches)",
                                  f"4 targets ({h}x{w}, may overlap)"]):
            ax.imshow(im)
            ax.set_xticks(np.arange(0, IMG + 1, PATCH) - 0.5, minor=True)
            ax.set_yticks(np.arange(0, IMG + 1, PATCH) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.5)
            ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
            ax.set_title(title if b == 0 else title.split(" (")[0] + f" ({n_ctx[b]} patches)"
                         if "context" in title else "", fontsize=9)
    fig.suptitle(f"I-JEPA multi-block masks, {GRID}x{GRID} grid of {PATCH}px patches", fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
