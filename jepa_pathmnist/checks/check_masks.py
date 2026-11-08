"""Checks for masks.py: targets are whole h x w patch blocks; context is exactly grid minus targets.

    uv run python checks/check_masks.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from masks import MultiBlockMasks


def main():
    grid, B = 14, 256
    N = grid * grid
    masks = MultiBlockMasks(grid)

    for _ in range(20):
        ctx_mask, ctx_keep, tgt_masks = masks.sample(B)
        assert len(tgt_masks) == 4 and ctx_mask.shape == ctx_keep.shape
        sizes = set()
        for b in range(B):
            tgt = set()
            for m in tgt_masks:
                r, c = m[b] // grid, m[b] % grid
                h, w = int(r.max() - r.min() + 1), int(c.max() - c.min() + 1)
                assert h * w == len(m[b]) == len(set(m[b].tolist())), "target is not a filled h x w block"
                sizes.add((h, w))
                tgt |= set(m[b].tolist())
            keep = ctx_keep[b]
            n_real = int(keep.sum())
            assert keep[:n_real].all() and not keep[n_real:].any(), "padding not at the end"
            ctx = ctx_mask[b][keep].tolist()
            assert len(ctx) == len(set(ctx)), "duplicate context patch"
            assert set(ctx) == set(range(N)) - tgt, "context != grid minus targets"
        assert len(sizes) == 1, f"target size differs within a batch: {sizes}"

    lens = ctx_keep.sum(1).float()
    print(f"target blocks {tgt_masks[0].shape[1]} patches = {tgt_masks[0].shape[1] / N:.0%} of grid; "
          f"context {lens.min():.0f}-{lens.max():.0f} patches (mean {lens.mean():.0f}), "
          f"padded to {ctx_mask.shape[1]}")

    # every patch is a target sometimes (positions cover the whole grid)
    seen = set()
    for _ in range(50):
        _, _, ts = masks.sample(8)
        seen |= set(torch.cat(ts).flatten().tolist())
    assert len(seen) == N

    # one picture: . context   # target
    img = [["?"] * grid for _ in range(grid)]
    for i in ctx_mask[0][ctx_keep[0]].tolist():
        img[i // grid][i % grid] = "."
    for m in tgt_masks:
        for i in m[0].tolist():
            img[i // grid][i % grid] = "#"
    print("\n".join(" ".join(r) for r in img))
    assert "?" not in "".join("".join(r) for r in img), "a patch is neither context nor target"
    print("OK")


if __name__ == "__main__":
    main()
