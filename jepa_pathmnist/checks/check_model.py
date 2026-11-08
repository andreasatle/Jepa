"""Checks for model.py: shapes, gradients, hidden patches invisible, padding has no effect.

    uv run python checks/check_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from masks import MultiBlockMasks
from model import Encoder, Predictor, apply_masks


def main():
    torch.manual_seed(0)
    B, img, patch, De = 4, 64, 8, 64
    enc = Encoder(img_size=img, patch_size=patch, in_chans=3, embed_dim=De, depth=2, num_heads=4)
    pred = Predictor(num_patches=enc.num_patches, encoder_dim=De, predictor_dim=32,
                     depth=2, num_heads=4)
    G, N = enc.grid, enc.num_patches
    imgs = torch.randn(B, 3, img, img)
    ctx_mask, ctx_keep, tgt_masks = MultiBlockMasks(G, seed=0).sample(B)
    L, K = ctx_mask.shape[1], tgt_masks[0].shape[1]
    assert not ctx_keep.all(), "want at least one padded context for this test"

    h = enc(imgs)
    z_ctx = enc(imgs, ctx_mask, ctx_keep)
    p = pred(z_ctx, ctx_mask, ctx_keep, tgt_masks)
    h_tgt = apply_masks(h, tgt_masks)
    assert h.shape == (B, N, De) and z_ctx.shape == (B, L, De)
    assert p.shape == h_tgt.shape == (4 * B, K, De)

    def blank(x, idx):
        """Zero the pixels of patches `idx` in image x (3, H, W)."""
        x = x.clone()
        for i in idx:
            r, c = divmod(int(i), G)
            x[:, r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = 0
        return x

    # 1. The context encoder never sees target patches: blank them, real outputs unchanged.
    tgt_all = torch.cat([m for m in tgt_masks], dim=1)
    imgs2 = torch.stack([blank(imgs[b], tgt_all[b]) for b in range(B)])
    z2 = enc(imgs2, ctx_mask, ctx_keep)
    assert torch.allclose(z2[ctx_keep], z_ctx[ctx_keep], atol=1e-5), "context encoder sees targets"

    # 2. Padding has no effect: point every padded slot at a TARGET patch (so if padding
    #    leaked, target content would reach the context), and add 5 extra padded slots.
    pad_mask = ctx_mask.clone()
    pad_mask[~ctx_keep] = tgt_all[:, :1].expand(-1, L)[~ctx_keep]
    pad_mask = torch.cat([pad_mask, tgt_all[:, :5]], dim=1)
    pad_keep = torch.cat([ctx_keep, torch.zeros(B, 5, dtype=torch.bool)], dim=1)
    z3 = enc(imgs, pad_mask, pad_keep)
    assert torch.allclose(z3[pad_keep], z_ctx[ctx_keep], atol=1e-5), "encoder: padding leaks"
    z3 = z3 + 100 * torch.randn_like(z3) * (~pad_keep)[..., None]     # garbage in padded slots
    p3 = pred(z3, pad_mask, pad_keep, tgt_masks)
    assert torch.allclose(p3, p, atol=1e-5), "predictor: padding leaks"

    # 3. Gradients reach the predictor and the encoder.
    ((p - h_tgt.detach()) ** 2).mean().backward()
    assert pred.mask_token.grad is not None and enc.patch_embed.weight.grad is not None
    print(f"h {tuple(h.shape)}  z_ctx {tuple(z_ctx.shape)} "
          f"(real {ctx_keep.sum(1).tolist()})  pred {tuple(p.shape)}")
    print("OK")


if __name__ == "__main__":
    main()
