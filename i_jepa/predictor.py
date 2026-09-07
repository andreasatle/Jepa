"""
I-JEPA predictor.

A narrow Transformer that takes the context encoder's output tokens (at
their real grid positions) plus a learned mask token stamped at every target
position, and predicts the target encoder's representation at those target
positions -- entirely in latent space, no pixels.

Deliberately narrower than the encoder (I-JEPA uses predictor_dim ~=
encoder_dim / 2 and ~6 layers): its whole job is "what would E have put
here, given the context", not to re-derive image features.

Shapes line up with masks.py and vit.apply_masks:

    context encoder:  z_ctx = vit(imgs, masks=enc_idx, key_pad=ctx_keep)  -> (B, L, De)
    target encoder:   h     = vit(imgs)                                   -> (B, N, De)
                      h_tgt = apply_masks(h, pred_masks)                  -> (n_pred*B, K_tgt, De)
    predictor:        p     = predictor(z_ctx, enc_idx, pred_masks, ctx_keep=ctx_keep)
                                                                          -> (n_pred*B, K_tgt, De)

so the loss is just `p` vs `h_tgt` (see loss.py).

The context is variable-length (grid minus the target union), right-padded
to the batch max; `ctx_keep` marks the real slots and is threaded into
attention so padding never reaches the target queries. Single context block
only.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vit import TransformerBlock, apply_masks, get_2d_sincos_pos_embed


class VisionTransformerPredictor(nn.Module):
    def __init__(
        self,
        num_patches: int,
        encoder_dim: int = 768,
        predictor_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        grid = int(round(num_patches ** 0.5))
        assert grid * grid == num_patches, "num_patches must be a perfect square"

        # encoder-dim tokens -> predictor-dim working space, and back again
        self.predictor_embed = nn.Linear(encoder_dim, predictor_dim, bias=True)
        self.predictor_proj = nn.Linear(predictor_dim, encoder_dim, bias=True)

        # the one learned parameter that stands in for "unknown patch here";
        # the same vector at every target slot, distinguished only by the
        # positional embedding added to it
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # fixed sincos positions, predictor width, gathered per-index at runtime
        pos_embed = get_2d_sincos_pos_embed(predictor_dim, grid, grid)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)  # (1, N, Dp)

        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.predictor_norm = nn.LayerNorm(predictor_dim)

    def forward(self, z_ctx: torch.Tensor, masks_ctx, masks_tgt,
                ctx_keep: torch.Tensor = None) -> torch.Tensor:
        """
        z_ctx:     (B, K_ctx, encoder_dim) -- context encoder output.
        masks_ctx: the context mask(s) (list of one (B, K_ctx) LongTensor, or
                   the bare tensor) -- the grid positions z_ctx came from.
        masks_tgt: the target masks (list of n_pred (B, K_tgt) LongTensors)
                   -- the grid positions to predict.
        ctx_keep:  optional (B, K_ctx) bool, True = real context token, False =
                   padding. When given, attention ignores the padded context
                   slots as keys (they never influence the target queries).
        returns:   (n_pred*B, K_tgt, encoder_dim), same layout as
                   apply_masks(target_encoder_output, masks_tgt).
        """
        if not isinstance(masks_ctx, (list, tuple)):
            masks_ctx = [masks_ctx]
        if not isinstance(masks_tgt, (list, tuple)):
            masks_tgt = [masks_tgt]
        assert len(masks_ctx) == 1, "this predictor supports a single context block (n_enc=1)"

        B = z_ctx.shape[0]
        n_tgt = len(masks_tgt)
        pos = self.pos_embed.expand(B, -1, -1)  # (B, N, Dp)

        # 1. project context tokens to predictor width, add their positions back
        x = self.predictor_embed(z_ctx)                     # (B, K_ctx, Dp)
        x = x + apply_masks(pos, masks_ctx)                 # positions at context slots
        K_ctx = x.shape[1]

        # 2. mask tokens: one per target slot, tagged with that slot's position
        tgt_pos = apply_masks(pos, masks_tgt)              # (n_tgt*B, K_tgt, Dp)
        mask_tokens = self.mask_token.expand(tgt_pos.shape[0], tgt_pos.shape[1], -1)
        mask_tokens = mask_tokens + tgt_pos               # (n_tgt*B, K_tgt, Dp)

        # 3. one copy of the context per target block, then concat context+queries
        x = x.repeat(n_tgt, 1, 1)                          # (n_tgt*B, K_ctx, Dp)
        x = torch.cat([x, mask_tokens], dim=1)            # (n_tgt*B, K_ctx+K_tgt, Dp)

        # attention bias: drop padded context keys; target-query columns valid
        attn_bias = None
        if ctx_keep is not None:
            keep = torch.cat([ctx_keep.repeat(n_tgt, 1),
                              torch.ones(n_tgt * B, tgt_pos.shape[1], dtype=torch.bool,
                                         device=x.device)], dim=1)         # (n_tgt*B, K_ctx+K_tgt)
            attn_bias = torch.where(keep[:, None, None, :], 0.0, torch.finfo(x.dtype).min)

        # 4. transformer, norm, drop the context slots, project back to encoder dim
        for blk in self.blocks:
            x = blk(x, attn_bias)
        x = self.predictor_norm(x)
        x = x[:, K_ctx:]                                   # (n_tgt*B, K_tgt, Dp)
        return self.predictor_proj(x)                      # (n_tgt*B, K_tgt, De)


if __name__ == "__main__":
    from vit import ViT
    from masks import MultiBlockMaskCollator
    from loss import ijepa_loss

    B, De = 4, 128
    grid = 14
    vit = ViT(img_size=224, patch_size=16, embed_dim=De, depth=4, num_heads=4)
    predictor = VisionTransformerPredictor(
        num_patches=grid * grid, encoder_dim=De,
        predictor_dim=64, depth=3, num_heads=4,
    )
    collator = MultiBlockMaskCollator(input_size=224, patch_size=16, n_pred=4)

    imgs = torch.randn(B, 3, 224, 224)
    enc_idx, ctx_keep, pred_masks = collator.sample(B, generator=torch.Generator().manual_seed(0))

    # --- one I-JEPA step, no EMA: same vit for context and target ---
    z_ctx = vit(imgs, masks=enc_idx, key_pad=ctx_keep)   # (B, L, De)   context encoder
    h = vit(imgs)                                        # (B, N, De)   target encoder (same weights)
    h_tgt = apply_masks(h, pred_masks)                   # (4B, K_tgt, De)
    p = predictor(z_ctx, enc_idx, pred_masks, ctx_keep=ctx_keep)  # (4B, K_tgt, De)

    print(f"z_ctx {tuple(z_ctx.shape)}   h_tgt {tuple(h_tgt.shape)}   pred {tuple(p.shape)}")
    assert p.shape == h_tgt.shape

    loss, parts = ijepa_loss(p, h_tgt, reg_embeddings=[z_ctx[ctx_keep], h], kappa=1.0)
    loss.backward()
    print("loss parts:", {k: round(float(v), 3) for k, v in parts.items()})

    # gradients reach the predictor AND both encoder passes (no stop-grad)
    assert predictor.mask_token.grad is not None
    assert vit.patch_embed.proj.weight.grad is not None
    print("OK")
