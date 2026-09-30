"""
The two networks of I-JEPA: the encoder (a plain ViT) and the predictor.

    encoder    image -> one D-dim embedding per patch. Used twice per step:
                 context encoder: sees only the context patches  (gets gradients)
                 target:          sees the full image            (output gradient-stopped)
    predictor  context embeddings + "which positions to fill in"
                 -> predicted embeddings at the target positions.

The loss (loss.py) compares the predictor's output with the target encoder's
embeddings at the target positions -- entirely in latent space, no pixels.
Training (optimizer, masks) lives elsewhere; this file is just the networks.

Masks are index tensors: a (B, K) LongTensor of patch indices per image, where
patches are numbered row-major over the grid (index = row * grid + col).

Padding. The context (grid minus the targets, see masks.py) has a different
length per image, so it is right-padded to a common length L, with a
`ctx_keep` (B, L) bool marking the real tokens. Inside attention, padded
tokens are removed as KEYS: no real token ever reads from them, so real
outputs are exactly what they would be without padding. Padded tokens still
produce outputs of their own (as queries); those are garbage and must be
ignored by the caller -- the predictor never reads them, and training only
uses the predictor's target outputs.

Adapted from ../i_jepa/vit.py and ../i_jepa/predictor.py; initialisation as in
the official I-JEPA code (init_weights).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional embedding: fixed 2-D sin-cos, no learnable parameters
# ---------------------------------------------------------------------------

def sincos_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    """(M,) positions -> (M, dim): sin and cos at dim/2 frequencies, high to low."""
    assert dim % 2 == 0
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2))
    args = positions.float()[:, None] * omega[None, :]           # (M, dim/2)
    return torch.cat([args.sin(), args.cos()], dim=1)             # (M, dim)


def sincos_2d(dim: int, grid: int) -> torch.Tensor:
    """(grid*grid, dim): first half of each vector encodes the row, second half the column."""
    assert dim % 4 == 0
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    return torch.cat([sincos_1d(dim // 2, rows.reshape(-1)),
                      sincos_1d(dim // 2, cols.reshape(-1))], dim=1)


# ---------------------------------------------------------------------------
# Selecting patches
# ---------------------------------------------------------------------------

def apply_masks(x: torch.Tensor, masks) -> torch.Tensor:
    """
    Keep a subset of tokens. x: (B, N, D).
    masks: one (B, K) index tensor -> (B, K, D); or a list of n such tensors ->
           each gathered separately, stacked along the batch dim -> (n*B, K, D).
    The list form is how several target blocks are cut out of one full pass.
    """
    if not isinstance(masks, (list, tuple)):
        masks = [masks]
    out = [torch.gather(x, 1, m[:, :, None].expand(-1, -1, x.shape[-1])) for m in masks]
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Transformer building blocks (standard pre-norm ViT layer)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention, written out so every step is visible."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, keep: torch.Tensor = None) -> torch.Tensor:
        """x: (B, N, D). keep: optional (B, N) bool, False = padding, never attended to."""
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                     # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5  # (B, heads, N, N)
        if keep is not None:                                      # padded keys get weight exp(-inf) = 0
            attn = attn.masked_fill(~keep[:, None, None, :], float("-inf"))
        out = attn.softmax(dim=-1) @ v                            # (B, heads, N, head_dim)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))    # heads concatenated back


class Block(nn.Module):
    """x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, keep: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), keep)
        x = x + self.mlp(self.norm2(x))
        return x


def init_weights(model: nn.Module, blocks: nn.ModuleList, std: float = 0.02) -> None:
    """
    Official I-JEPA initialisation.
      - Linear / Conv2d weights ~ truncated normal(0, 0.02), biases 0; LayerNorm (1, 0).
      - The two layers that write back into the residual stream (attention output
        proj, second MLP layer) of block l are divided by sqrt(2 l). Each block adds
        to the residual, so deeper blocks start with smaller updates and the
        stream's scale doesn't grow with depth at init.
    """
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    with torch.no_grad():
        for layer, blk in enumerate(blocks, start=1):
            blk.attn.proj.weight.div_((2.0 * layer) ** 0.5)
            blk.mlp[2].weight.div_((2.0 * layer) ** 0.5)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Plain ViT, no [CLS] token, no head.

    forward(imgs)               (B, C, H, W) -> (B, N, D)   full image (target)
    forward(imgs, masks, keep)  (B, C, H, W) -> (B, L, D)   only the context patches;
                                                            attention never sees the
                                                            hidden ones or the padding
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int,
                 embed_dim: int, depth: int, num_heads: int):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.embed_dim = embed_dim

        # Patchify + linear projection in one op: a conv with kernel = stride = patch size
        # is exactly "flatten each patch, multiply by a shared weight matrix".
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Buffer, not Parameter: moves with .to(device), never trained.
        self.register_buffer("pos_embed", sincos_2d(embed_dim, self.grid)[None], persistent=False)

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        init_weights(self, self.blocks)

    def forward(self, imgs: torch.Tensor, masks=None, keep: torch.Tensor = None) -> torch.Tensor:
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)    # (B, N, D), row-major patches
        x = x + self.pos_embed                                    # content + position
        # Drop patches AFTER adding positions, so each kept token knows where it sits.
        if masks is not None:
            x = apply_masks(x, masks)                             # (B, L, D)
        for blk in self.blocks:
            x = blk(x, keep)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    """
    A narrow transformer (paper: about half the encoder width, ~6 layers). Its job
    is only "what would the target encoder have put here, given the context".

    Input tokens = context embeddings (at their positions)
                 + one mask token per target patch (the same learned vector,
                   told apart only by the position added to it).
    After the transformer, the outputs at the mask-token slots are the predictions.

    Each target block is predicted separately: the context is copied once per
    block, so mask tokens of different blocks never attend to each other.
    """

    def __init__(self, num_patches: int, encoder_dim: int, predictor_dim: int,
                 depth: int, num_heads: int):
        super().__init__()
        grid = int(round(num_patches ** 0.5))
        assert grid * grid == num_patches

        self.embed = nn.Linear(encoder_dim, predictor_dim)       # encoder width -> predictor width
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.register_buffer("pos_embed", sincos_2d(predictor_dim, grid)[None], persistent=False)

        self.blocks = nn.ModuleList([Block(predictor_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj = nn.Linear(predictor_dim, encoder_dim)        # back to encoder width
        init_weights(self, self.blocks)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, z_ctx: torch.Tensor, ctx_mask: torch.Tensor, ctx_keep: torch.Tensor,
                tgt_masks) -> torch.Tensor:
        """
        z_ctx:     (B, L, De)  context encoder output (padded)
        ctx_mask:  (B, L)      grid positions z_ctx came from
        ctx_keep:  (B, L)      bool, False = padding slot
        tgt_masks: list of n (B, K_tgt) tensors, grid positions to predict
        returns:   (n*B, K_tgt, De), same layout as apply_masks(h_full, tgt_masks)
        """
        if not isinstance(tgt_masks, (list, tuple)):
            tgt_masks = [tgt_masks]
        B, n = z_ctx.shape[0], len(tgt_masks)
        pos = self.pos_embed.expand(B, -1, -1)                    # (B, N, Dp)

        ctx = self.embed(z_ctx) + apply_masks(pos, ctx_mask)      # (B, L, Dp)
        tgt_pos = apply_masks(pos, tgt_masks)                     # (n*B, K_tgt, Dp)
        queries = self.mask_token + tgt_pos                       # (n*B, K_tgt, Dp)

        x = torch.cat([ctx.repeat(n, 1, 1), queries], dim=1)      # (n*B, L+K_tgt, Dp)
        keep = torch.cat([ctx_keep.repeat(n, 1),                  # padding: never a key
                          torch.ones(queries.shape[:2], dtype=torch.bool, device=x.device)], dim=1)
        for blk in self.blocks:
            x = blk(x, keep)
        x = self.norm(x[:, ctx.shape[1]:])                        # keep only the target slots
        return self.proj(x)                                       # (n*B, K_tgt, De)
