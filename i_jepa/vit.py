"""
Plain Vision Transformer forward pass.

This is deliberately just the encoder: patchify -> linear projection ->
sincos positional embedding -> L transformer layers -> per-patch output.
No [CLS] token, no classification head.

This is the piece I-JEPA's context encoder and target encoder both are,
unmodified except for one hook: `forward` takes an optional `masks`
argument selecting which patches to keep. The *choice* of which patches
form the context block or the target blocks lives elsewhere (the mask
sampler); all this module does is, after fusing in positional info,
gather the requested subset and run the transformer over just that
shorter sequence. Pass `masks=None` and it behaves exactly as a plain
full-image ViT.

Typical I-JEPA usage:
  * target encoder: `h = vit(imgs)` (full pass), then slice target blocks
    out of `h` with `apply_masks(h, target_masks)`;
  * context encoder: `z = vit(imgs, masks=context_mask)` so attention
    only ever sees the context patches.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional embedding (fixed, no learnable parameters)
# ---------------------------------------------------------------------------

def get_1d_sincos_pos_embed(dim: int, positions: torch.Tensor) -> torch.Tensor:
    """
    dim: output dimension for each position (must be even)
    positions: (M,) tensor of position indices (e.g. row indices, or col indices)
    returns: (M, dim)
    """
    assert dim % 2 == 0
    omega = torch.arange(dim // 2, dtype=torch.float32)
    omega = omega / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)  # (dim/2,) -- frequencies, high to low

    positions = positions.float().reshape(-1)  # (M,)
    args = torch.einsum("m,d->md", positions, omega)  # (M, dim/2) outer product

    emb_sin = torch.sin(args)  # (M, dim/2)
    emb_cos = torch.cos(args)  # (M, dim/2)
    return torch.cat([emb_sin, emb_cos], dim=1)  # (M, dim)


def get_2d_sincos_pos_embed(dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    dim: total embedding dimension D (must be divisible by 4: half for row, half
         for col, and each half split again into sin/cos)
    grid_h, grid_w: number of patches along each spatial axis (e.g. 14, 14)
    returns: (grid_h * grid_w, dim) -- one row per patch, same order as the
             patchify step below (row-major: row 0 all cols, then row 1, ...)
    """
    assert dim % 4 == 0
    dim_half = dim // 2

    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)
    grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")  # each (grid_h, grid_w)

    emb_row = get_1d_sincos_pos_embed(dim_half, grid_row.reshape(-1))  # (N, dim/2)
    emb_col = get_1d_sincos_pos_embed(dim_half, grid_col.reshape(-1))  # (N, dim/2)

    return torch.cat([emb_row, emb_col], dim=1)  # (N, dim)


# ---------------------------------------------------------------------------
# Masking helper: keep only a subset of the patch tokens
# ---------------------------------------------------------------------------

def apply_masks(x: torch.Tensor, masks) -> torch.Tensor:
    """
    Gather a subset of patch tokens along the sequence dimension.

    x:     (B, N, D) patch tokens -- positional info is expected to be fused
           in already, so a gathered token still knows where it came from.
    masks: a LongTensor of shape (B, K) giving, per batch element, the K
           patch indices to keep; or a list/tuple of such tensors. Passing
           a list means "several different subsets of the same batch": each
           is gathered independently and the results are concatenated along
           the batch dim, yielding (len(masks) * B, K, D). I-JEPA uses this
           to pull all target blocks out of a single target-encoder pass.

    returns: (B, K, D) for a single mask tensor,
             (len(masks) * B, K, D) for a list of mask tensors.
    """
    if not isinstance(masks, (list, tuple)):
        masks = [masks]
    gathered = []
    for m in masks:
        idx = m.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (B, K) -> (B, K, D)
        gathered.append(torch.gather(x, dim=1, index=idx))
    return torch.cat(gathered, dim=0)


# ---------------------------------------------------------------------------
# Patch embedding: flatten + linear projection, implemented as one strided conv
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Splits an image into non-overlapping patches and linearly projects each
    to `embed_dim`. Implemented as a single Conv2d with kernel_size=stride=
    patch_size -- mathematically identical to "flatten each patch, then
    matrix-multiply by a shared weight", just faster.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.grid_size = img_size // patch_size  # patches per side
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, grid_size, grid_size)
        x = x.flatten(2)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)  -- one D-dim vector per patch, row-major order
        return x


# ---------------------------------------------------------------------------
# Transformer block: self-attention + MLP, pre-norm, residual connections
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        if attn_bias is not None:
            attn = attn + attn_bias  # additive: 0 to keep, large-negative to drop a key
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D) -- heads concatenated back
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """One layer: (Attn, residual) -> (MLP, residual), pre-norm."""

    def __init__(self, dim: int, num_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_bias)  # residual around attention
        x = x + self.mlp(self.norm2(x))              # residual around MLP
        return x


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class ViT(nn.Module):
    """
    Plain ViT encoder. No [CLS] token, no head.
    Input:  (B, C, H, W) image batch, plus optional `masks`
    Output: (B, N, D) -- one D-dim vector per patch (full image), or
            (B, K, D) / (len(masks) * B, K, D) if `masks` is given

    `key_pad`: optional (B, K) bool, True = real token, False = padding.
    When the kept sequence is variable-length across the batch it is padded
    to a common K; passing key_pad makes every attention layer ignore the
    padded positions as keys (they still occupy query rows, whose outputs
    the caller discards).
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        grid_size = self.patch_embed.grid_size

        # Fixed (non-learnable) sincos position embedding, computed once and
        # registered as a buffer so it moves with .to(device) but never gets
        # a gradient.
        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)  # (1, N, D)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)  # final norm, standard ViT practice

    def forward(self, x: torch.Tensor, masks=None, key_pad: torch.Tensor = None) -> torch.Tensor:
        x = self.patch_embed(x)          # (B, N, D) -- content only, all N patches
        x = x + self.pos_embed           # (B, N, D) -- content + position, fused

        # Drop to the requested subset *after* positions are fused in, so each
        # kept token still carries its absolute position. For the context
        # encoder this means the transformer below only ever attends within
        # the context block. masks=None -> full-image pass, unchanged.
        if masks is not None:
            x = apply_masks(x, masks)    # (B, K, D) or (len(masks) * B, K, D)

        attn_bias = None
        if key_pad is not None:
            neg = torch.finfo(x.dtype).min
            attn_bias = torch.where(key_pad[:, None, None, :], 0.0, neg)  # (B,1,1,K)

        for block in self.blocks:
            x = block(x, attn_bias)      # self-attention + MLP, L times
        x = self.norm(x)
        return x                         # (B, N, D) or masked shape


if __name__ == "__main__":
    # Smoke test: shapes should come out as expected for a 224x224 image,
    # 16x16 patches -> N=196, using a small config so it runs fast on CPU.
    model = ViT(img_size=224, patch_size=16, embed_dim=128, depth=4, num_heads=4)
    B, N, D = 2, model.patch_embed.num_patches, 128
    x = torch.randn(B, 3, 224, 224)  # batch of 2 images

    # 1. Full-image pass (plain ViT behaviour, masks=None).
    out = model(x)
    print(f"input:        {tuple(x.shape)}")
    print(f"full output:  {tuple(out.shape)}")
    assert out.shape == (B, N, D)

    # 2. Context-encoder pass: keep a subset of K patches per image.
    K = 50
    ctx_mask = torch.stack([torch.randperm(N)[:K] for _ in range(B)])  # (B, K)
    ctx = model(x, masks=ctx_mask)
    print(f"context out:  {tuple(ctx.shape)}  (masks=(B, K))")
    assert ctx.shape == (B, K, D)

    # 3. Target blocks: one full pass, then slice several blocks out of it.
    h = model(x)
    tgt_masks = [torch.stack([torch.randperm(N)[:K] for _ in range(B)])
                 for _ in range(4)]  # 4 target blocks
    tgt = apply_masks(h, tgt_masks)
    print(f"target out:   {tuple(tgt.shape)}  (list of 4 masks)")
    assert tgt.shape == (4 * B, K, D)

    print(f"N (num patches) = {N}")
    print("OK")