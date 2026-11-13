"""
The video encoder shared by both arms (V-JEPA and LeVJEPA): a small ViT over space-time patches.

    clip (B, T, 3, 64, 64) uint8  ->  tokens (B, N, D), one per tubelet of `tubelet` frames x patch x patch px

Tokens are numbered time-major, then row, then column: index = (t * grid + row) * grid + col. Positions are
fixed 3-D sin-cos vectors (a third of the channels each for time, row and column), added before any token is
dropped, so a kept token always knows where and when it sits. `forward(clips, keep_idx, valid)` runs only the
tokens in keep_idx (B, K), which is how V-JEPA's context encoder sees the visible part and how LeVJEPA drops 95%;
`valid` (B, K) bool marks padding (False), which no token attends to, so a padded row's real outputs are exactly
what they would be without the padding.

Pixel normalisation uses Push-T's own statistics (mostly white background, mean about 0.97), measured on 4,000
random frames; ImageNet's would leave the inputs far from zero mean and unit variance.

The transformer layer and the initialisation are copied from ../jepa_pathmnist/model.py (pre-norm ViT block;
the official I-JEPA init), with attention through torch's scaled_dot_product_attention for speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PIXEL_MEAN = (0.9691, 0.9782, 0.9746)
PIXEL_STD = (0.0979, 0.0706, 0.0772)


def sincos_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    """(M,) positions -> (M, dim): sin and cos at dim/2 frequencies, high to low."""
    assert dim % 2 == 0
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2))
    args = positions.float()[:, None] * omega[None, :]
    return torch.cat([args.sin(), args.cos()], dim=1)


def sincos_3d(dim: int, frames: int, grid: int) -> torch.Tensor:
    """(frames*grid*grid, dim): a third of each vector for time, a third for the row, a third for the column."""
    assert dim % 6 == 0
    t, r, c = torch.meshgrid(torch.arange(frames), torch.arange(grid), torch.arange(grid), indexing="ij")
    third = dim // 3
    return torch.cat([sincos_1d(third, t.reshape(-1)), sincos_1d(third, r.reshape(-1)),
                      sincos_1d(third, c.reshape(-1))], dim=1)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        mask = None if valid is None else valid[:, None, None, :]           # padded keys are never attended to
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)      # (B, heads, N, head_dim)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim), Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), valid)
        return x + self.mlp(self.norm2(x))


def init_weights(model: nn.Module, blocks: nn.ModuleList, std: float = 0.02) -> None:
    """Official I-JEPA init: truncated normal(0, 0.02), zero biases; residual-writing layers of block l / sqrt(2l)."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
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


class VideoEncoder(nn.Module):
    """Plain ViT over tubelets, no head. cls=True (LeVJEPA) prepends a learned [cls] token after the patch tokens
    are chosen; it has no position and is output 0, so LeVJEPA reads the clip's embedding there."""

    def __init__(self, frames: int = 16, img_size: int = 64, patch: int = 8, tubelet: int = 2,
                 dim: int = 192, depth: int = 6, heads: int = 3, cls: bool = False):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if cls else None
        assert frames % tubelet == 0 and img_size % patch == 0
        self.t_grid, self.grid, self.dim = frames // tubelet, img_size // patch, dim
        self.num_tokens = self.t_grid * self.grid ** 2
        # A Conv3d with kernel = stride = tubelet x patch x patch is "flatten each tubelet, shared linear map".
        self.patch_embed = nn.Conv3d(3, dim, kernel_size=(tubelet, patch, patch), stride=(tubelet, patch, patch))
        self.register_buffer("pos_embed", sincos_3d(dim, self.t_grid, self.grid)[None], persistent=False)
        self.register_buffer("mean", torch.tensor(PIXEL_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(PIXEL_STD).view(1, 3, 1, 1, 1), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        init_weights(self, self.blocks)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=.02)

    def tokens(self, clips: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) uint8 -> (B, N, D) embedded tubelets with positions, before the transformer."""
        x = clips.permute(0, 2, 1, 3, 4).float().div(255)                     # B, 3, T, H, W
        x = (x - self.mean) / self.std
        x = self.patch_embed(x).flatten(2).transpose(1, 2)                    # B, N, D, time-major
        return x + self.pos_embed

    def forward(self, clips: torch.Tensor, keep_idx: torch.Tensor | None = None,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        x = self.tokens(clips)
        if keep_idx is not None:                                              # (B, K): only these tokens
            x = torch.gather(x, 1, keep_idx[..., None].expand(-1, -1, self.dim))
        if self.cls is not None:
            x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
            if valid is not None:
                valid = torch.cat([valid.new_ones(len(x), 1), valid], 1)
        for blk in self.blocks:
            x = blk(x, valid)
        return self.norm(x)
