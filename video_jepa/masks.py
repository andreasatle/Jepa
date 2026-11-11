"""
V-JEPA's multi-block masks, scaled to our 8 x 8 patch grid and 8 time slots (16 frames, tubelets of 2).

A mask is a set of spatial blocks repeated through every time slot, so the model cannot copy a hidden patch
from the same place one frame earlier. Two kinds, as in the paper, one (context, target) pair of each per step:

    short-range   8 blocks, each about 15% of the frame's area     (union about 70% hidden)
    long-range    2 blocks, each about 70% of the frame's area     (union about 90% hidden)

Each block's aspect ratio is drawn from [0.75, 1.5]. Target = the tokens under the union of the blocks;
context = every other token. Neither encoder ever sees a token both ways.

Batching: as in the official collators, the block size (scale and aspect) is drawn once per batch and only
the positions per sample. The official collators then cut every sample's lists to the batch's shortest; on our
8 x 8 grid two 70% blocks sometimes cover nearly the whole frame, and cutting left every clip in the batch 16
context tokens. So lists are padded to the batch's longest instead, with a `valid` mask that attention and the
loss respect (a departure; each clip keeps all of its own context and targets).
"""

import math

import torch

GRID, T_SLOTS = 8, 8


class MultiBlock:
    def __init__(self, n_blocks: int, scale: float, aspect=(0.75, 1.5), grid: int = GRID, t_slots: int = T_SLOTS):
        self.n_blocks, self.scale, self.aspect, self.grid, self.t_slots = n_blocks, scale, aspect, grid, t_slots

    def size(self, g: torch.Generator) -> tuple[int, int]:
        """A block's (height, width) in patches: area scale * grid^2, aspect drawn log-uniformly."""
        area = self.scale * self.grid ** 2
        ar = math.exp(math.log(self.aspect[0]) + torch.rand(1, generator=g).item()
                      * (math.log(self.aspect[1]) - math.log(self.aspect[0])))
        return max(1, min(self.grid, round(math.sqrt(area * ar)))), max(1, min(self.grid, round(math.sqrt(area / ar))))

    def spatial(self, g: torch.Generator, hw: tuple[int, int]) -> torch.Tensor:
        """(grid, grid) bool: True = hidden; n_blocks blocks of size hw at random places."""
        hidden = torch.zeros(self.grid, self.grid, dtype=torch.bool)
        h, w = hw
        for _ in range(self.n_blocks):
            top = torch.randint(0, self.grid - h + 1, (1,), generator=g).item()
            left = torch.randint(0, self.grid - w + 1, (1,), generator=g).item()
            hidden[top:top + h, left:left + w] = True
        return hidden

    def sample(self, g: torch.Generator, hw: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One sample's (context indices, target indices) over the t_slots x grid x grid tokens, time-major."""
        hw = hw or self.size(g)
        while True:
            hid = self.spatial(g, hw)
            if 0 < hid.sum() < hid.numel():                     # keep at least one patch on each side
                break
        full = hid.flatten().repeat(self.t_slots)                # the same blocks in every time slot
        return (~full).nonzero().flatten(), full.nonzero().flatten()

    def batch(self, b: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ctx (b, K_ctx), ctx_valid, tgt (b, K_tgt), tgt_valid: index lists padded to the batch's longest
        (padding index 0, valid False)."""
        hw = self.size(g)
        pairs = [self.sample(g, hw) for _ in range(b)]
        return (*self.pad([c for c, _ in pairs]), *self.pad([t for _, t in pairs]))

    @staticmethod
    def pad(lists: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        k = max(len(x) for x in lists)
        idx = torch.zeros(len(lists), k, dtype=torch.long)
        valid = torch.zeros(len(lists), k, dtype=torch.bool)
        for i, x in enumerate(lists):
            idx[i, :len(x)], valid[i, :len(x)] = x, True
        return idx, valid


SHORT = MultiBlock(n_blocks=8, scale=0.15)
LONG = MultiBlock(n_blocks=2, scale=0.70)
