"""
Multi-block masking on the grid x grid patch grid (14 x 14 at 224px / 16px).

Every mask is a set of whole patches, given as integer patch indices
(index = row * grid + col, matching model.py). For every batch:

  1. Target blocks: n_pred (=4) rectangles of h x w patches. The size is drawn
     once per batch (all images share it), aiming at 15-20% of the grid with
     aspect ratio h/w in [0.75, 1.5], as in the official I-JEPA code; the
     position is drawn per image and per block. Blocks may overlap each other.
  2. Context: every patch of the grid minus exactly the target patches -- the
     union of the 4 blocks is removed, nothing else. Nothing is truncated.

Deviations from the official code (which draws an 85-100% context block and
then cuts every context to the shortest in the batch, dropping the
highest-numbered = bottom patches): the context is the exact complement of the
targets. Its length differs between images (targets overlap differently), so
contexts are right-padded to the longest in the batch, and `ctx_keep` marks
the real entries; model.py uses it to keep padding out of attention. Also,
block scale and aspect ratio are drawn independently (the official code reuses
one random number for both).

Output of `sample(B)`:
    ctx_mask   (B, L) LongTensor     context patch indices, sorted, padded with 0
    ctx_keep   (B, L) bool           True = real context patch, False = padding
    tgt_masks  list of n_pred (B, h*w) LongTensors, one per target block

Sampled in the training loop, not in a DataLoader collate_fn: worker processes
would each get a copy of the same generator and repeat each other's masks.
"""

import math

import torch


class MultiBlockMasks:
    def __init__(self, grid: int, n_pred: int = 4,
                 pred_scale=(0.15, 0.2), pred_aspect=(0.75, 1.5), seed: int = 0):
        self.grid = grid
        self.n_pred = n_pred
        self.pred_scale, self.pred_aspect = pred_scale, pred_aspect
        self.g = torch.Generator().manual_seed(seed)

    def _uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * torch.rand(1, generator=self.g).item()

    def block_size(self) -> tuple:
        """(h, w) in whole patches: about pred_scale of the grid, h/w about pred_aspect."""
        area = self.grid * self.grid * self._uniform(*self.pred_scale)
        ar = self._uniform(*self.pred_aspect)
        h = max(1, min(round(math.sqrt(area * ar)), self.grid))
        w = max(1, min(round(math.sqrt(area / ar)), self.grid))
        return h, w

    def _block(self, h: int, w: int) -> torch.Tensor:
        """h x w block at a random position fully inside the grid, as a (grid, grid) bool map."""
        top = torch.randint(0, self.grid - h + 1, (1,), generator=self.g).item()
        left = torch.randint(0, self.grid - w + 1, (1,), generator=self.g).item()
        block = torch.zeros(self.grid, self.grid, dtype=torch.bool)
        block[top:top + h, left:left + w] = True
        return block

    def sample(self, batch_size: int):
        h, w = self.block_size()
        ctx, tgt = [], []
        for _ in range(batch_size):
            blocks = [self._block(h, w) for _ in range(self.n_pred)]
            is_target = torch.stack(blocks).any(0)
            ctx.append((~is_target).flatten().nonzero().squeeze(1))              # grid \ targets
            tgt.append([b.flatten().nonzero().squeeze(1) for b in blocks])        # h*w each

        L = max(len(c) for c in ctx)
        ctx_mask = torch.zeros(batch_size, L, dtype=torch.long)
        ctx_keep = torch.zeros(batch_size, L, dtype=torch.bool)
        for i, c in enumerate(ctx):
            ctx_mask[i, :len(c)] = c
            ctx_keep[i, :len(c)] = True
        tgt_masks = [torch.stack([ts[j] for ts in tgt]) for j in range(self.n_pred)]
        return ctx_mask, ctx_keep, tgt_masks
