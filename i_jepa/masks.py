"""
I-JEPA mask sampler -- discrete, on the P x P patch grid (P = img_size /
patch_size; 14 for 224px / 16px patches).

One training step:

  * pick one integer target size (h, w), each side uniform in `pred_hw`
    (default ~0.2..0.55 of the patches-per-side P, i.e. 3..8 when P=14);
  * for each of `n_pred` (=4) target rectangles, drop an h x w box at a
    uniform integer top-left that keeps it fully inside the grid (boxes may
    overlap each other);
  * target patches  = union of the 4 boxes;
  * context patches = every patch NOT in that union -- the exact complement,
    nothing dropped.

Returns:
  enc_idx  : LongTensor (B, L)   -- context patch indices, right-padded to the
                                    batch's longest context with 0
  ctx_keep : bool     (B, L)     -- True where enc_idx is a real context patch,
                                    False on padding (feed to vit as key_pad)
  pred_masks : list of `n_pred` LongTensor (B, h*w) -- the target boxes

The four target boxes share one (h, w) per step, so pred_masks[j] stacks
cleanly and apply_masks(target, pred_masks) is rectangular. Only the context
length varies (per batch), handled by the pad + ctx_keep.
"""

from multiprocessing import Value

import torch
from torch.utils.data import default_collate


class MultiBlockMaskCollator:
    # target side length as a fraction of P when pred_hw is left unset
    HW_FRAC = (0.2, 0.55)   # -> (3, 8) at P=14

    def __init__(self, input_size=224, patch_size: int = 16,
                 pred_hw=None, n_pred: int = 4):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        assert input_size[0] % patch_size == 0 and input_size[1] % patch_size == 0
        assert input_size[0] == input_size[1], "square grid assumed"
        self.P = input_size[0] // patch_size
        self.n_patches = self.P * self.P

        if pred_hw is None:
            lo = max(1, round(self.HW_FRAC[0] * self.P))
            hi = max(lo + 1, round(self.HW_FRAC[1] * self.P))
            pred_hw = (lo, hi)
        self.hw_lo, self.hw_hi = pred_hw          # inclusive integer range for h and w
        self.n_pred = n_pred
        assert 1 <= self.hw_lo <= self.hw_hi <= self.P

        self._itr_counter = Value("i", -1)        # keeps DataLoader workers in sync

    # -- iteration counter ----------------------------------------------

    def step(self) -> int:
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    # -- geometry -----------------------------------------------------

    def _randint(self, lo, hi, g):  # inclusive
        return int(torch.randint(lo, hi + 1, (1,), generator=g).item())

    def _box_indices(self, h, w, g):
        """h x w box at a uniform in-grid position -> flat patch indices."""
        top = self._randint(0, self.P - h, g)
        left = self._randint(0, self.P - w, g)
        rows = torch.arange(top, top + h).unsqueeze(1)
        cols = torch.arange(left, left + w).unsqueeze(0)
        return (rows * self.P + cols).reshape(-1)

    def _sample_one(self, h, w, g):
        """One image -> (context_idx [variable len], [box_idx x n_pred])."""
        boxes = [self._box_indices(h, w, g) for _ in range(self.n_pred)]
        is_target = torch.zeros(self.n_patches, dtype=torch.bool)
        is_target[torch.cat(boxes)] = True
        context = (~is_target).nonzero(as_tuple=False).squeeze(-1)
        context = context[torch.randperm(context.numel(), generator=g)]
        return context, boxes

    def _sample_batch(self, batch_size: int, g):
        h = self._randint(self.hw_lo, self.hw_hi, g)
        w = self._randint(self.hw_lo, self.hw_hi, g)

        contexts, per_box = [], [[] for _ in range(self.n_pred)]
        for _ in range(batch_size):
            context, boxes = self._sample_one(h, w, g)
            contexts.append(context)
            for j, b in enumerate(boxes):
                per_box[j].append(b)

        # Pad to the FULL grid, not the batch max: the context is the exact
        # complement (never > n_patches), and a constant sequence length keeps
        # MPS/XLA from recompiling the graph every batch. ctx_keep masks the
        # padding out of attention, so the extra slots cost compute but change
        # nothing numerically.
        L = self.n_patches
        enc_idx = torch.zeros(batch_size, L, dtype=torch.long)
        ctx_keep = torch.zeros(batch_size, L, dtype=torch.bool)
        for i, c in enumerate(contexts):
            enc_idx[i, :c.numel()] = c
            ctx_keep[i, :c.numel()] = True

        pred_masks = [torch.stack(col) for col in per_box]   # (B, h*w) each
        return enc_idx, ctx_keep, pred_masks

    # -- public API -------------------------------------------------

    def sample(self, batch_size: int, generator=None):
        """-> (enc_idx (B, L), ctx_keep (B, L) bool, pred_masks list[(B, h*w)])."""
        if generator is None:
            generator = torch.Generator().manual_seed(self.step())
        return self._sample_batch(batch_size, generator)

    def __call__(self, batch):
        """DataLoader collate_fn -> (collated_batch, enc_idx, ctx_keep, pred_masks)."""
        collated_batch = default_collate(batch)
        generator = torch.Generator().manual_seed(self.step())
        enc_idx, ctx_keep, pred_masks = self._sample_batch(len(batch), generator)
        return collated_batch, enc_idx, ctx_keep, pred_masks


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vit import ViT, apply_masks

    B = 8
    col = MultiBlockMaskCollator(input_size=224, patch_size=16, pred_hw=(3, 8), n_pred=4)
    enc_idx, ctx_keep, pred_masks = col.sample(B, generator=torch.Generator().manual_seed(0))

    L = enc_idx.shape[1]
    print(f"enc_idx {tuple(enc_idx.shape)}   ctx_keep {tuple(ctx_keep.shape)}   "
          f"context lens {ctx_keep.sum(1).tolist()}")
    print(f"targets: {len(pred_masks)} x {tuple(pred_masks[0].shape)}")
    assert len(pred_masks) == 4 and all(t.shape[0] == B for t in pred_masks)

    # context == exact complement of the target union, per sample (no holes)
    for b in range(B):
        ctx = set(enc_idx[b][ctx_keep[b]].tolist())
        tgt = set().union(*(set(t[b].tolist()) for t in pred_masks))
        assert ctx.isdisjoint(tgt)
        assert ctx | tgt == set(range(col.n_patches)), "context is NOT the full complement"
    print("context == grid \\ target-union, exactly: OK")

    # every border patch is context unless it happens to fall in a box
    P = col.P
    border = {i for i in range(col.n_patches) if i // P in (0, P - 1) or i % P in (0, P - 1)}
    seen = set()
    for s in range(300):
        ei, ck, _ = col.sample(1, generator=torch.Generator().manual_seed(s))
        seen |= set(ei[0][ck[0]].tolist()) & border
    assert seen == border
    print(f"border patches ever in context over 300 draws: {len(seen)}/{len(border)}")

    # feeds the ViT: padded context + key_pad mask
    vit = ViT(img_size=224, patch_size=16, embed_dim=128, depth=4, num_heads=4)
    imgs = torch.randn(B, 3, 224, 224)
    z_ctx = vit(imgs, masks=enc_idx, key_pad=ctx_keep)   # (B, L, 128)
    h_tgt = apply_masks(vit(imgs), pred_masks)           # (4B, h*w, 128)
    print(f"context enc out: {tuple(z_ctx.shape)}   target slices: {tuple(h_tgt.shape)}")

    # widening the pad must not change the real-token outputs (attention
    # ignores padded keys)
    wide_idx = torch.cat([enc_idx, torch.zeros(B, 5, dtype=torch.long)], dim=1)
    wide_keep = torch.cat([ctx_keep, torch.zeros(B, 5, dtype=torch.bool)], dim=1)
    z_wide = vit(imgs, masks=wide_idx, key_pad=wide_keep)
    assert torch.allclose(z_ctx[ctx_keep], z_wide[wide_keep], atol=1e-4)
    print("padded positions do not leak into real tokens: OK")
    print("OK")
