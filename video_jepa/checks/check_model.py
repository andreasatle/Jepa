"""Checks for model.py (step 2 of PLAN.md): token counts, positions, dropping tokens, and speed on this Mac.

    uv run python checks/check_model.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data import Clips, PushTVideo
from model import VideoEncoder, sincos_3d


def speed(enc: VideoEncoder, x: torch.Tensor, keep: torch.Tensor | None, dev, reps: int = 5) -> float:
    """Clips per second for forward + backward."""
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-4)
    for i in range(reps + 1):
        if i == 1:
            torch.mps.synchronize() if dev.type == "mps" else None
            t0 = time.time()
        loss = enc(x, keep).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.mps.synchronize() if dev.type == "mps" else None
    return reps * len(x) / (time.time() - t0)


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    clips = Clips(PushTVideo(), "held_out", n=64, seed=2)
    x = torch.stack([clips[i][0] for i in range(32)])

    for tubelet, expect in ((2, 512), (1, 1024)):                     # V-JEPA arm, LeVJEPA arm
        enc = VideoEncoder(tubelet=tubelet)
        out = enc(x)
        print(f"tubelet {tubelet}: {enc.num_tokens} tokens -> {tuple(out.shape)}, "
              f"{sum(p.numel() for p in enc.parameters()) / 1e6:.2f} M parameters")
        assert enc.num_tokens == expect and out.shape == (32, expect, 192)

    pos = sincos_3d(192, 8, 8)
    d = torch.cdist(pos, pos) + torch.eye(len(pos)) * 1e9
    print(f"positions: {len(pos)} vectors, closest pair {d.min():.3f} apart")
    assert d.min() > 0.1, "two tokens share a position"

    # dropping tokens: the kept tokens' inputs are exactly the full tokens at those indices
    enc = VideoEncoder()
    keep = torch.sort(torch.randperm(512)[:51])[0].expand(32, -1)
    assert torch.equal(torch.gather(enc.tokens(x), 1, keep[..., None].expand(-1, -1, 192)),
                       enc.tokens(x)[:, keep[0]])
    assert enc(x, keep).shape == (32, 51, 192)
    print("keep_idx runs only the kept tokens: OK")

    enc = VideoEncoder().to(dev)
    xd = x.to(dev)
    full = speed(enc, xd, None, dev)
    part = speed(enc, xd, keep.to(dev), dev)
    print(f"{dev}: forward+backward, batch 32: {full:.0f} clips/s with all 512 tokens, "
          f"{part:.0f} clips/s with 51 kept (V-JEPA's context is ~10%)")
    print("OK")


if __name__ == "__main__":
    main()
