"""Checks for masks.py and vjepa.py (step 4 of PLAN.md), before any long training.

    uv run python checks/check_vjepa.py

1. Masks: hidden fraction per kind (paper: about 90% hidden on average); blocks repeated through every time
   slot; context and target disjoint and together every token; batches subsampled to equal lengths.
2. The untrained model's loss is finite, the predictor's output has the target's shape.
3. EMA: after one step, every target parameter equals m * old + (1 - m) * online.
4. A few steps on one fixed batch lower the loss (gradients reach encoder and predictor).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data import Clips, PushTVideo
from masks import GRID, LONG, SHORT, T_SLOTS
from vjepa import VJEPA, param_groups


def check_masks() -> None:
    g = torch.Generator().manual_seed(0)
    n = GRID * GRID * T_SLOTS
    for name, mk in (("short-range", SHORT), ("long-range", LONG)):
        fr = []
        for _ in range(2000):
            ctx, tgt = mk.sample(g)
            assert len(set(ctx.tolist()) & set(tgt.tolist())) == 0 and len(ctx) + len(tgt) == n
            hid = torch.zeros(n, dtype=torch.bool)
            hid[tgt] = True
            per_slot = hid.view(T_SLOTS, GRID * GRID)
            assert (per_slot == per_slot[0]).all(), "blocks must repeat through time"
            fr.append(len(tgt) / n)
        fr = torch.tensor(fr)
        ctx_b, cv, tgt_b, tv = mk.batch(64, g)
        print(f"{name}: hidden {fr.mean():.0%} on average (5-95%: {fr.quantile(.05):.0%}-{fr.quantile(.95):.0%}); "
              f"in a batch of 64, context {cv.sum(1).min()}-{cv.sum(1).max()} tokens per clip (padded to {cv.shape[1]})")
        both = (ctx_b[:, :, None] == tgt_b[:, None, :]) & cv[:, :, None] & tv[:, None, :]
        assert not both.any() and (cv.sum(1) + tv.sum(1) == n).all()
    # padding is invisible: a clip's real outputs are the same alone and inside a padded batch
    from model import VideoEncoder
    enc = VideoEncoder().eval()
    x = torch.randint(0, 256, (2, 16, 3, 64, 64), dtype=torch.uint8)
    c0, c1 = LONG.sample(g)[0], torch.arange(300)
    idx, valid = LONG.pad([c0, c1])
    alone = enc(x[:1], c0[None])
    padded = enc(x, idx, valid)[:1, :len(c0)]
    print(f"padding: max difference alone vs padded {(alone - padded).abs().max():.1e}")
    assert torch.allclose(alone, padded, atol=1e-4)
    print("masks: OK")


def main():
    check_masks()
    torch.manual_seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    clips = Clips(PushTVideo(), "train", n=64, seed=3)
    x = torch.stack([clips[i][0] for i in range(16)]).to(dev)
    g = torch.Generator().manual_seed(0)
    masks = [tuple(t.to(dev) for t in mk.batch(16, g)) for mk in (SHORT, LONG)]

    model = VJEPA().to(dev)
    loss, parts, h = model.loss(x, masks)
    ctx, cv, tgt, tv = masks[1]
    p = model.predictor(model.encoder(x, ctx, cv), ctx, cv, tgt, tv)
    print(f"untrained loss {loss.item():.4f} (short {parts[0].item():.4f}, long {parts[1].item():.4f}); "
          f"prediction {tuple(p.shape)} for up to {tgt.shape[1]} hidden tokens")
    assert torch.isfinite(loss) and p.shape == (16, tgt.shape[1], 192)

    opt = torch.optim.AdamW(param_groups(model.encoder, model.predictor), lr=3e-4)
    old = [t.detach().clone() for t in model.target.parameters()]
    opt.zero_grad()
    loss.backward()
    assert all(q.grad is not None for q in model.predictor.parameters()), "predictor gets no gradient"
    assert all(t.grad is None for t in model.target.parameters()), "target must get no gradient"
    opt.step()
    m = .998
    model.ema(m)
    err = max((t - (m * o0 + (1 - m) * o)).abs().max().item()
              for t, o0, o in zip(model.target.parameters(), old, model.encoder.parameters()))
    moved = max((t - o0).abs().max().item() for t, o0 in zip(model.target.parameters(), old))
    print(f"EMA: max deviation from m*old + (1-m)*online {err:.1e}; target moved by up to {moved:.1e}")
    assert err < 1e-6 and moved > 0

    first = None
    for _ in range(30):
        loss, _, _ = model.loss(x, masks)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.ema(m)
    print(f"30 steps on one batch: loss {first:.4f} -> {loss.item():.4f}")
    assert loss.item() < first
    print("OK")


if __name__ == "__main__":
    main()
