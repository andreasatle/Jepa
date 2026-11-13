"""Checks for levjepa.py (step 5 of PLAN.md), before any long training.

    uv run python checks/check_levjepa.py

1. Views: the global view keeps 5% of all 1,024 tokens; each local view keeps 5% of one s x s window that is
   the same in every frame; indices are valid and distinct.
2. SIGReg on known samples, against its exact expected values: a Gaussian sample scores
   integral (1 - e^{-t^2}) w(t) dt, a collapsed one B * integral (1 - e^{-t^2/2})^2 w(t) dt (w: the trapezoid
   weights times e^{-t^2/2}); shifted and squashed samples score far above the Gaussian.
3. The encoder with [cls]: output 0 is the [cls]; the untrained loss is finite; a checkpoint loads the way
   probes.py loads it; a few steps on one batch lower the loss.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data import Clips, PushTVideo
from levjepa import ENCODER, LeVJEPA, SIGReg, Views, param_groups
from model import VideoEncoder


def check_views() -> None:
    v, g = Views(), torch.Generator().manual_seed(0)
    for _ in range(20):
        glob, local = v.batch(32, g)
        assert glob.shape == (32, 51) and all(len(set(r.tolist())) == 51 for r in glob)
        assert len(local) == 4
        for idx in local:
            k = idx.shape[1]
            assert k in (20, 29), k                                      # 5% of 16 * 25 or 16 * 36
            s = 5 if k == 20 else 6
            t, rc = idx // 64, idx % 64
            r, c = rc // 8, rc % 8
            assert (idx < 1024).all() and (t < 16).all()
            span_r = r.max(1).values - r.min(1).values
            span_c = c.max(1).values - c.min(1).values
            assert (span_r < s).all() and (span_c < s).all(), "a local view stays inside one window"
            assert all(len(set(row.tolist())) == k for row in idx)
    print("views: global 51 of 1,024 tokens; local 20 or 29 tokens inside one 5x5 or 6x6 window  OK")


def check_sigreg() -> None:
    torch.manual_seed(0)
    sr = SIGReg()
    t, w = sr.t, sr.weights
    gauss_expected = ((1 - torch.exp(-t ** 2)) * w).sum().item()
    collapsed_per_n = ((1 - torch.exp(-t ** 2 / 2)) ** 2 * w).sum().item()
    b, d = 512, 256
    gauss = torch.stack([sr(torch.randn(b, d)) for _ in range(10)]).mean().item()
    collapsed = sr(torch.zeros(b, d)).item()
    shifted, squashed = sr(torch.randn(b, d) + 2).item(), sr(.1 * torch.randn(b, d)).item()
    print(f"sigreg: gaussian {gauss:.3f} (expected {gauss_expected:.3f}); collapsed {collapsed:.1f} "
          f"(expected {collapsed_per_n * b:.1f}); mean-shifted {shifted:.1f}; std 0.1 {squashed:.1f}")
    assert abs(gauss - gauss_expected) < .1 * gauss_expected
    assert abs(collapsed - collapsed_per_n * b) < 1e-3 * collapsed
    assert shifted > 20 * gauss and squashed > 20 * gauss
    print("sigreg: known samples  OK")


def main() -> None:
    check_views()
    check_sigreg()
    torch.manual_seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    clips = Clips(PushTVideo(), "train", n=64, seed=3)
    x = torch.stack([clips[i][0] for i in range(32)]).to(dev)
    model = LeVJEPA().to(dev)

    enc = model.encoder.eval()
    with torch.no_grad():
        full = enc(x[:2])
        assert full.shape == (2, 1025, 192), full.shape
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_levjepa_check.pt")
    torch.save({"encoder": enc.state_dict(), "config": ENCODER}, path)
    ck = torch.load(path, map_location="cpu")
    loaded = VideoEncoder(**ck["config"])
    loaded.load_state_dict(ck["encoder"])
    os.remove(path)
    with torch.no_grad():
        assert torch.allclose(loaded.to(dev).eval()(x[:2]), full, atol=1e-5)
    print("encoder: (B, 1 + 1,024, 192), [cls] first; loads as probes.py loads it  OK")

    model.train()
    g = torch.Generator().manual_seed(0)
    glob, local = Views().batch(len(x), g)
    glob, local = glob.to(dev), [v.to(dev) for v in local]
    opt = torch.optim.AdamW(param_groups(model), lr=3e-4)
    first = None
    for _ in range(30):
        loss, (inv, sig), _ = model.loss(x, glob, local)
        assert torch.isfinite(loss)
        first = first or (loss.item(), inv.item(), sig.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"30 steps on one batch: loss {first[0]:.4f} (inv {first[1]:.4f}, sig {first[2]:.3f}) -> "
          f"{loss.item():.4f} (inv {inv.item():.4f}, sig {sig.item():.3f})")
    assert loss.item() < first[0]
    print("OK")


if __name__ == "__main__":
    main()
