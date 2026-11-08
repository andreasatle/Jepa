"""Checks for LeJEPA (lejepa.py, the run's recipe): the multi-crop views and SIGReg's value on known samples.

    uv run python checks/check_lejepa.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torchvision.transforms import v2

from data import MultiCrop
from lejepa import Config
from loss import sigreg_loss


def check_views():
    """2 global views of the whole image and 6 local windows, cut at native resolution, never resized."""
    cfg = Config()
    ds = MultiCrop("train", cfg.img_size, cfg.local_size, cfg.global_views, cfg.local_views, cfg.zoom)
    views = ds[0]
    assert len(views) == cfg.global_views + cfg.local_views == 8
    assert all(v.shape == (3, 224, 224) for v in views[:2]), [tuple(v.shape) for v in views]
    assert all(v.shape == (3, 96, 96) for v in views[2:]), [tuple(v.shape) for v in views]
    assert not torch.equal(views[0], views[1]), "the two global views should differ (augmentation)"

    # Native resolution: the crop step alone returns the whole image (global) or an exact sub-window
    # of it (local), so no pixel is ever resampled.
    img = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    full = v2.ToImage()(img)
    g_crop, l_crop = ds.global_view.transforms[1], ds.local_view.transforms[1]
    assert isinstance(g_crop, v2.Identity) and torch.equal(g_crop(full), full)
    win = l_crop(full)
    hits = [(y, x) for y in range(224 - 95) for x in range(224 - 95)
            if torch.equal(full[:, y:y + 96, x:x + 96], win)]
    assert len(hits) == 1, "local view must be an exact native-resolution window"
    print("views: 2 x 224 + 6 x 96, native resolution  OK")


def check_sigreg():
    """n * Epps-Pulley: about 0.52 on a Gaussian sample, about 0.131 * n on a collapsed one, and large
    gaps in between, whatever the dimension."""
    torch.manual_seed(0)
    n, d = 512, 64
    gauss = torch.stack([sigreg_loss(torch.randn(n, d)) for _ in range(5)]).mean().item()
    collapsed = sigreg_loss(torch.zeros(n, d)).item()
    shifted = sigreg_loss(torch.randn(n, d) + 2.0).item()
    squashed = sigreg_loss(0.1 * torch.randn(n, d)).item()
    print(f"sigreg: gaussian {gauss:.3f}   collapsed {collapsed:.1f} (= {collapsed / n:.4f} n)"
          f"   mean-shifted {shifted:.1f}   std 0.1 {squashed:.1f}")
    assert 0.4 < gauss < 0.65, gauss
    assert abs(collapsed / n - 0.131) < 0.002, collapsed / n
    assert shifted > 20 * gauss and squashed > 20 * gauss
    print("sigreg: known samples  OK")


def main():
    check_sigreg()
    check_views()
    print("OK")


if __name__ == "__main__":
    main()
