"""Checks for data.py: image shape and that augmentation is live.

    uv run python checks/check_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data import PathMNIST


def main():
    ds = PathMNIST("train", img_size=224)
    x = ds[0]
    print(f"{len(ds)} images, sample {tuple(x.shape)} {x.dtype}, "
          f"range [{x.min():.2f}, {x.max():.2f}]")
    assert x.shape == (3, 224, 224)
    assert not torch.equal(ds[5], ds[5]), "augmentation should differ between draws"
    print("OK")


if __name__ == "__main__":
    main()
