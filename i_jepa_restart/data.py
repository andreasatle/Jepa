"""
PathMNIST images for pretraining (no labels).

Source: the 224px PathMNIST split (MedMNIST+), materialised once as an
uncompressed .npy in ~/.medmnist and memory-mapped, so it costs ~0 RAM.
(medmnist ships a compressed .npz; loading it pulls ~13.5 GB into memory.)
The old ../i_jepa run already wrote train and test; other splits are written
on first use.

Augmentation, as in the official I-JEPA code: RandomResizedCrop(scale 0.3-1.0)
to `img_size`, nothing else -- no flips, no colour jitter; the masking is the
task. Normalised with the ImageNet mean/std, also as in the official code.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

ROOT = os.path.expanduser("~/.medmnist")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _npy_path(split: str) -> str:
    path = os.path.join(ROOT, f"pathmnist_224_{split}_images.npy")
    if not os.path.exists(path):
        from medmnist import PathMNIST
        print(f"[data] writing {path} (one-time) ...")
        ds = PathMNIST(split=split, size=224, download=True, root=ROOT)
        np.save(path, np.ascontiguousarray(ds.imgs))          # (N, 224, 224, 3) uint8
    return path


class PathMNIST(Dataset):
    """Yields (3, img_size, img_size) float images, no labels."""

    def __init__(self, split: str, img_size: int, crop_scale=(0.3, 1.0)):
        self.images = np.load(_npy_path(split), mmap_mode="r")
        self.transform = v2.Compose([
            v2.ToImage(),                                              # HWC uint8 -> CHW
            v2.RandomResizedCrop(img_size, scale=crop_scale, antialias=True),
            v2.ToDtype(torch.float32, scale=True),                     # -> [0, 1]
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.transform(np.array(self.images[i]))              # copy one image off the memmap
