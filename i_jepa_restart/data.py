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


class MultiCrop(PathMNIST):
    """
    LeJEPA's views (the authors' DINO-style recipe): yields a list of `n_global`
    (3, img_size, img_size) views, crop scale 0.3-1.0, then `n_local` (3, local_size,
    local_size) views, scale 0.05-0.3. Each view: horizontal flip, colour jitter
    (p 0.8; brightness 0.4, contrast 0.4, saturation 0.2, hue 0.1), grayscale (p 0.2),
    Gaussian blur (p 0.5), solarize (p 0.2). Colour augmentation matters here: without
    it, stain colour alone tells views of the same image apart from others.

    zoom=False: no resizing, so a cell is always the same number of pixels (PathMNIST
    has one fixed magnification, and size is diagnostic). Global views are the whole
    image; local views are random local_size windows at native resolution.
    """

    def __init__(self, split: str, img_size: int, local_size: int, n_global: int, n_local: int,
                 zoom: bool = True):
        super().__init__(split, img_size)
        if zoom:
            self.global_view = self.view(v2.RandomResizedCrop(img_size, scale=(0.3, 1.0), antialias=True))
            self.local_view = self.view(v2.RandomResizedCrop(local_size, scale=(0.05, 0.3), antialias=True))
        else:
            self.global_view = self.view(v2.Identity())
            self.local_view = self.view(v2.RandomCrop(local_size))
        self.n_global, self.n_local = n_global, n_local

    @staticmethod
    def view(crop) -> v2.Compose:
        return v2.Compose([
            v2.ToImage(),
            crop,
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomSolarize(threshold=128, p=0.2),                   # on uint8, before scaling
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __getitem__(self, i: int) -> list[torch.Tensor]:
        img = np.array(self.images[i])
        return ([self.global_view(img) for _ in range(self.n_global)]
                + [self.local_view(img) for _ in range(self.n_local)])


class RawPathMNIST(PathMNIST):
    """Yields the untouched (224, 224, 3) uint8 image; NativeWindows makes the views per batch."""

    def __getitem__(self, i: int) -> np.ndarray:
        return np.array(self.images[i])


class NativeWindows:
    """
    collate_fn for RawPathMNIST: LeJEPA views as windows at native resolution, never
    resized, so a cell is always the same number of pixels, but views still see more
    or less tissue. Per batch, one global size is drawn from `global_sizes` and one
    local size from `local_sizes` (views in one slot must share a size to be stacked);
    each window's position is drawn per image. Then the same flips and colour
    augmentation as MultiCrop. Returns a list of n_global + n_local (B, 3, s, s) tensors.
    torch's generator, not numpy's: DataLoader seeds it per worker.
    """

    def __init__(self, global_sizes: list[int], local_sizes: list[int], n_global: int, n_local: int):
        self.global_sizes, self.local_sizes = global_sizes, local_sizes
        self.n_global, self.n_local = n_global, n_local
        self.augment = MultiCrop.view(v2.Identity())

    @staticmethod
    def pick(options: list[int]) -> int:
        return options[int(torch.randint(len(options), ()))]

    @staticmethod
    def window(img: np.ndarray, size: int) -> np.ndarray:
        y, x = (int(torch.randint(n - size + 1, ())) for n in img.shape[:2])
        return np.ascontiguousarray(img[y:y + size, x:x + size])

    def __call__(self, images: list[np.ndarray]) -> list[torch.Tensor]:
        sizes = ([self.pick(self.global_sizes)] * self.n_global
                 + [self.pick(self.local_sizes)] * self.n_local)
        return [torch.stack([self.augment(self.window(img, s)) for img in images]) for s in sizes]
