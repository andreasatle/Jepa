"""
PathMNIST-224 for I-JEPA pretraining.

medmnist ships the 224px split as a *compressed* .npz, so np.load pulls the
whole ~13.5 GB train array into RAM (mmap_mode is a no-op on a compressed
member). On a 24 GB unified-memory Mac that alone puts training into swap.

So: use medmnist once to materialise each split to an uncompressed .npy next
to the source, then memory-map that for training -- data RAM ~0, pages in
per sample, and the .npy memmap is picklable so num_workers > 0 works with
no per-worker copy.

    first run:  ~25 s decompress + ~13.5 GB disk write, once per split
    after:      near-zero RAM

Augmentation is light (block masking is the real difficulty):
RandomResizedCrop(scale=(0.3, 1.0)) + H/V flips (histology has no canonical
orientation). Output: (3, 224, 224) float in [-1, 1].
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

DEFAULT_ROOT = os.path.expanduser("~/.medmnist")


def _ensure_npy(split: str, root: str) -> str:
    """Decompress `{split}_images` to a sibling .npy once; return its path."""
    out = os.path.join(root, f"pathmnist_224_{split}_images.npy")
    if os.path.exists(out):
        return out
    from medmnist import PathMNIST
    print(f"[data] materialising '{split}' -> {out}  (one-time: ~25 s + ~13 GB write) ...")
    ds = PathMNIST(split=split, size=224, download=True, root=root)  # decompresses into RAM
    np.save(out, np.ascontiguousarray(ds.imgs))                      # (N, 224, 224, 3) uint8
    del ds
    print(f"[data] wrote {out}  ({os.path.getsize(out) / 1e9:.1f} GB)")
    return out


class PathMNIST224(Dataset):
    """split in {'train','val','test'}. Yields just the image tensor, no label."""

    def __init__(self, split: str = "train", augment: bool = True, img_size: int = 224,
                 root: str = DEFAULT_ROOT):
        assert split in ("train", "val", "test")
        self.images = np.load(_ensure_npy(split, root), mmap_mode="r")  # (N,224,224,3) uint8 memmap

        tfms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]  # HWC uint8 -> CHW [0,1]
        if augment:
            tfms += [
                v2.RandomResizedCrop(img_size, scale=(0.3, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
            ]
        tfms += [v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]  # -> [-1, 1]
        self.transform = v2.Compose(tfms)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        img = np.array(self.images[i])   # copy one frame off the memmap
        return self.transform(img)       # (3, 224, 224) float32


if __name__ == "__main__":
    ds = PathMNIST224("train")
    print(f"train frames: {len(ds)}  (backing array is a memmap: {isinstance(ds.images, np.memmap)})")
    x = ds[0]
    print(f"sample: {tuple(x.shape)}  {x.dtype}  min={x.min():.3f} max={x.max():.3f}")
    assert x.shape == (3, 224, 224)
    a, b = ds[10], ds[10]
    print(f"two draws of idx 10 differ (aug live): {not torch.equal(a, b)}")

    val = PathMNIST224("val", augment=False)
    print(f"val (no aug): {len(val)} frames, sample {tuple(val[0].shape)}")
    print("OK")
