"""
The label-scarce test (PLAN.md goal 2): with only a small fraction of the train labels
(default 1%, about 900 images, stratified by class), does a linear classifier on frozen
LeJEPA features beat a network trained from scratch on those same labels?

Three arms, each on the same labelled subset (one subset per seed):

  probe     frozen encoder features (mean-pooled tokens, as probes/probe.py), linear
            probe and k-NN fitted on the subset only. With --random: the floor.
  scratch   a network trained end to end on the subset: --arch vit (the same ViT-Tiny
            as the encoder, plus a linear head on the mean-pooled tokens) or --arch
            resnet18 (the strong standard baseline). Same augmentations as LeJEPA's
            global views (flip, colour jitter, grayscale, blur, solarize; no resizing).
            Fixed recipe, no early stopping: in a label-scarce setting the 10k val labels
            can't be used to pick a checkpoint, so every arm is scored once at the end.
            --arch resnet18-imagenet: the same ResNet-18 starting from ImageNet weights and
            fine-tuned (lr 1e-4, the standard for fine-tuning): what pathology uses in practice.
  imagenet  frozen ImageNet ResNet-18 features (512-d, global average pool), the same linear
            probe and k-NN as the probe arm: the like-for-like comparison with frozen LeJEPA.

    uv run python probes/scarce.py probe runs/lejepa-nozoom/encoder_e010.pt --seed 0
    uv run python probes/scarce.py probe runs/lejepa-nozoom/encoder_e010.pt --random --seed 0
    uv run python probes/scarce.py scratch --arch resnet18 --seed 0
    uv run python probes/scarce.py imagenet --seed 0
    options: --frac F (default 0.01)   --split val|test (test only for the final number)
    The scratch arm scores val and test both at the end of its fixed recipe (nothing is chosen
    from either), so a network is trained once.

Results are appended to probes/scarce_results.csv.
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from data import MultiCrop, _npy_path
from model import Encoder
from probe import LABELS, Labelled, embed, knn_probe, linear_probe

RESULTS = os.path.join(HERE, "scarce_results.csv")


def labelled_subset(frac: float, seed: int) -> np.ndarray:
    """Indices of a class-stratified subset of the train split: round(frac * n_c) per class."""
    labels = np.load(LABELS)["train_labels"][:, 0]
    rng = np.random.default_rng(seed)
    idx = [rng.choice(np.flatnonzero(labels == c), max(1, round(frac * (labels == c).sum())),
                      replace=False) for c in range(9)]
    return np.sort(np.concatenate(idx))


def record(arm: str, args, n: int, acc: float, extra: str = "") -> None:
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arm", "frac", "seed", "n_labels", "split", "accuracy", "note"])
        w.writerow([arm, args.frac, args.seed, n, args.split, f"{acc:.4f}", extra])


# ---- probe arm: frozen features, fitted on the subset ------------------------------------

class ImageNetFeatures(nn.Module):
    """ImageNet ResNet-18 without its head, as one 'token' per image so probe.embed's mean-pool fits."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x).unsqueeze(1)                 # (B, 1, 512)


def run_probe(args):
    device = pick_device()
    if args.arm == "imagenet":
        encoder, name = ImageNetFeatures(), "imagenet-resnet18"
    else:
        ck = torch.load(args.ckpt, map_location="cpu")
        cfg = ck["config"]
        torch.manual_seed(cfg["seed"])
        encoder = Encoder(cfg["img_size"], cfg["patch_size"], 3, cfg["embed_dim"], cfg["depth"],
                          cfg["num_heads"])
        if not args.random:
            encoder.load_state_dict(ck["encoder"])
        run = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
        name = f"{run}_random" if args.random else f"{run}_e{ck['epoch']:03d}"
    encoder.to(device).eval()

    feats = {}
    for split in ["train", args.split]:                 # same cache as probes/probe.py
        path = os.path.join(HERE, "cache", f"{name}_{split}_all.pt")
        if not os.path.exists(path):
            t = time.time()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(embed(encoder, split, 0, device), path)
            print(f"embedded {split} in {time.time() - t:.0f}s")
        feats[split] = torch.load(path)

    idx = torch.from_numpy(labelled_subset(args.frac, args.seed))
    xtr, ytr = feats["train"][0][idx], feats["train"][1][idx]
    xev, yev = feats[args.split]
    lin_train, lin = linear_probe(xtr, ytr, xev, yev)
    knn = knn_probe(xtr, ytr, xev, yev, k=min(20, len(idx)))
    print(f"{name}  {len(idx)} labels  [{args.split}]  linear {lin:.3f} (train {lin_train:.3f})"
          f"   k-NN {knn:.3f}")
    record(f"probe-linear {name}", args, len(idx), lin, f"train {lin_train:.3f}")
    record(f"probe-knn {name}", args, len(idx), knn)


# ---- scratch arm: a network trained end to end on the subset -----------------------------

class AugmentedSubset(Dataset):
    """The labelled train subset, with LeJEPA's global-view augmentations (no resizing)."""

    def __init__(self, idx: np.ndarray):
        self.images = np.load(_npy_path("train"), mmap_mode="r")
        self.labels = np.load(LABELS)["train_labels"][:, 0].astype(np.int64)
        self.idx = idx
        self.transform = MultiCrop.view(v2.Identity())

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.transform(np.array(self.images[j])), self.labels[j]


class ViTClassifier(nn.Module):
    """The pretraining encoder's architecture (ViT-Tiny), random init, linear head on mean-pooled tokens."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(224, 16, 3, 192, 12, 3)
        self.head = nn.Linear(192, 9)

    def forward(self, x):
        return self.head(self.encoder(x).mean(1))


def build(arch: str) -> nn.Module:
    if arch == "vit":
        return ViTClassifier()
    if arch == "resnet18-imagenet":
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(net.fc.in_features, 9)
        return net
    return torchvision.models.resnet18(weights=None, num_classes=9)


@torch.no_grad()
def evaluate(model, split, device) -> float:
    model.eval()
    loader = DataLoader(Labelled(split), batch_size=256, num_workers=4, multiprocessing_context="fork")
    correct = total = 0
    for x, y in loader:
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        total += len(y)
    return correct / total


def run_scratch(args):
    torch.manual_seed(args.seed)
    idx = labelled_subset(args.frac, args.seed)
    loader = DataLoader(AugmentedSubset(idx), batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=6, multiprocessing_context="fork", persistent_workers=True)
    it = iter(loader)                                   # fork the workers before MPS is touched
    device = pick_device()
    model = build(args.arch).to(device)
    lr = args.lr or (1e-4 if args.arch == "resnet18-imagenet" else 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps, warmup = args.epochs * len(loader), 5 * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
    t = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot = correct = n = 0
        for _ in range(len(loader)):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        if epoch % 10 == 9 or epoch == args.epochs - 1:
            print(f"e{epoch + 1}  loss {tot / n:.3f}  train acc {correct / n:.3f}  {time.time() - t:.0f}s",
                  flush=True)
    for split in ["val", "test"]:
        acc = evaluate(model, split, device)
        print(f"scratch-{args.arch}  {len(idx)} labels  [{split}]  accuracy {acc:.3f}")
        args.split = split
        record(f"scratch-{args.arch}", args, len(idx), acc,
               f"train acc {correct / n:.3f}, {args.epochs} ep, lr {lr:g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["probe", "scratch", "imagenet"])
    ap.add_argument("ckpt", nargs="?", help="probe arm: encoder snapshot (encoder_eXXX.pt)")
    ap.add_argument("--random", action="store_true", help="probe arm: same config, random init (the floor)")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "vit", "resnet18-imagenet"])
    ap.add_argument("--frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0, help="which labelled subset (and init)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, help="default 1e-3 from scratch, 1e-4 fine-tuning ImageNet weights")
    args = ap.parse_args()
    run_scratch(args) if args.arm == "scratch" else run_probe(args)


def pick_device() -> torch.device:
    """Called only after the DataLoader workers are forked: forking after MPS is touched segfaults them."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


if __name__ == "__main__":
    main()
