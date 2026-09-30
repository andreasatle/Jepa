"""
Frozen-encoder probes: does the embedding separate the 9 tissue classes?

  1. Freeze the encoder. Embed every image: full image (no masking, no
     augmentation), mean of the 196 patch embeddings -> one vector per image.
  2. Linear probe: multinomial logistic regression on the train-split vectors
     (standardised), scored on the eval split. Linear = straight boundaries only,
     so success means the classes were already separated in the embedding.
  3. k-NN probe: no training at all. Each eval image gets the majority label of
     its k nearest train images (cosine similarity). Tests the raw geometry.

Always read the numbers against the floor: the same probes on a randomly
initialised encoder of the same size (--random).

    uv run python probes/probe.py runs/<run>/encoder_e010.pt            # trained
    uv run python probes/probe.py runs/<run>/encoder_e010.pt --random   # floor, same config
    options: --split val|test (default val; test only for the final number)
             --limit N        (use N images per split; quick sanity runs)

Features are cached in probes/cache/, results appended to probes/results.csv.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from data import IMAGENET_MEAN, IMAGENET_STD, ROOT, _npy_path
from model import Encoder

LABELS = os.path.join(ROOT, "pathmnist_224.npz")


class Labelled(Dataset):
    """Images without augmentation (just normalised), plus labels."""

    def __init__(self, split: str, limit: int = 0):
        self.images = np.load(_npy_path(split), mmap_mode="r")
        self.labels = np.load(LABELS)[f"{split}_labels"][:, 0].astype(np.int64)
        self.n = min(limit, len(self.labels)) if limit else len(self.labels)
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                     v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.transform(np.array(self.images[i])), self.labels[i]


@torch.no_grad()
def embed(encoder, split, limit, device, batch_size=256):
    """(N, D) mean-pooled patch embeddings and (N,) labels."""
    loader = DataLoader(Labelled(split, limit), batch_size=batch_size, num_workers=4,
                        multiprocessing_context="fork")
    feats, labels = [], []
    for x, y in loader:
        feats.append(encoder(x.to(device)).mean(1).float().cpu())    # (B, N, D) -> (B, D)
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe(xtr, ytr, xev, yev, weight_decay=1e-4, iters=500):
    """Logistic regression, full batch, L-BFGS. Returns (train acc, eval acc)."""
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xev = (xtr - mu) / sd, (xev - mu) / sd                   # train statistics only
    lin = torch.nn.Linear(xtr.shape[1], 9)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=iters, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(xtr), ytr) + weight_decay * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return ((lin(xtr).argmax(1) == ytr).float().mean().item(),
                (lin(xev).argmax(1) == yev).float().mean().item())


@torch.no_grad()
def knn_probe(xtr, ytr, xev, yev, k=20, chunk=1024):
    """Majority vote of the k most cosine-similar train images. Returns eval acc."""
    xtr, xev = F.normalize(xtr, dim=1), F.normalize(xev, dim=1)
    correct = 0
    for i in range(0, len(xev), chunk):
        nn_idx = (xev[i:i + chunk] @ xtr.T).topk(k, dim=1).indices    # (chunk, k)
        votes = F.one_hot(ytr[nn_idx], 9).sum(1)                        # (chunk, 9)
        correct += (votes.argmax(1) == yev[i:i + chunk]).sum().item()
    return correct / len(xev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="encoder snapshot from train.py (encoder_eXXX.pt)")
    ap.add_argument("--random", action="store_true", help="same config, random init (the floor)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck["config"]
    torch.manual_seed(cfg["seed"])
    encoder = Encoder(cfg["img_size"], cfg["patch_size"], 3, cfg["embed_dim"], cfg["depth"],
                      cfg["num_heads"])
    if not args.random:
        encoder.load_state_dict(ck["encoder"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder.to(device).eval()

    run = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    name = f"{run}_random" if args.random else f"{run}_e{ck['epoch']:03d}"
    feats = {}
    for split in ["train", args.split]:
        path = os.path.join(HERE, "cache", f"{name}_{split}_{args.limit or 'all'}.pt")
        if os.path.exists(path):
            feats[split] = torch.load(path)
        else:
            t = time.time()
            feats[split] = embed(encoder, split, args.limit, device)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(feats[split], path)
            print(f"embedded {split}: {len(feats[split][1])} images in {time.time() - t:.0f}s")

    (xtr, ytr), (xev, yev) = feats["train"], feats[args.split]
    lin_train, lin_eval = linear_probe(xtr, ytr, xev, yev)
    knn = knn_probe(xtr, ytr, xev, yev)
    print(f"{name}  [{args.split}]  linear {lin_eval:.3f} (train {lin_train:.3f})   k-NN {knn:.3f}")

    out = os.path.join(HERE, "results.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["encoder", "split", "limit", "linear", "linear_train", "knn"])
        w.writerow([name, args.split, args.limit or "all", f"{lin_eval:.4f}",
                    f"{lin_train:.4f}", f"{knn:.4f}"])


if __name__ == "__main__":
    main()
