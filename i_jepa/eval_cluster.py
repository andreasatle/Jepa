"""
Cluster check: project the frozen encoder's mean-pooled features to 2D and
colour by PathMNIST tissue class.

Loads i_jepa/jepa_ckpt.pt (or a random-init encoder if absent -- useful as a
"before" baseline). Extracts  feat = E(img).mean(over patches)  for a subset
of the test split, runs PCA -> t-SNE, scatter-plots by class, and prints two
numbers on the raw features:
  * silhouette score   (how separated the 9 class clouds are, -1..1)
  * 5-NN accuracy       (30% held-out split; chance = 1/9 ~ 0.11)

Runs on CPU by default so it doesn't fight a training run for the GPU.

    python i_jepa/eval_cluster.py [--n 3000] [--device cpu|mps] [--ckpt PATH]
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medmnist import INFO, PathMNIST
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torchvision.transforms import v2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from train import Config
from vit import ViT

OUT = os.path.join(_HERE, "cluster.png")


def build_encoder(ckpt_path, device):
    cfg = Config()
    enc = ViT(img_size=cfg.img_size, patch_size=cfg.patch, in_chans=3,
              embed_dim=cfg.embed_dim, depth=cfg.depth, num_heads=cfg.num_heads)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        enc.load_state_dict(ck["encoder"])
        tag = f"ckpt epoch {ck.get('epoch', '?')}, step {ck.get('step', '?')}"
    else:
        tag = "RANDOM INIT (no checkpoint)"
    return enc.eval().to(device), tag


@torch.no_grad()
def extract(enc, ds, idx, device, batch):
    feats, labels = [], []
    for i in range(0, len(idx), batch):
        items = [ds[int(j)] for j in idx[i:i + batch]]
        x = torch.stack([it[0] for it in items]).to(device)
        feats.append(enc(x).mean(dim=1).cpu())          # (b, 196, D) -> (b, D)
        labels.extend(int(np.asarray(it[1]).ravel()[0]) for it in items)  # label is shape (1,)
        print(f"\r  features {min(i + batch, len(idx))}/{len(idx)}", end="", flush=True)
    print()
    return torch.cat(feats).numpy(), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "jepa_ckpt.pt"))
    args = ap.parse_args()

    device = torch.device(args.device)
    enc, tag = build_encoder(args.ckpt, device)
    print(f"encoder: {tag}   device={device}")

    tf = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    ds = PathMNIST(split="test", size=224, download=True, transform=tf)  # (img, label[1])
    idx = np.random.RandomState(0).permutation(len(ds))[:args.n]

    feat, y = extract(enc, ds, idx, device, args.batch)
    fz = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)

    sil = silhouette_score(fz, y)
    xtr, xte, ytr, yte = train_test_split(fz, y, test_size=0.3, stratify=y, random_state=0)
    acc = KNeighborsClassifier(n_neighbors=5).fit(xtr, ytr).score(xte, yte)
    print(f"silhouette {sil:.3f}   5-NN acc {acc:.3f}  (chance {1/9:.3f})")

    p = PCA(n_components=min(50, fz.shape[1]), random_state=0).fit_transform(fz)
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(p)

    names = [INFO["pathmnist"]["label"][str(k)] for k in range(9)]
    fig, ax = plt.subplots(figsize=(9, 8))
    for k in range(9):
        m = y == k
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6, label=f"{k}  {names[k]}")
    ax.legend(markerscale=2.5, fontsize=8, loc="best", framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"PathMNIST test -- mean-pooled E features\n"
                 f"{tag}   |   silhouette {sil:.3f}   5-NN acc {acc:.3f}")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
