"""
Frozen-encoder downstream evaluation for the I-JEPA PathMNIST encoder.

`eval_cluster.py` asks "do the features cluster?" -- this asks the question the
pretraining actually exists to answer: "are they good for the task?", i.e.
9-way colorectal tissue classification. Everything here treats E as a fixed
feature extractor; nothing backprops into the ViT.

Three probes, reported in one table:

  * 5-NN            -- non-parametric, zero training. Same number eval_cluster
                       prints, repeated here so the probes sit side by side.
  * linear          -- multinomial logistic regression on the mean-pooled
                       tokens  feat = E(img).mean(patches), with a small C
                       sweep on a held-out slice of train. The canonical SSL
                       metric: how linearly separable are the classes.
  * attentive       -- (--tokens) a one-query cross-attention head pools the
                       196 patch tokens instead of averaging them, then a
                       linear layer. This is what the I-JEPA paper reports;
                       it usually beats mean-pool + linear by a few points
                       because averaging discards which patch said what.
                       Trains a tiny head only -- E stays frozen.

Plus:

  * confusion matrix -- 9x9 per probe, row-normalised. Expect the mass off the
                        diagonal to sit in the {adeno epithelium, normal
                        mucosa, cancer-associated stroma, smooth muscle} block:
                        those four are morphologically close and are the whole
                        ballgame once the easy classes are solved.
  * effective rank   -- entropy / participation-ratio / stable rank of the
                        centered raw features. z_std (in train.py) says per-dim
                        variance is ~1; this says how many dimensions are
                        actually *used*. A fall here across checkpoints while
                        z_std holds is the low-rank "easy solution" the EMA-free
                        loss can slide into (see loss.py).
  * label efficiency -- (--label-efficiency) refit the linear probe on
                        1% / 10% / ... / 100% of the labels, plot test acc vs
                        fraction. Run once normally and once with --ckpt none
                        (random init), then pass the saved npz to --overlay to
                        draw both curves on one axis. That gap is the argument
                        for having pretrained at all.

Every run appends a line to i_jepa/probe_out/probe_log.csv (keyed by checkpoint
step, one row per tag+tokens-mode -- re-running replaces its row). Run this once
per epoch against the live jepa_ckpt.pt and --plot-log renders acc + effective
rank vs pretraining step: the curve to watch instead of eyeballing t-SNE.

Mean-pooled features are cached under i_jepa/probe_cache/ keyed by split +
checkpoint step, so repeated runs skip the ViT. Token grids (--tokens) are
cached as an fp16 memmap. Delete probe_cache/ to force re-extraction.

    uv run python i_jepa/eval_probe.py                        # kNN + linear + confusion
    uv run python i_jepa/eval_probe.py --tokens --device mps  # + attentive probe
    uv run python i_jepa/eval_probe.py --plot-log             # + redraw the per-epoch curve
    uv run python i_jepa/eval_probe.py --label-efficiency
    uv run python i_jepa/eval_probe.py --ckpt none --label-efficiency  # random baseline
    uv run python i_jepa/eval_probe.py --label-efficiency \
        --overlay i_jepa/probe_out/randinit_labeff.npz
"""

import argparse
import csv
import glob
import os
import sys
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medmnist import INFO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from train import Config
from vit import ViT
from data import PathMNIST224, DEFAULT_ROOT

CACHE = os.path.join(_HERE, "probe_cache")
OUT = os.path.join(_HERE, "probe_out")
NAMES = [INFO["pathmnist"]["label"][str(k)] for k in range(9)]
RANDOM_TOKENS_CAP = 20000  # fp16 token memmap for the full 90k train split is ~13 GB


# ---------------------------------------------------------------------------
# Encoder + data
# ---------------------------------------------------------------------------

def build_encoder(ckpt_path, device):
    """Load E from a checkpoint, or a seeded random-init ViT if ckpt_path is
    'none'/'random' or missing. Returns (encoder, tag, pretty, epoch, step);
    epoch/step are -1 for random init."""
    cfg = Config()
    torch.manual_seed(0)
    enc = ViT(img_size=cfg.img_size, patch_size=cfg.patch, in_chans=3,
              embed_dim=cfg.embed_dim, depth=cfg.depth, num_heads=cfg.num_heads)
    is_random = (not ckpt_path) or ckpt_path.lower() in ("none", "random", "randinit")
    if not is_random and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        enc.load_state_dict(ck["encoder"])
        epoch, step = int(ck.get("epoch", -1)), int(ck.get("step", -1))
        tag = f"e{epoch}s{step}"
        pretty = f"ckpt epoch {epoch}, step {step}"
    else:
        tag, pretty, epoch, step = "randinit", "RANDOM INIT", -1, -1
    return enc.eval().to(device), tag, pretty, epoch, step


def load_labels(split):
    """(N,) int64 labels for a split, read straight from the medmnist npz (only
    the tiny labels member decompresses) and cached as a .npy."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"labels_{split}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    cands = sorted(glob.glob(os.path.join(DEFAULT_ROOT, "pathmnist*.npz")))
    cands.sort(key=lambda p: ("224" not in p, p))  # prefer the 224 npz
    if not cands:
        raise FileNotFoundError(
            f"no pathmnist*.npz in {DEFAULT_ROOT} -- run i_jepa/train.py once so "
            "medmnist downloads it, then retry.")
    with np.load(cands[0]) as z:
        y = z[f"{split}_labels"].reshape(-1).astype(np.int64)
    np.save(cache, y)
    return y


@torch.no_grad()
def get_features(enc, split, n, device, batch, tag, want_tokens):
    """Mean-pooled features (n, D) for `n` random samples of a split (n<=0 or
    n>=N means all), with labels. If want_tokens, also an fp16 memmap of the
    full token grids (n, 196, D). Both are cached under CACHE keyed by tag+n."""
    os.makedirs(CACHE, exist_ok=True)
    ds = PathMNIST224(split, augment=False)
    N = len(ds)
    take = N if (n is None or n <= 0 or n >= N) else n
    idx = np.random.RandomState(0).permutation(N)[:take]
    y = load_labels(split)[idx]

    mean_path = os.path.join(CACHE, f"mean_{split}_{tag}_n{take}.npz")
    tok_path = os.path.join(CACHE, f"tok_{split}_{tag}_n{take}.npy")
    have_mean = os.path.exists(mean_path)
    have_tok = os.path.exists(tok_path)
    if have_mean and (not want_tokens or have_tok):
        mean = np.load(mean_path)["feat"]
        tok = np.load(tok_path, mmap_mode="r") if want_tokens else None
        return mean, tok, y

    D = Config().embed_dim
    ntok = enc.patch_embed.num_patches
    mean = np.empty((take, D), np.float32)
    tok = (np.lib.format.open_memmap(tok_path, mode="w+", dtype=np.float16,
                                     shape=(take, ntok, D))
           if want_tokens else None)
    t0 = time.time()
    for i in range(0, take, batch):
        sl = idx[i:i + batch]
        x = torch.stack([ds[int(j)] for j in sl]).to(device)
        h = enc(x)                                   # (b, ntok, D)
        mean[i:i + len(sl)] = h.mean(1).float().cpu().numpy()
        if want_tokens:
            tok[i:i + len(sl)] = h.half().cpu().numpy()
        done = min(i + batch, take)
        print(f"\r  [{split}] {done}/{take}  {done / max(1e-6, time.time() - t0):.0f} img/s",
              end="", flush=True)
    print()
    np.savez(mean_path, feat=mean)
    if want_tokens:
        tok.flush()
        tok = np.load(tok_path, mmap_mode="r")
    return mean, tok, y


def standardize(train, *others):
    """z-score every column by train statistics; apply the same shift/scale to
    the other arrays (no test leakage)."""
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return tuple((a - mu) / sd for a in (train, *others))


def feature_geometry(x, cap=20000):
    """How many feature dimensions actually carry information, from the singular
    spectrum of the CENTERED RAW features (standardizing first would force every
    dim to unit variance and erase the signal). Three views:

      * eff_rank_entropy  exp(H) of the normalised singular values
                          (Roy & Vetterli 2007) -- equals `dims` iff perfectly
                          flat, collapses toward 1 as energy concentrates.
      * eff_rank_pr       participation ratio (sum s^2)^2 / sum s^4.
      * stable_rank       sum s^2 / max s^2.

    A fall in these across checkpoints while train.py's z_std stays ~1 is the
    low-rank "easy solution" the EMA-free loss can drift into (see loss.py):
    per-dim variance still ~1, but only a handful of directions are used.
    """
    xc = x - x.mean(0)
    if len(xc) > cap:
        xc = xc[np.random.RandomState(0).choice(len(xc), cap, replace=False)]
    s = np.linalg.svd(xc, compute_uv=False)
    s = s[s > 0]
    p = s / s.sum()
    return {
        "dims": int(x.shape[1]),
        "eff_rank_entropy": float(np.exp(-(p * np.log(p)).sum())),
        "eff_rank_pr": float((s ** 2).sum() ** 2 / (s ** 4).sum()),
        "stable_rank": float((s ** 2).sum() / (s ** 2).max()),
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def knn_probe(xtr, ytr, xte, yte, k):
    clf = KNeighborsClassifier(n_neighbors=k).fit(xtr, ytr)
    pred = clf.predict(xte)
    return (pred == yte).mean(), pred


def linear_probe(xtr, ytr, xte, yte, cs=(0.01, 0.1, 1.0, 10.0)):
    """Logistic regression with a C sweep picked on a 15% slice of train, then
    refit at the best C on all of train."""
    xt, xv, yt, yv = train_test_split(xtr, ytr, test_size=0.15, stratify=ytr, random_state=0)
    scored = []
    for c in cs:
        v = LogisticRegression(C=c, max_iter=2000).fit(xt, yt).score(xv, yv)
        scored.append((v, c))
    best_c = max(scored)[1]
    clf = LogisticRegression(C=best_c, max_iter=2000).fit(xtr, ytr)
    pred = clf.predict(xte)
    return (pred == yte).mean(), pred, best_c, scored


class AttentivePool(torch.nn.Module):
    """One learned query cross-attends over the patch tokens -> a single
    pooled vector -> linear classifier. The only trained parameters."""

    def __init__(self, dim, n_cls, heads=6, p=0.1):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = torch.nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        self.norm = torch.nn.LayerNorm(dim)
        self.drop = torch.nn.Dropout(p)
        self.fc = torch.nn.Linear(dim, n_cls)

    def forward(self, tok):                          # (B, N, D)
        q = self.q.expand(tok.size(0), -1, -1)
        pooled, _ = self.attn(q, tok, tok, need_weights=False)
        return self.fc(self.drop(self.norm(pooled.squeeze(1))))


def attentive_probe(tok_tr, ytr, tok_te, yte, device, epochs, bs=256, lr=1e-3):
    """Train an AttentivePool head on cached token grids. tok_tr/tok_te are
    (n, N, D) fp16 memmaps; rows align with ytr/yte."""
    dim = int(tok_tr.shape[-1])
    tr_i, va_i = train_test_split(np.arange(len(ytr)), test_size=0.15,
                                  stratify=ytr, random_state=0)
    ytr_t = torch.tensor(ytr, device=device)

    def gather(mm, rows):
        rows = np.sort(rows)
        return torch.from_numpy(np.asarray(mm[rows]).astype(np.float32)).to(device), rows

    head = AttentivePool(dim, 9).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = torch.nn.CrossEntropyLoss()
    rng = np.random.RandomState(0)

    best_va, best_state = -1.0, None
    for ep in range(epochs):
        head.train()
        perm = rng.permutation(tr_i)
        for i in range(0, len(perm), bs):
            xb, rows = gather(tok_tr, perm[i:i + bs])
            loss = lossf(head(xb), ytr_t[rows])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            hit = tot = 0
            for i in range(0, len(va_i), bs):
                xb, rows = gather(tok_tr, va_i[i:i + bs])
                hit += (head(xb).argmax(1).cpu().numpy() == ytr[rows]).sum()
                tot += len(rows)
        va = hit / tot
        print(f"\r  attentive ep {ep + 1}/{epochs}  val acc {va:.4f}", end="", flush=True)
        if va > best_va:
            best_va = va
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    print()

    head.load_state_dict(best_state)
    head.eval()
    preds = np.empty(len(yte), np.int64)
    with torch.no_grad():
        for i in range(0, len(yte), bs):                 # test rows are contiguous & in order
            xb = torch.from_numpy(np.asarray(tok_te[i:i + bs]).astype(np.float32)).to(device)
            preds[i:i + bs] = head(xb).argmax(1).cpu().numpy()
    return (preds == yte).mean(), preds, best_va


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def confusion_plot(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(9)).astype(float)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    im = ax.imshow(cmn, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_yticklabels([f"{k}  {NAMES[k]}" for k in range(9)], fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] < 0.6 else "black")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def label_efficiency(xtr, ytr, xte, yte, fractions):
    accs = []
    for f in fractions:
        if f >= 1.0:
            xs, ys = xtr, ytr
        else:
            xs, _, ys, _ = train_test_split(xtr, ytr, train_size=f, stratify=ytr,
                                            random_state=0)
        a = (LogisticRegression(C=1.0, max_iter=2000).fit(xs, ys).predict(xte) == yte).mean()
        accs.append(a)
        print(f"  {f * 100:6.2f}%   n={len(ys):6d}   test acc {a:.4f}")
    return np.array(fractions, float), np.array(accs, float)


def label_efficiency_plot(fr, acc, pretty, path, overlay=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fr * 100, acc, "o-", label=pretty)
    if overlay is not None:
        o = np.load(overlay)
        ax.plot(o["fractions"] * 100, o["acc"], "s--", label=str(o["pretty"]))
    ax.set_xscale("log")
    ax.set_xlabel("labelled fraction of train (%)")
    ax.set_ylabel("linear-probe test accuracy")
    ax.set_title("PathMNIST label efficiency -- frozen E + linear probe")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


LOG_COLS = ["tag", "epoch", "step", "n_train", "n_test", "tokens", "dims",
            "eff_rank_entropy", "eff_rank_pr", "stable_rank",
            "knn_acc", "knn_f1", "linear_acc", "linear_f1", "linear_C",
            "attentive_acc", "attentive_f1", "attentive_val"]


def append_log(row, path):
    """One line per run into probe_out/probe_log.csv. Re-running on the same
    checkpoint (same tag+tokens) overwrites its previous line rather than
    stacking duplicates."""
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as fh:
            rows = [r for r in csv.DictReader(fh)
                    if not (r.get("tag") == row["tag"]
                            and r.get("tokens") == str(row["tokens"]))]
    rows.append({k: row.get(k, "") for k in LOG_COLS})
    rows.sort(key=lambda r: (int(r["step"]) if str(r["step"]).lstrip("-").isdigit() else 0))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"logged -> {path}")


def plot_log(path, out):
    """Redraw probe_log.csv as accuracy-vs-pretraining-step, with effective rank
    on a second axis. Random-init rows (step -1) are dropped."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["step"] not in ("", "-1")]
    if not rows:
        print("plot_log: no checkpoint rows yet")
        return
    seen = {}
    for r in rows:                        # keep the last row per step
        seen[int(r["step"])] = r
    steps = sorted(seen)
    rows = [seen[s] for s in steps]

    def col(name):
        return [float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, mk in [("knn_acc", "o-"), ("linear_acc", "s-"), ("attentive_acc", "^-")]:
        y = col(name)
        if np.isfinite(y).any():
            ax.plot(steps, y, mk, label=name.replace("_acc", ""))
    ax.set_xlabel("pretraining step")
    ax.set_ylabel("test accuracy")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    ax2 = ax.twinx()
    ax2.plot(steps, col("eff_rank_entropy"), "x--", color="0.5", label="eff rank (entropy)")
    ax2.set_ylabel("effective rank (entropy)")
    ax2.legend(loc="lower right")

    ax.set_title("frozen-probe accuracy vs pretraining step")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "jepa_ckpt.pt"),
                    help="checkpoint path, or 'none' for a seeded random-init encoder")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=0, help="0 = all of train")
    ap.add_argument("--n-test", type=int, default=0, help="0 = all of test")
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--tokens", action="store_true",
                    help="also cache token grids and run the attentive probe")
    ap.add_argument("--epochs", type=int, default=40, help="attentive-probe epochs")
    ap.add_argument("--label-efficiency", action="store_true")
    ap.add_argument("--fractions", default="0.01,0.05,0.1,0.25,1.0")
    ap.add_argument("--overlay", default="", help="a *_labeff.npz to draw as a reference curve")
    ap.add_argument("--plot-log", action="store_true",
                    help="after logging this run, redraw probe_out/probe_log.csv as acc-vs-step")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    logcsv = os.path.join(OUT, "probe_log.csv")
    device = torch.device(args.device)
    enc, tag, pretty, ck_epoch, ck_step = build_encoder(args.ckpt, device)
    print(f"encoder: {pretty}   device={device}   tag={tag}")

    n_train = args.n_train
    if args.tokens and (n_train <= 0 or n_train > RANDOM_TOKENS_CAP):
        n_train = RANDOM_TOKENS_CAP
        print(f"--tokens: capping n-train at {RANDOM_TOKENS_CAP} "
              f"(fp16 token memmap for the full split would be ~13 GB)")

    xtr, tok_tr, ytr = get_features(enc, "train", n_train, device, args.batch, tag, args.tokens)
    xte, tok_te, yte = get_features(enc, "test", args.n_test, device, args.batch, tag, args.tokens)
    print(f"features: train {xtr.shape}  test {xte.shape}")

    geo = feature_geometry(xtr)
    print(f"feature geometry: dims {geo['dims']}   "
          f"eff-rank(entropy) {geo['eff_rank_entropy']:.1f}   "
          f"eff-rank(PR) {geo['eff_rank_pr']:.1f}   stable-rank {geo['stable_rank']:.1f}")

    zt, ze = standardize(xtr, xte)
    rows = []

    knn_acc, knn_pred = knn_probe(zt, ytr, ze, yte, args.knn_k)
    rows.append((f"{args.knn_k}-NN", knn_acc, f1_score(yte, knn_pred, average="macro")))

    lin_acc, lin_pred, best_c, sweep = linear_probe(zt, ytr, ze, yte)
    rows.append(("linear", lin_acc, f1_score(yte, lin_pred, average="macro")))
    print(f"linear-probe C sweep (val acc): " +
          "  ".join(f"C={c}:{v:.3f}" for v, c in sweep) + f"  -> best C={best_c}")
    confusion_plot(yte, lin_pred, f"linear probe -- {pretty}  (acc {lin_acc:.3f})",
                   os.path.join(OUT, f"{tag}_confusion_linear.png"))

    att_acc = att_f1 = att_va = None
    if args.tokens:
        att_acc, att_pred, att_va = attentive_probe(tok_tr, ytr, tok_te, yte, device, args.epochs)
        att_f1 = f1_score(yte, att_pred, average="macro")
        rows.append(("attentive", att_acc, att_f1))
        confusion_plot(yte, att_pred, f"attentive probe -- {pretty}  (acc {att_acc:.3f})",
                       os.path.join(OUT, f"{tag}_confusion_attentive.png"))

    print(f"\n=== frozen-probe results -- {pretty} ===")
    print(f"{'probe':12s} {'test acc':>10s} {'macro-F1':>10s}")
    for name, acc, f1 in rows:
        print(f"{name:12s} {acc:10.4f} {f1:10.4f}")

    summ = os.path.join(OUT, f"{tag}_summary.txt")
    with open(summ, "w") as fh:
        fh.write(f"encoder: {pretty}\ntrain {xtr.shape}  test {xte.shape}\n")
        fh.write(f"eff-rank(entropy) {geo['eff_rank_entropy']:.2f}  "
                 f"eff-rank(PR) {geo['eff_rank_pr']:.2f}  "
                 f"stable-rank {geo['stable_rank']:.2f}\n\n")
        fh.write(f"{'probe':12s} {'test acc':>10s} {'macro-F1':>10s}\n")
        for name, acc, f1 in rows:
            fh.write(f"{name:12s} {acc:10.4f} {f1:10.4f}\n")
    print(f"wrote {summ}")

    r2 = lambda v: "" if v is None else round(float(v), 2)
    r4 = lambda v: "" if v is None else round(float(v), 4)
    append_log({
        "tag": tag, "epoch": ck_epoch, "step": ck_step,
        "n_train": len(ytr), "n_test": len(yte), "tokens": int(args.tokens),
        "dims": geo["dims"], "eff_rank_entropy": r2(geo["eff_rank_entropy"]),
        "eff_rank_pr": r2(geo["eff_rank_pr"]), "stable_rank": r2(geo["stable_rank"]),
        "knn_acc": r4(knn_acc), "knn_f1": r4(rows[0][2]),
        "linear_acc": r4(lin_acc), "linear_f1": r4(rows[1][2]), "linear_C": best_c,
        "attentive_acc": r4(att_acc), "attentive_f1": r4(att_f1), "attentive_val": r4(att_va),
    }, logcsv)

    if args.label_efficiency:
        fr = [float(x) for x in args.fractions.split(",")]
        print(f"\nlabel efficiency ({pretty}):")
        fr, acc = label_efficiency(zt, ytr, ze, yte, fr)
        npz = os.path.join(OUT, f"{tag}_labeff.npz")
        np.savez(npz, fractions=fr, acc=acc, pretty=pretty)
        label_efficiency_plot(fr, acc, pretty, os.path.join(OUT, f"{tag}_labeff.png"),
                              overlay=args.overlay or None)
        print(f"wrote {npz}")

    if args.plot_log:
        plot_log(logcsv, os.path.join(OUT, "probe_log.png"))


if __name__ == "__main__":
    main()
