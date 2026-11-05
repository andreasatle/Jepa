"""
Gromov delta-hyperbolicity diagnostic for the ball-in-cup-hyperbolic
encoder, run periodically against checkpoints (same role in the eval
suite as eval_linear_probe_strided.py / eval_mlp_probe_strided.py --
a held-out-data check you re-run as training progresses).

Deliberately measures EUCLIDEAN distance between the RAW (pre-exp0)
encoder outputs, not hyperbolic.dist between exp0-projected points.
Reasoning: the true Poincare ball is already maximally delta-hyperbolic
by construction -- it has a fixed, known delta (a property of the model
space itself), so testing with hyperbolic.dist as the metric would just
re-derive that built-in constant and say nothing about what the encoder
learned. Testing the raw Euclidean latent is the informative version: it
answers "is training organizing the raw embedding more tree-like than it
was incidentally under v1", directly comparable to the v1 baseline of
rel_delta ~ 0.15-0.27 (see gromov-delta-hyperbolicity.md / the one-off
hyperbolicity_test.py run on jepa_checkpoint_strided.pt from ball_in_cup/).

delta-hyperbolicity (four-point condition): for points w,x,y,z with
pairwise distances d(.,.), form the three pairwise sums
    S1 = d(w,x) + d(y,z)
    S2 = d(w,y) + d(x,z)
    S3 = d(w,z) + d(x,y)
sort them S(1) <= S(2) <= S(3), and delta(w,x,y,z) = (S(3) - S(2)) / 2.
delta = sup over all quadruples. Exact computation is O(n^4), so delta is
estimated by sampling many random quadruples (standard practice).

    rel_delta = 2 * delta / diam(X)   in [0, 1]

near 0: strongly tree-like / negatively curved. near 1: flat/Euclidean-like.

Run (after jepa_train_strided.py has produced jepa_checkpoint_strided.pt):
    python3 hyperbolicity_eval.py
"""
import pathlib

import numpy as np
import torch

import jepa_train as jt

CHECKPOINT_PATH = "jepa_checkpoint_strided.pt"
EVAL_DIR = pathlib.Path("jepa_eval_data_strided")
N_SAMPLES = 50000
SEED = 0


def load_encoder(device):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    latent_dim = ckpt.get("latent_dim", jt.LATENT_DIM)
    E = jt.Encoder(3, latent_dim).to(device)
    E.load_state_dict(ckpt["E"])
    E.eval()
    return E, ckpt.get("epoch", "?")


def encode_eval_episodes(E, device):
    all_z = []
    files = sorted(EVAL_DIR.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found in {EVAL_DIR}/")
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            frames = d["frames"]  # (T, H, W, 3) uint8
            tensors = jt._frames_to_tensor(frames).to(device)  # (T, 3, H, W)
            z = E(tensors).cpu().numpy()  # (T, D) -- raw, pre-exp0
            all_z.append(z)
    return np.concatenate(all_z, axis=0)


def rel_delta_hyperbolicity(Z, n_samples=N_SAMPLES, seed=SEED):
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(Z, metric="euclidean"))
    n = D.shape[0]
    diam = D.max()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_samples, 4))
    mask = np.all(idx[:, 1:] != idx[:, :1], axis=1)
    for i in range(4):
        for j in range(i + 1, 4):
            mask &= idx[:, i] != idx[:, j]
    idx = idx[mask]

    w, x, y, z_ = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    S1 = D[w, x] + D[y, z_]
    S2 = D[w, y] + D[x, z_]
    S3 = D[w, z_] + D[x, y]
    S = np.stack([S1, S2, S3], axis=1)
    S.sort(axis=1)
    deltas = (S[:, 2] - S[:, 1]) / 2.0

    delta_max = deltas.max()
    delta_p99 = np.percentile(deltas, 99)
    return {
        "n_points": n,
        "diam": diam,
        "n_quadruples": len(deltas),
        "delta_max": delta_max,
        "delta_p99": delta_p99,
        "rel_delta_max": 2 * delta_max / diam,
        "rel_delta_p99": 2 * delta_p99 / diam,
    }


def main():
    device = torch.device("cpu")  # this test is O(n) encode + O(samples) distance work; cpu is fine
    E, epoch = load_encoder(device)
    print(f"Loaded encoder from {CHECKPOINT_PATH} (epoch={epoch})")

    Z = encode_eval_episodes(E, device)
    print(f"Encoded {Z.shape[0]} held-out frames, latent_dim={Z.shape[1]} (raw, pre-exp0)")

    stats = rel_delta_hyperbolicity(Z)
    print(f"Diameter (max pairwise distance): {stats['diam']:.4f}")
    print(f"Sampled {stats['n_quadruples']} quadruples")
    print(f"delta (max):  {stats['delta_max']:.4f}   rel_delta = {stats['rel_delta_max']:.4f}")
    print(f"delta (p99):  {stats['delta_p99']:.4f}   rel_delta = {stats['rel_delta_p99']:.4f}")
    print(f"\n(v1 baseline for comparison, from the incidental one-off check: rel_delta ~ 0.15-0.27)")


if __name__ == "__main__":
    main()
