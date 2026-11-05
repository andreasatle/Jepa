"""
Gromov delta-hyperbolicity test on the ball-in-cup JEPA encoder's latent space.

Loads the trained encoder E from jepa_checkpoint_strided.pt, runs it over
every frame of the 20 held-out eval episodes to get 32-dim embeddings, then
estimates the four-point (Gromov) delta-hyperbolicity of the resulting point
cloud under Euclidean distance in that latent space.

delta-hyperbolicity (four-point condition): for points w,x,y,z with pairwise
distances d(.,.), form the three pairwise sums
    S1 = d(w,x) + d(y,z)
    S2 = d(w,y) + d(x,z)
    S3 = d(w,z) + d(x,y)
sort them S(1) <= S(2) <= S(3), and delta(w,x,y,z) = (S(3) - S(2)) / 2.
delta = sup over all quadruples. A space is "more hyperbolic" (more
tree-like) the smaller delta is relative to the diameter of the point set;
Khrulkov et al. (2020) report the scale-invariant ratio

    rel_delta = 2 * delta / diam(X)

which lies in [0, 1]: values near 0 indicate strongly negatively-curved
(tree-like) geometry, values near 1 are consistent with flat/Euclidean data.
Exact computation is O(n^4), so delta is estimated by sampling many random
quadruples (standard practice for this test).
"""
import sys
import pathlib
import numpy as np
import torch

sys.path.insert(0, "/mnt/user-data/uploads/Jepa/ball_in_cup")
import jepa_train as jt

CKPT = "/mnt/user-data/uploads/Jepa/ball_in_cup/jepa_checkpoint_strided.pt"
EVAL_DIR = pathlib.Path("/mnt/user-data/uploads/Jepa/ball_in_cup/jepa_eval_data_strided")

device = torch.device("cpu")
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
latent_dim = ckpt.get("latent_dim", jt.LATENT_DIM)

E = jt.Encoder(3, latent_dim).to(device)
E.load_state_dict(ckpt["E"])
E.eval()

all_z = []
episode_ids = []
files = sorted(EVAL_DIR.glob("episode_*.npz"))
print(f"Encoding {len(files)} episodes...")
with torch.no_grad():
    for ep_idx, f in enumerate(files):
        d = np.load(f)
        frames = d["frames"]  # (T, H, W, 3) uint8
        tensors = jt._frames_to_tensor(frames).to(device)  # (T, 3, H, W)
        z = E(tensors).cpu().numpy()  # (T, D)
        all_z.append(z)
        episode_ids.append(np.full(len(z), ep_idx))

Z = np.concatenate(all_z, axis=0)
ep_ids = np.concatenate(episode_ids, axis=0)
print(f"Latent points: {Z.shape[0]}, dim: {Z.shape[1]}")

# Pairwise Euclidean distances (full matrix is fine at this size)
from scipy.spatial.distance import pdist, squareform
D = squareform(pdist(Z, metric="euclidean"))
n = D.shape[0]
diam = D.max()
print(f"Diameter (max pairwise distance): {diam:.4f}")
print(f"Mean pairwise distance: {D[np.triu_indices(n, k=1)].mean():.4f}")

rng = np.random.default_rng(0)
n_samples = 50000
idx = rng.integers(0, n, size=(n_samples, 4))
# drop quadruples with repeated indices
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
delta_mean = deltas.mean()

rel_delta_max = 2 * delta_max / diam
rel_delta_p99 = 2 * delta_p99 / diam

print(f"\nSampled {len(deltas)} quadruples")
print(f"delta (max over sample):  {delta_max:.4f}   rel_delta = 2*delta/diam = {rel_delta_max:.4f}")
print(f"delta (99th percentile):  {delta_p99:.4f}   rel_delta = {rel_delta_p99:.4f}")
print(f"delta (mean):             {delta_mean:.4f}")

np.savez("/mnt/user-data/outputs/latent_points.npz", Z=Z, ep_ids=ep_ids)
print("\nSaved latent embeddings to latent_points.npz")
