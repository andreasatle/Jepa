"""
Nonlinear (MLP) position probe for the STRIDED model's encoder --
diagnostic only, does NOT touch the planner/plan_step pipeline yet.

Motivation: drift_diagnostic_strided.py showed ball position error is
already ~0.08m at tau=0 (a single real frame, no rollout at all) --
that's purely a single-frame z -> position decoding problem. The
existing position_probe_strided.npz is a LINEAR (affine) map fit by
least squares; its R^2 (~0.96 train / ~0.94 test, mean across
cup_x/cup_z/ball_x/ball_z) is good but not 1.0. This script fits a small
MLP instead, on the exact same single-frame z -> position task
("Probe 1" in eval_linear_probe.py), with the same episode train/test
split (same SEED) so the R^2 numbers are directly comparable.

If MLP R^2 is meaningfully higher than the linear probe's: the position
info was in z all along, just not linearly extractable -- swapping the
planner's decode_position over to this probe should shrink the ~0.08m
floor. If MLP R^2 is about the same: the missing info genuinely isn't in
z (or isn't at this latent_dim/resolution), and the fix has to happen
upstream in the encoder, not in how we read it out.

Run (after eval_linear_probe_strided.py has been run at least once, so
jepa_eval_data/ and jepa_checkpoint_strided.pt both exist):
    uv run python eval_mlp_probe_strided.py
"""

import numpy as np
import torch
import torch.nn as nn

import eval_linear_probe as elp
import eval_linear_probe_strided as elps

HIDDEN = 128
EPOCHS = 300
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
OUT_PATH = "position_probe_strided_mlp.pt"


class MLPProbe(nn.Module):
    """Single-frame z -> position, only nonlinearity added vs the linear
    probe. Normalizes inputs/outputs internally (mean/std stored as
    buffers) so it can be called directly on raw z, like decode_position
    does for the linear W -- same calling convention, drop-in later."""

    def __init__(self, latent_dim, hidden, z_mean, z_std, y_mean, y_std):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )
        self.register_buffer("z_mean", torch.as_tensor(z_mean, dtype=torch.float32))
        self.register_buffer("z_std", torch.as_tensor(z_std, dtype=torch.float32))
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))

    def forward(self, z):
        z_n = (z - self.z_mean) / self.z_std
        y_n = self.net(z_n)
        return y_n * self.y_std + self.y_mean


def train_mlp_probe(train_z, train_y, test_z, test_y, latent_dim, device):
    z_mean = train_z.mean(axis=0)
    z_std = train_z.std(axis=0) + 1e-6
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0) + 1e-6

    model = MLPProbe(latent_dim, HIDDEN, z_mean, z_std, y_mean, y_std).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_z_t = torch.from_numpy(train_z).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().to(device)
    test_z_t = torch.from_numpy(test_z).float().to(device)
    test_y_t = torch.from_numpy(test_y).float().to(device)

    n = train_z_t.shape[0]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            zb, yb = train_z_t[idx], train_y_t[idx]
            pred = model(zb)
            loss = ((pred - yb) ** 2 / model.y_std ** 2).mean()  # normalized MSE
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if epoch % 50 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                train_pred = model(train_z_t).cpu().numpy()
                test_pred = model(test_z_t).cpu().numpy()
            train_r2 = elp.r_squared(train_y, train_pred).mean()
            test_r2 = elp.r_squared(test_y, test_pred).mean()
            print(f"  epoch {epoch:4d}/{EPOCHS}  train_loss={total_loss / n:.4f}  "
                  f"train_R2={train_r2:.4f}  test_R2={test_r2:.4f}")

    model.eval()
    return model


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    E, k_context, latent_dim = elps.load_encoder(device)
    print(f"Loaded encoder from {elps.CHECKPOINT_PATH} (latent_dim={latent_dim})")

    episodes = elp.load_episodes(elp.EVAL_DATA_DIR)
    print(f"Loaded {len(episodes)} eval episodes from {elp.EVAL_DATA_DIR}/")

    rng = np.random.default_rng(elp.SEED)
    order = rng.permutation(len(episodes))
    n_train = max(1, int(round(len(episodes) * elp.TRAIN_EPISODE_FRACTION)))
    if len(episodes) > 1:
        n_train = min(n_train, len(episodes) - 1)
    train_idx = set(order[:n_train].tolist())

    single_train_z, single_train_pos = [], []
    single_test_z, single_test_pos = [], []
    for i, (frames, positions, velocities) in enumerate(episodes):
        z_all = elp.encode_all_frames(E, frames, device)
        if i in train_idx:
            single_train_z.append(z_all)
            single_train_pos.append(positions)
        else:
            single_test_z.append(z_all)
            single_test_pos.append(positions)

    train_z = np.concatenate(single_train_z)
    train_y = np.concatenate(single_train_pos)
    test_z = np.concatenate(single_test_z)
    test_y = np.concatenate(single_test_pos)
    print(f"{len(train_idx)} train episodes, {len(episodes) - len(train_idx)} held-out episodes  "
          f"({len(train_z)}/{len(test_z)} single frames, train/test)")

    print("\nTraining MLP probe (single-frame z -> position)...")
    model = train_mlp_probe(train_z, train_y, test_z, test_y, latent_dim, device)

    with torch.no_grad():
        train_pred = model(torch.from_numpy(train_z).float().to(device)).cpu().numpy()
        test_pred = model(torch.from_numpy(test_z).float().to(device)).cpu().numpy()
    train_r2 = elp.r_squared(train_y, train_pred)
    test_r2 = elp.r_squared(test_y, test_pred)

    print(f"\nMLP probe -- final per-dim R^2")
    print(f"{'dim':12s} {'train R^2':>10s} {'test R^2':>10s}")
    for label, tr, te in zip(elp.POSITION_LABELS, train_r2, test_r2):
        print(f"{label:12s} {tr:10.4f} {te:10.4f}")
    print(f"{'mean':12s} {train_r2.mean():10.4f} {test_r2.mean():10.4f}")

    # Also report mean L2 position error in meters, directly comparable
    # to drift_diagnostic_strided.py's tau=0 number (0.0827m for the
    # linear probe) -- R^2 alone doesn't say much about the actual scale
    # of the residual error in meters.
    test_err = np.linalg.norm(test_pred[:, 2:4] - test_y[:, 2:4], axis=1)  # ball only
    print(f"\nBall position L2 error (test set): mean={test_err.mean():.4f}m  "
          f"median={np.median(test_err):.4f}m  max={test_err.max():.4f}m")
    print("(compare to drift_diagnostic_strided.py's tau=0 linear-probe number: 0.0827m)")

    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "latent_dim": latent_dim,
            "hidden": HIDDEN,
        },
        OUT_PATH,
    )
    print(f"\nSaved MLP probe to {OUT_PATH}")


if __name__ == "__main__":
    main()
