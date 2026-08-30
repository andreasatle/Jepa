"""
Linear probe evaluation for the trained JEPA encoder -- step 4.

The training diagnostics (avg_inv_loss, avg_cov_penalty) only tell us the
optimization behaved -- prediction got more accurate, embeddings didn't
collapse. They don't tell us whether z = E(x) actually captures the
physics (ball/cup position and velocity) rather than satisfying the loss
some other, less useful way.

This script checks that directly: fit the simplest possible model -- a
single linear (affine) map -- from some representation to the true state,
on some episodes, and see how well it predicts the true state on *held-out*
episodes it never saw.

Two separate probes, because E is single-frame now (see jepa_train.py's
K_CONTEXT restructuring) and that changes what's fair to test:

  1. single-frame z = E(x_t)  ->  position only.
     A sanity check that spatial encoding still works. Not expected to
     recover velocity -- under the current architecture E is deliberately
     never given more than one frame, so a single z has no way to see
     motion. That's not a failure mode to probe for anymore; it's the
     design working as intended (see jepa_train.py's docstring on why
     velocity moved out of E and into P).

  2. the K_CONTEXT-frame window [z_{t-k+1},...,z_t], concatenated  ->
     position AND velocity, at time t.
     This is the actual test of whether the restructuring solved anything:
     it probes the *same* information P itself is given, so a good score
     here means the information P needs to reconstruct velocity is
     linearly present in the window, not that P is doing something a
     probe could never detect. (A bad score wouldn't prove the opposite --
     P is nonlinear and could still be using something a linear probe
     can't see -- but a good score is real, positive evidence the fix
     worked.)

Held out by *episode*, not by transition -- adjacent frames within an
episode are highly correlated, so splitting at the transition level would
leak near-duplicate states between train and test and inflate R^2. Both
probes use the same episode split, so their scores are directly comparable.

The fit itself uses np.linalg.lstsq, which solves via SVD and truncates
small singular values internally -- the discrete, hard-cutoff cousin of
the Tikhonov damping from a few messages back. No separate regularization
parameter to pick; numpy's default rcond threshold handles the
near-singular case for us.

Run (after collect_eval_data.py and a trained jepa_checkpoint.pt both
exist):
    python3 eval_linear_probe.py
"""

import pathlib

import numpy as np
import torch

import jepa_train as jt

EVAL_DATA_DIR = pathlib.Path("jepa_eval_data")
CHECKPOINT_PATH = "jepa_checkpoint.pt"
TRAIN_EPISODE_FRACTION = 0.8
SEED = 0

POSITION_LABELS = ["cup_x", "cup_z", "ball_x", "ball_z"]
STATE_LABELS = POSITION_LABELS + [
    "cup_x_vel", "cup_z_vel", "ball_x_vel", "ball_z_vel",
]


def load_encoder(device):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    k_context = ckpt.get("k_context", jt.K_CONTEXT)
    latent_dim = ckpt.get("latent_dim", jt.LATENT_DIM)
    E = jt.Encoder(3, latent_dim).to(device)   # always single-frame now
    E.load_state_dict(ckpt["E"])
    E.eval()
    return E, k_context, latent_dim


def load_episodes(data_dir):
    paths = sorted(pathlib.Path(data_dir).glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No episode_*.npz found in {data_dir}/ -- run collect_eval_data.py first."
        )
    episodes = []
    for path in paths:
        data = np.load(path)
        episodes.append((data["frames"], data["positions"], data["velocities"]))
    return episodes


def encode_all_frames(E, frames, device):
    """z for every single frame in the episode -- no windowing here, E
    takes one frame at a time. frames: (n, H, W, 3) uint8. Returns
    (n, D) numpy array, z[t] = E(frames[t])."""
    tensors = jt._frames_to_tensor(frames).to(device)   # (n, 3, H, W)
    with torch.no_grad():
        z = E(tensors).cpu().numpy()
    return z


def windowed_features(z_all, positions, velocities, k_context):
    """Concatenates each K_CONTEXT-length run of consecutive z's into one
    vector, paired with the full (position, velocity) state at the last
    frame in that window -- the same pairing P itself is trained on.
    Returns (windows, states), windows: (n - k + 1, k*D), states: (n - k + 1, 8)."""
    n = z_all.shape[0]
    windows, states = [], []
    for t in range(k_context - 1, n):
        windows.append(z_all[t - k_context + 1 : t + 1].reshape(-1))
        states.append(np.concatenate([positions[t], velocities[t]]))
    return np.stack(windows), np.stack(states)


def r_squared(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot


def fit_and_report(train_X, train_y, test_X, test_y, labels, title):
    A_train = np.concatenate([train_X, np.ones((len(train_X), 1), dtype=np.float32)], axis=1)
    A_test = np.concatenate([test_X, np.ones((len(test_X), 1), dtype=np.float32)], axis=1)

    W, _residuals, rank, _sv = np.linalg.lstsq(A_train, train_y, rcond=None)

    train_pred = A_train @ W
    test_pred = A_test @ W
    train_r2 = r_squared(train_y, train_pred)
    test_r2 = r_squared(test_y, test_pred)

    print(f"\n{title}")
    print(f"Fit rank: {rank}/{A_train.shape[1]} (feature dim + 1 bias column)")
    print(f"{'dim':12s} {'train R^2':>10s} {'test R^2':>10s}")
    for label, tr, te in zip(labels, train_r2, test_r2):
        print(f"{label:12s} {tr:10.4f} {te:10.4f}")
    print(f"{'mean':12s} {train_r2.mean():10.4f} {test_r2.mean():10.4f}")
    return W


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    E, k_context, latent_dim = load_encoder(device)
    print(
        f"Loaded encoder from {CHECKPOINT_PATH} "
        f"(k_context={k_context}, latent_dim={latent_dim})"
    )

    episodes = load_episodes(EVAL_DATA_DIR)
    print(f"Loaded {len(episodes)} eval episodes from {EVAL_DATA_DIR}/")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(episodes))
    n_train = max(1, int(round(len(episodes) * TRAIN_EPISODE_FRACTION)))
    if len(episodes) > 1:
        n_train = min(n_train, len(episodes) - 1)  # keep at least 1 held out
    train_idx = set(order[:n_train].tolist())

    single_train_z, single_train_pos = [], []
    single_test_z, single_test_pos = [], []
    win_train_z, win_train_state = [], []
    win_test_z, win_test_state = [], []

    for i, (frames, positions, velocities) in enumerate(episodes):
        z_all = encode_all_frames(E, frames, device)   # (n, D)
        win_z, win_state = windowed_features(z_all, positions, velocities, k_context)

        if i in train_idx:
            single_train_z.append(z_all)
            single_train_pos.append(positions)
            win_train_z.append(win_z)
            win_train_state.append(win_state)
        else:
            single_test_z.append(z_all)
            single_test_pos.append(positions)
            win_test_z.append(win_z)
            win_test_state.append(win_state)

    single_train_z = np.concatenate(single_train_z)
    single_train_pos = np.concatenate(single_train_pos)
    single_test_z = np.concatenate(single_test_z)
    single_test_pos = np.concatenate(single_test_pos)

    win_train_z = np.concatenate(win_train_z)
    win_train_state = np.concatenate(win_train_state)
    win_test_z = np.concatenate(win_test_z)
    win_test_state = np.concatenate(win_test_state)

    print(
        f"{len(train_idx)} train episodes, {len(episodes) - len(train_idx)} held-out episodes  "
        f"({len(single_train_z)}/{len(single_test_z)} single frames, "
        f"{len(win_train_z)}/{len(win_test_z)} {k_context}-frame windows, train/test)"
    )

    W_pos = fit_and_report(
        single_train_z, single_train_pos, single_test_z, single_test_pos,
        POSITION_LABELS,
        "Probe 1: single-frame z -> position (sanity check, velocity not expected here)",
    )
    W_state = fit_and_report(
        win_train_z, win_train_state, win_test_z, win_test_state,
        STATE_LABELS,
        f"Probe 2: {k_context}-frame window of z's -> position + velocity "
        "(the real test)",
    )

    # Persist both fitted probes -- so cem_planner.py (and anything else
    # downstream) doesn't need to refit on every run. W already includes
    # the bias row (see fit_and_report: A = [X | 1]), so decoding is just
    # [z | 1] @ W -- see cem_planner.py's decode_position().
    np.savez("position_probe.npz", W=W_pos, labels=np.array(POSITION_LABELS), latent_dim=latent_dim)
    np.savez(
        "state_probe.npz", W=W_state, labels=np.array(STATE_LABELS),
        k_context=k_context, latent_dim=latent_dim,
    )
    print(f"\nSaved position_probe.npz (32->4) and state_probe.npz ({k_context}*32->8).")


if __name__ == "__main__":
    main()