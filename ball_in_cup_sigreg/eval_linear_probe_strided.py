"""
Refits the position/state probes for the STRIDED model's encoder
(jepa_checkpoint_strided.pt) -- a separate probe from position_probe.npz,
since the strided model has its own E (same architecture as the
fine-grained one, different trained weights).

Reuses jepa_eval_data/ AS-IS (the existing, fine-grained eval data) -- E
encodes one frame at a time regardless of what stride P was trained at,
so no new data collection is needed just for this probe.

Reuses eval_linear_probe.py's helpers via import rather than duplicating
them -- only the checkpoint path and output filenames differ here.

Run (after jepa_train_strided.py has produced jepa_checkpoint_strided.pt):
    python3 eval_linear_probe_strided.py
"""

import numpy as np
import torch

import eval_linear_probe as elp
import jepa_train as jt

CHECKPOINT_PATH = "jepa_checkpoint_strided.pt"


def load_encoder(device):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    k_context = ckpt.get("k_context", jt.K_CONTEXT)
    latent_dim = ckpt.get("latent_dim", jt.LATENT_DIM)
    E = jt.Encoder(3, latent_dim).to(device)
    E.load_state_dict(ckpt["E"])
    E.eval()
    return E, k_context, latent_dim


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    E, k_context, latent_dim = load_encoder(device)
    print(f"Loaded encoder from {CHECKPOINT_PATH} (k_context={k_context}, latent_dim={latent_dim})")

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
    win_train_z, win_train_state = [], []
    win_test_z, win_test_state = [], []

    for i, (frames, positions, velocities) in enumerate(episodes):
        z_all = elp.encode_all_frames(E, frames, device)
        win_z, win_state = elp.windowed_features(z_all, positions, velocities, k_context)
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

    W_pos = elp.fit_and_report(
        single_train_z, single_train_pos, single_test_z, single_test_pos,
        elp.POSITION_LABELS,
        "Probe 1 (strided model): single-frame z -> position",
    )
    W_state = elp.fit_and_report(
        win_train_z, win_train_state, win_test_z, win_test_state,
        elp.STATE_LABELS,
        f"Probe 2 (strided model): {k_context}-frame window of z's -> position + velocity",
    )

    np.savez(
        "position_probe_strided.npz", W=W_pos, labels=np.array(elp.POSITION_LABELS),
        latent_dim=latent_dim,
    )
    np.savez(
        "state_probe_strided.npz", W=W_state, labels=np.array(elp.STATE_LABELS),
        k_context=k_context, latent_dim=latent_dim,
    )
    print("\nSaved position_probe_strided.npz and state_probe_strided.npz.")


if __name__ == "__main__":
    main()
