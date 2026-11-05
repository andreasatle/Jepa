"""
Training P on the PREDICTOR_STRIDE-spaced data from collect_data_strided.py
-- one P call now spans PREDICTOR_STRIDE real control-steps (action held
constant across them), instead of 1. Everything about E/P's architecture,
the VICReg anti-collapse loss, and the multi-step (UNROLL_STEPS) exposure-
bias fix from jepa_train.py v7 carries over unchanged -- those are reused
directly via `import jepa_train as jt` rather than copy-pasted, since none
of that depends on what one P call represents in real time. Only two
things are new here: the dataset (windows/targets spaced `PREDICTOR_STRIDE`
apart instead of 1) and the output paths (separate checkpoint/log, so this
never collides with the fine-grained model from jepa_train.py).

Why this exists (see project_brief.md / conversation): drift_diagnostic.py
showed the fine-grained P's rollout becomes unusable well before the
100+ steps ball-in-cup's "pump the swing" strategy needs. Coarsening P's
own step means fewer P-calls are needed to span the same real duration --
PREDICTOR_STRIDE=5 means one P call now covers what used to take 5, so a
150-real-step horizon becomes a 30-P-call rollout.

Requires collect_data_strided.py to have been run first (produces
jepa_data_strided/ -- NOT the same directory as jepa_data/, and NOT
interchangeable with it: that data has a fresh i.i.d. random action every
single real step, so it can't supply a genuine "held constant" action for
this dataset's targets).

Run:
    uv run python jepa_train_strided.py
"""

import csv
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import jepa_train as jt
import hyperbolic as hyp

DATA_DIR = pathlib.Path("jepa_data_strided")
PREDICTOR_STRIDE = 5      # must match collect_data_strided.py's value --
                          # this is a property of the DATA (how often a new
                          # action was actually drawn), not something this
                          # script can change independently.
UNROLL_STEPS = 5          # same multi-step exposure-bias fix as
                          # jepa_train.py v7, stacked on top of the stride
                          # -- each of these U calls covers PREDICTOR_STRIDE
                          # real steps, so U*PREDICTOR_STRIDE real steps get
                          # supervised per training example (25, by default).

BATCH_SIZE = jt.BATCH_SIZE
LR = jt.LR
KAPPA = jt.KAPPA
N_EPOCHS = jt.N_EPOCHS

CHECKPOINT_PATH = "jepa_checkpoint_strided.pt"
LOG_PATH = "jepa_training_log_strided.csv"
SEED = 0


class StridedJepaTransitionDataset(Dataset):
    """Like jepa_train.JepaTransitionDataset, but context/targets are
    spaced PREDICTOR_STRIDE real steps apart instead of 1, matching what
    one P call is being trained to represent.

    An example starting at real index t needs:
      context:     frames[t-(k-1)*s], ..., frames[t-s], frames[t]   (k frames)
      action_seq:  actions[t], actions[t+s], ..., actions[t+(u-1)*s]  (u actions --
                   each is the held-constant value for its stride-block,
                   see collect_data_strided.py)
      target_seq:  frames[t+s], frames[t+2s], ..., frames[t+u*s]   (u REAL frames)
    where k=k_context, u=unroll_steps, s=stride. t ranges over
    [(k-1)*s, T - u*s] within an episode of T actions / T+1 frames.
    """

    def __init__(self, data_dir, k_context, unroll_steps, stride):
        self.k_context = k_context
        self.unroll_steps = unroll_steps
        self.stride = stride
        self.episodes = []
        self.index = []

        paths = sorted(pathlib.Path(data_dir).glob("episode_*.npz"))
        if not paths:
            raise FileNotFoundError(
                f"No episode_*.npz files found in {data_dir}/ -- "
                "run collect_data_strided.py first."
            )

        for ep_idx, path in enumerate(paths):
            data = np.load(path)
            frames = data["frames"]
            actions = data["actions"]
            self.episodes.append((frames, actions))

            T = actions.shape[0]
            k, u, s = k_context, unroll_steps, stride
            for t in range((k - 1) * s, T - u * s + 1):
                self.index.append((ep_idx, t))

        print(
            f"Loaded {len(paths)} episodes, {len(self.index)} examples "
            f"(k_context={k_context}, unroll_steps={unroll_steps}, stride={stride})."
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        frames, actions = self.episodes[ep_idx]
        k, u, s = self.k_context, self.unroll_steps, self.stride

        context_idx = [t - (k - 1 - j) * s for j in range(k)]
        context = frames[context_idx]                          # (k, H, W, 3)
        target_seq = np.stack([frames[t + (i + 1) * s] for i in range(u)])   # (u, H, W, 3)
        action_seq = np.stack([actions[t + i * s] for i in range(u)])        # (u, action_dim)

        context = jt._frames_to_tensor(context)
        target_seq = jt._frames_to_tensor(target_seq)
        action_seq = torch.from_numpy(action_seq.astype(np.float32))

        return context, action_seq, target_seq


def train():
    print(f"jepa_train_strided.py -- PREDICTOR_STRIDE={PREDICTOR_STRIDE}, "
          f"UNROLL_STEPS={UNROLL_STEPS} (jepa_train.py version: {jt.VERSION})")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    E, P = jt.make_models(device)

    dataset = StridedJepaTransitionDataset(DATA_DIR, jt.K_CONTEXT, UNROLL_STEPS, PREDICTOR_STRIDE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    optimizer = torch.optim.Adam(list(E.parameters()) + list(P.parameters()), lr=LR)

    # Resume support, same pattern as jepa_train.py -- deliberately a
    # SEPARATE checkpoint file, so this never overwrites or gets confused
    # with the fine-grained (non-strided) model.
    checkpoint_path = pathlib.Path(CHECKPOINT_PATH)
    start_epoch = 0
    if checkpoint_path.exists():
        print(f"Found {CHECKPOINT_PATH} -- resuming from it.")
        ckpt = torch.load(checkpoint_path, map_location=device)
        E.load_state_dict(ckpt["E"])
        P.load_state_dict(ckpt["P"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Already trained through epoch {start_epoch}/{N_EPOCHS}.")
    else:
        print(f"No {CHECKPOINT_PATH} found -- starting fresh.")

    log_path = pathlib.Path(LOG_PATH)
    write_header = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["epoch", "avg_loss", "avg_inv_loss", "avg_cov_penalty", "avg_raw_z_norm"])

    for epoch in range(start_epoch, N_EPOCHS):
        epoch_losses, epoch_inv_losses, epoch_cov_penalties, epoch_z_norms = [], [], [], []

        for context, action_seq, target_seq in loader:
            context = context.to(device)
            action_seq = action_seq.to(device)
            target_seq = target_seq.to(device)

            n, k, c, h, w = context.shape
            u = action_seq.shape[1]

            frames_flat = torch.cat(
                [context.reshape(n * k, c, h, w), target_seq.reshape(n * u, c, h, w)], dim=0
            )
            z_all = E(frames_flat)
            z_context = z_all[: n * k].reshape(n, k, -1)
            z_targets = z_all[n * k :].reshape(n, u, -1)
            z_last_real = z_context[:, -1, :]

            # Hyperbolic swap: E and P still output raw, unconstrained Euclidean
            # vectors (unchanged) -- cov_penalty_term below keeps operating on
            # those directly, so the anti-collapse term never fights the ball's
            # geometry. exp0 is applied only transiently, right here, to get a
            # point actually on the manifold for the distance computation.
            #
            # dist(...)**2 rather than dist(...) itself: hyperbolic distance is
            # built from arccosh, which -- just like the plain Euclidean length
            # functional the very first derivation in this project started from
            # -- has a sqrt-like singularity in its gradient as the argument
            # approaches its branch point (distance -> 0). Squaring it is the
            # same "energy vs length" fix as back then: same minimizer (zero
            # distance is still zero either way), smooth gradient throughout.
            inv_losses = []
            for i in range(u):
                z_pred = P(z_context, action_seq[:, i, :])
                z_pred_ball = hyp.exp0(z_pred)
                z_target_ball = hyp.exp0(z_targets[:, i, :])
                inv_losses.append(0.5 * hyp.dist(z_pred_ball, z_target_ball).pow(2).mean())
                z_context = torch.cat([z_context[:, 1:, :], z_pred.unsqueeze(1)], dim=1)
            inv_loss = torch.stack(inv_losses).mean()

            cov_penalty = jt.cov_penalty_term(z_last_real) + torch.stack(
                [jt.cov_penalty_term(z_targets[:, i, :]) for i in range(u)]
            ).mean()

            loss = inv_loss + (KAPPA / 2) * cov_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_inv_losses.append(inv_loss.item())
            epoch_cov_penalties.append(cov_penalty.item())
            epoch_z_norms.append(z_last_real.detach().norm(dim=-1).mean().item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_inv_loss = sum(epoch_inv_losses) / len(epoch_inv_losses)
        avg_cov_penalty = sum(epoch_cov_penalties) / len(epoch_cov_penalties)
        avg_z_norm = sum(epoch_z_norms) / len(epoch_z_norms)
        print(
            f"epoch {epoch + 1}/{N_EPOCHS}  avg_loss={avg_loss:.6f}  "
            f"avg_inv_loss={avg_inv_loss:.6f}  avg_cov_penalty={avg_cov_penalty:.6f}  "
            f"avg_raw_z_norm={avg_z_norm:.3f}"
        )
        log_writer.writerow([epoch, avg_loss, avg_inv_loss, avg_cov_penalty, avg_z_norm])
        log_file.flush()

        torch.save(
            {
                "E": E.state_dict(),
                "P": P.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "k_context": jt.K_CONTEXT,
                "latent_dim": jt.LATENT_DIM,
                "predictor_stride": PREDICTOR_STRIDE,
            },
            CHECKPOINT_PATH,
        )

    log_file.close()
    print(f"Checkpoint saved to {CHECKPOINT_PATH}, log saved to {LOG_PATH}")


if __name__ == "__main__":
    train()
