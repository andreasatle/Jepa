"""
v3, phase 2 -- joint fine-tuning of E together with ALL FIVE multi-stride
predictors (P_1, P_2, P_4, P_8, P_16) at once, instead of E staying frozen
(jepa_train_multistride.py).

Why: jepa_train_multistride.py's frozen-encoder results were already good
(flat, no blowup, held-out ball error actually improving with stride
through stride=8) -- so this isn't a fix for a confirmed failure, it's a
deliberate second experiment: one shared E trained against ALL five
resolutions' objectives at once should (a) give E explicit pressure to
keep whatever information the LONGER strides need, closing the
"nearsighted encoder" gap freezing couldn't rule out, and (b) pull the
five predictors toward mutual consistency for free, as a side effect of
sharing gradients through the same E, rather than needing a separate
telescoping-consistency loss.

Design (agreed with Andreas):
  - One shared anchor real-time index t0 per training example. All five
    (context, action-block, target) windows for that example are built
    from the SAME t0, so one update step sees every resolution's view of
    the same moment. t0's valid range is set by the MOST restrictive
    stride (16): needs (k_context-1)*16 + UNROLL_STEPS*16 = 112 real
    steps of room, out of jepa_data/'s 200 -- that range comfortably
    contains what every smaller stride needs too.
  - Loss: mean (not sum) of the five strides' own hyperbolic
    squared-distance losses -- equal weighting to start, and averaging
    (not summing) keeps the overall loss scale comparable to a
    single-stride run, so KAPPA doesn't need retuning.
  - E is trainable again -> VICReg covariance penalty comes back (mean
    over the five strides' own cov terms, each computed the same way as
    jepa_train.py / jepa_train_strided.py: cov(z_last_real) + mean_i
    cov(z_targets[:,i])), since collapse is only ever a risk while E is
    being updated.
  - Warm start: E from jepa_checkpoint_strided.pt (v2), each P_s from its
    already-trained predictor_stride{s}.pt (jepa_train_multistride.py) --
    fine-tuning from a reasonable starting point, not from scratch.
  - Separate checkpoint files (jepa_checkpoint_joint.pt,
    predictor_stride{s}_joint.pt) so today's frozen-encoder baseline
    stays intact for comparison.

Heads up on cost: this calls E once PER STRIDE per batch (5 forward+
backward passes through the conv stack instead of the 1 that
jepa_train_multistride.py used, since E was frozen there) -- expect this
to run noticeably slower per epoch than either jepa_train_strided.py or
jepa_train_multistride.py. N_EPOCHS defaults lower for this reason; bump
it (same resume support as the other scripts) if the loss is still moving
when it finishes.

Run:
    uv run python jepa_train_joint.py
"""

import csv
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import jepa_train as jt
import hyperbolic as hyp

DATA_DIR = pathlib.Path("jepa_data")
BASE_ENCODER_CHECKPOINT = "jepa_checkpoint_strided.pt"   # warm-start source for E
JOINT_ENCODER_CHECKPOINT = "jepa_checkpoint_joint.pt"

STRIDE_LEVELS = [1, 2, 4, 8, 16]
UNROLL_STEPS = 5
N_EPOCHS = 1700  # cumulative epoch count (resume uses absolute epoch #, not epochs-since-new-data) -- ~480+ epochs on the 5x-bigger dataset from here; bump further or stop early based on joint_snapshots + eval checks

# Periodic, never-overwritten snapshots -- separate from the per-epoch
# resume checkpoint above (JOINT_ENCODER_CHECKPOINT / predictor_stride{s}_joint.pt),
# which gets replaced every epoch and therefore can't recover an earlier,
# possibly-better-generalizing point once training has moved past it.
SNAPSHOT_EVERY = 50  # epochs between snapshots; adjust for disk usage vs. granularity
SNAPSHOT_DIR = pathlib.Path("joint_snapshots")

BATCH_SIZE = jt.BATCH_SIZE
LR = jt.LR
KAPPA = jt.KAPPA
SEED = 0

LOG_PATH = "jepa_training_log_joint.csv"


class JointMultiResDataset(Dataset):
    """One example = one real anchor index t0, with a full (context,
    action-block, target) window built for EVERY stride in STRIDE_LEVELS
    off that same t0. Valid t0 range is set by the largest stride, so
    every smaller stride's window fits automatically."""

    def __init__(self, data_dir, k_context, unroll_steps, stride_levels):
        self.k = k_context
        self.u = unroll_steps
        self.strides = stride_levels
        self.episodes = []
        self.index = []

        paths = sorted(pathlib.Path(data_dir).glob("episode_*.npz"))
        if not paths:
            raise FileNotFoundError(f"No episode_*.npz files found in {data_dir}/")

        max_s = max(stride_levels)
        for ep_idx, path in enumerate(paths):
            data = np.load(path)
            frames = data["frames"]
            actions = data["actions"]
            self.episodes.append((frames, actions))

            T = actions.shape[0]
            k, u = k_context, unroll_steps
            lo = (k - 1) * max_s
            hi = T - u * max_s + 1
            for t0 in range(lo, hi):
                self.index.append((ep_idx, t0))

        print(
            f"Loaded {len(paths)} episodes, {len(self.index)} joint examples "
            f"(strides={stride_levels}, k_context={k_context}, unroll_steps={unroll_steps})."
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t0 = self.index[idx]
        frames, actions = self.episodes[ep_idx]
        k, u = self.k, self.u

        item = {}
        for s in self.strides:
            context_idx = [t0 - (k - 1 - j) * s for j in range(k)]
            context = frames[context_idx]
            target_seq = np.stack([frames[t0 + (i + 1) * s] for i in range(u)])
            action_block_seq = np.stack(
                [actions[t0 + i * s: t0 + (i + 1) * s].reshape(-1) for i in range(u)]
            )
            item[f"context_{s}"] = jt._frames_to_tensor(context)
            item[f"targets_{s}"] = jt._frames_to_tensor(target_seq)
            item[f"actions_{s}"] = torch.from_numpy(action_block_seq.astype(np.float32))

        return item


def load_or_warmstart(device):
    E = jt.Encoder(3, jt.LATENT_DIM).to(device)
    predictors = {
        s: jt.Predictor(jt.LATENT_DIM, s * jt.ACTION_DIM, jt.PREDICTOR_HIDDEN, jt.K_CONTEXT).to(device)
        for s in STRIDE_LEVELS
    }

    joint_path = pathlib.Path(JOINT_ENCODER_CHECKPOINT)
    if joint_path.exists():
        print(f"Found {JOINT_ENCODER_CHECKPOINT} -- resuming the joint run from it.")
        ckpt = torch.load(joint_path, map_location=device)
        E.load_state_dict(ckpt["E"])
        start_epoch = ckpt.get("epoch", -1) + 1
        for s in STRIDE_LEVELS:
            p_ckpt = torch.load(f"predictor_stride{s}_joint.pt", map_location=device)
            predictors[s].load_state_dict(p_ckpt["P"])
        optimizer_state = ckpt.get("optimizer")
        print(f"Already trained through epoch {start_epoch}/{N_EPOCHS}.")
    else:
        print(f"No {JOINT_ENCODER_CHECKPOINT} found -- warm-starting from the frozen-encoder-era checkpoints.")
        base_ckpt = torch.load(BASE_ENCODER_CHECKPOINT, map_location=device)
        E.load_state_dict(base_ckpt["E"])
        start_epoch = 0
        for s in STRIDE_LEVELS:
            p_ckpt = torch.load(f"predictor_stride{s}.pt", map_location=device)
            predictors[s].load_state_dict(p_ckpt["P"])
        optimizer_state = None

    return E, predictors, start_epoch, optimizer_state


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    E, predictors, start_epoch, optimizer_state = load_or_warmstart(device)
    optimizer = torch.optim.Adam(
        list(E.parameters()) + [p for P in predictors.values() for p in P.parameters()],
        lr=LR,
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    dataset = JointMultiResDataset(DATA_DIR, jt.K_CONTEXT, UNROLL_STEPS, STRIDE_LEVELS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    log_path = pathlib.Path(LOG_PATH)
    write_header = not log_path.exists()
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(
            ["epoch", "avg_loss", "avg_inv_loss", "avg_cov_penalty"]
            + [f"avg_inv_loss_s{s}" for s in STRIDE_LEVELS]
        )

    for epoch in range(start_epoch, N_EPOCHS):
        epoch_losses, epoch_inv, epoch_cov = [], [], []
        epoch_inv_per_stride = {s: [] for s in STRIDE_LEVELS}

        for batch in loader:
            inv_per_stride = []
            cov_per_stride = []

            for s in STRIDE_LEVELS:
                context = batch[f"context_{s}"].to(device)
                action_seq = batch[f"actions_{s}"].to(device)
                target_seq = batch[f"targets_{s}"].to(device)

                n, k, c, h, w = context.shape
                u = action_seq.shape[1]

                frames_flat = torch.cat(
                    [context.reshape(n * k, c, h, w), target_seq.reshape(n * u, c, h, w)], dim=0
                )
                z_all = E(frames_flat)
                z_context = z_all[: n * k].reshape(n, k, -1)
                z_targets = z_all[n * k:].reshape(n, u, -1)
                z_last_real = z_context[:, -1, :]

                P = predictors[s]
                step_losses = []
                for i in range(u):
                    z_pred = P(z_context, action_seq[:, i, :])
                    z_pred_ball = hyp.exp0(z_pred)
                    z_target_ball = hyp.exp0(z_targets[:, i, :])
                    step_losses.append(0.5 * hyp.dist(z_pred_ball, z_target_ball).pow(2).mean())
                    z_context = torch.cat([z_context[:, 1:, :], z_pred.unsqueeze(1)], dim=1)
                inv_loss_s = torch.stack(step_losses).mean()

                cov_s = jt.cov_penalty_term(z_last_real) + torch.stack(
                    [jt.cov_penalty_term(z_targets[:, i, :]) for i in range(u)]
                ).mean()

                inv_per_stride.append(inv_loss_s)
                cov_per_stride.append(cov_s)
                epoch_inv_per_stride[s].append(inv_loss_s.item())

            inv_loss = torch.stack(inv_per_stride).mean()
            cov_penalty = torch.stack(cov_per_stride).mean()
            loss = inv_loss + (KAPPA / 2) * cov_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_inv.append(inv_loss.item())
            epoch_cov.append(cov_penalty.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_inv = sum(epoch_inv) / len(epoch_inv)
        avg_cov = sum(epoch_cov) / len(epoch_cov)
        per_stride_avgs = [sum(v) / len(v) for v in epoch_inv_per_stride.values()]

        per_stride_str = "  ".join(f"s{s}={v:.4f}" for s, v in zip(STRIDE_LEVELS, per_stride_avgs))
        print(
            f"epoch {epoch + 1}/{N_EPOCHS}  avg_loss={avg_loss:.6f}  avg_inv_loss={avg_inv:.6f}  "
            f"avg_cov_penalty={avg_cov:.6f}  {per_stride_str}"
        )
        log_writer.writerow([epoch, avg_loss, avg_inv, avg_cov] + per_stride_avgs)
        log_file.flush()

        torch.save(
            {"E": E.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
            JOINT_ENCODER_CHECKPOINT,
        )
        for s in STRIDE_LEVELS:
            torch.save(
                {"P": predictors[s].state_dict(), "epoch": epoch, "stride": s},
                f"predictor_stride{s}_joint.pt",
            )

        if (epoch + 1) % SNAPSHOT_EVERY == 0:
            snap_epoch = epoch + 1
            torch.save(
                {"E": E.state_dict(), "epoch": epoch},
                SNAPSHOT_DIR / f"jepa_checkpoint_joint_ep{snap_epoch:05d}.pt",
            )
            for s in STRIDE_LEVELS:
                torch.save(
                    {"P": predictors[s].state_dict(), "epoch": epoch, "stride": s},
                    SNAPSHOT_DIR / f"predictor_stride{s}_joint_ep{snap_epoch:05d}.pt",
                )
            print(f"  -> snapshot saved at epoch {snap_epoch} in {SNAPSHOT_DIR}/")

    log_file.close()
    print(f"Checkpoints saved: {JOINT_ENCODER_CHECKPOINT}, predictor_stride*_joint.pt; log: {LOG_PATH}")


if __name__ == "__main__":
    train()
