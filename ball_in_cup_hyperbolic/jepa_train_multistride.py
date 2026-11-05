"""
v3 -- hierarchical multi-stride predictors P_1, P_2, P_4, ... trained on
top of the FROZEN v2 encoder from jepa_train_strided.py's hyperbolic
checkpoint.

Design (agreed with Andreas, see ball-in-cup-v2-roadmap.md's "nearsighted
encoder" note for the risk we're deliberately treating this as a first
diagnostic for, not a final architectural commitment):

  - E is loaded from jepa_checkpoint_strided.pt (v2, hyperbolic) and
    frozen -- not in the optimizer, forward pass wrapped in no_grad().
    Consequence: the VICReg covariance penalty existed only to keep E
    itself from collapsing while it was being trained jointly with P.
    With E fixed, that risk is gone, so the loss here is JUST the
    hyperbolic squared-distance term -- no cov_penalty.

  - One stride s = "how many real control-steps one P_s call spans" and
    ALSO "how many real actions get consumed in one P_s call" (the full
    heterogeneous sequence of actions actually taken across that jump,
    concatenated -- not one action repeated, since that's what actually
    happened in the data and what CEM will want to feed in at inference:
    a candidate future action sequence, not a constant). jt.Predictor's
    signature already takes an arbitrary action_dim, so no new model
    class is needed -- P_s is just jt.Predictor(LATENT_DIM,
    s * jt.ACTION_DIM, jt.PREDICTOR_HIDDEN, jt.K_CONTEXT), one instance
    (separate weights, not shared) per stride level.

  - K_CONTEXT stays 3, but now spaced s real steps apart (matching what
    one P_s call represents) rather than the native dt -- same
    "coarse-grained window" idea jepa_train_strided.py used, just no
    longer hardcoded to s=5. Stride 5 in v2 was purely a numerical
    convenience (fewer P-calls needed to survive a long rollout); it has
    no other significance and isn't reused here.

  - Stride set: 1, 2, 4, 8, 16 -- five doublings, reaching well past
    where the v2 single-stride (s=5) model stood. Each level's window
    still fits comfortably inside jepa_data/'s 200-real-step episodes
    (stride 16 needs (k_context-1+UNROLL_STEPS)*16 = 112 steps per
    example, out of 200). Extending further (64, 128) later is cheap --
    just add to STRIDE_LEVELS -- not attempted yet.

  - Same multi-step (unroll) exposure-bias fix as v1/v2: UNROLL_STEPS
    chained P_s calls per training example, predicted z fed back into
    the (s-spaced) context window between calls, targets always real.

Requires jepa_data/ (i.i.d. per-step random actions) and
ENCODER_CHECKPOINT to already exist.

REPURPOSED for the "refreeze E, retrain predictors" contingency (see
conversation): after growing jepa_data/ to 500 episodes and jointly
fine-tuning E+all predictors (jepa_train_joint.py), held-out drift
diagnostics showed s8/s16 roughly 2x worse than they were as a frozen-
encoder baseline, and plateaued across two checks ~30 epochs apart --
consistent with predictors chasing a co-adapting E (a harder target than
a fixed one). This run now freezes E at its CURRENT joint-fine-tuned
state (ENCODER_CHECKPOINT = jepa_checkpoint_joint.pt, not the original
pre-joint jepa_checkpoint_strided.pt) and warm-starts each predictor from
its joint-phase weights (predictor_stride{s}_joint.pt), giving them a
stable target to re-converge against for a while, before (if it helps)
going back to joint fine-tuning. Writes to predictor_stride{s}_refrozen.pt
-- a THIRD file, distinct from both predictor_stride{s}.pt (the original
pre-joint frozen-baseline weights) and predictor_stride{s}_joint.pt (the
current joint weights) -- so neither existing checkpoint set is touched;
both remain full fallbacks if this doesn't help either.

Also adds LR decay (CosineAnnealingLR), absent from this whole project
until now: the joint run's flat LR=3e-4 held constant for its entire
1500+ epoch run, and its per-epoch loss was oscillating in a narrow band
rather than trending down for the last ~300 epochs -- the classic sign
of an LR too coarse to make further progress this late in a run, distinct
from (and possibly compounding) the moving-target problem above.

Run:
    uv run python jepa_train_multistride.py
"""

import csv
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import jepa_train as jt
import hyperbolic as hyp

DATA_DIR = pathlib.Path("jepa_data")
ENCODER_CHECKPOINT = "jepa_checkpoint_joint.pt"   # current joint-fine-tuned
                                                    # E, frozen from here --
                                                    # see module docstring's
                                                    # "REPURPOSED" section.
                                                    # NOT jepa_checkpoint_strided.pt
                                                    # (the original pre-joint E)
                                                    # anymore.

STRIDE_LEVELS = [1, 2, 4, 8, 16]   # doubling; extend later (see module docstring)
UNROLL_STEPS = 5            # same per-level exposure-bias fix as v1/v2
N_EPOCHS = 500              # same default as v1/v2 -- checkpoints resume,
                             # so this is safe to interrupt early per level

BATCH_SIZE = jt.BATCH_SIZE
LR = jt.LR
LR_MIN_FRAC = 0.01          # CosineAnnealingLR decays from LR down to
                             # LR * LR_MIN_FRAC over N_EPOCHS -- see module
                             # docstring's LR-decay paragraph.
SEED = 0

CHECKPOINT_SUFFIX = "_refrozen"   # predictor_stride{s}_refrozen.pt -- this
                                    # refreeze phase's own checkpoints, never
                                    # overwrites predictor_stride{s}.pt or
                                    # predictor_stride{s}_joint.pt.
WARMSTART_SUFFIX = "_joint"        # predictor_stride{s}_joint.pt -- weights
                                    # to warm-start from if no _refrozen
                                    # checkpoint exists yet for this stride.


class MultiStrideDataset(Dataset):
    """Like StridedJepaTransitionDataset, but the action fed to each P_s
    call is the full block of `stride` REAL actions spanning that jump
    (flattened to stride * ACTION_DIM), read straight from jepa_data/'s
    i.i.d.-per-step actions -- not a single held-constant action."""

    def __init__(self, data_dir, k_context, unroll_steps, stride):
        self.k_context = k_context
        self.unroll_steps = unroll_steps
        self.stride = stride
        self.episodes = []
        self.index = []

        paths = sorted(pathlib.Path(data_dir).glob("episode_*.npz"))
        if not paths:
            raise FileNotFoundError(f"No episode_*.npz files found in {data_dir}/")

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
            f"(stride={stride}, k_context={k_context}, unroll_steps={unroll_steps})."
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        frames, actions = self.episodes[ep_idx]
        k, u, s = self.k_context, self.unroll_steps, self.stride

        context_idx = [t - (k - 1 - j) * s for j in range(k)]
        context = frames[context_idx]                                        # (k, H, W, 3)
        target_seq = np.stack([frames[t + (i + 1) * s] for i in range(u)])    # (u, H, W, 3)
        action_block_seq = np.stack(
            [actions[t + i * s: t + (i + 1) * s].reshape(-1) for i in range(u)]
        )                                                                     # (u, s * action_dim)

        context = jt._frames_to_tensor(context)
        target_seq = jt._frames_to_tensor(target_seq)
        action_block_seq = torch.from_numpy(action_block_seq.astype(np.float32))

        return context, action_block_seq, target_seq


def load_frozen_encoder(device):
    ckpt = torch.load(ENCODER_CHECKPOINT, map_location=device)
    E = jt.Encoder(3, jt.LATENT_DIM).to(device)
    E.load_state_dict(ckpt["E"])
    E.eval()
    for p in E.parameters():
        p.requires_grad_(False)
    print(f"Loaded frozen E from {ENCODER_CHECKPOINT} (epoch {ckpt.get('epoch')}).")
    return E


def train_one_stride(E, device, stride):
    checkpoint_path = pathlib.Path(f"predictor_stride{stride}{CHECKPOINT_SUFFIX}.pt")
    warmstart_path = pathlib.Path(f"predictor_stride{stride}{WARMSTART_SUFFIX}.pt")
    log_path = pathlib.Path(f"predictor_stride{stride}{CHECKPOINT_SUFFIX}_log.csv")

    print(f"\n=== stride={stride} ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    P = jt.Predictor(jt.LATENT_DIM, stride * jt.ACTION_DIM, jt.PREDICTOR_HIDDEN, jt.K_CONTEXT).to(device)
    optimizer = torch.optim.Adam(P.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * LR_MIN_FRAC)

    dataset = MultiStrideDataset(DATA_DIR, jt.K_CONTEXT, UNROLL_STEPS, stride)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    start_epoch = 0
    if checkpoint_path.exists():
        print(f"Found {checkpoint_path} -- resuming this refreeze phase from it.")
        ckpt = torch.load(checkpoint_path, map_location=device)
        P.load_state_dict(ckpt["P"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Already trained through epoch {start_epoch}/{N_EPOCHS} of the refreeze phase.")
    elif warmstart_path.exists():
        print(
            f"No {checkpoint_path} yet -- warm-starting P from {warmstart_path} "
            "(joint-phase weights); optimizer/scheduler/epoch count start fresh "
            "for this refreeze phase (predictor_stride{s}_joint.pt only ever "
            "saved P's weights, no optimizer state to inherit -- see module "
            "docstring)."
        )
        wckpt = torch.load(warmstart_path, map_location=device)
        P.load_state_dict(wckpt["P"])
    else:
        print(f"No {checkpoint_path} or {warmstart_path} found -- starting fresh.")

    write_header = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["epoch", "avg_inv_loss", "lr"])

    for epoch in range(start_epoch, N_EPOCHS):
        epoch_losses = []

        for context, action_block_seq, target_seq in loader:
            context = context.to(device)
            action_block_seq = action_block_seq.to(device)
            target_seq = target_seq.to(device)

            n, k, c, h, w = context.shape
            u = action_block_seq.shape[1]

            with torch.no_grad():
                frames_flat = torch.cat(
                    [context.reshape(n * k, c, h, w), target_seq.reshape(n * u, c, h, w)], dim=0
                )
                z_all = E(frames_flat)
                z_context = z_all[: n * k].reshape(n, k, -1)
                z_targets = z_all[n * k:].reshape(n, u, -1)

            inv_losses = []
            for i in range(u):
                z_pred = P(z_context, action_block_seq[:, i, :])
                z_pred_ball = hyp.exp0(z_pred)
                z_target_ball = hyp.exp0(z_targets[:, i, :])
                inv_losses.append(0.5 * hyp.dist(z_pred_ball, z_target_ball).pow(2).mean())
                z_context = torch.cat([z_context[:, 1:, :], z_pred.unsqueeze(1)], dim=1)
            loss = torch.stack(inv_losses).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"stride={stride}  epoch {epoch + 1}/{N_EPOCHS}  avg_inv_loss={avg_loss:.6f}  lr={current_lr:.2e}")
        log_writer.writerow([epoch, avg_loss, current_lr])
        log_file.flush()

        torch.save(
            {
                "P": P.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "stride": stride,
            },
            checkpoint_path,
        )

    log_file.close()
    print(f"stride={stride}: checkpoint saved to {checkpoint_path}, log saved to {log_path}")


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    E = load_frozen_encoder(device)
    for stride in STRIDE_LEVELS:
        train_one_stride(E, device, stride)


if __name__ == "__main__":
    train()
