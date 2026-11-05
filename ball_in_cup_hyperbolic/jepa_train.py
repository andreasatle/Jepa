"""
JEPA pretraining for ball_in_cup-catch -- step 3 of the project.

Two parts, as requested:
  Part 1: the models (E, P) and how they're applied.
  Part 2: the dataset built from jepa_data/*.npz, and the training loop.

Architecture (revised): E is single-frame -- one image in, one z out, purely
spatial, no temporal information in any individual z. P is given a short
window of K_CONTEXT consecutive single-frame embeddings (plus the action)
and does all the temporal reasoning itself:

    z_{t-k+1}, ..., z_t   = E(x_{t-k+1}), ..., E(x_t)      -- k = K_CONTEXT
    z_pred                = P([z_{t-k+1}, ..., z_t], a_t)
    z_target               = E(x_{t+1})

This replaced an earlier version where E itself took a stack of frames
concatenated on the channel dimension and did the temporal compression
internally. The switch was motivated by a linear probe against ground-truth
state: with the old single-vector-per-window scheme, a linear map from z to
true position recovered it well (test R^2 ~0.85-0.90), but velocity was
essentially unrecoverable (test R^2 ~0). Reasoning for why, and why this
restructuring is the principled fix rather than just a bigger network:
ball_in_cup is governed by Newton's second law, a second-order ODE in
position. The standard way to make a second-order system Markovian -- i.e.
have a "current state" from which the future is fully determined -- is to
augment it to phase space (x, v), since position alone can't predict the
future (same position, different velocity, different outcome). A single
frame (or a single frame-stack compressed to one vector) forces E to
pre-decide, once and for all, how to pack that phase-space information into
one representation; asking P to look at several past z's directly instead
lets it reconstruct velocity itself, the same way any multistep/finite-
difference integrator does, except learned and nonlinear instead of
Taylor-derived and linear.

K_CONTEXT = 3, not 2: with only two points, velocity is estimated as
(x_t - x_{t-1})/dt; with dm_control's control_timestep = 0.02s, dividing by
dt^2 amplifies any position-estimation noise by a factor of ~2500. Using the
outer two of three points, (x_t - x_{t-2})/(2*dt), roughly halves the
resulting noise's standard deviation (a 4x variance reduction) for the same
underlying position noise. Three points, not more: (x, v) is a *complete*
phase space for a second-order system -- acceleration is a function of
(x, v, action) via F=ma, not an independent state component, so there's no
first-principles reason to go beyond what's needed for a robust velocity
estimate.

Anti-collapse mechanism: VICReg (your formulation) -- no target network, no
EMA, no stop-gradient. The same shared encoder E is applied to every frame
in the context window and to the target frame; collapse is excluded
directly by penalizing the covariance matrices of the most recent context
frame's embedding and the target embedding for straying from the identity:

    L = 1/2 * ||P([z_{t-k+1},...,z_t], a_t) - E(x_{t+1})||^2
        + kappa/2 * ( pen(z_t) + pen(E(x_{t+1})) )

where pen(z), for the D x D matrix C = Cov(z), is NOT a plain
||C - I||_F^2. It's split by diagonal vs off-diagonal:

    pen(z) = sum_{j != k} C_jk^2            (off-diagonal, as-is)
             + sum_j ( sqrt(C_jj) - 1 )^2   (diagonal, sqrt first)

Why the diagonal is treated differently: a first smoke test of the plain
||C - I||_F^2 version collapsed to near-zero output and got stuck there.
Verified numerically (not just suspected): C_jj is the batch variance
sigma_j^2, already a squared/degree-2-in-z quantity, and squaring
(C_jj - 1) again makes the diagonal penalty degree-4 in z -- its gradient
w.r.t. z is proportional to z itself, so as z collapses toward 0 the
restoring force that's supposed to fight collapse vanishes exactly when
it's needed most (checked: gradient norm ~500x weaker at ||z||~0.001 than
at ||z||~1, for a fixed random z). The off-diagonal term doesn't have this
problem -- its target (0) already coincides with where collapse sends it,
so there's nothing to fix there. sqrt(C_jj) = sigma_j undoes the squaring
before penalizing, which restores a non-vanishing gradient at collapse
(checked: ~0.25 at ||z||~0.001, same order as at ||z||~1) while staying
fully squared/smooth/symmetric -- no hinge, and (unlike a hinge) it also
pulls back oversized variance instead of only pushing up from below.

Activations: SiLU throughout (encoder and predictor), not ReLU -- avoids
ever landing on ReLU's exact-zero dead-neuron case, for free, since there's
no benefit to ReLU here worth trading for that risk.

Normalization: none. BatchNorm was added earlier specifically to stabilize
EMA/stop-gradient training (per Richemond et al., BYOL-style methods need
it in practice) -- that justification doesn't apply to VICReg, whose
anti-collapse guarantee doesn't depend on batch statistics. Removing it
also fixes the batch-size-1 crash in BatchNorm1d we hit before, which
matters once we're running the predictor on single states for planning.

Multi-step (unrolled) training -- why, added in v7: drift_diagnostic.py
(rolling P forward autoregressively on real recorded actions and comparing
to ground truth) found P's rollout error doesn't just grow with horizon --
it explodes exponentially, reaching literally kilometers of predicted
position error by ~150 steps. The root cause: P is only ever trained
one-step, with a 100% REAL context window every time (teacher forcing) --
it has never once seen its own predicted output sitting in that window
during training, so once a rollout starts feeding predictions back into
themselves (unavoidable at inference -- that's what autoregressive rollout
IS), they're off-distribution for P and its response to that is unstable
rather than just noisier. This is "exposure bias", a known failure mode of
teacher-forced sequence models.

The fix: train P on UNROLL_STEPS=U successive calls per example instead of
1, feeding the PREDICTED (not real) embedding back into the context window
between calls -- exactly mirroring what jepa_evaluator.rollout does at
inference, except now with gradients flowing through it. The targets at
every one of the U steps are still real frames (we have ground truth for
all of them), so this doesn't relax supervision, it only changes what P
sees as *input* on steps 2..U. u=1 exactly reproduces the original
single-step loss -- useful as a regression check that this refactor didn't
silently change anything at U=1.

Install:
    uv add torch

Run (reads jepa_data/*.npz produced by collect_data.py):
    python3 jepa_train.py

NOTE: checkpoints from earlier versions are NOT compatible -- E now takes a
single frame (3 channels) instead of a frame-stack, and P takes K_CONTEXT
separate embeddings instead of one. Delete jepa_checkpoint.pt and
jepa_training_log.csv before running this version.

Saves:
    jepa_checkpoint.pt     -- E, P, optimizer, epoch
    jepa_training_log.csv  -- epoch, avg_loss, avg_inv_loss, avg_cov_penalty,
                               avg_raw_z_norm

Version history (printed at the start of every run, so you can always tell
which version actually produced a given output):
    v1  raw MSE loss on the EMA target (original version)
    v2  L2-normalize before the loss (fixes norm explosion) +
        avg_raw_z_norm diagnostic
    v3  + avg_z_dir_std diagnostic (directional-collapse check)
    v4  + BatchNorm in the encoder's conv layers and the predictor's
        hidden layers (fixes the directional collapse v3 caught)
    v5  resume support -- checkpoints (model + optimizer + epoch) and the
        log are saved every epoch, and reloaded automatically at the start
        of train() if jepa_checkpoint.pt already exists, so N_EPOCHS can be
        raised and the run continued instead of restarted from scratch
    v6  fix: loading a pre-v5 checkpoint (no "optimizer"/"epoch" keys) no
        longer crashes -- falls back gracefully and says what was and
        wasn't restored
    v7  multi-step (unrolled) training loss, UNROLL_STEPS=U -- P is called
        U times per example, and after each call the PREDICTED embedding
        (not the real one) is slid into the context window for the next
        call, so P is trained against its own compounding error instead
        of only ever seeing ground-truth context. Targets stay real at
        every step (still real frames encoded by E). U=1 reproduces the
        original single-step loss exactly. Motivated by
        drift_diagnostic.py showing catastrophic (unbounded, exponential)
        rollout error at inference time -- P had never once seen its own
        output during training (exposure bias). Doesn't touch E/P's
        architecture -- v6 checkpoints still load fine under v7.
"""

import csv
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

VERSION = "v7"

# ---------------------------------------------------------------------------
# Hyperparameters -- all the design choices from jepa_theory_and_design.md
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path("jepa_data")
K_CONTEXT = 3             # k -- consecutive single-frame embeddings fed to
                          # P (not to E -- see module docstring). 3, not 2,
                          # for the noise-robustness reason explained there.
LATENT_DIM = 32           # D -- the actual bottleneck width (see doc, sec 3.1)
ACTION_DIM = 2            # cup's x/z motors
PREDICTOR_HIDDEN = 256    # H
IMAGE_SIZE = 64

BATCH_SIZE = 256
LR = 3e-4
KAPPA = 1.0               # weight on the covariance penalty -- tunable.
                          # Worth knowing going in: with D=32, the off-
                          # diagonal sum has D*(D-1)=992 entries against the
                          # diagonal's 32, so the off-diagonal term will
                          # naturally dominate the total purely by entry
                          # count unless per-entry magnitudes differ a lot.
                          # Watch avg_cov_penalty -- adjust KAPPA if it
                          # isn't converging.
UNROLL_STEPS = 5          # U -- number of successive P calls trained
                          # together per example, feeding P's own
                          # predictions back in between calls (see module
                          # docstring's "Multi-step (unrolled) training"
                          # section). U=1 reproduces the original
                          # single-step loss exactly. Purely a training-
                          # loop/dataset-windowing choice -- doesn't touch
                          # E/P's architecture, so this is safe to change
                          # across a resume, unlike K_CONTEXT etc.

N_EPOCHS = 350

CHECKPOINT_PATH = "jepa_checkpoint.pt"
LOG_PATH = "jepa_training_log.csv"
SEED = 0


# ===========================================================================
# PART 1 -- Models: E, P, and how they're applied
# ===========================================================================

class Encoder(nn.Module):
    """E -- maps a SINGLE frame (3 channels, 64x64) down to a LATENT_DIM
    vector. Purely spatial -- no temporal information in an individual z;
    that's P's job now (see module docstring). The conv stack halves
    spatial resolution 4 times (64->32->16->8->4) while growing channel
    count; the actual dimensionality bottleneck is the final Linear layer,
    not the conv stack itself (see design doc, section 3.1). No BatchNorm --
    see module docstring. SiLU instead of ReLU throughout."""

    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """P -- given K_CONTEXT consecutive embeddings [z_{t-k+1},...,z_t] and
    action a_t, predicts z_{t+1}. All temporal reasoning (velocity, in
    particular) happens here, not in E -- see module docstring. Operates
    purely on vectors, never touches pixels. No BatchNorm, SiLU instead of
    ReLU."""

    def __init__(self, latent_dim, action_dim, hidden_dim, k_context):
        super().__init__()
        self.k_context = k_context
        self.net = nn.Sequential(
            nn.Linear(k_context * latent_dim + action_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_context, a):
        """z_context: (N, k_context, latent_dim). a: (N, action_dim)."""
        z_flat = z_context.reshape(z_context.shape[0], -1)
        return self.net(torch.cat([z_flat, a], dim=-1))


def covariance_matrix(z):
    """Cov(z) over the batch dimension -- z is (N, D), returns (D, D).
    Mean-centers over the batch first, standard sample-covariance formula
    (divide by N-1)."""
    n = z.shape[0]
    z_centered = z - z.mean(dim=0, keepdim=True)
    return (z_centered.T @ z_centered) / (n - 1)


def cov_penalty_term(z, eps=1e-8):
    """One branch's contribution to the anti-collapse penalty (see module
    docstring for why the diagonal is treated differently from the
    off-diagonal). z is (N, D)."""
    cov = covariance_matrix(z)
    diag = torch.diagonal(cov)
    off_diag = cov - torch.diag(diag)

    off_diag_term = off_diag.pow(2).sum()
    # sqrt before penalizing -- undoes the diagonal's built-in squaring, so
    # the gradient doesn't vanish right where collapse needs fighting. eps
    # keeps sqrt's gradient finite at exactly C_jj = 0.
    diag_term = ((diag + eps).sqrt() - 1).pow(2).sum()

    return off_diag_term + diag_term


def make_models(device):
    """Builds E (single-frame, shared across every frame it ever sees -- no
    separate target network) and P (consumes K_CONTEXT embeddings at once).

    Application, per transition (context frames x_{t-k+1..t}, a_t, x_{t+1}):
        z_context = E(x_{t-k+1}), ..., E(x_t)     -- trained by backprop
        z_target  = E(x_{t+1})                    -- SAME E, also backprop
        z_pred    = P(z_context, a_t)              -- trained by backprop
        loss      = 1/2 * ||z_pred - z_target||^2
                    + kappa/2 * (cov_penalty_term(z_context[-1])
                                 + cov_penalty_term(z_target))
    """
    in_channels = 3   # single frame now, not 3 * (frame-stack size)
    E = Encoder(in_channels, LATENT_DIM).to(device)
    P = Predictor(LATENT_DIM, ACTION_DIM, PREDICTOR_HIDDEN, K_CONTEXT).to(device)
    return E, P


# ===========================================================================
# PART 2 -- Data and the optimization loop
# ===========================================================================

def _frames_to_tensor(frames_uint8):
    """(n, H, W, 3) uint8 -> (n, 3, H, W) float32 tensor in [0, 1]. Works
    for a single frame too if you pass an (1, H, W, 3) array."""
    arr = frames_uint8.astype(np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))          # (n, 3, H, W)
    return torch.from_numpy(arr)


class JepaTransitionDataset(Dataset):
    """Loads every episode_*.npz in data_dir and builds all valid
    (context_frames, action_seq, target_seq) examples, with K_CONTEXT
    consecutive frames of context (returned as separate frames, not
    channel-stacked -- E is single-frame now, P does the temporal work)
    and UNROLL_STEPS consecutive (action, target_frame) pairs for the
    multi-step unrolled loss (see module docstring).

    An example starting at time t needs frames[t-k+1 .. t] as context (k
    separate frames), actions[t .. t+u-1] as the u actions fed to P one
    per unroll step, and frames[t+1 .. t+u] as the u REAL target frames
    (one per unroll step -- only P's own context window gets fed
    predictions during unrolling, never the targets) -- so t ranges over
    [k-1, T-u] within an episode of T actions / T+1 frames. u=1 reduces
    exactly to the original single-step dataset.
    """

    def __init__(self, data_dir, k_context, unroll_steps):
        self.k_context = k_context
        self.unroll_steps = unroll_steps
        self.episodes = []   # list of (frames, actions) numpy arrays
        self.index = []      # list of (episode_idx, t)

        paths = sorted(pathlib.Path(data_dir).glob("episode_*.npz"))
        if not paths:
            raise FileNotFoundError(
                f"No episode_*.npz files found in {data_dir}/ -- "
                "run collect_data.py first."
            )

        for ep_idx, path in enumerate(paths):
            data = np.load(path)
            frames = data["frames"]     # (T+1, H, W, 3) uint8
            actions = data["actions"]   # (T, action_dim) float32
            self.episodes.append((frames, actions))

            T = actions.shape[0]
            k, u = k_context, unroll_steps
            for t in range(k - 1, T - u + 1):
                self.index.append((ep_idx, t))

        print(
            f"Loaded {len(paths)} episodes, {len(self.index)} "
            f"{unroll_steps}-step-unrolled examples."
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        frames, actions = self.episodes[ep_idx]
        k, u = self.k_context, self.unroll_steps

        context = frames[t - k + 1 : t + 1]      # (k, H, W, 3)
        target_seq = frames[t + 1 : t + 1 + u]   # (u, H, W, 3) -- REAL frames
        action_seq = actions[t : t + u]          # (u, action_dim)

        context = _frames_to_tensor(context)       # (k, 3, H, W)
        target_seq = _frames_to_tensor(target_seq)  # (u, 3, H, W)
        action_seq = torch.from_numpy(action_seq.astype(np.float32))  # (u, action_dim)

        return context, action_seq, target_seq


def train():
    print(f"jepa_train.py version: {VERSION}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():   # Apple Silicon
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    E, P = make_models(device)

    dataset = JepaTransitionDataset(DATA_DIR, K_CONTEXT, UNROLL_STEPS)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
    )

    optimizer = torch.optim.Adam(
        list(E.parameters()) + list(P.parameters()), lr=LR
    )

    # Resume support. N_EPOCHS is a *total* target, not "epochs to add": if
    # jepa_checkpoint.pt exists, we load it (models + optimizer + which
    # epoch it stopped at) and continue counting from there. To force a
    # fresh run, delete jepa_checkpoint.pt (and jepa_training_log.csv,
    # unless you want old rows kept alongside new ones -- the log is
    # appended to, not overwritten).
    #
    # Caveat: only works if LATENT_DIM / K_CONTEXT / PREDICTOR_HIDDEN /
    # ACTION_DIM match what the checkpoint was saved with. KAPPA, LR,
    # N_EPOCHS, BATCH_SIZE are all safe to change between resumes.
    checkpoint_path = pathlib.Path(CHECKPOINT_PATH)
    start_epoch = 0
    if checkpoint_path.exists():
        print(f"Found {CHECKPOINT_PATH} -- resuming from it.")
        ckpt = torch.load(checkpoint_path, map_location=device)
        try:
            E.load_state_dict(ckpt["E"])
            P.load_state_dict(ckpt["P"])
        except (KeyError, RuntimeError) as e:
            print(
                f"Could not load {CHECKPOINT_PATH} -- it looks like it's "
                "from a different architecture (e.g. the old frame-stacked-"
                f"E version). Delete {CHECKPOINT_PATH} and "
                f"{LOG_PATH}, then re-run to start fresh. "
                f"(Underlying error: {e})"
            )
            raise

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print(
                "Checkpoint has no optimizer state -- model weights "
                "restored, Adam's momentum buffers start fresh instead."
            )

        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            print(f"Already trained through epoch {start_epoch}/{N_EPOCHS}.")
        else:
            start_epoch = 0

        if start_epoch >= N_EPOCHS:
            print(
                f"N_EPOCHS ({N_EPOCHS}) already reached -- raise it at the "
                "top of the script if you want to keep training."
            )
    else:
        print(f"No {CHECKPOINT_PATH} found -- starting fresh.")

    log_path = pathlib.Path(LOG_PATH)
    write_header = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(
            ["epoch", "avg_loss", "avg_inv_loss", "avg_cov_penalty", "avg_raw_z_norm"]
        )

    for epoch in range(start_epoch, N_EPOCHS):
        epoch_losses = []
        epoch_inv_losses = []
        epoch_cov_penalties = []
        epoch_z_norms = []

        for context, action_seq, target_seq in loader:
            context = context.to(device)         # (N, k, 3, H, W)
            action_seq = action_seq.to(device)   # (N, U, action_dim)
            target_seq = target_seq.to(device)   # (N, U, 3, H, W)

            n, k, c, h, w = context.shape
            u = action_seq.shape[1]

            # One E forward pass over every REAL frame in the batch --
            # context frames and all U target frames together -- then
            # split. Same E throughout; targets stay real at every unroll
            # step (only P's own sliding context window gets fed
            # predictions, never the targets -- see module docstring).
            frames_flat = torch.cat(
                [context.reshape(n * k, c, h, w), target_seq.reshape(n * u, c, h, w)],
                dim=0,
            )
            z_all = E(frames_flat)
            z_context = z_all[: n * k].reshape(n, k, -1)   # (N, k, D)
            z_targets = z_all[n * k :].reshape(n, u, -1)    # (N, U, D)
            z_last_real = z_context[:, -1, :]               # most recent REAL
                                                               # context frame --
                                                               # same role as
                                                               # z_last before

            inv_losses = []
            for i in range(u):
                z_pred = P(z_context, action_seq[:, i, :])
                inv_losses.append(
                    0.5 * (z_pred - z_targets[:, i, :]).pow(2).sum(dim=-1).mean()
                )
                # Slide the PREDICTED embedding into the context window for
                # the next unroll step -- not the real target. This is the
                # actual exposure-bias fix (see module docstring); at u=1
                # this line never runs, so u=1 is exactly the old loss.
                z_context = torch.cat([z_context[:, 1:, :], z_pred.unsqueeze(1)], dim=1)
            inv_loss = torch.stack(inv_losses).mean()  # averaged over U so
                                                          # overall loss scale
                                                          # doesn't shift when
                                                          # U is changed

            # Covariance term: z_last_real's penalty plus the U real
            # targets' penalties, averaged (not summed) over the U targets
            # -- at u=1 this is exactly cov_penalty_term(z_last) +
            # cov_penalty_term(z_target) as before; for u>1 it stays the
            # same overall scale instead of growing with U. Still applied
            # only to REAL E-encoded embeddings, never to P's predictions
            # -- VICReg's anti-collapse guarantee is about E's output
            # space, not P's.
            cov_penalty = cov_penalty_term(z_last_real) + torch.stack(
                [cov_penalty_term(z_targets[:, i, :]) for i in range(u)]
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
            f"avg_inv_loss={avg_inv_loss:.6f}  "
            f"avg_cov_penalty={avg_cov_penalty:.6f}  "
            f"avg_raw_z_norm={avg_z_norm:.3f}"
        )
        # Note the flipped reading compared to the old EMA diagnostics:
        # avg_cov_penalty going DOWN toward 0 is the goal now (Cov(z) -> I,
        # i.e. healthy, non-collapsed embeddings), not a red flag. The old
        # "loss hit exactly 0 and stayed there" collapse signature would
        # show up here as avg_cov_penalty staying persistently HIGH while
        # avg_inv_loss drops -- i.e. the model finding it "easy" to predict
        # only because it never actually satisfied the anti-collapse term.

        log_writer.writerow(
            [epoch, avg_loss, avg_inv_loss, avg_cov_penalty, avg_z_norm]
        )
        log_file.flush()

        torch.save(
            {
                "E": E.state_dict(),
                "P": P.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "k_context": K_CONTEXT,
                "latent_dim": LATENT_DIM,
            },
            CHECKPOINT_PATH,
        )

    log_file.close()
    print(f"Checkpoint saved to {CHECKPOINT_PATH}, log saved to {LOG_PATH}")


if __name__ == "__main__":
    train()