"""
jepa_model.py

Maze JEPA -- the model itself: E (encoder) + P (predictor), plus the
geometry-handling and anti-collapse pieces they're built from. Split out
of jepa_train.py (which now only handles data loading, the training loop,
and evaluation) because this is the part other scripts should import --
a future planner or diagnostic script needs the model, not the training
loop. See WORKPLAN.md's "Phase 2 architecture" section for the full
design reasoning.

E: a learned embedding table, one free vector per grid cell. There's no
need for a CNN/MLP encoder here the way ball-in-cup needs one for pixels
-- the maze's "observation" already IS the state (a discrete cell index).
E's output is always the raw/tangent-space vector; `to_geometry()` maps it
onto the manifold (identity for Euclidean, `hyp.exp0` for hyperbolic) only
at the point it's used for a distance computation -- see the anti-collapse
note below for why this split matters.

P: (E(start_cell), horizon) -> predicted latent, same raw/tangent
convention as E. horizon is embedded and concatenated with the start
embedding, then a small MLP.

P is deliberately NOT conditioned on the action sequence -- see
collect_data.py's docstring for why (conditioning on actions in a
deterministic maze removes all genuine ambiguity, defeating the point of
this whole test).

ANTI-COLLAPSE (added after the first real run -- see jepa_train.py's git
history / WORKPLAN.md for the story): the first version had NO
anti-collapse mechanism, and it collapsed -- E mapped every cell to
(nearly) the same point, so P trivially "predicted" it too, regardless of
which cell was actually reached. This is a well-known JEPA/BYOL/SimSiam
failure mode. Fixed with VICReg, ported unchanged from
ball_in_cup_hyperbolic/jepa_train.py (see that project's module docstring
for the full derivation of why the diagonal covariance term needs a sqrt
before penalizing). Applied to E's RAW output for the start and end cells
in each training batch -- never to P's prediction, since VICReg's
guarantee is about E's output space, not P's:

    L = inv_loss(P(E(start), H), E(end))
        + kappa/2 * (cov_penalty(E(start)) + cov_penalty(E(end)))
"""
import torch
import torch.nn as nn

import hyperbolic as hyp

LATENT_DIM = 12  # bumped from 8 (the setting that won the 8/16/32 sweep --
                 # see WORKPLAN.md's "Dimension re-sweep, completed" --
                 # dim=8: hyperbolic wins broadly; dim=16: mixed; dim=32:
                 # Euclidean wins broadly). "Wins the comparison" isn't
                 # the same as "has enough capacity in absolute terms" --
                 # P_step's training loss plateaued around 0.4-1.0 at
                 # dim=8 even after ruling out undertraining (10000
                 # steps, no better than 2000), suggesting dim=8 may not
                 # have had room to represent one-step transitions
                 # precisely, even though it was enough for Phase 2's
                 # coarser multi-step reachability target. 12 is a
                 # modest bump, still short of where the original sweep
                 # found hyperbolic's relative advantage fading (dim=16
                 # was mixed). Per the hyperbolic-only scope decision,
                 # this is being tried for hyperbolic specifically, not
                 # re-tested against Euclidean.
HIDDEN_DIM = 32

CHECKPOINT_PATH = "jepa_checkpoint_{geometry}.pt"           # written by jepa_train.py
STEP_CHECKPOINT_PATH = "jepa_step_checkpoint_{geometry}.pt"  # written by jepa_train_step.py


def cell_index(cell, size):
    """(row, col) -> flat index into E's embedding table. The canonical
    indexing scheme for this maze -- any script working with E needs
    this, not just training."""
    r, c = cell
    return r * size + c


def to_geometry(v, geometry):
    """Maps a raw/tangent-space vector onto the manifold: identity for
    Euclidean, hyp.exp0 for hyperbolic. Kept as an explicit step (rather
    than folded into Encoder/Predictor) so the anti-collapse penalty can
    be applied to the pre-mapped tangent vectors, matching
    ball_in_cup_hyperbolic/jepa_train_joint.py's convention."""
    return hyp.exp0(v) if geometry == "hyperbolic" else v


def squared_dist(a, b, geometry):
    """a, b already mapped onto the geometry (post to_geometry())."""
    if geometry == "hyperbolic":
        return hyp.dist(a, b) ** 2
    return ((a - b) ** 2).sum(dim=-1)


def covariance_matrix(z):
    """Cov(z) over the batch dimension -- z is (N, D), returns (D, D).
    Identical to ball_in_cup_hyperbolic/jepa_train.py's version."""
    n = z.shape[0]
    z_centered = z - z.mean(dim=0, keepdim=True)
    return (z_centered.T @ z_centered) / (n - 1)


def cov_penalty_term(z, eps=1e-8):
    """VICReg anti-collapse penalty for one branch, identical formula to
    ball_in_cup_hyperbolic/jepa_train.py (see that project's module
    docstring for the full derivation of why the diagonal needs a sqrt
    before penalizing). z is (N, D), the RAW/tangent-space embedding."""
    cov = covariance_matrix(z)
    diag = torch.diagonal(cov)
    off_diag = cov - torch.diag(diag)

    off_diag_term = off_diag.pow(2).sum()
    diag_term = ((diag + eps).sqrt() - 1).pow(2).sum()

    return off_diag_term + diag_term


class Encoder(nn.Module):
    """One free vector per grid cell. Always returns the raw/tangent
    vector -- see to_geometry()."""

    def __init__(self, num_cells, dim=LATENT_DIM):
        super().__init__()
        self.table = nn.Embedding(num_cells, dim)
        nn.init.normal_(self.table.weight, std=0.5)

    def forward(self, idx):
        return self.table(idx)


class Predictor(nn.Module):
    """(start latent, horizon) -> predicted end latent, raw/tangent
    vector -- see to_geometry()."""

    def __init__(self, max_horizon, dim=LATENT_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.horizon_embed = nn.Embedding(max_horizon + 1, hidden)
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, start_raw, horizon_idx):
        h = self.horizon_embed(horizon_idx)
        return self.net(torch.cat([start_raw, h], dim=-1))


class OneStepPredictor(nn.Module):
    """(E(cell), action) -> predicted E(next cell), raw/tangent vector.
    Distinct from Predictor above: Predictor answers "where do H random
    steps typically end up" (no action input, deliberately, for the
    Phase 0/2 geometry comparison -- see collect_data.py's docstring).
    OneStepPredictor answers the opposite question -- "given this ONE
    action, where do I actually go" -- which is what a planner needs.
    The maze's transitions are deterministic given (cell, action) (a wall
    bump is still a deterministic no-op), so there's no multimodality
    here and no VICReg term is needed: this is plain supervised
    regression against a single correct target, trained with E FROZEN
    (loaded from a jepa_train.py checkpoint) so it's testing whether the
    already-learned representation supports one-step prediction, not
    learning a new representation from scratch. See jepa_train_step.py
    and maze_planner.py."""

    def __init__(self, num_actions, dim=LATENT_DIM, hidden=HIDDEN_DIM, residual=False):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, hidden)
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        self.residual = residual  # False preserves the original behavior exactly
                                    # (predict the whole next raw vector from
                                    # scratch). True instead predicts start_raw
                                    # PLUS a small learned delta -- real edges are
                                    # short (mean ~0.34 in the current treeloss E)
                                    # relative to a typical embedding's own norm
                                    # (mean ~1.5, see treeloss_norm_check.py), so
                                    # asking the network for a small delta instead
                                    # of a whole new absolute position may be a
                                    # substantially easier regression target --
                                    # same underlying prediction, different
                                    # parameterization. See pstep_onestep_error_check.py
                                    # (P_step's error was a stable ~2.3-2.45x the true
                                    # edge length across LR/hidden-width/dagger
                                    # ablations, none of which moved it).

    def forward(self, start_raw, action_idx):
        a = self.action_embed(action_idx)
        out = self.net(torch.cat([start_raw, a], dim=-1))
        return start_raw + out if self.residual else out


def load_encoder_checkpoint(geometry, path=CHECKPOINT_PATH):
    """Loads a frozen, eval-mode Encoder from a jepa_train.py checkpoint.
    Returns (E, checkpoint_dict) -- the dict is returned too since
    callers often need size/max_horizon alongside the model. Centralized
    here rather than duplicated in jepa_train_step.py, maze_planner.py,
    and any future diagnostic -- checkpoint loading is a model concern,
    not a training-script concern, same reasoning as this file's split
    from jepa_train.py in the first place."""
    ckpt = torch.load(path.format(geometry=geometry), map_location="cpu")
    E = Encoder(ckpt["num_cells"], dim=ckpt["dim"])
    E.load_state_dict(ckpt["E"])
    for param in E.parameters():
        param.requires_grad = False
    E.eval()
    return E, ckpt


def load_step_checkpoint(geometry, path=STEP_CHECKPOINT_PATH):
    """Loads a frozen, eval-mode OneStepPredictor from a
    jepa_train_step.py checkpoint. Returns (P_step, checkpoint_dict)."""
    ckpt = torch.load(path.format(geometry=geometry), map_location="cpu")
    P_step = OneStepPredictor(num_actions=ckpt["num_actions"], dim=ckpt["dim"],
                               hidden=ckpt.get("hidden", HIDDEN_DIM),  # old checkpoints
                               # (pre this field) were all trained at the default HIDDEN_DIM
                               residual=ckpt.get("residual", False))  # old checkpoints
                               # (pre this field) were all trained non-residual
    P_step.load_state_dict(ckpt["P_step"])
    for param in P_step.parameters():
        param.requires_grad = False
    P_step.eval()
    return P_step, ckpt
