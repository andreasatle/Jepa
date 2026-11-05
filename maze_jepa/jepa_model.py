"""
jepa_model.py

Maze JEPA -- the model itself: E (encoder) + Predictor, plus the
geometry-handling and anti-collapse pieces they're built from.

E: one vector per grid cell, split into two parts -- a FIXED positional
part (the literal (row, col) grid coordinates, centered and scaled, not
learned) and a free/learned part (nn.Embedding, the rest of LATENT_DIM).
Why: a pure free-embedding table gives the Predictor nothing to lean on
except memorizing every (cell, action) -> next_cell pair from scratch,
and the cheapest thing gradient descent finds is "predict roughly no
change" (see WORKPLAN.md's identity-collapse finding: 89% accuracy on
wall-bump/no-op pairs, 2% on real moves). Baking literal grid position
into E means "move by the given delta" is close to an exact vector
addition on the 2 positional dims (since action_delta -- see Predictor
below -- is itself an exact integer grid shift), leaving the network's
free capacity to focus on the actually-hard part: the wall-bump
exceptions, where the real answer deviates from that default. E's
output is always the raw/tangent-space vector; `to_geometry()` maps it
onto the manifold (identity for Euclidean, `hyp.exp0` for hyperbolic)
only at the point it's used for a distance computation -- see the
anti-collapse note below for why this split matters.

Predictor: (E(cell), action_delta) -> predicted E(next cell), same
raw/tangent convention as E. action_delta is the actual (row, col)
movement direction (Action.value in env.py -- e.g. UP=(-1,0)), fed in
directly and concatenated with the start embedding, rather than routed
through a per-action lookup table. A lookup table (nn.Embedding over 4
action indices) would give the network 4 totally unrelated learned
vectors with no structure between them; the raw delta instead lets the
network see, for free, that UP and DOWN are literally opposite vectors
(negatives of each other), AND lets it add directly onto E's positional
dims. Note E's positional encoding and the Predictor's action_delta
input are complementary, not redundant -- one says where you are, the
other says which way you're moving; the vector-addition trick needs
both operands, so this doesn't replace passing the action in.

ANTI-COLLAPSE (see WORKPLAN.md / the old maze/ project's history for the
full story): a self-predictive objective with nothing tying E to real
distances has a well-known degenerate solution (collapse all cells toward
one point, "predict no change" becomes trivially accurate) -- a
well-known JEPA/BYOL/SimSiam failure mode. `cov_penalty_term`/
`covariance_matrix` (VICReg) guard against that. The off-diagonal term is
normalized by embedding dimension D, matching the VICReg paper's own
formula -- without that division, the raw sum over the D*(D-1)
off-diagonal covariance entries grows roughly quadratically with D, while
the prediction loss (a per-dimension squared distance) only grows
linearly with D, so kappa's effective anti-collapse strength would
otherwise silently depend on LATENT_DIM rather than being a stable,
comparable knob across dimension choices. (Applies to E's full raw
output, positional dims included -- those are fixed/non-trainable, so
the penalty can't change them, but it can still push the free dims to
decorrelate from them.)
"""
import torch
import torch.nn as nn

import hyperbolic as hyp

LATENT_DIM = 12
HIDDEN_DIM = 128  # bumped 32->128 to test whether junction-cell failures (degree
                  # 3-4, multiple simultaneous real targets from one start point) are
                  # a capacity bottleneck in the shared Predictor MLP -- isolated as
                  # its own variable rather than changed together with anything else
POS_DIMS = 2  # (row, col) -- see Encoder's docstring

CHECKPOINT_PATH = "jepa_checkpoint_{geometry}.pt"           # written by the E-training script
STEP_CHECKPOINT_PATH = "jepa_step_checkpoint_{geometry}.pt"  # written by the Predictor-training script


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
    """Cov(z) over the batch dimension -- z is (N, D), returns (D, D)."""
    n = z.shape[0]
    z_centered = z - z.mean(dim=0, keepdim=True)
    return (z_centered.T @ z_centered) / (n - 1)


def cov_penalty_term(z, eps=1e-8):
    """VICReg anti-collapse penalty for one branch. z is (N, D), the
    RAW/tangent-space embedding. The off-diagonal term is divided by D
    (embedding dimension) -- matching the VICReg paper's own formula --
    so its scale doesn't grow quadratically with D the way an
    unnormalized sum over the D*(D-1) off-diagonal entries would; see
    the module docstring's ANTI-COLLAPSE note for why that matters for
    kappa's effective weight."""
    cov = covariance_matrix(z)
    dim = z.shape[-1]
    diag = torch.diagonal(cov)
    off_diag = cov - torch.diag(diag)

    off_diag_term = off_diag.pow(2).sum() / dim
    diag_term = ((diag + eps).sqrt() - 1).pow(2).sum()

    return off_diag_term + diag_term


class Encoder(nn.Module):
    """One vector per grid cell = [learned free part] ++ [fixed (row,
    col) position, centered and scaled]. See module docstring for why
    the positional part is fixed rather than learned: it's meant to be
    an exact, stable target for the Predictor's vector-addition
    shortcut, not something that drifts during training."""

    def __init__(self, num_cells, size, dim=LATENT_DIM):
        super().__init__()
        assert dim > POS_DIMS, f"dim={dim} must be > POS_DIMS={POS_DIMS}"
        self.size = size
        free_dim = dim - POS_DIMS
        self.table = nn.Embedding(num_cells, free_dim)
        nn.init.normal_(self.table.weight, std=0.5)

        idx = torch.arange(num_cells)
        rows = (idx // size).float()
        cols = (idx % size).float()
        center = (size - 1) / 2
        # Centered (so the grid's middle sits near the origin, matching
        # the free dims' zero-mean init) and scaled by size (so a
        # one-cell step is a small, consistent fraction of the vector's
        # own range, comparable in magnitude to the free dims' std=0.5
        # init, rather than raw integers 0..size-1 dominating it).
        pos = torch.stack([rows - center, cols - center], dim=-1) / size
        self.register_buffer("pos", pos)  # (num_cells, POS_DIMS) -- NOT a parameter

    def forward(self, idx):
        return torch.cat([self.table(idx), self.pos[idx]], dim=-1)


class Predictor(nn.Module):
    """(E(cell), action_delta) -> predicted E(next cell), raw/tangent
    vector. action_delta is a raw (row, col) direction vector (shape
    (..., 2)), NOT a categorical action index -- see the module
    docstring for why. No multimodality here (transitions are
    deterministic given (cell, action)), so this is plain supervised
    regression against a single correct target."""

    def __init__(self, dim=LATENT_DIM, hidden=HIDDEN_DIM, residual=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        self.residual = residual  # False: predict the whole next raw vector from
                                    # scratch. True: predict start_raw PLUS a small
                                    # learned delta instead -- real edges are short
                                    # relative to a typical embedding's own norm, so
                                    # this may be an easier regression target, same
                                    # underlying prediction, different
                                    # parameterization. The old maze/ project found
                                    # residual=True broke on size-10 mazes (cells
                                    # whose raw embedding norm is smaller than a
                                    # typical edge length); residual=False is the
                                    # config that worked there.

    def forward(self, start_raw, action_delta):
        out = self.net(torch.cat([start_raw, action_delta], dim=-1))
        return start_raw + out if self.residual else out


def load_encoder_checkpoint(geometry, path=CHECKPOINT_PATH):
    """Loads a frozen, eval-mode Encoder from an E-training checkpoint.
    Returns (E, checkpoint_dict) -- the dict is returned too since
    callers often need size/max_horizon alongside the model."""
    ckpt = torch.load(path.format(geometry=geometry), map_location="cpu")
    E = Encoder(ckpt["num_cells"], ckpt["size"], dim=ckpt["dim"])
    E.load_state_dict(ckpt["E"])
    for param in E.parameters():
        param.requires_grad = False
    E.eval()
    return E, ckpt


def load_step_checkpoint(geometry, path=STEP_CHECKPOINT_PATH):
    """Loads a frozen, eval-mode Predictor from a Predictor-training
    checkpoint. Returns (Predictor, checkpoint_dict)."""
    ckpt = torch.load(path.format(geometry=geometry), map_location="cpu")
    P = Predictor(dim=ckpt["dim"],
                  hidden=ckpt.get("hidden", HIDDEN_DIM),
                  residual=ckpt.get("residual", False))
    P.load_state_dict(ckpt["P_step"])
    for param in P.parameters():
        param.requires_grad = False
    P.eval()
    return P, ckpt
