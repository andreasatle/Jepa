"""
I-JEPA loss, written as the three VICReg terms (Bardes et al., 2022).

VICReg = Variance + Invariance + Covariance. In I-JEPA the three terms mean:

    invariance(pred, target)  the predictor's guess for each hidden patch should
                              equal the target embedding of that patch. This is
                              the actual I-JEPA objective. As in the official
                              code: the target is LayerNormed over its feature
                              dim (no affine), then compared with smooth-L1.
                              (The paper's text says L2; the code uses
                              smooth-L1, which is L2 for small errors and L1
                              for large ones.)

    variance(Z)               every embedding dimension should keep a spread
                              (std ~ 1) across the rows of Z. Blocks the trivial
                              solution "map everything to one constant vector",
                              which makes invariance exactly 0.

    covariance(Z)             embedding dimensions should be decorrelated. Blocks
                              the subtler collapse "all dimensions carry the same
                              signal", where variance is fine but the effective
                              rank is 1.

Total loss:

    L = invariance(pred, target) + var_weight * sum_Z variance(Z)
                                 + cov_weight * sum_Z covariance(Z)

With var_weight = cov_weight = 0 this is the official I-JEPA loss, where an EMA
target encoder is the only thing preventing collapse. This project replaces the
EMA with the variance and covariance terms, applied to ONLINE
(gradient-carrying) embeddings, never to the (stop-gradient) target.

What to pass as Z. Each tensor has shape (..., D); all leading dims are flattened
into rows, so a row is one "sample" of the embedding:
  - token-level:  (B, K, D) context tokens -> every (image, patch) is a row.
                  Keeps patches different from each other -- but is blind to
                  all images collapsing to the same *mean* embedding.
  - image-level:  (B, D) mean-pooled tokens -> every image is a row. Protects
                  between-image spread, which is what a downstream probe uses.
                  (Old v2 lesson: token-level alone let the pooled embedding
                  collapse.)

Formulas, for Z with M rows and D columns and C = Cov(Z) (D x D):

    variance(Z)   = sum_j ( sqrt(C_jj + eps) - 1 )^2
    covariance(Z) = ( sum_{j != k} C_jk^2 ) / D

Two deliberate choices, both inherited from the old code:
  - variance is symmetric, pulling std toward 1 from both sides. Canonical
    VICReg uses a one-sided hinge, mean_j relu(1 - std_j), which only pushes up.
    Taking sqrt first matters: the gradient of (C_jj - 1)^2 w.r.t. z shrinks to
    0 as the embedding collapses, while that of (std_j - 1)^2 does not.
  - covariance is divided by D (as in canonical VICReg): it is a sum over ~D^2
    entries and would otherwise swamp the other terms.
"""

import torch
import torch.nn.functional as F


def covariance_matrix(z: torch.Tensor) -> torch.Tensor:
    """Sample covariance over rows. z: (M, D) -> (D, D)."""
    z = z - z.mean(dim=0, keepdim=True)
    return (z.T @ z) / (z.shape[0] - 1)


def invariance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """smooth-L1 between the prediction and the LayerNormed target, both (..., D)."""
    target = F.layer_norm(target, (target.shape[-1],))    # per token: mean 0, std 1 over features
    return F.smooth_l1_loss(pred, target)


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Pull the std of every embedding dimension toward 1. z: (..., D)."""
    z = z.reshape(-1, z.shape[-1])
    var = z.var(dim=0)                      # = diag(Cov(z)), without the D x D matrix
    return ((var + eps).sqrt() - 1).pow(2).sum()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Push off-diagonal covariances toward 0. z: (..., D)."""
    z = z.reshape(-1, z.shape[-1])
    cov = covariance_matrix(z)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / z.shape[-1]


def sigreg_loss(z: torch.Tensor, num_slices: int = 1024, num_points: int = 17,
                t_max: float = 5.0) -> torch.Tensor:
    """
    SIGReg (LeJEPA, Balestriero & LeCun 2025, arXiv 2511.08544): push the embedding
    distribution toward an isotropic Gaussian N(0, I), the one the paper proves
    minimises downstream prediction risk. Collapse is then a bad *score*, not a
    path the training dynamics happen to avoid, so no EMA or stop-gradient is needed.

    z: (M, D), one row per sample (here: one mean-pooled embedding per image).

    Sliced: project z onto `num_slices` random unit directions a (resampled every
    call), and test each 1-D projection x = z a against N(0, 1) with the
    Epps-Pulley statistic, which compares characteristic functions:

        EP = M * integral |phi_hat(t) - exp(-t^2/2)|^2 w(t) dt,   w(t) = exp(-t^2)
        phi_hat(t) = mean_j exp(i t x_j) = mean cos(t x) + i mean sin(t x)

    The integrand is even in t, so it is integrated over [0, t_max] with the
    trapezoid rule and doubled (the paper: 17 points over [-5, 5]). Averaged over
    directions, as in the paper. A Gaussian match on every 1-D projection pins
    down the whole distribution (Cramer-Wold), unlike VICReg, which only
    constrains each dimension's spread and the pairwise correlations.
    """
    M, D = z.shape
    a = torch.randn(D, num_slices, device=z.device)
    a = a / a.norm(dim=0, keepdim=True)
    x = z.float() @ a                                          # (M, S)
    t = torch.linspace(0, t_max, num_points, device=z.device)  # (T,)
    tx = x.unsqueeze(-1) * t                                   # (M, S, T)
    re = torch.cos(tx).mean(0) - torch.exp(-t ** 2 / 2)        # (S, T)
    im = torch.sin(tx).mean(0)
    integrand = (re ** 2 + im ** 2) * torch.exp(-t ** 2)
    ep = 2 * torch.trapezoid(integrand, t, dim=-1)             # (S,)
    return M * ep.mean()


def center_per_position(h: torch.Tensor) -> torch.Tensor:
    """
    h: (B, N, D) full-image tokens. Subtract, at every position, the mean over the
    B images. What is left is only how images differ at the same position, so an
    embedding that depends on position alone becomes 0 and fails the variance term.
    (Plain token-level statistics count position-to-position differences as healthy
    variance -- the loophole pure VICReg exploited in run `target-grad`.)
    """
    return h - h.mean(dim=0, keepdim=True)


def vicreg_loss(pred, target, regularized=(), var_weight: float = 0.0,
                cov_weight: float = 0.0):
    """
    pred, target: (..., D) predictor output and target embeddings of the hidden
                  patches. `target` is the encoder's own full-image output
                  (gradient-stopped unless train.py's target_grad is on).
    regularized:  iterable of ONLINE embedding tensors (..., D) that the
                  variance / covariance terms act on (see module docstring).
    var_weight, cov_weight: 0 switches the term off (no compute).

    Returns (loss, parts): parts is a dict of detached scalars for logging.
    """
    inv = invariance_loss(pred, target)
    loss = inv
    parts = {"inv": inv.detach()}

    if var_weight:
        var = sum(variance_loss(z) for z in regularized)
        loss = loss + var_weight * var
        parts["var"] = var.detach()

    if cov_weight:
        cov = sum(covariance_loss(z) for z in regularized)
        loss = loss + cov_weight * cov
        parts["cov"] = cov.detach()

    parts["loss"] = loss.detach()
    return loss, parts
