"""
Anti-collapse loss for I-JEPA, VICReg-style -- no EMA, no target network, no
stop-gradient. The same encoder E produces both the context tokens and the
target tokens, and gradients flow through both passes; collapse is blocked
directly by regularizing E's output covariance instead of by a momentum
teacher. Formulation from ball_in_cup/jepa_train.py, lifted to patch tokens.

Total loss:

    L = ( 1 - cos(pred_i, target_i) ) averaged over i        (prediction)
        + kappa/2 * sum_Z pen(Z)                              (anti-collapse)

The prediction term is on L2-NORMALISED pred/target: with the same encoder on
both sides and no stop-grad, a raw ||pred - target||^2 is minimised for free
by shrinking every embedding toward a constant, i.e. it rewards collapse.
Comparing directions (1 - cos) removes that reward; magnitude is left to the
covariance penalty below, which is evaluated on the UN-normalised embeddings.

pen(Z), for D-dim embeddings stacked as rows of Z (M, D) and C = Cov(Z):

    pen(Z) = ( sum_{j!=k} C_jk^2 ) / D             (off-diagonal: decorrelate)
             + sum_j ( sqrt(C_jj + eps) - 1 )^2    (diagonal: unit variance,
                                                    sqrt FIRST so the restoring
                                                    force doesn't vanish at
                                                    collapse)

The off-diagonal is a sum over ~D^2 terms; dividing by D (as canonical VICReg
does) keeps its gradient from swamping everything and from spiking. The
diagonal is kept exactly as ball_in_cup: sqrt(C_jj) = sigma_j undoes the
built-in squaring so the gradient stays finite as sigma_j -> 0, while
remaining smooth/symmetric (it also pulls back oversized variance).
"""

import torch
import torch.nn.functional as F


def covariance_matrix(z: torch.Tensor) -> torch.Tensor:
    """Sample covariance over the first (batch) dimension. z: (M, D) -> (D, D)."""
    m = z.shape[0]
    z = z - z.mean(dim=0, keepdim=True)
    return (z.T @ z) / (m - 1)


def vicreg_penalty(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Anti-collapse penalty for one embedding tensor. Any leading dims are
    flattened into the batch, so (B, K, D) token sets and plain (B, D)
    vectors are both fine -- for I-JEPA every (sample, patch) is a row of Z.
    """
    z = z.reshape(-1, z.shape[-1])
    d = z.shape[-1]
    cov = covariance_matrix(z)
    diag = torch.diagonal(cov)
    off_diag = cov - torch.diag(diag)

    off_diag_term = off_diag.pow(2).sum() / d
    diag_term = ((diag + eps).sqrt() - 1).pow(2).sum()
    return off_diag_term + diag_term


def ijepa_loss(pred, target, reg_embeddings, kappa: float = 1.0, eps: float = 1e-4,
               pooled_embeddings=None, kappa_pooled: float = 0.0):
    """
    pred, target:   (..., D) -- predictor output and target-encoder slices.
                    L2-normalised here, so the prediction term is 1 - cos.
                    target is NOT detached: gradients flow into E through it.
    reg_embeddings: iterable of UN-normalised embedding tensors for the
                    covariance penalty -- typically [context_tokens,
                    target_encoder_output]. Flattened to (sample*patch, D), so
                    this term only sees the PATCH-TOKEN distribution.
    kappa:          weight on the token-level anti-collapse penalty.
    pooled_embeddings: iterable of UN-normalised (B, D) tensors -- typically the
                    mean-pooled per-image embedding h.mean(1). Same VICReg
                    penalty, but now the variance term protects BETWEEN-IMAGE
                    variance (the quantity the downstream probe actually uses;
                    the token-level term above is blind to it -- it can stay
                    happy while every image's mean collapses to one point).
    kappa_pooled:   weight on that image-level penalty. 0.0 -> disabled (old
                    behaviour, token-level only).

    returns (loss, parts) with `parts` a dict of detached scalars for logging.
    """
    pn = F.normalize(pred, dim=-1)
    tn = F.normalize(target, dim=-1)
    inv = 0.5 * (pn - tn).pow(2).sum(dim=-1).mean()          # = 1 - cos(pred, target)

    pen = sum(vicreg_penalty(z, eps) for z in reg_embeddings)
    loss = inv + 0.5 * kappa * pen
    parts = {"inv": inv.detach(), "pen": pen.detach()}

    if pooled_embeddings and kappa_pooled:
        pen_pool = sum(vicreg_penalty(z, eps) for z in pooled_embeddings)
        loss = loss + 0.5 * kappa_pooled * pen_pool
        parts["pen_pool"] = pen_pool.detach()

    parts["loss"] = loss.detach()
    return loss, parts


if __name__ == "__main__":
    torch.manual_seed(0)
    D, M = 384, 8192

    z_good = torch.randn(M, D)
    z_bad = (torch.randn(M, D) * 1e-3).requires_grad_(True)
    pen_good = vicreg_penalty(z_good)
    pen_bad = vicreg_penalty(z_bad)
    pen_bad.backward()
    print(f"pen(healthy)   = {pen_good.item():.3f}")
    print(f"pen(collapsed) = {pen_bad.item():.1f}   grad norm = {z_bad.grad.norm().item():.4f}")
    assert pen_good.item() < pen_bad.item()
    assert z_bad.grad.norm().item() > 1e-3, "diagonal gradient vanished at collapse"

    # prediction term: identical dirs -> 0, opposite -> 2, orthogonal -> 1
    a = torch.randn(4, 10, D, requires_grad=True)
    for tgt, expect in [(a.detach(), 0.0), (-a.detach(), 2.0)]:
        v, _ = ijepa_loss(a, tgt, reg_embeddings=[a])
        # subtract the penalty contribution to isolate inv
        inv_only = _["inv"].item()
        print(f"inv(cos)  target={'a' if expect==0 else '-a'}  -> {inv_only:.3f}  (expect {expect})")
        assert abs(inv_only - expect) < 1e-2
    v, parts = ijepa_loss(a, torch.randn_like(a), reg_embeddings=[a], kappa=1.0)
    v.backward()
    print("loss parts:", {k: round(float(x), 3) for k, x in parts.items()})

    # pooled term: adds pen_pool and raises the loss
    pooled = a.detach().mean(dim=1).requires_grad_(True)
    v0, p0 = ijepa_loss(a.detach(), torch.randn(4, 10, D), reg_embeddings=[a.detach()],
                        kappa=1.0)
    v1, p1 = ijepa_loss(a.detach(), torch.randn(4, 10, D), reg_embeddings=[a.detach()],
                        kappa=1.0, pooled_embeddings=[pooled], kappa_pooled=1.0)
    assert "pen_pool" in p1 and "pen_pool" not in p0
    assert v1.item() > v0.item()
    v1.backward()
    assert pooled.grad is not None and pooled.grad.norm().item() > 0
    print("pooled term OK")
    print("OK")
