"""
Anti-collapse loss for I-JEPA -- single encoder, no EMA / momentum teacher.
Collapse is blocked by a STOP-GRAD on the prediction target plus VICReg on the
online embeddings. Covariance formulation from ball_in_cup/jepa_train.py.

Total loss (v3):

    L = smooth_l1( pred_i, LN(target_i) ) averaged over i     (prediction)
        + kappa/2 * sum_Z pen(Z)                              (token-level VICReg)
        + kappa_pooled/2 * sum_Z pen(Z_pooled)               (image-level VICReg)

The target is `h.detach()` -- the same encoder, stop-gradded (SimSiam's
essential ingredient). Because the target cannot be pulled back toward the
prediction, matching magnitudes directly (`smooth_l1`) no longer rewards
shrinking both sides to a constant, so no L2-normalisation is needed; the
target is LayerNormed over its feature dim (no affine, standard I-JEPA) just to
keep its scale stable. `predictor_loss="cos"` + a non-detached target is still
supported for the pre-v3 recipe. VICReg is evaluated on the UN-normalised
ONLINE embeddings (gradients flow), never on the detached target.

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


def ijepa_loss(pred, target, reg_embeddings=None, kappa: float = 1.0, eps: float = 1e-4,
               pooled_embeddings=None, kappa_pooled: float = 0.0,
               predictor_loss: str = "cos", target_layernorm: bool = False):
    """
    pred, target:   (..., D) -- predictor output and target slices.
    predictor_loss: "cos"       -> 0.5 * ||norm(pred) - norm(target)||^2 (= 1 - cos).
                                   For a NON-detached target: comparing directions
                                   removes the reward for shrinking both sides.
                    "smooth_l1" -> F.smooth_l1_loss(pred, target). For a
                                   STOP-GRADDED target (target = h.detach()):
                                   nothing to gain from shrinking, match
                                   magnitudes directly.
    target_layernorm: LayerNorm `target` over its feature dim (no affine) before
                    the loss -- standard I-JEPA target normalisation, keeps the
                    smooth_l1 target scale stable.
    reg_embeddings: iterable of UN-normalised ONLINE embedding tensors for the
                    token-level covariance penalty -- typically
                    [context_tokens, full_image_tokens]. Flattened to
                    (sample*patch, D). None / kappa 0 -> skipped.
    kappa:          weight on the token-level penalty.
    pooled_embeddings: iterable of UN-normalised (B, D) tensors -- typically the
                    mean-pooled per-image embedding h.mean(1). Same VICReg
                    penalty, but the variance term now protects BETWEEN-IMAGE
                    variance (what the downstream probe uses; the token-level
                    term is blind to it -- it stays happy while every image's
                    mean collapses to one point).
    kappa_pooled:   weight on that image-level penalty. 0.0 -> skipped.

    returns (loss, parts) with `parts` a dict of detached scalars for logging.
    """
    if target_layernorm:
        target = F.layer_norm(target, (target.shape[-1],))

    if predictor_loss == "cos":
        pn = F.normalize(pred, dim=-1)
        tn = F.normalize(target, dim=-1)
        inv = 0.5 * (pn - tn).pow(2).sum(dim=-1).mean()      # = 1 - cos(pred, target)
    elif predictor_loss == "smooth_l1":
        inv = F.smooth_l1_loss(pred, target)
    else:
        raise ValueError(f"predictor_loss must be 'cos' or 'smooth_l1', got {predictor_loss!r}")

    loss = inv
    parts = {"inv": inv.detach()}

    if reg_embeddings and kappa:
        pen = sum(vicreg_penalty(z, eps) for z in reg_embeddings)
        loss = loss + 0.5 * kappa * pen
        parts["pen"] = pen.detach()

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

    # v3 path: smooth_l1 + LayerNormed stop-grad target, VICReg on online embeds
    ctx = torch.randn(4, 10, D, requires_grad=True)
    pred = torch.randn(4, 10, D, requires_grad=True)
    tgt = ctx.detach() * 5.0                                  # different scale on purpose
    v, parts = ijepa_loss(pred, tgt, reg_embeddings=[ctx],
                          pooled_embeddings=[ctx.mean(1)], kappa=1.0, kappa_pooled=1.0,
                          predictor_loss="smooth_l1", target_layernorm=True)
    v.backward()
    assert {"inv", "pen", "pen_pool", "loss"} <= parts.keys()
    assert ctx.grad.norm().item() > 0 and pred.grad.norm().item() > 0
    v_match, _ = ijepa_loss(tgt / tgt.std(), tgt, predictor_loss="smooth_l1",
                            target_layernorm=True)
    v_off, _ = ijepa_loss(torch.zeros_like(tgt), tgt, predictor_loss="smooth_l1",
                          target_layernorm=True)
    assert v_match.item() < v_off.item()
    try:
        ijepa_loss(pred, tgt, predictor_loss="l2")
        raise AssertionError("bad predictor_loss should raise")
    except ValueError:
        pass
    print("smooth_l1 / stop-grad path OK")
    print("OK")
