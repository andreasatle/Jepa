"""Checks for loss.py: each penalty reacts to the collapse it targets, and gradients flow.

    uv run python checks/check_loss.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from loss import center_per_position, covariance_loss, invariance_loss, variance_loss, vicreg_loss


def main():
    torch.manual_seed(0)
    D, M = 384, 8192

    # Healthy embedding (unit variance, uncorrelated) -> both penalties near 0.
    z_good = torch.randn(M, D)
    print(f"healthy:   var={variance_loss(z_good):.3f}  cov={covariance_loss(z_good):.3f}")

    # Collapsed embedding -> large variance penalty, and the gradient must not vanish.
    z_bad = (torch.randn(M, D) * 1e-3).requires_grad_(True)
    v_bad = variance_loss(z_bad)
    v_bad.backward()
    print(f"collapsed: var={v_bad:.1f}  grad norm={z_bad.grad.norm():.4f}")
    assert v_bad > variance_loss(z_good)
    assert z_bad.grad.norm() > 1e-3, "variance gradient vanished at collapse"

    # Rank-1 embedding (every dim is the same signal) -> variance fine, covariance large.
    z_rank1 = torch.randn(M, 1).expand(M, D)
    print(f"rank-1:    var={variance_loss(z_rank1):.3f}  cov={covariance_loss(z_rank1):.1f}")
    assert covariance_loss(z_rank1) > 100 * covariance_loss(z_good)

    # Invariance: 0 when the prediction equals the normalised target, whatever the target's scale.
    t = torch.randn(4, 10, D)
    assert invariance_loss(F.layer_norm(t, (D,)), 5 * t + 3) < 1e-8
    assert invariance_loss(torch.zeros_like(t), t) > 0.1

    # Weights of 0 skip the terms; nonzero weights add them and carry gradient.
    ctx = torch.randn(4, 10, D, requires_grad=True)
    pred = torch.randn(4, 10, D, requires_grad=True)
    _, p0 = vicreg_loss(pred, t, [ctx])
    assert p0.keys() == {"inv", "loss"}
    loss, p1 = vicreg_loss(pred, t, [ctx, ctx.mean(1)], var_weight=1.0, cov_weight=1.0)
    loss.backward()
    assert p1.keys() == {"inv", "var", "cov", "loss"}
    assert ctx.grad.norm() > 0 and pred.grad.norm() > 0
    print("parts:", {k: round(float(x), 3) for k, x in p1.items()})
    # Position loophole: embeddings that depend on position only (same for every image)
    # look healthy to token-level VICReg but must fail after per-position centring.
    B, N = 64, 196
    pos_only = torch.randn(1, N, D).expand(B, N, D)                   # no image content at all
    content = torch.randn(B, N, D)                                    # differs between images
    print(f"position-only: token-level var={variance_loss(pos_only):.2f}  "
          f"per-position var={variance_loss(center_per_position(pos_only)):.1f}")
    assert variance_loss(pos_only) < 1.0, "token-level stats should be fooled (that is the loophole)"
    assert variance_loss(center_per_position(pos_only)) > 0.9 * D, "per-position should see collapse"
    assert variance_loss(center_per_position(content)) < 1.0, "real content should still pass"
    print("OK")


if __name__ == "__main__":
    main()
