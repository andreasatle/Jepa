"""Checks for the EMA target encoder (train.ema_update): the update rule, and that the target never gets gradients.

    uv run python checks/check_ema.py
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import Encoder
from train import ema_update


def params(m):
    return [p.detach().clone() for p in m.parameters()]


def main():
    torch.manual_seed(0)
    online = Encoder(32, 8, 3, embed_dim=16, depth=1, num_heads=2)
    target = copy.deepcopy(online).requires_grad_(False)
    with torch.no_grad():
        for p in online.parameters():
            p.add_(torch.randn_like(p))                     # make online != target
    t0, o0 = params(target), params(online)

    # m = 1: target unchanged
    ema_update(target, online, 1.0)
    assert all(torch.equal(a, b) for a, b in zip(params(target), t0))
    # m = 0.5: exact average
    ema_update(target, online, 0.5)
    assert all(torch.allclose(a, (t + o) / 2) for a, t, o in zip(params(target), t0, o0))
    # m = 0: exact copy of online
    ema_update(target, online, 0.0)
    assert all(torch.equal(a, o) for a, o in zip(params(target), o0))
    # online is never modified by the update
    assert all(torch.equal(a, o) for a, o in zip(params(online), o0))

    # target gets no gradients from a loss through it; online does
    x = torch.randn(2, 3, 32, 32)
    (target(x).sum() + online(x).sum()).backward()
    assert all(p.grad is None for p in target.parameters())
    assert all(p.grad is not None for p in online.parameters())
    print("OK")


if __name__ == "__main__":
    main()
