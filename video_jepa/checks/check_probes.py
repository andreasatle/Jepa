"""Checks for probes.py (step 2 of PLAN.md): the probes work when the answer is in the features.

    uv run python checks/check_probes.py

Ground-truth "tokens": one token per frame of the clip, holding that frame's true state (pusher x, y; T x, y;
T angle as sin, cos; standardised) plus a sin-cos code of the frame's time. Velocity is not in any token,
only in the difference between the last two, which is exactly what a video encoder has to provide.

- Attentive probe on these tokens: position and velocity should be near perfect (it can attend to the last
  two frames with different heads and subtract); contact, a nonlinear function of the geometry, high.
- Linear probe on the last frame's state alone: position near perfect; velocity near zero (one frame has none).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data import PushTVideo
from model import sincos_1d
from probes import attentive_probe, device, linear_probe, print_rows, probe_sets, summarise


def state_tokens(video: PushTVideo, ends: np.ndarray, mu: np.ndarray, sd: np.ndarray, dim: int = 192) -> torch.Tensor:
    rows = np.stack([PushTVideo.clip_rows(int(e)) for e in ends])            # (n, 16)
    s = video.state[rows]                                                     # (n, 16, 7)
    feat = np.concatenate([s[..., 0:4], np.sin(s[..., 4:5]), np.cos(s[..., 4:5])], -1)
    feat = (feat - mu) / sd
    time_code = sincos_1d(dim - 6, torch.arange(rows.shape[1])).numpy()     # (16, dim-6)
    tok = np.concatenate([feat, np.broadcast_to(time_code, feat.shape[:2] + time_code.shape[1:])], -1)
    return torch.from_numpy(tok.astype(np.float32)).half()


def main():
    video = PushTVideo()
    tr, ev = probe_sets(video)
    s = video.state[tr.ends]
    f = np.concatenate([s[:, 0:4], np.sin(s[:, 4:5]), np.cos(s[:, 4:5])], -1)
    mu, sd = f.mean(0), f.std(0)
    ttr, tev = state_tokens(video, tr.ends, mu, sd), state_tokens(video, ev.ends, mu, sd)
    ytr, yev = torch.from_numpy(tr.targets), torch.from_numpy(ev.targets)

    rows = [summarise("state, last frame, linear", linear_probe(ttr[:, -1, :6].float(), ytr, tev[:, -1, :6].float()), yev),
            summarise("state tokens, attentive", attentive_probe(ttr, ytr, tev, device()), yev)]
    print_rows(rows)
    lin, att = rows
    assert lin["position"] > .97, "linear probe misses position given the state itself"
    assert lin["velocity"] < .2, "one frame should not give velocity"
    assert att["position"] > .95 and att["velocity"] > .9, "attentive probe fails on ground-truth tokens"
    assert att["contact_acc"] > .85, "attentive probe misses contact given the geometry"
    print("OK")


if __name__ == "__main__":
    main()
