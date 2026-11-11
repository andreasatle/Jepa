"""
Step 3: the image baseline. The Push-T world model's image encoder (../pusht/runs/sigreg-s0, SIGReg at the
paper's weight, which solved 39 of 50 puzzles) embeds each frame of a clip on its own; the same probes then
read the clip's last frame from those per-frame embeddings. This is the honest image-based answer for velocity:
nothing in it saw two frames at once, so velocity can only come from comparing frames.

  linear (last 3)   ridge on the last three frames' embeddings side by side (0.4 s of motion)
  linear (last 1)   ridge on the last frame's embedding alone: velocity should be near the floor
  attentive         the attentive probe over all 16 frame embeddings, each plus a sin-cos code of its time

The encoder is imported from ../pusht (model.py), the one place this project reads another's code: it is the
baseline itself, unchanged. Features are its output after the projector (128-d), the embedding its planner
used, in eval mode (BatchNorm on running statistics).

    uv run python baseline_image.py
"""

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PUSHT = os.path.join(HERE, "..", "pusht")
sys.path.insert(0, PUSHT)
from model import Encoder as PushTEncoder  # noqa: E402  (../pusht/model.py)

sys.path.remove(PUSHT)
sys.modules.pop("model")
from data import FRAMES, PushTVideo  # noqa: E402
from model import sincos_1d  # noqa: E402  (this project's model.py)
from probes import attentive_probe, device, linear_probe, print_rows, probe_sets, summarise  # noqa: E402

CHECKPOINT = os.path.join(PUSHT, "runs", "sigreg-s0", "model.pt")


@torch.no_grad()
def frame_embeddings(enc: PushTEncoder, video: PushTVideo, ends: np.ndarray, dev, batch: int = 4096) -> torch.Tensor:
    """(n, FRAMES, 128): the encoder applied to every frame of every clip."""
    rows = np.stack([PushTVideo.clip_rows(int(e)) for e in ends]).reshape(-1)
    out = []
    for i in range(0, len(rows), batch):
        x = torch.from_numpy(np.array(video.frames[np.sort(rows[i:i + batch])]))
        order = np.argsort(np.argsort(rows[i:i + batch]))                       # undo the sort
        x = x[order].to(dev).permute(0, 3, 1, 2).float().div(255)              # as pusht's train.to_input
        out.append(enc(x).cpu())
    return torch.cat(out).reshape(len(ends), FRAMES, -1)


def main():
    dev = device()
    enc = PushTEncoder()
    enc.load_state_dict(torch.load(CHECKPOINT, map_location="cpu")["E"])
    enc.eval().to(dev)
    video = PushTVideo()
    tr, ev = probe_sets(video)
    t0 = time.time()
    ftr, fev = frame_embeddings(enc, video, tr.ends, dev), frame_embeddings(enc, video, ev.ends, dev)
    print(f"embedded {len(tr)} + {len(ev)} clips x {FRAMES} frames in {time.time() - t0:.0f} s, dim {ftr.shape[-1]}")
    ytr, yev = torch.from_numpy(tr.targets), torch.from_numpy(ev.targets)

    rows = [summarise("image, linear (last 1)", linear_probe(ftr[:, -1], ytr, fev[:, -1]), yev),
            summarise("image, linear (last 3)", linear_probe(ftr[:, -3:].flatten(1), ytr, fev[:, -3:].flatten(1)), yev)]
    code = sincos_1d(ftr.shape[-1], torch.arange(FRAMES))[None]                # time of each frame
    scale = ftr.std()
    rows.append(summarise("image, attentive (16 frames)",
                          attentive_probe((ftr / scale + code).half(), ytr, (fev / scale + code).half(), dev), yev))
    print_rows(rows)
    xy = ["pusher_x", "pusher_y", "t_x", "t_y"]
    print("x, y positions only (R^2): " + ", ".join(f"{r['probe']} {np.mean([r[t] for t in xy]):.3f}" for r in rows))
    assert np.mean([rows[1][t] for t in xy]) > .85, "the world model's encoder should know where things are"


if __name__ == "__main__":
    main()
