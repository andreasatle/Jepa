"""
treeloss_norm_check.py

Diagnostic for the dim-bump experiment (see WORKPLAN.md): doubling
LATENT_DIM from 12 to 24 changed almost nothing (same 4 edges broke,
nearly identical values, same loss floor ~44.0). Hypothesis: the real
ceiling isn't dimension count, it's exp0's tanh(|v|) saturating -- as
raw embedding norm |v| grows, tanh(|v|) approaches 1 with vanishing
gradient, so points get numerically stuck partway to the Poincare ball
boundary long before the loss's far-distance targets (up to bfs
diameter 46) would require. This checks that directly: reports the raw
(pre-tanh) embedding norm distribution and what practical max pairwise
distance that implies, to see if points are already near saturation
rather than just undertrained.

Run:
    uv run python treeloss_norm_check.py --geometry hyperbolic
"""
import argparse

import torch

from jepa_model import load_encoder_checkpoint

MAX_NORM = 1.0 - 1e-5  # same EPS as hyperbolic.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="hyperbolic")
    ap.add_argument("--encoder-checkpoint", default=None,
                     help="override checkpoint path template (default jepa_checkpoint_treeloss_{geometry}.pt)")
    args = ap.parse_args()

    E, ckpt = load_encoder_checkpoint(
        args.geometry,
        path=args.encoder_checkpoint or "jepa_checkpoint_treeloss_{geometry}.pt",
    )
    raw = E.table.weight.detach()  # (num_cells, dim) -- pre-tanh, pre-clip tangent vectors
    norms = raw.norm(dim=-1)

    print(f"[{args.geometry}] dim={ckpt['dim']}, {len(norms)} cells")
    print(f"  raw (pre-tanh) norm: min={norms.min():.3f}  mean={norms.mean():.3f}  "
          f"median={norms.median():.3f}  max={norms.max():.3f}")

    ball_radius = torch.tanh(norms)
    print(f"  post-tanh ball radius |exp0(v)|: min={ball_radius.min():.5f}  "
          f"mean={ball_radius.mean():.5f}  max={ball_radius.max():.5f}  "
          f"(boundary is 1.0, numerical clip at {MAX_NORM})")

    dist_from_origin = 2 * torch.atanh(ball_radius.clamp_max(MAX_NORM))
    print(f"  implied dist(0, point): min={dist_from_origin.min():.3f}  "
          f"mean={dist_from_origin.mean():.3f}  max={dist_from_origin.max():.3f}")

    # theoretical max pairwise distance if two points both sat exactly at
    # MAX_NORM in opposite directions (the absolute ceiling this
    # parameterization could ever reach, numerics aside)
    import math
    theoretical_ceiling = math.acosh(1 + 2 * (2 * MAX_NORM) ** 2 / ((1 - MAX_NORM ** 2) ** 2))
    print(f"  theoretical max pairwise distance at this EPS (both points at boundary, opposite dirs): "
          f"{theoretical_ceiling:.2f}")
    print(f"  fraction of cells with raw norm > 4 (tanh(4)=0.9993, already near-saturated): "
          f"{(norms > 4).float().mean()*100:.1f}%")


if __name__ == "__main__":
    main()
