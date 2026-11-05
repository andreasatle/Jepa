"""
checks/check_unit_ball.py

One-off sanity check: does E's learned embedding, once mapped onto the
manifold via to_geometry(), actually stay strictly inside the unit ball
(Poincare ball radius 1)? Every hyperbolic embedding MUST satisfy this by
construction (hyp.exp0/hyp.clip enforce it), so this isn't really testing
"did it escape the ball" -- it can't -- it's testing HOW CLOSE TO THE
BOUNDARY the embeddings sit, since distance/arcosh both blow up in
precision cost as norm -> 1 (see hyperbolic.py's EPS/MAX_NORM comments
and the old maze/ project's dist_scale fix for this exact numerical
concern).

Run (from inside maze_jepa/):
    uv run python checks/check_unit_ball.py --geometry hyperbolic
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from jepa_model import load_encoder_checkpoint, to_geometry

ap = argparse.ArgumentParser()
ap.add_argument("--geometry", default="hyperbolic")
args = ap.parse_args()

E, ckpt = load_encoder_checkpoint(args.geometry)
with torch.no_grad():
    raw = E.table.weight
    geo = to_geometry(raw, args.geometry)
    norms = geo.norm(dim=-1)

print(f"{norms.shape[0]} cell embeddings, geometry={args.geometry}")
print(f"norm min={norms.min().item():.6f} max={norms.max().item():.6f} mean={norms.mean().item():.6f}")
print(f"all strictly inside unit ball (norm < 1): {bool((norms < 1).all())}")
print(f"cells with norm > 0.99 (near boundary, precision risk): {int((norms > 0.99).sum())}")
