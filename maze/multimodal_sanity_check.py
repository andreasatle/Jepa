"""
multimodal_sanity_check.py

Maze JEPA -- Phase 0: parametric sweep on synthetic multimodal targets.
See WORKPLAN.md for the full reasoning. Two design traps were caught and
fixed before this script was written, both confirmed numerically first with
a plain-numpy check, not assumed:

1) Exactly two equally-weighted modes force the squared-distance minimizer
   to the exact metric midpoint between them, in ANY geodesic space with a
   reflection symmetry swapping the two modes -- Euclidean and hyperbolic
   both qualify. A two-mode test is therefore guaranteed to show identical
   "collapse to the middle" behavior in both geometries, regardless of
   whether hyperbolic geometry helps anywhere. Not usable.

2) The natural-looking fix -- more modes, arranged symmetrically (a regular
   k-gon) -- has the exact same flaw: the k-gon's own rotational symmetry
   forces the minimizer to the centroid in BOTH geometries too, confirmed
   the same way. Also not usable.

The fix that actually works: modes placed asymmetrically (random, not on a
regular polygon), so no isometry of the ambient space maps the mode set
back onto itself, and nothing forces the two geometries to the same answer.
Confirmed this lets the two geometries genuinely diverge before writing the
full sweep below.

WHAT THIS TESTS:
For k >= 3 randomly-placed target modes, optimize a single point x to
minimize expected squared distance to a uniformly-drawn mode, once under
ordinary Euclidean distance and once under hyperbolic.dist (this project's
own hyperbolic.py, copied in from ball_in_cup_hyperbolic/). For each
(k, dim, separation), average over several random mode placements and
report how close the optimum lands to the nearest actual mode, relative to
the typical spacing between modes -- lower is better (closer to something
real, not lost in the empty middle) -- for both geometries side by side.

Run (from inside maze/):
    uv run python3 multimodal_sanity_check.py
"""
import itertools
import torch
from scipy import stats

import hyperbolic as hyp

DIMS = [2, 4, 8]
K_MODES = [3, 5, 8]
SEPARATIONS = [0.3, 0.8, 1.5, 2.5]  # tangent-space NORM, dim-independent (see random_modes)
N_TRIALS = 20         # random mode placements averaged per (k, dim, sep)
N_STEPS = 1500
LR = 0.05


def random_modes(k, dim, separation, generator):
    # Each mode gets tangent-space norm exactly = separation, independent of
    # dim -- NOT i.i.d. per-axis noise scaled by separation, which would make
    # the typical norm grow like separation*sqrt(dim) and silently saturate
    # tanh (exp0) for most dim/separation combinations tested (confirmed
    # numerically: tanh(sep*sqrt(dim)) >= 0.97 for most settings originally
    # swept here, collapsing nearly all modes onto the ball boundary
    # regardless of their intended separation -- not a real geometry result).
    raw = torch.randn(k, dim, generator=generator)
    directions = raw / raw.norm(dim=-1, keepdim=True)
    return directions * separation


def fit_euclidean(modes, steps=N_STEPS, lr=LR):
    x = modes.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((x.unsqueeze(0) - modes) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
    return x.detach()


def fit_hyperbolic(modes_tangent, steps=N_STEPS, lr=LR):
    modes_ball = hyp.exp0(modes_tangent)
    v0 = modes_tangent.mean(0) * 0.3  # safely inside the ball as a start
    v = v0.clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        x = hyp.exp0(v.unsqueeze(0))[0]
        d = hyp.dist(x.unsqueeze(0).expand_as(modes_ball), modes_ball)
        loss = (d ** 2).mean()
        loss.backward()
        opt.step()
    return hyp.exp0(v.detach().unsqueeze(0))[0], modes_ball


def nearest_over_spacing_euclidean(x, modes):
    near = torch.linalg.norm(modes - x.unsqueeze(0), dim=-1).min()
    spacing = torch.pdist(modes).mean()
    return (near / spacing).item()


def nearest_over_spacing_hyperbolic(x, modes_ball):
    near = hyp.dist(x.unsqueeze(0).expand_as(modes_ball), modes_ball).min()
    k = modes_ball.shape[0]
    pw = torch.stack([
        hyp.dist(modes_ball[i:i + 1], modes_ball[j:j + 1])[0]
        for i in range(k) for j in range(i + 1, k)
    ])
    return (near / pw.mean()).item()


def main():
    gen = torch.Generator().manual_seed(0)
    header = f"{'k':>3} {'dim':>4} {'sep':>5}  {'EUCL':>16}  {'HYP':>16}  {'paired diff (E-H)':>18}  {'p (ttest/wilcoxon)':>19}"
    print(header)
    for k, dim, sep in itertools.product(K_MODES, DIMS, SEPARATIONS):
        ratios_e, ratios_h = [], []
        for _ in range(N_TRIALS):
            modes = random_modes(k, dim, sep, gen)
            x_e = fit_euclidean(modes)
            ratios_e.append(nearest_over_spacing_euclidean(x_e, modes))
            x_h, modes_ball = fit_hyperbolic(modes)
            ratios_h.append(nearest_over_spacing_hyperbolic(x_h, modes_ball))
        re = torch.tensor(ratios_e)
        rh = torch.tensor(ratios_h)
        diff = re - rh  # positive means HYP landed closer to a mode (better)
        t_p = stats.ttest_rel(ratios_e, ratios_h).pvalue
        try:
            w_p = stats.wilcoxon(ratios_e, ratios_h).pvalue
        except ValueError:
            w_p = float("nan")  # all-zero-difference edge case
        flag = "*" if t_p < 0.05 and diff.mean() > 0 else ("v" if t_p < 0.05 and diff.mean() < 0 else " ")
        print(f"{k:>3} {dim:>4} {sep:>5.1f}  {re.mean():>7.4f}+-{re.std():.4f}  {rh.mean():>7.4f}+-{rh.std():.4f}  {diff.mean():>9.4f}+-{diff.std():.4f}  {t_p:>8.4f}/{w_p:.4f} {flag}")
    print()
    print("* = HYP significantly closer to nearest mode (p<0.05, paired t-test)")
    print("v = EUCL significantly closer to nearest mode (p<0.05, paired t-test)")


if __name__ == "__main__":
    main()
