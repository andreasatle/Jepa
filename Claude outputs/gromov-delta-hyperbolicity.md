# Gromov δ-Hyperbolicity

## What it is

Gromov δ-hyperbolicity is a way to measure how "tree-like" a metric space is, and its key feature — the reason it matters beyond the Riemannian setting of the tutorial doc — is that it needs nothing but a set of points and a distance function `d(x,y)` between them. No manifold structure, no smoothness, no connection or curvature tensor required. It applies equally to a discrete graph, a phylogenetic tree, a point cloud extracted from a neural network's latent space, or a genuine Riemannian manifold like `H^n`. This is the sense in which it's a strictly more general tool than sectional curvature: sectional curvature needs `M` to be a smooth Riemannian manifold in the first place, while δ-hyperbolicity is defined for *any* metric space `(X, d)`.

## The four-point condition

For any four points `w, x, y, z ∈ X`, form the three ways of pairing them up and summing distances:

```
S1 = d(w,x) + d(y,z)
S2 = d(w,y) + d(x,z)
S3 = d(w,z) + d(x,y)
```

Sort these three numbers so `S(1) ≤ S(2) ≤ S(3)`, and define the hyperbolicity of that particular quadruple as half the gap between the two largest sums:

```
δ(w,x,y,z) = ( S(3) − S(2) ) / 2
```

The δ-hyperbolicity of the whole space is the supremum of this quantity over every possible quadruple of points: `δ = sup δ(w,x,y,z)`.

## Why this characterizes "tree-likeness"

In an exact metric tree (the shortest-path metric on a graph with no cycles), the four-point condition holds with `δ = 0` for every quadruple, always — this is not approximately true, it's exact, and it's actually an *if and only if*: a metric space embeds isometrically into a metric tree exactly when it is 0-hyperbolic. The intuition is that in a tree, for any four points, two of the three pairing sums are always forced to be equal to each other and at least as large as the third (the two "long way around" pairings tie, and the "direct/crossing" pairing is shorter) — geodesics in a tree only ever meet at branch points, so there is no room for all three pairings to disagree independently the way they can in, say, flat Euclidean space. Moving away from `δ = 0`, the size of `δ` measures how far the space deviates from that rigid tree structure — small `δ` relative to the scale of the space means "close to a tree," large `δ` means "genuinely non-tree-like," with Euclidean space `R^n` being the standard example of an *un*bounded, non-hyperbolic space (δ grows without bound as you look at larger and larger regions of `R^n`).

This connects directly back to the Riemannian material: any complete, simply connected Riemannian manifold with sectional curvature bounded above by some negative constant (which includes `H^n` itself, curvature exactly `−1`) is Gromov hyperbolic in this sense, with a *finite, bounded* `δ` that does not grow with the size of the region you look at — this is the same "thin triangles" phenomenon that shows up as the CAT(-1) property in comparison geometry, and it is one of the deep links between the local, infinitesimal notion of curvature from the tutorial doc and this purely metric, coordinate-free notion of hyperbolicity. Trees are the extreme case (`δ = 0`, "infinitely negatively curved" in an informal sense); genuine constant-curvature hyperbolic space sits close to that extreme with some fixed finite `δ`; flat and positively curved spaces are unbounded or otherwise fail to be hyperbolic in this sense at all.

## The scale-invariant version: relative delta

Raw `δ` has units of distance, so it isn't comparable across datasets or embeddings at different scales — a latent space that's simply been rescaled by a factor of 10 would report a `δ` ten times larger despite being geometrically identical. The standard fix (used by Khrulkov et al., *Hyperbolic Image Embeddings*, CVPR 2020, the source of the test as applied to learned representations) is to normalize by the diameter of the point set:

```
rel_delta = 2 * delta / diam(X)
```

This lies in `[0, 1]`: `rel_delta = 0` for an exact tree, and larger values indicate progressively less tree-like, more "spread out" (Euclidean-like) geometry. This is the number worth comparing across different runs, checkpoints, or datasets — not raw `δ`.

## Estimating it in practice

Computing `δ` exactly requires checking *every* quadruple of points, which is `O(n^4)` — infeasible once `n` is more than a few hundred points. The standard practical approach is to randomly sample a large number of quadruples (tens of thousands is typical) and use a statistic of the sampled `δ(w,x,y,z)` values as a proxy for the true supremum. Two choices are common: the sample maximum, or a high percentile (e.g. the 99th) of the sampled values — the percentile is generally preferred as the headline number because the true maximum is a single potentially noisy outlier, while a high percentile is a more stable summary of "how hyperbolic the space typically looks," at the cost of being a slight underestimate of the true worst-case `δ`. Either way, because sampling only ever checks a subset of all quadruples, **any value obtained this way is a lower bound on the true `δ`** — the real supremum, checked over every quadruple, can only be equal to or larger than what sampling finds.

## What's fundamental vs. what's a convenience/implementation choice

Fundamental to the mathematics — true of the metric space itself, not a choice made when testing it: the four-point condition and the definition of `δ` as its supremum; the fact that `δ = 0` exactly characterizes metric trees; the fact that constant-negative-curvature Riemannian manifolds are hyperbolic in this sense with a curvature-dependent bound on `δ`.

Convenience/implementation choices — decisions made when *estimating* `δ` for a real dataset, which don't change the underlying definition: which distance function `d` is used (Euclidean distance in a learned latent space is one legitimate choice among others — nothing forces it); how many quadruples are sampled and how they're chosen (uniform random is standard, but not the only option); whether the sample maximum or a percentile is reported as the headline `δ`; which subset of the data (which episodes, which checkpoint) is included in the point cloud being tested.

## Reference: all the formulas together

```
Four-point hyperbolicity of a quadruple:
    S1 = d(w,x) + d(y,z)
    S2 = d(w,y) + d(x,z)
    S3 = d(w,z) + d(x,y)
    sort so S(1) <= S(2) <= S(3)
    delta(w,x,y,z) = ( S(3) - S(2) ) / 2

Hyperbolicity of the space:
    delta = sup over all quadruples of delta(w,x,y,z)

Scale-invariant relative hyperbolicity:
    rel_delta = 2 * delta / diam(X)          (range [0,1]; 0 = exact tree)

Complexity:
    exact:      O(n^4)  -- infeasible beyond a few hundred points
    estimated:  sample many quadruples at random, take max or a high
                percentile (e.g. 99th) of the sampled delta values --
                always a lower bound on the true delta
```

## Applied to the ball-in-cup JEPA encoder

As a concrete instance: `E`'s 32-dimensional latent space, Euclidean distance, 20 held-out eval episodes (4020 points), 50,000 sampled quadruples, gave:

```
diameter:            10.57
delta (99th pctile):  0.81   ->  rel_delta = 0.153
delta (max sampled):  1.44   ->  rel_delta = 0.273
```

`rel_delta` in the 0.15–0.27 range is a genuinely low reading — this encoder was trained with an ordinary Euclidean loss, no hyperbolic structure imposed, so the latent space organizing itself with a noticeably tree-like shape is an emergent property of the data (branching ball-in-cup trajectories), not something engineered in.
