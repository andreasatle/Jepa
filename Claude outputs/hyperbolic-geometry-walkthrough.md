# Why Hyperbolic Geometry "Falls Apart": A Full Walkthrough

This is the specific derivation, done carefully and in order, of why hyperbolic space hands you closed-form geodesics and distances almost for free. It builds directly on the general machinery already derived elsewhere in this set of docs (the Euler-Lagrange derivation, the Christoffel-symbol formula, and the general fact that any geodesic has constant speed) — nothing here is taken on faith that wasn't earned in those.

## 1. The space and the metric

We work on the open unit ball `B^n = {x in R^n : |x| < 1}` — an ordinary open subset of `R^n`, nothing exotic about the underlying set. What makes it hyperbolic is the metric:

```
g_x(u,w) = lambda(x)^2 <u,w>,      lambda(x) = 2 / (1 - |x|^2)
```

`<u,w>` is the plain Euclidean dot product, so `g_x` is the flat metric multiplied by one position-dependent scalar. This is a *conformal* metric: angles are preserved, lengths are rescaled by `lambda(x)`, more aggressively as `x` approaches the boundary (`lambda -> infinity` as `|x| -> 1`).

Fundamental vs. convenience, first instance: the ball is one particular chart for this geometry — the hyperboloid and half-space models are isometric to it, related by explicit coordinate changes, so anything derived here transfers automatically. What's fundamental is the isometry class itself, guaranteed unique by the Killing–Hopf classification: the one complete, simply connected space of constant curvature `-1`.

## 2. Christoffel symbols for this metric

Write `phi = ln(lambda)`. For any conformally flat metric `g_ij = lambda^2 delta_ij`, plugging into the general Christoffel formula and cancelling the `lambda^2` factors gives the standard identity

```
Gamma^k_ij = delta^k_i (d_j phi) + delta^k_j (d_i phi) - delta_ij (d_k phi)
```

— true for any conformal factor, in any dimension, not yet specific to this one. Now specialize: `phi = ln(2) - ln(1-|x|^2)`, so `d_k phi = 2 x_k / (1-|x|^2)`. Substituting:

```
Gamma^k_ij = ( 2 / (1 - |x|^2) ) * [ delta^k_i x_j + delta^k_j x_i - delta_ij x_k ]
```

This is the full, explicit Christoffel field for the Poincaré ball, in any dimension `n`.

## 3. Radial lines are geodesics — the symmetry argument

Before grinding through the geodesic equation, notice something structural: `lambda(x)` depends only on `|x|`, so the metric is invariant under any rotation about the origin. Consider a straight Euclidean ray through the origin, `x(t) = r(t) e` for a fixed unit vector `e`. There is no preferred direction for it to bend toward — every direction perpendicular to `e` looks identical by the rotational symmetry — so by symmetry alone, a geodesic that starts at the origin heading along `e` must continue along `e` forever. This tells us *the shape* of the solution (a straight ray in these coordinates) without yet solving anything; only the *speed law* `r(t)` remains to be found.

(This can also be checked directly by substituting `x(t) = r(t)e` into the geodesic equation from Section 2 and confirming the angular part is automatically satisfied — that computation is in the appendix of the companion derivation doc, for anyone who wants the non-symmetry-based confirmation.)

## 4. Solving for `r(t)`, the fast way

Here's where the general fact about geodesics pays off directly: **any geodesic has constant Riemannian speed**, a consequence of the geodesic equation being equivalent to "parallel transport of the velocity along itself" combined with the Levi-Civita connection being metric-compatible (parallel transport preserves lengths measured by `g`). We don't need to solve a second-order ODE and discover this afterward — we can use it as an input from the start.

For our radial curve, the Riemannian speed is

```
|x'(t)|_g = lambda(r(t)) * r'(t) = ( 2 / (1 - r(t)^2) ) * r'(t)
```

Constant speed means this equals some fixed number, call it `C`, for the whole curve. That's now a first-order, separable ODE:

```
2 r' / (1 - r^2) = C   =>   d/dt [ artanh(r) ] = C/2
```

Integrate directly: `artanh(r(t)) = (C/2) t + const`. With `r(0) = 0` (starting at the origin), `const = 0`. Writing `c = C/2`:

```
r(t) = tanh(c t)
```

No second-order analysis, no need to spot a clever substitution after the fact — the constant-speed fact, already established in general, hands you a first-order ODE that integrates in one line.

## 5. The exponential map `exp_0(v)`

`exp_0(v) = gamma_v(1)`, where `gamma_v(0) = 0` and the coordinate derivative `gamma_v'(0) = v`. Writing `v = c*e` for a unit vector `e`, this is exactly the curve just solved, evaluated at `t=1`:

```
exp_0(v) = r(1) e = tanh(c) e = tanh(|v|) * v / |v|          (exp_0(0) = 0)
```

This is precisely the closed-form projection used as the encoder's final step in the roadmap — not a heuristic squashing function, but the literal solution of the geodesic equation.

## 6. Distance: from the origin, then everywhere else

**From the origin.** Since `r(t) = tanh(ct)` has constant Riemannian speed `2c` (check: `lambda(r(t))*r'(t) = (2/(1-r^2)) * c(1-r^2) = 2c`, confirming the constant from Section 4 was `C=2c`), the length of the segment from `t=0` to `t=1` is simply `speed * interval = 2c`. Writing `rho = tanh(c)` (the endpoint's Euclidean radius), `c = artanh(rho)`, so:

```
dist(0, x) = 2 * artanh(|x|)
```

**Everywhere else.** This is the general IVP-vs-BVP split in action. Section 5 solved the *initial-value* problem (start at `0`, given a velocity, integrate forward) in closed form. For two arbitrary points `x, y`, what you actually want is the *boundary-value* problem — find the geodesic connecting them — which is generically much harder. But the ball's isometry group (rotations, plus "hyperbolic translations" `T_x` taking `0` to any point `x`) acts transitively, so every boundary-value problem can be converted into an already-solved initial-value problem: translate `x` to the origin with `T_x^{-1}`, and read off the from-origin distance to the translated point:

```
dist(x, y) = dist(0, T_x^{-1}(y))
```

Working out `T_x^{-1}(y)` explicitly is pure algebra (Mobius addition), not new geometric content, and gives the standard closed forms:

```
Mobius addition:   x (+) y = [ (1+2<x,y>+|y|^2) x + (1-|x|^2) y ] / (1+2<x,y>+|x|^2|y|^2)
dist(x,y) = 2 * artanh( |(-x)(+)y| )  =  arcosh( 1 + 2|x-y|^2 / ((1-|x|^2)(1-|y|^2)) )
```

The two forms on the right are algebraically the same number; which one gets coded up is a convenience choice (the `arcosh` form is what most libraries actually use, since it avoids computing a Mobius sum).

## 7. Confirming the curvature really is `-1`

The 2D conformal-metric curvature formula, `K = -(1/lambda^2) * Delta(ln lambda)`, applied to `phi = ln(2) - ln(1-x^2-y^2)`:

```
Delta(phi) = 4 / (1-r^2)^2     (computed directly from d_x phi = 2x/(1-r^2), etc.)
lambda^2   = 4 / (1-r^2)^2
K = -[(1-r^2)^2/4] * [4/(1-r^2)^2] = -1
```

exactly, everywhere. For `n>2`, the isometry group found in Section 6 acts transitively on the unit tangent bundle (every point, every direction, looks the same), which forces constant sectional curvature; combined with the 2D value above, that constant is `-1` in every dimension. Same Killing–Hopf logic as always: maximal symmetry plus one measured value pins down the whole space.

## 8. Why this is usable in a training loop

Two payoffs, both direct consequences of the above, covered in full in the companion docs:

The Christoffel array that a *generic* `n`-dimensional metric would need — `O(n^3)` numbers per point, half a billion for `n=1000` — never gets built here. Every entry is a few arithmetic operations on `x` from the formula in Section 2, and in practice you don't even need entries: `exp_0` and `dist` are already closed-form solutions of the geodesic equation, so the array is bypassed entirely.

Optimization stays plain Euclidean Adam, because `exp_0` and `dist` are ordinary smooth maps between Euclidean spaces from the optimizer's point of view — the network's actual weights never leave `R^N`, so there's no retraction or parallel transport to implement. (Full argument in `hyperbolic-geometry-derivations.md`, Section 8.) Points get clipped away from `|x|=1` purely because `dist(0,x) = 2*artanh(|x|) -> infinity` there and floating point can't resolve it — a numerics safeguard, not a geometry change.

## Reference: every formula, together

```
Metric:                 g_x(u,w) = lambda(x)^2 <u,w>,   lambda(x) = 2/(1-|x|^2)

Christoffel symbols:    Gamma^k_ij = (2/(1-|x|^2)) * [ delta^k_i x_j + delta^k_j x_i - delta_ij x_k ]

Radial geodesic:        x(t) = tanh(c t) * e        (constant Riemannian speed 2c)

Exponential map:        exp_0(v) = tanh(|v|) * v / |v|          (exp_0(0) = 0)

Distance from origin:   dist(0, x) = 2 * artanh(|x|)

Mobius addition:        x (+) y = [ (1+2<x,y>+|y|^2) x + (1-|x|^2) y ] / (1+2<x,y>+|x|^2|y|^2)

General distance:       dist(x,y) = 2 * artanh( |(-x)(+)y| )
                                   = arcosh( 1 + 2|x-y|^2 / ((1-|x|^2)(1-|y|^2)) )

Curvature:              K = -1   (constant, all dimensions)

Why it's cheap:         Christoffel array never built (closed-form exp_0/dist bypass it);
                         optimizer stays plain Euclidean (weights never leave R^N);
                         boundary clipping is numerics only (dist(0,x) -> infinity as |x| -> 1)
```
