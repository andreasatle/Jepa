# Hyperbolic Geometry: The Practical Toolkit, Derived From First Principles

This picks up directly from the Riemannian geometry tutorial doc, and derives the specific pieces that make hyperbolic space usable inside a training loop: the closed-form exponential map $\exp_0(v)$, the closed-form distance $\operatorname{dist}(x,y)$, why boundary clipping is needed, and why plain Adam still works even though the output lives on a curved space. Every one of these is a *consequence* of constant curvature, not a separate assumption bolted on — the point of this doc is to show exactly which line of algebra that consequence comes from.

## 1. Where we start: the general machinery

Three definitions from the tutorial doc, unchanged, are all we need as inputs.

$$
\begin{aligned}
\text{Christoffel symbols:}\quad \Gamma^k_{ij} &= \frac12\sum_l g^{kl}(\partial_i g_{jl}+\partial_j g_{il}-\partial_l g_{ij}) \\
\text{Geodesic equation:}\quad \frac{d^2x^k}{dt^2}+\sum_{ij}\Gamma^k_{ij}\frac{dx^i}{dt}\frac{dx^j}{dt} &= 0 \\
\text{Exponential map:}\quad \exp_p(v) &= \gamma_v(1),\quad \gamma_v(0)=p,\quad \gamma_v'(0)=v
\end{aligned}
$$

Distance is, as always, the length of a length-minimizing geodesic between two points. In general this whole pipeline — get the metric, grind out Christoffel symbols, solve a nonlinear ODE for the geodesic, integrate its length — has no closed form and gets solved numerically. What makes hyperbolic space different is that the ODE in the last step turns out to be solvable by hand. That is the entire content of "hyperbolic space is special," made precise below.

## 2. The model: the Poincaré ball and its metric

We work on the open unit ball $\mathbb{B}^n = \{x\in \mathbb{R}^n : \lVert x\rVert < 1\}$, with metric

$$
g_x(u,w)=\lambda(x)^2\langle u,w\rangle,\qquad \lambda(x)=\frac{2}{1-\lVert x\rVert^2}
$$

This is a *conformal* metric: it's just the flat Euclidean metric multiplied by a position-dependent scalar $\lambda(x)^2$, so it preserves angles but stretches lengths, more and more as $x$ approaches the boundary.

Fundamental vs. convenience, first instance: choosing *this particular chart* (the ball, as opposed to the upper half-space or the hyperboloid model) is a convenience choice — those models are isometric to this one, related by explicit coordinate changes, so any fact derived here transfers to them automatically. What's fundamental is the isometry class itself: the unique (up to isometry) complete, simply connected Riemannian manifold of constant curvature $-1$, which is what the Killing–Hopf classification in the tutorial doc guarantees exists and is unique. We picked the ball model because it's bounded (convenient for a neural net's output range) and conformal (angle-preserving, so directions behave intuitively).

## 3. Christoffel symbols for a conformal metric

Write $\phi = \ln(\lambda)$. Since $g_{ij} = \lambda^2 \delta_{ij}$, we get $g^{ij} = \lambda^{-2} \delta_{ij}$, and

$$
\partial_i g_{jl}=2\lambda(\partial_i\lambda)\delta_{jl}=2\lambda^2(\partial_i\phi)\delta_{jl}
$$

Substituting into the Christoffel formula and cancelling the $\lambda^2$ factors:

$$
\Gamma^k_{ij}=\delta^k_i(\partial_j\phi)+\delta^k_j(\partial_i\phi)-\delta_{ij}(\partial_k\phi)
$$

This holds for *any* conformally flat metric, in any dimension — it's a standalone algebraic fact about the Christoffel formula, not yet specific to hyperbolic space.

Now specialize: $\phi = \ln(2) - \ln(1 - \lVert x\rVert^2)$, so $\partial_k \phi = 2 x_k / (1 - \lVert x\rVert^2)$. Plugging in:

$$
\Gamma^k_{ij}=\frac{2}{1-\lVert x\rVert^2}\left[\delta^k_i x_j+\delta^k_j x_i-\delta_{ij}x_k\right]
$$

This is the exact, fully explicit Christoffel symbol field for the Poincaré ball, in any dimension $n$.

## 4. Solving the geodesic equation through the origin

By the rotational symmetry of $\lambda(x)$ (it depends only on $\lVert x\rVert$), a straight Euclidean line through the origin, $x(t) = r(t) e$ for a fixed unit vector $e$, should be a geodesic — there's no preferred direction to bend toward. We can verify this directly and get the exact speed law for free.

Substitute $x(t) = r(t) e$ into the geodesic equation. First compute $\Gamma^k_{ij} e^i e^j$ using the formula above, with $x = r e$:

$$
\begin{aligned}
\Gamma^k_{ij}e^i e^j &= \frac{2}{1-r^2}\left[2e^k\langle x,e\rangle-x^k\right] \\
&= \frac{2}{1-r^2}\left[2e^k r-r e^k\right] \qquad (\langle x,e\rangle=r,\ x^k=r e^k) \\
&= \frac{2r}{1-r^2}e^k
\end{aligned}
$$

So the geodesic equation reduces to a single scalar ODE for $r(t)$:

$$
r''+\frac{2r(r')^2}{1-r^2}=0
$$

Substitute $u(t) = \operatorname{artanh}(r(t))$, i.e. $u' = r'/(1-r^2)$. Differentiating again:

$$
u''=\frac{r''(1-r^2)+2r(r')^2}{(1-r^2)^2}
$$

The numerator is exactly the left side of the ODE above (multiplied by $1-r^2$), which is zero. So $u'' = 0$: **$u(t)$ is linear in $t$.**

With $r(0) = 0$ (so $u(0) = 0$) and writing $c = r'(0)$ for the initial Euclidean-coordinate speed, $u(t) = c t$, i.e.

$$
r(t)=\tanh(ct)
$$

This confirms radial lines are geodesics, and gives their exact parametrization — no numerical ODE solver needed, because the substitution $u = \operatorname{artanh}(r)$ linearizes it exactly.

## 5. The exponential map $\exp_0(v)$

By definition, $\exp_0(v) = \gamma_v(1)$ where $\gamma_v(0) = 0$ and the coordinate derivative $\gamma_v'(0) = v$. Writing $v = c e$ with $e$ a unit vector, this is exactly the curve solved above, so:

$$
\exp_0(v)=r(1)e=\tanh(c)e=\tanh(\lVert v\rVert)\frac{v}{\lVert v\rVert}
$$

with the convention $\exp_0(0) = 0$ (the formula is smooth there too — $\tanh(\lVert v\rVert) v/\lVert v\rVert \to v$ as $v \to 0$). This is precisely the projection used as the encoder's final step in the roadmap: an ordinary Euclidean vector $v$ in, a point inside the ball out, and the whole derivation above is where that specific $\tanh$ comes from — it is not a heuristic squashing function, it is the literal solution to the geodesic equation.

## 6. Distance: from the origin, then in general

**From the origin.** The parametrization $r(t) = \tanh(ct)$ was affine (constant Riemannian speed), which we can check directly: $\lVert \gamma'(t)\rVert_g = \lambda(r(t)) \lVert r'(t)\rVert = (2/(1-r^2)) \cdot c(1-r^2) = 2c$, a constant. So the Riemannian length of the segment from $t=0$ to $t=1$ is $2c$, and its endpoint sits at Euclidean radius $\tanh(c)$. Writing $\rho = \tanh(c)$, i.e. $c = \operatorname{artanh}(\rho)$:

$$
\operatorname{dist}(0,x)=2\operatorname{artanh}(\lVert x\rVert)
$$

(Consistency check: this says $\exp_0(v)$ sits at distance $2\lVert v\rVert$ from the origin, matching $\lVert v\rVert_g = \lambda(0)\lVert v\rVert = 2\lVert v\rVert$ — the Riemannian norm of $v$, exactly as an exponential map should give.)

**In general.** The isometry group of the ball acts transitively — for any point $x$, there is an isometry $T_x$ taking $0$ to $x$ (a "hyperbolic translation," on top of the ordinary rotations about $0$, which are isometries because $\lambda$ depends only on $\lVert x\rVert$). So the distance between any two points reduces to a distance-from-origin computation after moving one point there:

$$
\operatorname{dist}(x,y)=\operatorname{dist}(0,T_x^{-1}(y))
$$

Working out $T_x^{-1}(y)$ explicitly and substituting into the formula above is pure algebra from here (no new geometric content), and gives the two standard closed forms — stated here, not re-derived symbol-by-symbol, since it's bookkeeping rather than geometry:

$$
\begin{aligned}
\text{Möbius addition:}\quad x\oplus y &= \frac{(1+2\langle x,y\rangle+\lVert y\rVert^2)x+(1-\lVert x\rVert^2)y}{1+2\langle x,y\rangle+\lVert x\rVert^2\lVert y\rVert^2} \\
\operatorname{dist}(x,y) &= 2\operatorname{artanh}(\lVert(-x)\oplus y\rVert) \\
&= \operatorname{arcosh}\left(1+\frac{2\lVert x-y\rVert^2}{(1-\lVert x\rVert^2)(1-\lVert y\rVert^2)}\right)
\end{aligned}
$$

Both forms are algebraically the same number; which one a given implementation uses is a convenience choice (the $\operatorname{arcosh}$ form avoids computing a Möbius sum explicitly and is what most hyperbolic-embedding codebases actually call).

## 7. Checking the curvature is really constant $-1$

The 2D case has a clean, classical formula for the Gaussian curvature of a conformal metric $ds^2 = \lambda^2(dx^2+dy^2)$:

$$
K=-\frac{1}{\lambda^2}\Delta(\ln\lambda),\qquad \Delta=\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}\quad\text{(flat Laplacian)}
$$

With $\phi = \ln(\lambda) = \ln(2) - \ln(1-x^2-y^2)$ and $r^2 = x^2+y^2$:

$$
\begin{aligned}
\partial_x\phi &= \frac{2x}{1-r^2} \\
\partial_x^2\phi &= \frac{2(1-r^2)+4x^2}{(1-r^2)^2}\qquad\text{(similarly for }y\text{)} \\
\Delta\phi &= \frac{4-4r^2+4x^2+4y^2}{(1-r^2)^2}=\frac{4}{(1-r^2)^2} \\
\lambda^2 &= \frac{4}{(1-r^2)^2} \\
K &= -\frac{(1-r^2)^2}{4}\frac{4}{(1-r^2)^2}=-1
\end{aligned}
$$

Exactly $-1$, everywhere on the ball, confirmed by direct computation. For dimension $n > 2$ the full Ricci-tensor bookkeeping is heavier, but it's not needed: the isometry group found in section 6 (rotations plus hyperbolic translations) acts transitively on the unit tangent bundle — every point, in every direction, looks the same — and a manifold with that much symmetry necessarily has constant sectional curvature. Combined with the 2D value above (any 2-plane section gives the same computation), the curvature is $-1$ in every dimension. This is the same Killing–Hopf logic from the tutorial doc: maximal symmetry plus one measured value pins down the whole space.

## 8. Why plain Adam still works: the "wrapped" parametrization

This is where the roadmap's "no Riemannian SGD needed" claim comes from, and it's a calculus fact rather than a geometry fact.

Genuine Riemannian optimization is needed when the *trainable parameters themselves* are constrained to lie on a manifold — for example, optimizing directly over points on a sphere. There, an ordinary gradient step $\theta - \eta \cdot \operatorname{grad}(f)$ walks off the manifold, so every step needs a retraction (an exponential map) back onto it, the gradient itself needs correcting by the inverse metric ($g^{-1} \operatorname{grad}(f)$, not just $\operatorname{grad}(f)$), and momentum-based methods need parallel transport to keep running averages meaningful across tangent spaces at different points.

None of that applies here, because the network's actual weights — conv filters, MLP weights — are never constrained to the ball. They stay ordinary free elements of $\mathbb{R}^N$ through the entire optimization. The ball only ever shows up as the *range* of $\exp_0$, applied once per forward pass to the network's raw output $v$. Formally: the loss is $L = \operatorname{dist}(\exp_0(v_{\mathrm{pred}}), \exp_0(v_{\mathrm{target}}))$, and

$$
\frac{dL}{d\theta}=\frac{dL}{dz}\frac{dz}{dv}\frac{dv}{d\theta}
$$

$\exp_0$ is just a smooth map $\mathbb{R}^n \to \mathbb{R}^n$ (forgetting, for calculus purposes, that we intend to interpret its output as living on a curved space) — it has an ordinary Jacobian, and the chain rule doesn't care what geometric meaning we assign to its output afterward. Likewise $\operatorname{dist}(x,y)$ is an ordinary smooth scalar-valued function of two Euclidean vectors (the $\operatorname{arcosh}$ formula from section 6), so its gradient with respect to $x$ and $y$ is an ordinary Euclidean gradient. Nowhere in this chain does an update step move a point *along* the manifold — the manifold is touched only in the forward pass, as a value, never as the space the optimizer walks in. That is precisely why plain Adam, unmodified, is correct here.

## 9. Why points get clipped away from the boundary

From section 6, $\operatorname{dist}(0,x) = 2\cdot \operatorname{artanh}(\lVert x\rVert)$, and $\operatorname{artanh}(\rho) \to \infty$ as $\rho \to 1$. So points near the Euclidean boundary of the ball are not "almost outside the space" — they represent points that are genuinely, arbitrarily far away in the actual hyperbolic distance. The manifold is complete (Hopf–Rinow, from the tutorial doc) with infinite diameter; the Euclidean boundary is a coordinate artifact of this particular chart, not a true edge of the space.

Numerically this is a problem: $\operatorname{artanh}$ and $\lambda(x) = 2/(1-\lVert x\rVert^2)$ both blow up as $\lVert x\rVert \to 1$, and floating point loses precision in $1 - \lVert x\rVert^2$ well before actually reaching the boundary, producing `inf`/`NaN`. Clipping $\lVert x\rVert$ to something like $1 - 10^{-5}$ is purely a numerical safeguard against that — it changes nothing about the geometry, only about which finite region of it is numerically trustworthy.

## 10. Fundamental vs. convenience, in one place

Fundamental — true of constant-curvature $-1$ space itself, independent of any implementation choice: that the geodesic ODE linearizes exactly (section 4); the resulting $\tanh$/$\operatorname{artanh}$ closed forms for $\exp_0$ and $\operatorname{dist}$ (sections 5–6); the value $K=-1$ and its model-independence (section 7); the fact that no retraction or parallel transport is needed for optimizing parameters that merely *feed into* a map onto the manifold, rather than living on it (section 8).

Convenience — a choice made in how the fundamental facts get coded up, not a separate geometric claim: using the ball model rather than the hyperboloid or half-space model (section 2); using the $\operatorname{arcosh}$ form of $\operatorname{dist}(x,y)$ rather than the Möbius-addition form (section 6); the specific clipping threshold used to avoid boundary blowup (section 9); which of the two algebraically-equal distance formulas a given library implements.

## 11. Reference: every formula, together

**Metric**

$$
g_x(u,w)=\lambda(x)^2\langle u,w\rangle,\qquad \lambda(x)=\frac{2}{1-\lVert x\rVert^2}
$$

**Christoffel symbols**

$$
\Gamma^k_{ij}=\frac{2}{1-\lVert x\rVert^2}\left[\delta^k_i x_j+\delta^k_j x_i-\delta_{ij}x_k\right]
$$

**Radial geodesic**

$$
x(t)=\tanh(ct)e,\qquad \lVert e\rVert=1,\quad c=\text{initial Euclidean speed}
$$

**Exponential map**

$$
\exp_0(v)=\tanh(\lVert v\rVert)\frac{v}{\lVert v\rVert},\qquad \exp_0(0)=0
$$

**Distance from origin**

$$
\operatorname{dist}(0,x)=2\operatorname{artanh}(\lVert x\rVert)
$$

**Möbius addition**

$$
x\oplus y=\frac{(1+2\langle x,y\rangle+\lVert y\rVert^2)x+(1-\lVert x\rVert^2)y}{1+2\langle x,y\rangle+\lVert x\rVert^2\lVert y\rVert^2}
$$

**General distance**

$$
\begin{aligned}
\operatorname{dist}(x,y)&=2\operatorname{artanh}(\lVert(-x)\oplus y\rVert) \\
&=\operatorname{arcosh}\left(1+\frac{2\lVert x-y\rVert^2}{(1-\lVert x\rVert^2)(1-\lVert y\rVert^2)}\right)
\end{aligned}
$$

**Curvature**

$$
K=-1\qquad\text{(constant, all dimensions)}
$$

**Optimization:** Plain Euclidean Adam on $\theta$; $\exp_0$ and $\operatorname{dist}$ are ordinary smooth maps between Euclidean spaces from the optimizer's point of view; no retraction, no parallel transport, because $\theta$ never leaves $\mathbb{R}^N$.

**Numerics:** Clip $\lVert x\rVert\leq 1-\varepsilon$ (e.g. $\varepsilon\approx 10^{-5}$) before calling $\operatorname{dist}$ or $\exp_0$, since $\lambda(x)$ and $\operatorname{artanh}(\lVert x\rVert)$ diverge as $\lVert x\rVert\to 1$.
