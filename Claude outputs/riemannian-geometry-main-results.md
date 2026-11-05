# Riemannian Geometry: Main Results — General, then Spherical and Hyperbolic

This consolidates the derivations from the earlier docs into one reference: general Riemannian geometry first, then two parallel specializations — spherical ($\kappa=+1$) and hyperbolic ($\kappa=-1$). The two specializations are deliberately built the same way, step for step, so the symmetry between them (and the one sign that breaks it) is visible directly. Full rigor for the Euler-Lagrange derivation and the energy-vs-length justification lives in the earlier appendix docs; this file states the results and the main derivation steps, not every proof of every lemma.

## Part I — General Riemannian Geometry

### 1. The metric tensor

At each point $p$ of a manifold $M$, the metric is an inner product on the tangent space:

$$g_p : T_pM \times T_pM \to \mathbb{R}, \qquad g_p(u,w) = g_{ij}(p)\, u^i w^j$$

symmetric ($g_{ij}=g_{ji}$) and positive definite. Positive-definiteness alone guarantees $g$ is invertible (no nonzero vector maps to zero, so all eigenvalues are positive); symmetry is what makes $g$ a genuine inner product and keeps $g^{-1}$ itself symmetric.

### 2. Angle and length

$$\cos\theta = \frac{g_p(u,w)}{\sqrt{g_p(u,u)}\sqrt{g_p(w,w)}}, \qquad \|u\|_g = \sqrt{g_p(u,u)}$$

For a **conformal** metric $g_p(u,w) = \lambda(p)^2\langle u,w\rangle$ (a scalar multiple of a flat reference metric), the $\lambda(p)^2$ cancels exactly in the angle formula — conformal metrics preserve angles, even though they rescale lengths by a single factor of $\lambda(p)$. This is a general fact about any metric of this form, used repeatedly below.

### 3. Connections and parallel transport

Differentiating a vector field along a curve requires comparing tangent vectors at infinitesimally nearby points — but $T_pM$ and $T_qM$ ($p\ne q$) are a priori unrelated vector spaces, so the naive difference quotient doesn't parse. An **affine connection** $\nabla$ supplies the missing rule: it takes a direction $X$ and a vector field $Y$ and produces $\nabla_XY$, linear in $Y$, Leibniz in $Y$ ($\nabla_X(fY)=X(f)Y+f\nabla_XY$), and depending on $X$ only pointwise. The **Fundamental Theorem of Riemannian Geometry**: among all connections, exactly one is compatible with $g$ — the Levi-Civita connection, the unique connection that is both

$$\text{torsion-free: } \nabla_XY-\nabla_YX=[X,Y] \qquad\qquad \text{metric-compatible: } Xg(Y,Z)=g(\nabla_XY,Z)+g(Y,\nabla_XZ) \ \ (\nabla g = 0)$$

Metric-compatibility is what makes this the *geometrically correct* choice: it guarantees parallel transport preserves lengths and angles. A vector field $V(t)$ along a curve is **parallel** if $DV/dt \equiv 0$; given $v_0 \in T_{\gamma(0)}M$, a unique parallel field exists with $V(0)=v_0$ (linear ODE), giving the parallel transport map $P_\gamma : T_{\gamma(0)}M \to T_{\gamma(1)}M$ — an isometry of tangent spaces that, unlike in $\mathbb R^n$, depends on the *path*, not just the endpoints, whenever the enclosed region has nonzero curvature. This path-dependence (holonomy) is one of the cleanest operational readings of what curvature means.

One caveat worth keeping precise: despite carrying indices like a tensor, $\Gamma^k_{ij}$ **is not itself a tensor** — it doesn't transform tensorially under a coordinate change (an extra term involving second derivatives of the transition map appears). You can always find coordinates where all $\Gamma$'s vanish at a single point (normal coordinates), even on a genuinely curved manifold — impossible for a nonzero tensor. Curvature, defined next, is built from $\Gamma$ and its derivatives in exactly the combination that cancels this coordinate artifact, which is what makes curvature the honest, coordinate-independent quantity and $\Gamma$ merely useful bookkeeping.

### 4. The Levi-Civita connection and Christoffel symbols

$$\Gamma^k_{ij} = \frac{1}{2} g^{kl}\left( \partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij} \right)$$

This isn't asserted — it's forced out of the calculus of variations. Extremizing the energy functional $E[x] = \int \frac{1}{2} g_{ij}\dot x^i \dot x^j\, dt$ via the Euler–Lagrange equation

$$\frac{d}{dt}\left(\frac{\partial \mathcal L}{\partial \dot x^i}\right) - \frac{\partial \mathcal L}{\partial x^i} = 0, \qquad \mathcal L = \frac{1}{2} g_{ij}\dot x^i \dot x^j$$

and combining the position-derivative term with the time-derivative-of-momentum term (symmetrizing a relabeled dummy-index pair along the way) produces exactly this combination. (Minimizing energy instead of length is valid, not just convenient: by Cauchy–Schwarz, $E[x] \ge L[x]^2/(2(b-a))$ with equality iff constant speed, so an energy-minimizer is automatically a length-minimizer with the constant-speed parametrization built in — full proof in the companion appendix.)

### 5. The geodesic equation

$$\frac{d^2x^k}{dt^2} + \Gamma^k_{ij}\,\frac{dx^i}{dt}\frac{dx^j}{dt} = 0$$

Same shape as Newton's second law with zero force, $m\ddot x = F$, $F=0$ — the Christoffel term is exactly the correction between naive coordinate acceleration and true (covariant) acceleration, the same role centrifugal/Coriolis terms play for a free particle described in polar or rotating coordinates.

### 6. Geodesics have constant speed

A completely general fact, not special to any one geometry: any solution of the geodesic equation has constant Riemannian speed $\|\dot x(t)\|_g$. This follows because the geodesic equation says the velocity is parallel-transported along itself ($D\dot x/dt = 0$), and the Levi-Civita connection is metric-compatible ($\nabla g = 0$), so parallel transport preserves lengths. This shortcut — using the constant-speed fact as an input rather than deriving it after the fact — is what makes both specializations below solvable by a one-line first-order integration instead of a harder second-order analysis.

### 7. Distance, via geodesics

$$d(p,q) = \inf_{\gamma} L[\gamma], \qquad L[\gamma] = \int \sqrt{g_{ij}\dot\gamma^i\dot\gamma^j}\, dt$$

the infimum taken over all curves connecting $p$ and $q$. Geodesics are the *answer* to this infimum, not the definition: locally, short geodesics minimize length; globally, on a complete manifold, the Hopf–Rinow theorem guarantees a minimizing geodesic exists between any two points, so $d(p,q)$ equals that geodesic's length. Computing this splits into two different problems: the **initial-value problem** (given $p$ and a velocity $v$, integrate forward to get $\exp_p(v)$ — generally easy) and the **boundary-value problem** (given $p$ and $q$, find the $v$ with $\exp_p(v)=q$ — generally hard, a nonlinear root-find). Everything that follows is valuable precisely because, for constant curvature, symmetry converts every boundary-value problem into an already-solved initial-value problem.

### 8. Curvature, the main structural theorems, and the space-form trichotomy

**The curvature tensor.** Curvature is defined by how badly covariant derivatives fail to commute:

$$R(X,Y)Z = \nabla_X\nabla_YZ - \nabla_Y\nabla_XZ - \nabla_{[X,Y]}Z$$

$R\equiv 0$ exactly characterizes local flatness (coordinates with $g_{ij}$ constant, everywhere nearby) — $R$ is precisely the obstruction that keeps second derivatives of the metric from being killable by a coordinate change, even though first derivatives always can be (normal coordinates). The full tensor is usually summarized by contractions: **sectional curvature** $K(\sigma) = g(R(u,v)v,u)$ for orthonormal $u,v$ spanning a 2-plane $\sigma$ (literally ordinary surface curvature, measured on every 2-plane through every point — it determines $R$ completely); **Ricci curvature** $\mathrm{Ric}(v,v)=\sum_i K(\mathrm{span}(v,e_i))$, a trace of $R$; and **scalar curvature** $S=\mathrm{tr}(\mathrm{Ric})$, a single number per point.

**The sign dictionary**, the fact everything downstream hangs on:

$$
\begin{array}{l}
K>0 \ (\text{spheres}):\ \text{geodesics converge; triangle angle sum} > \pi;\ \text{circumference} < 2\pi r;\ \text{tends compact} \\
K=0 \ (\text{flat}):\ \text{geodesics stay parallel; angle sum} = \pi;\ \text{circumference} = 2\pi r \\
K<0 \ (\text{hyperbolic}):\ \text{geodesics diverge exponentially; angle sum} < \pi;\ \text{circumference} \sim \sinh(r),\ \text{exponential in } r
\end{array}
$$

**Three theorems** turn this local, infinitesimal data into global, qualitative facts (all assume completeness — false without it):

*Hopf–Rinow.* For connected $M$: $(M,d)$ complete as a metric space $\iff$ geodesically complete $\iff$ closed bounded sets compact; and if complete, any two points are joined by **at least one** minimizing geodesic. This is the theorem that hands you existence of shortest paths for free, and it's why completeness is the standing hypothesis everywhere else.

*Gauss–Bonnet* (2D, historically first): for a closed surface, $\int_M K\, dA = 2\pi\chi(M)$, tying curvature (local, analytic) to the Euler characteristic (global, topological) — why a torus admits a flat metric but a sphere provably cannot.

*Cartan–Hadamard.* Complete, simply connected, $K\le 0$ everywhere $\Rightarrow$ $\exp_p : T_pM \to M$ is a diffeomorphism for every $p$: $M\cong\mathbb R^n$, any two points joined by a **unique**, globally minimizing geodesic — no antipodal-point pathology is possible.

*Bonnet–Myers* (the positive-curvature counterpart): $K \ge \kappa > 0$ everywhere $\Rightarrow$ $M$ is compact with diameter $\le \pi/\sqrt\kappa$ — the opposite global regime from Cartan–Hadamard.

**The space-form trichotomy.** A *space form* is a manifold of *constant* sectional curvature — the most symmetric case the theory admits. The **Killing–Hopf theorem**: any complete, simply connected space form is isometric to exactly one of three models, up to rescaling to fix $\kappa=+1,0,-1$:

$$\kappa > 0 \Rightarrow S^n \qquad \kappa = 0 \Rightarrow \mathbb R^n \qquad \kappa < 0 \Rightarrow H^n$$

(Every other constant-curvature space, not simply connected, is a quotient of one of these three by a discrete isometry group — e.g. flat tori are quotients of $\mathbb R^n$.) This is a trichotomy — pick exactly one — not a decomposition of a general manifold into pieces of each, and it's the pivot for the rest of this doc: Parts II and III work out the $\kappa=+1$ and $\kappa=-1$ cases in full, in parallel. Because $K\equiv -1 \le 0$, Cartan–Hadamard applies to $H^n$ unconditionally — unique, globally minimizing geodesics everywhere, none of the sphere's antipodal pathology; because $K\equiv +1 > 0$, Bonnet–Myers applies to $S^n$ unconditionally — compact, diameter exactly $\pi$. Both facts reappear as concrete computations in Parts II and III below, not separately-proved special cases.

### 9. The shared tool: conformal metrics

Both specializations use a metric of the form $g_x(u,w) = \lambda(x)^2\langle u,w\rangle$. Writing $\phi = \ln\lambda$, the general Christoffel formula collapses (for *any* conformal factor, in any dimension) to:

$$\Gamma^k_{ij} = \delta^k_i\, \partial_j\phi + \delta^k_j\, \partial_i\phi - \delta_{ij}\, \partial_k\phi$$

and, in 2D, the Gaussian curvature has the clean closed form:

$$K = -\frac{1}{\lambda^2}\,\Delta(\ln\lambda), \qquad \Delta = \partial_x^2 + \partial_y^2$$

Everything below is just choosing $\lambda$ and turning the crank.

---

## Part II — Spherical Geometry ($\kappa = +1$)

### 1. The model and the metric

Stereographic projection of the unit sphere $S^n$ onto $\mathbb{R}^n$ pulls back the round metric to a conformal metric on all of $\mathbb{R}^n$:

$$g_x(u,w) = \lambda(x)^2\langle u,w\rangle, \qquad \lambda(x) = \frac{2}{1+|x|^2}$$

Note the single sign flip from the hyperbolic case's $\lambda(x)=2/(1-|x|^2)$ — same construction, opposite sign in the denominator, and (as will follow) that one sign is the entire difference between a bounded, positively-curved space and an unbounded, negatively-curved one.

### 2. Christoffel symbols

$\phi = \ln 2 - \ln(1+|x|^2) \Rightarrow \partial_k\phi = -\dfrac{2x_k}{1+|x|^2}$. Substituting into the conformal formula from Part I:

$$\Gamma^k_{ij} = \frac{2}{1+|x|^2}\Big[\, \delta_{ij}\,x_k - \delta^k_i\,x_j - \delta^k_j\,x_i \,\Big]$$

— the same bracket as the hyperbolic case, sign-reversed, with $1-|x|^2 \to 1+|x|^2$.

### 3. Radial geodesics, the fast way

For $x(t) = r(t)e$ ($e$ a unit vector), constant Riemannian speed gives $\lambda(r)\,r' = C$, i.e.

$$\frac{2r'}{1+r^2} = C$$

Since $\frac{d}{dr}[\arctan r] = \frac{1}{1+r^2}$, this integrates directly: $\arctan(r(t)) = \tfrac{C}{2}t$, so with $c = C/2$:

$$r(t) = \tan(ct)$$

— the same derivation as the hyperbolic case, with $\tan$ in place of $\tanh$ because the separable ODE has $1+r^2$ instead of $1-r^2$ in the denominator.

### 4. The exponential map

$$\exp_0(v) = \tan(|v|)\,\frac{v}{|v|}, \qquad \text{valid for } |v| < \frac{\pi}{2}$$

This is where the sign flip has a real consequence: $\tan$ blows up at $ct = \pi/2$, unlike $\tanh$, which stays bounded forever. $\exp_0$ is only a valid (injective) chart for $|v| < \pi/2$ — beyond that, the geodesic has passed the antipodal point and the stereographic chart can no longer represent it. This is a concrete instance of the injectivity radius / conjugate point phenomenon from the general theory: the sphere's injectivity radius is *finite*.

### 5. Distance

Same constant-speed bookkeeping as before ($\lambda(r)r' = 2c$, confirmed constant) gives, from the origin:

$$d(0,x) = 2\arctan(|x|)$$

bounded above by $\pi$ as $|x|\to\infty$ — the sphere has finite diameter $\pi$, exactly the geodesic distance to the antipodal point. Via the same rotation/translation isometry-transitivity argument as the hyperbolic case, the general two-point distance (standard stereographic/chordal distance formula, stated here rather than re-derived symbol by symbol) is:

$$d(x,y) = \arccos\!\left( 1 - \frac{2|x-y|^2}{(1+|x|^2)(1+|y|^2)} \right)$$

(Consistency check: setting $x=0$ and using the double-angle identity $\cos(2\theta) = \frac{1-\tan^2\theta}{1+\tan^2\theta}$ with $\theta = \arctan|x|$ recovers $d(0,x)=2\arctan|x|$ exactly.)

### 6. Curvature check

With $\phi = \ln 2 - \ln(1+r^2)$ in 2D: $\Delta\phi = -4/(1+r^2)^2$, $\lambda^2 = 4/(1+r^2)^2$, so

$$K = -\frac{1}{\lambda^2}\Delta\phi = -\left[\frac{(1+r^2)^2}{4}\right]\left[\frac{-4}{(1+r^2)^2}\right] = +1$$

confirmed. Constant across the whole space (same maximal-symmetry argument as the hyperbolic case), and constant across dimension by Killing–Hopf.

### 7. Consequences already predicted by Part I

A geodesic circle of radius $r$ (unit sphere) has circumference $2\pi\sin(r)$ and encloses area $2\pi(1-\cos r)$ — both *less* than the flat-space formulas, exactly the $K>0$ sign-dictionary prediction from Part I §8. And since $K\equiv+1>0$ everywhere, Bonnet–Myers applies unconditionally: $M$ is compact with diameter $\le \pi/\sqrt{1}=\pi$ — not merely consistent with, but the *same fact as*, the $d(0,x)\to\pi$ bound already found in §5 by direct computation.

---

## Part III — Hyperbolic Geometry ($\kappa = -1$)

### 1. The model and the metric

$$g_x(u,w) = \lambda(x)^2\langle u,w\rangle, \qquad \lambda(x) = \frac{2}{1-|x|^2}, \qquad x \in B^n = \{x\in\mathbb R^n : |x|<1\}$$

the Poincaré ball model.

### 2. Christoffel symbols

$\phi = \ln 2 - \ln(1-|x|^2) \Rightarrow \partial_k\phi = \dfrac{2x_k}{1-|x|^2}$:

$$\Gamma^k_{ij} = \frac{2}{1-|x|^2}\Big[\, \delta^k_i\,x_j + \delta^k_j\,x_i - \delta_{ij}\,x_k \,\Big]$$

### 3. Radial geodesics, the fast way

Constant speed gives $\dfrac{2r'}{1-r^2} = C$. Since $\dfrac{d}{dr}[\operatorname{artanh} r] = \dfrac{1}{1-r^2}$:

$$r(t) = \tanh(ct), \qquad c = C/2$$

### 4. The exponential map

$$\exp_0(v) = \tanh(|v|)\,\frac{v}{|v|}, \qquad \text{valid for all } v \in \mathbb R^n$$

No boundary on $|v|$ this time — $\tanh$ never blows up, so $\exp_0$ is injective everywhere. This is the geometric reason hyperbolic space has *infinite* injectivity radius, the polar opposite of the spherical case above.

### 5. Distance

From the origin:

$$d(0,x) = 2\operatorname{artanh}(|x|)$$

unbounded as $|x|\to 1$ (the ball's boundary sits at infinite distance — a coordinate artifact, not a true edge, consistent with completeness). General two-point distance, via Möbius-addition isometries (stated, not re-derived symbol by symbol — pure algebra from the origin case):

$$x \oplus y = \frac{(1+2\langle x,y\rangle+|y|^2)\,x + (1-|x|^2)\,y}{1+2\langle x,y\rangle + |x|^2|y|^2}, \qquad d(x,y) = 2\operatorname{artanh}\big(|(-x)\oplus y|\big) = \operatorname{arcosh}\!\left(1 + \frac{2|x-y|^2}{(1-|x|^2)(1-|y|^2)}\right)$$

### 6. Curvature check

$$K = -\frac{1}{\lambda^2}\Delta\phi = -\left[\frac{(1-r^2)^2}{4}\right]\left[\frac{4}{(1-r^2)^2}\right] = -1$$

### 7. Consequences already predicted by Part I

A geodesic circle of radius $r$ has circumference $2\pi\sinh(r)$ and encloses area $2\pi(\cosh r-1)$ — both exponential in $r$, exactly the $K<0$ sign-dictionary prediction. And since $K\equiv-1\le0$ everywhere, Cartan–Hadamard applies unconditionally: $\exp_0$ is a global diffeomorphism $\mathbb R^n \to B^n$, every two points have a *unique*, globally minimizing geodesic, with no antipodal-point pathology of any kind — the direct opposite of the sphere's finite injectivity radius in Part II §4.

**Other models, briefly.** The Poincaré ball isn't the only chart for this space — the hyperboloid (Minkowski) model realizes $H^n$ as $\{x:\langle x,x\rangle=-1,\ x_0>0\}$ inside Minkowski space with the induced metric (isometry group literally $O(n,1)$, geodesics = intersections with 2-planes through the origin); the upper half-space model uses $g=(1/x_n^2)g_{\text{Eucl}}$ on $\{x_n>0\}$, also conformal; the Klein model uses the same ball but a *non*-conformal metric under which geodesics are literally straight Euclidean segments (useful for visual collinearity, at the cost of angle distortion). All four are isometric; the Poincaré ball is used throughout this doc because its conformal structure is what makes the Part I §9 toolkit apply directly.

### 8. Why this is usable in a training loop

A generic $n$-dimensional metric needs $O(n^3)$ Christoffel components per point — half a billion for $n=1000$. Here every component is a few arithmetic operations on $x$, and in practice you never even touch them: $\exp_0$ and $\operatorname{dist}$ are already the closed-form solutions. Optimization stays plain Euclidean Adam, because $\exp_0$ and $\operatorname{dist}$ are ordinary smooth maps between Euclidean spaces from the optimizer's point of view — the network's weights never leave $\mathbb R^N$, so no retraction or parallel transport is needed (the "wrapped parametrization" trick). Points get clipped away from $|x|=1$ purely because $\operatorname{artanh}$ and $\lambda(x)$ diverge there and floating point can't resolve it — a numerics safeguard, not a change to the geometry.

---

## Part IV — Side by side

$$
\begin{array}{l|l|l}
 & \textbf{Spherical } (\kappa=+1) & \textbf{Hyperbolic } (\kappa=-1) \\ \hline
\text{Metric} & \lambda(x)=\dfrac{2}{1+|x|^2} & \lambda(x)=\dfrac{2}{1-|x|^2} \\[8pt]
\text{Christoffel} & \dfrac{2}{1+|x|^2}\big[\delta_{ij}x_k-\delta^k_ix_j-\delta^k_jx_i\big] & \dfrac{2}{1-|x|^2}\big[\delta^k_ix_j+\delta^k_jx_i-\delta_{ij}x_k\big] \\[8pt]
\text{Radial geodesic} & r(t)=\tan(ct) & r(t)=\tanh(ct) \\[6pt]
\exp_0(v) & \tan(|v|)\,v/|v|,\ \ |v|<\pi/2 & \tanh(|v|)\,v/|v|,\ \ \text{all } v \\[6pt]
d(0,x) & 2\arctan(|x|) \le \pi & 2\operatorname{artanh}(|x|),\ \text{unbounded} \\[6pt]
\text{Diameter} & \pi \ (\text{finite}) & \infty \\[6pt]
\text{Injectivity radius} & \pi \ (\text{finite}) & \infty \\[6pt]
\text{Curvature } K & +1 & -1
\end{array}
$$

The entire spherical/hyperbolic distinction, mechanically, comes down to one sign in $\lambda(x)$'s denominator — and that single sign is what turns a bounded, closed space with a finite cut locus into an unbounded space with none.

---

## Appendix — Gromov $\delta$-hyperbolicity (the practical diagnostic)

A metric-space-only tool (no manifold structure required), used to check whether a learned embedding's geometry is worth exploiting before committing to it. For four points $w,x,y,z$, form the three pairwise-distance sums, sort them $S_{(1)}\le S_{(2)}\le S_{(3)}$, and define

$$\delta(w,x,y,z) = \frac{S_{(3)} - S_{(2)}}{2}, \qquad \delta = \sup_{w,x,y,z} \delta(w,x,y,z), \qquad \operatorname{rel\_delta} = \frac{2\delta}{\operatorname{diam}(X)} \in [0,1]$$

$\operatorname{rel\_delta}=0$ exactly characterizes metric trees; complete simply connected manifolds with curvature bounded above by a negative constant (including hyperbolic space) are Gromov hyperbolic with a *finite* $\delta$, independent of scale — the source of the "thin triangles" phenomenon. Exact computation is $O(n^4)$; in practice, $\delta$ is estimated by sampling tens of thousands of random quadruples and taking the sample maximum or a high percentile, which is always a lower bound on the true value. Applied to the ball-in-cup encoder's 32-dim latent space (Euclidean distance, 20 held-out episodes, 4020 points, 50,000 sampled quadruples): diameter $10.57$, $\operatorname{rel\_delta} \approx 0.15$–$0.27$ — low enough to indicate the latent space already organizes tree-like, without anything forcing it to, which is the empirical justification for the hyperbolic rebuild in the roadmap doc.
