# Riemannian Geometry to Hyperbolic Geometry

2026-09-17 · @Andreas Atle

## 1. Purpose and Motivation

Euclidean geometry works because $R³$ comes with a built-in, global way to measure distance and angle: the standard inner product, the same at every point. Riemannian geometry exists because most of the spaces you actually want to do geometry on — curved surfaces, the configuration space of a mechanical system, spacetime, statistical manifolds, the space of positive-definite matrices — do not have that luxury. There is no single global coordinate chart in which "distance" looks like Pythagoras. What they do have, locally, is a notion of length and angle at each point, varying smoothly from point to point. A Riemannian manifold is exactly the object that packages this: a smooth manifold $M$ plus a smoothly varying inner product $g_p$ on each tangent space $T_pM$.

The purpose of the theory is to redo everything you know from vector calculus and Euclidean geometry — lengths, angles, straight lines, parallelism, curvature, volumes, the Laplacian — using only $g$ and the manifold's smooth structure, with no reference to any ambient Euclidean space the manifold might happen to sit inside. This is the difference between **extrinsic** geometry (how a surface bends inside $R³$, à la classical differential geometry of surfaces) and **intrinsic** geometry (what an inhabitant confined to the surface, measuring only with a ruler and protractor that live in the surface, could ever discover). Gauss's Theorema Egregium — that Gaussian curvature of a surface is computable from the metric alone, without reference to the embedding — is the historical seed of the whole subject: it says curvature is an intrinsic fact, not an artifact of how you happened to embed the space.

Once the machinery is in place, it does three things for you. First, it tells you what a "straight line" means when there is no ambient flat space to draw one in — the geodesic. Second, it gives you a precise, coordinate-free measure of how far the space is from being flat — the curvature tensor — and shows that this single local invariant controls global qualitative behavior (whether geodesics converge or diverge, whether the space can close up on itself, how volume grows with radius). Third, it lets you transport the entire toolkit of analysis — gradients, divergence, the Laplacian, integration by parts — onto curved spaces.

Hyperbolic geometry is where this machinery becomes vivid rather than abstract. It is the unique (up to scale and isometry) simply connected Riemannian manifold of constant negative curvature, and constant negative curvature is the regime where every one of the "different from $R³$" phenomena below — divergent geodesics, exponential volume growth, non-uniqueness of parallel lines — is maximally, rather than marginally, true. Building up the general theory first and specializing at the end (rather than starting from hyperbolic geometry's classical axiomatic presentation) is the right order: every fact about $Hⁿ$ below is a specific instance of a general Riemannian theorem, not a fact that needs its own separate proof.

## 2. Manifolds and the Metric Tensor

A smooth $n$-manifold $M$ is a space that locally looks like $Rⁿ$: it is covered by charts mapping open sets of $M$ to $Rⁿ$ with smooth transition maps on overlaps. At each point $p$ there is a tangent space $T_pM$, an $n$-dimensional vector space of "directions you can move in at $p$" — formally, equivalence classes of curves through $p$, or equivalently, derivations at $p$. In local coordinates $(x¹,...,xⁿ)$, $T_pM$ has basis $∂/∂x¹, ..., ∂/∂xⁿ$. Unlike $Rⁿ$, where every tangent space is canonically identified with $Rⁿ$ itself, $T_pM$ and $T_qM$ for $p ≠ q$ are a priori unrelated vector spaces — there is no god-given way to compare a vector at $p$ to a vector at $q$. This single fact is the source of almost everything that makes Riemannian geometry different from vector calculus, and it is worth sitting with before moving on.

A Riemannian metric $g$ is an assignment, to each point $p$, of a symmetric positive-definite bilinear form $g_p$ on $T_pM × T_pM$, varying smoothly in $p$. In local coordinates $g_p(∂_i, ∂_j) = g_{ij}(p)$, a symmetric positive-definite matrix depending smoothly on the point, and for tangent vectors $v = vⁱ ∂_i, w = wʲ ∂_j$:

$$
g_p(v, w) = g_{ij}(p) · v^i w^j
$$

Everything metric is built from $g_ij$. The length of a tangent vector is $|v| = √(g_p(v,v))$. The length of a curve $γ(t)$, $t ∈ [a,b]$, is:

$$
L(γ) = ∫ₐᵇ √( g(γ̇(t), γ̇(t)) ) dt
$$

The angle between two curves crossing at $p$ is recovered from $g_p$ exactly as in Euclidean space; the Riemannian distance between two points is $d(p,q)$ = inf over $γ$ of $L(γ)$, the infimum over curves joining them; and the metric determines a volume form $dV_g = √(\text{det} g_{ij}) dx¹∧...∧dxⁿ$, which is what makes integration on $M$ meaningful.

Two examples anchor the definition. On $Rⁿ$ itself, $g_ij = δ_ij$ (the identity matrix) in Cartesian coordinates, constant everywhere — this is the flat, Euclidean metric, and it is the special case where all the machinery below trivializes. On the round sphere $S²$ of radius $r$, using spherical coordinates $(θ,φ)$:

$$
g = r² dθ² + r² sin²θ dφ²
$$

visibly not constant, and visibly encoding the fact that circles of latitude shrink near the poles. The metric is the **only** data a Riemannian manifold carries beyond its smooth structure; every geometric notion from here on — geodesic, curvature, volume — is defined purely in terms of $g$.

## 3. What Is Genuinely Different from $R³$

It's worth being precise about which features of $R³$ are special to it, because the general theory is built by figuring out how to live without them.

**No canonical way to compare tangent spaces.** In $R³$, "the vector $(1,0,0)$" means the same thing regardless of which point you attach it to, because all tangent spaces are canonically identified with $R³$ via translation. On a general manifold, $T_pM$ and $T_qM$ are different vector spaces with no preferred isomorphism between them. This means the ordinary directional derivative of a vector field — the limit of a difference quotient $(V(γ(t)) − V(γ(0))) / t$ — doesn't even parse: you cannot subtract a vector living in $T_γ(t)M$ from one living in $T_γ(0)M$. Fixing this requires new structure — a *connection* — which is entirely absent as a concept in elementary $R³$ calculus because it's trivially supplied for free there.

**No global flat coordinates.** $R³$ has Cartesian coordinates in which $g_ij = δ_ij$ everywhere, simultaneously, globally. On a curved manifold you can always find coordinates in which $g_ij(p) = δ_ij$ at a single point $p$ (normal coordinates), and even arrange the first derivatives $∂_k g_ij$ to vanish there — but generically the second derivatives cannot be killed. That obstruction is exactly the curvature tensor. So flatness at a point is free (it's just a choice of coordinates); flatness in a neighborhood is not — it's the actual geometric content of the space being $R³$ and not something else.

**Parallel transport is path-dependent.** In $R³$, "moving a vector without rotating it" (parallel transport) is trivial and path-independent — slide it along, keep the components constant. On a curved manifold, parallel-transporting a vector around a closed loop generally returns a different vector than you started with (this is holonomy). Transport a vector around a spherical triangle on $S²$ and it comes back rotated by an angle equal to the triangle's excess area times curvature. This path-dependence is not a technical nuisance; it is one of the two or three cleanest operational definitions of what curvature means.

**Straight lines bifurcate into two different things.** In $R³$, "the line through $p$ in direction $v$", "the shortest path between two points", and "the locally length-critical path" are all the same object. On a curved manifold, geodesics (locally length-critical paths — the correct generalization of "straight") need not be globally shortest (great circles on a sphere stop minimizing past the antipodal point), and uniqueness of the geodesic between two far-apart points can fail entirely.

**Distances and volumes stop obeying flat-space formulas.** The circumference of a circle of radius $r$ is $2πr$ only in the flat case; on a positively curved space it is smaller, and on a negatively curved space (as you'll see with hyperbolic space) it grows faster than linearly in $r$ — exponentially, in fact. Every one of these is a symptom of the same underlying cause, made precise in Sections 4–6: curvature.

## 4. Connections and Covariant Differentiation

To differentiate a vector field along a curve, we need a rule for comparing tangent vectors at infinitesimally nearby points. This rule is called an (affine) connection $∇$. It takes a vector field $Y$ and a direction $X$ and produces a new vector field $∇_X Y$, satisfying the properties you'd want a derivative to have: linearity in $Y$, a Leibniz rule $∇_X(fY) = X(f)Y + f∇_X Y$, and function-linearity in $X$, meaning $∇_X Y$ at $p$ only depends on $X_p$, not on how $X$ extends nearby. Along a curve $γ(t)$, this gives the covariant derivative $DV/dt = ∇_γ̇ V$ of a vector field $V(t) ∈ T_γ(t)M$ defined along $γ$ — the correctly generalized notion of "rate of change," replacing the naive difference quotient that failed in Section 3.

A connection by itself is extra structure not determined by the metric — but the **Fundamental Theorem of Riemannian Geometry** says that among all possible connections, exactly one is compatible with $g$ in the right way: there is a unique connection $∇$ (the Levi-Civita connection) that is torsion-free and metric-compatible:

$$
\text{torsion-free}: \quad       ∇_X Y − ∇_Y X = [X, Y] \\
\text{metric-compatible}: \quad  X g(Y,Z) = g(∇_X Y, Z) + g(Y, ∇_X Z) \quad (i.e. ∇g = 0)
$$

Metric compatibility is what makes this the geometrically correct choice: it guarantees that parallel transport preserves lengths and angles, exactly as rigid transport does in $R³$. In local coordinates, $∇$ is encoded by the Christoffel symbols $Γ^k_ij$, computed directly from the metric and its first derivatives:

$$
Γ^k_{ij} = (1/2) g^{kl} ( ∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij} )
$$

and covariant differentiation of a vector field $V = V^k ∂_k$ in the direction $∂_i$ reads $∇_i V^k = ∂_i V^k + Γ^k_ij V^j$ — the extra $Γ$ term is precisely the correction that ordinary calculus is missing on a curved space, and it vanishes identically when $g_ij$ is constant, i.e. in flat Euclidean coordinates.

A vector field $V(t)$ along $γ$ is **parallel** if $DV/dt ≡ 0$; given any vector $v₀ ∈ T_γ(0)M$ there is a unique parallel field along $γ$ with $V(0) = v₀$ (a linear ODE, so existence and uniqueness are automatic), giving the **parallel transport map** $P_γ : T_γ(0)M → T_γ(1)M$. This is finally the rigorous replacement for "sliding a vector along without rotating it": it is an isometry of tangent spaces (metric compatibility), it is the identity when $M = Rⁿ$ and $γ$ is a straight line with the flat connection, and — critically, and unlike in $Rⁿ$ — it depends on the path $γ$, not just its endpoints, whenever the region enclosed has nonzero curvature.

## 5. Geodesics

A geodesic is a curve $γ(t)$ whose own velocity is parallel along itself: $Dγ̇/dt = 0$. This is the covariant-derivative-aware version of "constant velocity, no acceleration," and in local coordinates it expands, using the Christoffel symbols from Section 4, into the **geodesic equation**:

$$
γ̈^k + Γ^k_{ij} γ̇^i γ̇^j = 0,        k = 1, ..., n
$$

This is a second-order ODE, so by standard existence/uniqueness theory, for every point $p$ and vector $v ∈ T_pM$ there is a unique geodesic $γ_p,v(t)$ with $γ(0) = p$, $γ̇(0) = v$, defined for $t$ in some interval around $0$. Collecting these into a single map gives the **exponential map** $exp_p : T_pM → M$, $exp_p(v) = γ_p,v(1)$ — the single most important construction in the subject, because it is the precise sense in which "shoot a straight line from $p$ in direction $v$" makes sense on a curved manifold. When $M = Rⁿ$ with the flat metric, $exp_p(v) = p + v$ exactly, recovering ordinary straight lines; $exp_p$ is the correct nonlinear generalization.

Geodesics are the critical points of the **energy functional**:

$$
E(γ) = (1/2) ∫ₐᵇ g(γ̇, γ̇) dt
$$

among curves with fixed endpoints (this is the first-variation calculation that derives the geodesic equation in the first place, and it is the cleanest way to see that geodesics generalize "shortest path"). But — and this is a point worth being very deliberate about, because it's exactly where curved geometry parts ways with flat intuition — a geodesic is only **locally** length-minimizing in general, not globally. On the round sphere, great circles are geodesics, and an arc of a great circle minimizes length between its endpoints only up to the antipodal point; beyond that, the "short way around" is shorter. So: geodesic ≠ globally shortest path, in general; geodesic = locally shortest path, always (for the Levi-Civita connection).

A manifold is **geodesically complete** if every geodesic extends for all $t ∈ R$ (equivalently, $exp_p$ is defined on the whole of $T_pM$ for every $p$) — this rules out pathologies like $R²$ with a point removed, where a geodesic aimed at the missing point simply runs out of manifold to be defined on. Completeness is the right finiteness hypothesis for almost every global theorem in the subject (Section 7), and both $Rⁿ$ and hyperbolic space $Hⁿ$ are complete, so this hypothesis will be satisfied for free once we get there.

## 6. Curvature: Measuring the Failure of Flatness

Curvature is defined by how badly covariant derivatives fail to commute. The **Riemann curvature tensor** is:

$$
R(X,Y)Z = ∇_X∇_Y Z − ∇_Y∇_X Z − ∇_{[X,Y]} Z
$$

$R ≡ 0$ exactly characterizes local flatness — exactly when $M$ admits, around every point, coordinates in which $g_ij$ is the constant identity matrix. So $R$ is precisely the obstruction identified informally in Section 3: the thing that keeps second derivatives of the metric from being killable by a coordinate change. Two equivalent operational readings of $R$ are worth keeping alongside the formula: $R(X,Y)Z$ measures the discrepancy in parallel-transporting $Z$ around an infinitesimal loop spanned by $X,Y$ (link to the holonomy discussion in Section 3), and, via the Jacobi equation, it directly governs whether initially parallel geodesics converge or diverge.

The full 4-index tensor $R$ is unwieldy to reason about directly, so it's usually summarized by contractions. The most geometrically transparent is **sectional curvature**: for a 2-plane $σ ⊂ T_pM$ spanned by orthonormal $u, v$:

$$
K(σ) = g( R(u,v)v, u )
$$

which is exactly the Gaussian curvature of the little 2-dimensional surface swept out by geodesics through $p$ tangent to $σ$ — so sectional curvature is literally "ordinary surface curvature, measured on every 2-plane through every point," and it determines $R$ completely (knowing $K(σ)$ for all $σ$ recovers the whole tensor). Averaging over planes gives **Ricci curvature**, $Ric(v,v) = sum over i of K(span(v,e_i))$ (trace of $R$ over one pair of slots), and averaging again gives the **scalar curvature** $S = tr(Ric)$, a single number at each point.

The sign of curvature has direct, memorable geometric meaning, and this is the fact that everything downstream (Section 7, then hyperbolic geometry) hangs on:

- $K > 0$ (spheres): initially parallel geodesics converge and cross; geodesic triangles have angle sum $> π$; circles of radius $r$ have circumference $< 2πr$; the space tends to be compact.
- $K = 0$ (flat, $Rⁿ$): geodesics that start parallel stay parallel; angle sum $= π$; circumference $= 2πr$ exactly.
- $K < 0$ (hyperbolic): initially parallel geodesics diverge, at an exponential rate; geodesic triangles have angle sum $< π$; circles of radius $r$ have circumference growing like $sinh(r)$ — exponentially in $r$, not linearly. This last fact — exponential rather than polynomial volume growth — is the single most consequential feature of negative curvature, and it's the reason hyperbolic space looks locally tree-like and cannot be isometrically embedded in any $R^N$ globally the way a sphere can.

## 7. Main Structural Results

Three theorems do most of the work of turning the local, infinitesimal data ($g$, then $∇$, then $R$) into global, qualitative statements about the shape of the space. All three assume completeness (Section 5) — without it, they're false.

**Hopf–Rinow.** For a connected Riemannian manifold, the following are equivalent: $(M,d)$ is a complete metric space; $M$ is geodesically complete; closed bounded subsets of $M$ are compact. And if $M$ is complete, any two points are joined by **at least one** minimizing geodesic (a geodesic whose length realizes the distance $d(p,q)$) — existence of shortest paths, which is not automatic on an incomplete space (think of two points in $R²$ minus a segment between them). This theorem is the justification for treating completeness as the standing hypothesis everywhere else: it converts an analytic condition (Cauchy sequences converge) into a geometric one (geodesics don't run out) and hands you existence of minimizers for free.

**Gauss–Bonnet** (the 2-dimensional case, historically first): for a closed surface $M$:

$$
∫_M K dA = 2π · χ(M)
$$

where $χ(M)$ is the Euler characteristic, a topological invariant. This is the prototype for an entire family of results relating curvature (analytic/geometric, local) to topology (global, coordinate-free) — it's why a torus ($χ = 0$) admits a flat metric but a sphere ($χ = 2$) provably cannot, no matter how you try to deform it.

**Cartan–Hadamard.** If $M$ is complete, simply connected, and has sectional curvature $K ≤ 0$ everywhere, then $exp_p : T_pM → M$ is a diffeomorphism for every $p$ — in particular $M$ is diffeomorphic to $Rⁿ$, any two points are joined by a **unique** geodesic, and that geodesic is globally minimizing (no "short way around" pathology from Section 5 is possible). This is the theorem that makes nonpositively curved spaces tractable, and it is the direct bridge to the next section: it says that as long as curvature stays $≤ 0$, geodesics behave exactly the way flat-space intuition wants them to (unique connecting line, globally shortest), while everything quantitative about lengths, angles, and volume growth still reflects the negative curvature. (For contrast: Bonnet–Myers shows that a **positive** curvature lower bound $K ≥ κ > 0$ forces $M$ to be compact with bounded diameter — the opposite global regime.)

## 8. From Constant Curvature to Hyperbolic Geometry

A **space form** is a Riemannian manifold of **constant** sectional curvature: $K(σ) = κ$ for every 2-plane $σ$ at every point, a single number. This is the most symmetric case the general theory admits (the isometry group has the largest possible dimension for an $n$-manifold). The **Killing–Hopf theorem** classifies these completely: every complete, simply connected space form of curvature $κ$ is isometric to exactly one of three models, up to rescaling the metric to fix $κ = +1, 0,$ or $−1$:

$$
κ > 0:  S^n        κ = 0:  R^n        κ < 0:  H^n
$$

Every other complete space form (constant curvature but not simply connected) is a quotient of one of these three by a discrete group of isometries acting freely and properly discontinuously — e.g. flat tori are quotients of $Rⁿ$. So the entire classification of constant-curvature geometry reduces to understanding these three model spaces and their quotients, and hyperbolic space $Hⁿ$ is, by definition, the $κ = −1$ member of this list: **the** complete, simply connected Riemannian $n$-manifold of constant sectional curvature $−1$.

Because $K ≡ −1 ≤ 0$, Cartan–Hadamard (Section 7) applies to $Hⁿ$ unconditionally: $exp_p$ is a global diffeomorphism $T_pHⁿ ≅ Rⁿ → Hⁿ$, any two points have a unique connecting geodesic, and it's globally minimizing — geodesics in hyperbolic space have none of the sphere's antipodal-point pathology from Section 5. And because $K$ is not just $≤ 0$ but exactly $−1$, all the exponential phenomena from Section 6 are exact, not just asymptotic: geodesic triangles have angle-sum defect exactly proportional to area (a sharp Gauss–Bonnet), and a circle of geodesic radius $r$ has circumference exactly $2π sinh(r)$ and encloses area exactly $2π(cosh(r) − 1)$ — both genuinely exponential in $r$, in contrast to the linear/quadratic flat-space formulas.

$Hⁿ$ has no canonical coordinate model the way $Rⁿ$ has Cartesian coordinates — instead there are several standard, isometric realizations, each useful for different computations:

- **Hyperboloid (Minkowski) model:** the upper sheet of the two-sheeted hyperboloid ${x : ⟨x,x⟩ = −1, x₀ > 0}$ in Minkowski space $R^(n,1)$ (signature $(n,1)$), with the metric induced from the Minkowski form. This is the most algebraically transparent model — the isometry group is literally the Lorentz group $O(n,1)$ — and geodesics are exactly the intersections of the hyperboloid with 2-planes through the origin.
- **Poincaré ball model:** the open unit ball $Bⁿ ⊂ Rⁿ$ with the conformal metric $g = 4/(1−|x|²)² · g_Eucl$. Angles agree with the Euclidean angles you'd naively measure (conformal), but lengths blow up near the boundary sphere, which sits at infinite distance. Geodesics are diameters of the ball and circular arcs meeting the boundary sphere at right angles.
- **Poincaré upper half-space model:** ${x ∈ Rⁿ : x_n > 0}$ with $g = (1/x_n²) · g_Eucl$, also conformal; geodesics are vertical half-lines and semicircles centered on the boundary hyperplane ${x_n = 0}$.
- **Klein (Beltrami–Klein) model:** the open unit ball with a *non*-conformal metric, chosen so that geodesics are literally straight Euclidean line segments — useful for visually deciding whether points are collinear, at the cost of angles being distorted.

All four are isometric to each other and to the abstract "constant curvature $−1$" space form; which one to compute in is a matter of convenience, exactly the way choosing Cartesian vs. polar coordinates is a matter of convenience in $R²$. That every fact above — unique minimizing geodesics, exponential circumference growth, the specific $sinh$/$cosh$ formulas — is a direct instance of general Riemannian theorems (Cartan–Hadamard, the sectional curvature sign dictionary, Gauss–Bonnet) rather than a separately-proved fact about hyperbolic geometry specifically, is exactly the payoff promised in Section 1 for building the general theory first.
