# The Geodesic Equation From the Calculus of Variations

The tutorial doc and the hyperbolic-geometry derivation doc both *use* the geodesic equation $\frac{d^2 x^k}{dt^2} + \Gamma^k_{ij} \frac{dx^i}{dt}\frac{dx^j}{dt} = 0$ as a starting point. This doc derives *why* that's the right equation — starting only from "a geodesic extremizes length" and the calculus of variations, showing that the Christoffel-symbol combination is forced out by the algebra rather than assumed. Indices $i,j,k,l,m$ run over $1,\ldots,n$; repeated upper/lower index pairs are summed (Einstein convention), matching the tutorial doc's conventions.

## 1. The length functional, and a shortcut

For a curve $x(t)$ on the manifold, its length is

$$
L[x] = \int_a^b \sqrt{g_{ij}(x) \frac{dx^i}{dt}\frac{dx^j}{dt}} dt
$$

This is the direct, geometrically meaningful quantity — but the square root makes the calculus of variations messy, and length is unchanged by reparametrizing the curve, which means the length-minimizing *curve* doesn't pick out a unique speed law along it. Both problems disappear if we extremize the **energy** functional instead:

$$
E[x] = \int_a^b \frac{1}{2} g_{ij}(x) \frac{dx^i}{dt}\frac{dx^j}{dt} dt
$$

The reason this replacement works is an exact identity. Write the Riemannian speed as $v(t)=\lVert\dot{x}(t)\rVert_g$ and fix the travel time $T=b-a>0$. Since $L[x]=\int_a^b v(t)\,dt$ and $E[x]=\tfrac12\int_a^b v(t)^2\,dt$,

$$
\boxed{
E[x]=\frac{L[x]^2}{2T}
+\frac12\int_a^b\left(v(t)-\frac{L[x]}{T}\right)^2\,dt.
}
$$

To check it, expand the square and use $\int_a^b v(t)\,dt=L[x]$:

$$
\frac12\int_a^b\left(v(t)-\frac{L[x]}{T}\right)^2\,dt
=E[x]-\frac{L[x]^2}{T}+\frac{L[x]^2}{2T}
=E[x]-\frac{L[x]^2}{2T}.
$$

The first term in the boxed identity depends only on the path's length; the second measures how unevenly its speed differs from the average speed $L[x]/T$. It is nonnegative and vanishes exactly for constant speed. Thus $E[x]\geq L[x]^2/(2T)$, the same bound given by Cauchy–Schwarz, with its equality condition now explicit.

Every regular path can be reparametrized at constant speed without changing its length. For that fixed path, the smallest energy is therefore $L[x]^2/(2T)$. Minimizing energy over paths with fixed endpoints and fixed travel time consequently selects the same geometric paths as minimizing length, together with their constant-speed parametrization. This is why dropping the square root is legitimate for the minimization problem.

The Euler-Lagrange equation below imposes stationarity, which is necessary for a smooth minimum but does not guarantee a global minimum. Its solutions here are affinely parametrized geodesics, with constant Riemannian speed.

## 2. The Euler-Lagrange equation

Write the integrand as a Lagrangian depending on position and velocity, $\mathcal{L}(x, \dot{x}) = \frac{1}{2} g_{ij}(x) \dot{x}^i \dot{x}^j$. A curve extremizing $\int \mathcal{L}\,dt$ with fixed endpoints must satisfy, for every index $s$:

$$
\frac{\partial \mathcal{L}}{\partial x^s} - \frac{d}{dt} [ \frac{\partial \mathcal{L}}{\partial \dot{x}^s} ] = 0
$$

This follows by perturbing the curve while holding its endpoints fixed, differentiating the functional, and integrating by parts; [Appendix A](#appendix-a-deriving-the-euler-lagrange-equation) gives the derivation. Everything below works out what this equation says for this specific $\mathcal{L}$.

## 3. Term A: the position derivative

The velocities $\dot{x}^i$ don't depend on position, but the metric components $g_{ij}(x)$ do, so only the metric is differentiated:

$$
\frac{\partial \mathcal{L}}{\partial x^s} = \frac{1}{2} (\frac{\partial g_{ij}}{\partial x^s}) \dot{x}^i \dot{x}^j = \frac{1}{2} g_{ij,s} \dot{x}^i \dot{x}^j
$$

(using the shorthand $g_{ij,s} = \frac{\partial g_{ij}}{\partial x^s}$).

## 4. Term B: the velocity derivative, then its time derivative

First differentiate with respect to velocity. Only the two factors matching index $s$ survive:

$$
\frac{\partial \mathcal{L}}{\partial \dot{x}^s} = \frac{1}{2} g_{ij} ( \delta^i_s \dot{x}^j + \dot{x}^i \delta^j_s ) = \frac{1}{2}( g_{sj} \dot{x}^j + g_{is} \dot{x}^i )
$$

Since $g$ is symmetric ($g_{sj} = g_{js}$), both terms are the same sum relabeled, so

$$
\frac{\partial \mathcal{L}}{\partial \dot{x}^s} = g_{sj} \dot{x}^j
$$

Now differentiate this in $t$. Because $g_{sj}$ depends on $x(t)$, the chain rule brings in a second term:

$$
\frac{d}{dt} ( g_{sj} \dot{x}^j ) = g_{sj} \ddot{x}^j + g_{sj,m} \dot{x}^m \dot{x}^j
$$

## 5. Assembling and symmetrizing

Substitute Terms A and B into the Euler-Lagrange equation:

$$
\frac{1}{2} g_{ij,s} \dot{x}^i \dot{x}^j - g_{sj} \ddot{x}^j - g_{sj,m} \dot{x}^m \dot{x}^j = 0
$$

Rearrange, and split the last term into two equal halves so everything can be grouped under one symmetric expression in $i,j$:

$$
g_{sj,m} \dot{x}^m \dot{x}^j = \frac{1}{2} g_{sj,m} \dot{x}^m \dot{x}^j + \frac{1}{2} g_{sm,j} \dot{x}^m \dot{x}^j
$$

giving

$$
g_{sj} \ddot{x}^j + \frac{1}{2}( g_{sj,m} + g_{sm,j} - g_{mj,s} ) \dot{x}^m \dot{x}^j = 0
$$

## 6. Isolating the acceleration: Christoffel symbols emerge

Multiply through by the inverse metric $g^{ks}$ (which satisfies $g^{ks} g_{sj} = \delta^k_j$) to free up the acceleration term:

$$
\ddot{x}^k + \frac{1}{2} g^{ks} ( g_{sj,m} + g_{sm,j} - g_{mj,s} ) \dot{x}^m \dot{x}^j = 0
$$

The bracketed expression, by definition, *is* the Christoffel symbol of the second kind:

$$
\Gamma^k_{mj} = \frac{1}{2} g^{ks} ( g_{sj,m} + g_{sm,j} - g_{mj,s} )
$$

Nothing was assumed about this combination in advance — it is exactly the object the Euler-Lagrange equation produces when you extremize a metric-quadratic Lagrangian. That's the sense in which the Christoffel symbol formula is *derived* here rather than handed down.

## 7. The geodesic equation

Substituting back:

$$
\frac{d^2 x^k}{dt^2} + \Gamma^k_{mj} \frac{dx^m}{dt}\frac{dx^j}{dt} = 0
$$

— the same equation used throughout the tutorial doc and the hyperbolic-geometry doc, now obtained from "extremize length" rather than assumed.

## Fundamental vs. convenience

Fundamental: that extremizing any Lagrangian of the form $\frac{1}{2} g_{ij} \dot{x}^i \dot{x}^j$ under Euler-Lagrange necessarily produces the symmetric combination $g_{sj,m} + g_{sm,j} - g_{mj,s}$ — this is forced by the algebra, not a modeling choice, and it's the reason the Christoffel symbol has exactly that formula and no other. Convenience: extremizing energy instead of length — it changes nothing about which curves are geodesics, only which parametrization of them you get directly (constant-speed, for free) and how easy the calculus is along the way.

## Reference: the full derivation, start to finish

**Length functional**

$$
L[x] = \int_a^b \sqrt{g_{ij} \dot{x}^i \dot{x}^j}\,dt
$$

**Energy functional** (same minimizers, constant speed)

$$
E[x] = \int_a^b \frac{1}{2} g_{ij} \dot{x}^i \dot{x}^j\,dt
$$

**Euler-Lagrange**

$$
\frac{\partial \mathcal{L}}{\partial x^s} - \frac{d}{dt}[ \frac{\partial \mathcal{L}}{\partial \dot{x}^s} ] = 0,\qquad \mathcal{L} = \frac{1}{2} g_{ij} \dot{x}^i \dot{x}^j
$$

**Term A**

$$
\frac{\partial \mathcal{L}}{\partial x^s} = \frac{1}{2} g_{ij,s} \dot{x}^i \dot{x}^j
$$

**Term B**

$$
\frac{\partial \mathcal{L}}{\partial \dot{x}^s} = g_{sj} \dot{x}^j
$$

$$
\frac{d}{dt}[ \frac{\partial \mathcal{L}}{\partial \dot{x}^s} ] = g_{sj} \ddot{x}^j + g_{sj,m} \dot{x}^m \dot{x}^j
$$

**Assemble + symmetrize**

$$
g_{sj} \ddot{x}^j + \frac{1}{2}( g_{sj,m} + g_{sm,j} - g_{mj,s} ) \dot{x}^m \dot{x}^j = 0
$$

**Raise index with $g^{ks}$**

$$
\ddot{x}^k + \frac{1}{2} g^{ks}( g_{sj,m} + g_{sm,j} - g_{mj,s} ) \dot{x}^m \dot{x}^j = 0
$$

**Christoffel symbol**

$$
\Gamma^k_{mj} = \frac{1}{2} g^{ks} ( g_{sj,m} + g_{sm,j} - g_{mj,s} )
$$

**Geodesic equation**

$$
\frac{d^2 x^k}{dt^2} + \Gamma^k_{mj} \frac{dx^m}{dt}\frac{dx^j}{dt} = 0
$$

## Appendix A: Deriving the Euler-Lagrange equation

Start with a general functional

$$
S[x]=\int_a^b \mathcal{L}(t,x(t),\dot{x}(t))\,dt.
$$

Here $S$ names the functional, $\mathcal{L}$ its integrand, and $L[x]$ still denotes length. Assume $\mathcal{L}$ and the curve are sufficiently smooth for the derivatives below (twice continuously differentiable is enough). Work in a coordinate chart; the argument is local, so it also applies along a curve that requires several charts. The explicit $t$ argument allows a time-dependent Lagrangian, although the metric Lagrangian in section 2 has no explicit time dependence.

**1. Vary the curve, keeping the endpoints fixed.**

Choose any smooth vector-valued perturbation $\eta(t)$ with $\eta(a)=\eta(b)=0$, and form a one-parameter family of nearby curves:

$$
x_\varepsilon^s(t)=x^s(t)+\varepsilon\eta^s(t),\qquad
\dot{x}_\varepsilon^s(t)=\dot{x}^s(t)+\varepsilon\dot{\eta}^s(t).
$$

For sufficiently small $\varepsilon$, these are admissible nearby curves in the chart. If $x$ is a local minimum or maximum of $S$, then the ordinary function $\varepsilon\mapsto S[x_\varepsilon]$ has an extremum at zero. Its derivative must therefore vanish for every allowed $\eta$. This derivative is called the *first variation*:

$$
\delta S[x;\eta]
:=\left.\frac{d}{d\varepsilon}S[x_\varepsilon]\right|_{\varepsilon=0}
=0.
$$

**2. Differentiate under the integral.**

Apply the chain rule to the position and velocity arguments of $\mathcal{L}$, keeping $t$ fixed:

$$
\delta S[x;\eta]
=\int_a^b\left(
\frac{\partial\mathcal{L}}{\partial x^s}\eta^s
+\frac{\partial\mathcal{L}}{\partial\dot{x}^s}\dot{\eta}^s
\right)\,dt.
$$

The index $s$ is summed here, and every derivative of $\mathcal{L}$ is evaluated along the original curve $(t,x(t),\dot{x}(t))$. Position and velocity are independent arguments when taking these partial derivatives; their variations along a perturbed curve are related by $\delta\dot{x}=d(\delta x)/dt$.

**3. Integrate the velocity term by parts.**

We want both terms to multiply $\eta^s$, so move the time derivative off $\dot{\eta}^s$:

$$
\int_a^b\frac{\partial\mathcal{L}}{\partial\dot{x}^s}\dot{\eta}^s\,dt
=\left[\frac{\partial\mathcal{L}}{\partial\dot{x}^s}\eta^s\right]_a^b
-\int_a^b\frac{d}{dt}\left(\frac{\partial\mathcal{L}}{\partial\dot{x}^s}\right)\eta^s\,dt.
$$

The boundary term vanishes because $\eta(a)=\eta(b)=0$. Thus stationarity becomes

$$
0=\delta S[x;\eta]
=\int_a^b\left[
\frac{\partial\mathcal{L}}{\partial x^s}
-\frac{d}{dt}\left(\frac{\partial\mathcal{L}}{\partial\dot{x}^s}\right)
\right]\eta^s\,dt
\qquad\text{for every allowed }\eta.
$$

**4. Use the freedom to vary the curve locally.**

Why must the bracket vanish pointwise, rather than merely integrate to zero? We can choose one component of $\eta$ at a time and support it in an arbitrarily small interval inside $(a,b)$. If the corresponding continuous bracket were positive at some point, it would stay positive in a small neighborhood. A nonnegative smooth perturbation supported there would make the integral positive, contradicting stationarity. The same argument excludes a negative bracket. This is the *fundamental lemma of the calculus of variations*.

Consequently, for each $s$ separately,

$$
\boxed{
\frac{\partial\mathcal{L}}{\partial x^s}
-\frac{d}{dt}\left(\frac{\partial\mathcal{L}}{\partial\dot{x}^s}\right)=0
}
\qquad\text{(no sum over }s\text{)}.
$$

This is the Euler-Lagrange equation used in section 2. It is a necessary condition for a smooth extremum, not a guarantee of a minimum: a stationary curve can also be a saddle point. Substituting $\mathcal{L}=\tfrac12 g_{ij}(x)\dot{x}^i\dot{x}^j$ gives the geodesic equation through the algebra in sections 3–7.
