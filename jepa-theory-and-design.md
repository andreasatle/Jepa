# JEPA for Ball-in-Cup: Theory and Design Choices

This document covers the two separate design axes you identified — the **cost
function** (how we measure and prevent collapse) and the **model
architecture** (what E and P actually are as networks) — plus how they fit
together into one training objective for our `ball_in_cup-catch` data.

---

## 1. Notation and What We're Building

JEPA stands for **Joint-Embedding Predictive Architecture**. Two networks:

- **E — the Encoder.** Maps a raw observation (one or more image frames) to a
  low-dimensional vector, the *embedding* or *latent representation*,
  $z = E(x)$.
- **P — the Predictor.** Given the current embedding and an action, predicts
  the *next* embedding: $ẑ_{t+1} = P(z_t, a_t)$.

The defining trait of JEPA, as opposed to a generative video-prediction
model, is that P never touches pixels. It predicts *in embedding space*
only. Nothing in this design ever reconstructs an image — the training
signal comes entirely from comparing predicted embeddings to encoded real
embeddings. This is why collapse is the central danger: if E can cheat by
mapping every image to the same constant vector, the prediction loss
trivially hits zero without learning anything about the physics.

That collapse risk is exactly the "cost function" problem below. The
"architecture" problem — what E and P look like as networks — is separate,
and doesn't interact with collapse at all. You were right to split these
apart; they're solved independently.

---

## 2. The Cost Function

### 2.1 Why the naive loss collapses

The obvious loss is just prediction error:
$$
    L_{naive} = (1/N) Σ_i \| P(E(x_t^i), a_t^i) − E(x_{t+1}^i) \|²
$$

The problem: E appears on *both* sides. If E collapses to a constant
function (outputs the same vector regardless of input), both sides of the
subtraction become that same constant, the loss is exactly zero, and
gradient descent has no reason to avoid this — it's a perfectly valid, if
useless, global minimum. Something extra has to make collapse a bad outcome
for the optimizer, not just an unhelpful one.

### 2.2 Option A — EMA / stop-gradient (what most JEPA papers do)

The standard fix, used in I-JEPA, V-JEPA, BYOL, and most of the modern
self-supervised literature: keep **two** copies of the encoder. An "online"
encoder E_θ, trained by gradient descent as usual, and a "target" encoder
E_φ, which is *not* trained by gradient descent — instead it's updated after
every step as an exponential moving average of the online encoder's
weights:

    φ ← τ·φ + (1−τ)·θ        (τ close to 1, e.g. 0.996)

The target branch also has gradients blocked (`stop-gradient`), so the loss

    L = ‖ P(E_θ(x_t), a_t) − E_φ(x_{t+1}) ‖²

only ever updates θ, never φ directly.

Why this tends not to collapse, in brief (we went deep on this earlier):
it's *not* that collapse becomes mathematically impossible. It's a
dynamical-stability argument — under stop-gradient plus a deliberately
low-capacity predictor, the collapsed solution is an unstable fixed point of
the training dynamics, so gradient descent drifts away from it in practice.
The EMA update itself can even be derived as the exact solution to a
proximal-regularized optimization problem, which makes it look principled —
but that reformulation still admits collapse as a valid stationary point.
The guarantee is empirical and dynamical, not a proof that the loss
landscape excludes collapse.

This is the "cheat" you flagged: it works very reliably in practice, but the
reason it works is a statement about training *trajectories*, not about the
optimization problem being solved.

### 2.3 Option B — VICReg

VICReg (Bardes et al., 2022) makes collapse literally impossible to reach as a minimizer, by adding two explicit regularization terms to the loss. No second network, no momentum, no stop-gradient.

One shared encoder E, used identically on both the current and next frame-stack:

$$
    z_t = E(x_t)      z_{t+1} = E(x_{t+1})      ẑ_{t+1} = P(z_t, a_t)
$$

For a minibatch of N transitions, three terms:

$$
    L = λ·s  +  μ·(v(Z_t) + v(Z_{t+1}))  +  ν·(c(Z_t) + c(Z_{t+1}))
$$

**Invariance term** (the actual prediction loss):

$$
    s = (1/N) Σ_i ‖ ẑ_{t+1}^i − z_{t+1}^i ‖²
$$

**Variance term** — for embedding matrix `Z ∈ ℝ^{N×D}` (N samples in the batch, D latent dimensions), compute each dimension's standard deviation *across the batch*:

$$
    σ_j = sqrt(Var_i(Z_{i,j}) + ε) \text{ for } j = 1..D \\
    v(Z) = (1/D) Σ_j \max(0, γ − σ_j) \quad (γ = \text{target std, e. g. } 1) \\
$$

This is a hinge: zero penalty once a dimension's spread across the batch reaches γ, growing penalty as it shrinks toward zero. If E collapsed to a constant, every σ_j → 0 and this term is pinned at its maximum — it cannot be traded away by making s smaller.

**Covariance term** — decorrelates the D dimensions so information can't all pile onto one axis while the rest go along for free:

$$
    C(Z) = (1/(N−1)) Σ_i (Z_i − Z̄)(Z_i − Z̄)ᵀ ∈ ℝ^{D×D} \\
    c(Z) = (1/D) Σ_{j≠k} C(Z)_{j,k}²            
$$

The variance and covariance terms are applied to the raw encoder outputs $Z_t$ and $Z_{t+1}$, not to the predictor's output — they constrain what E is allowed to produce, independent of how well P is currently predicting.

### 2.4 Comparison and choice

| | EMA / stop-gradient | VICReg |
|---|---|---|
| Extra network copy | Yes (target encoder) | No |
| Extra hyperparameter | Momentum τ | Loss weights λ, μ, ν, target std γ |
| Why collapse is avoided | Empirical/dynamical (unstable fixed point) | Provable (collapse maximally penalized) |
| Needs asymmetric predictor design | Yes, this matters a lot | No, incidental |
| Needs sizeable batch | No | Yes — variance/covariance are batch statistics |
| Conceptual honesty | "Works, mechanism is a stability argument" | "Collapse literally isn't a minimizer" |

Given your stated preference — not wanting to lean on a mechanism whose justification is "empirically it doesn't collapse" when a mathematically direct alternative exists — **we're using VICReg.** The one real cost is needing a reasonably sized batch (variance/covariance are meaningless for a batch of 1), which is a non-issue given we have 20,000 transitions sitting in `jepa_data/`.

---

## 3. The Model Architecture

This is the entirely separate question of *what E and P are as networks* — none of the above math cares whether E is a CNN, a transformer, or a lookup table, as long as it's differentiable.

### 3.1 Encoder E — a CNN

Since the input is raw pixels (64×64×3 per frame), a CNN is the natural choice: convolutional layers exploit the fact that nearby pixels are correlated and that useful features (edges, the ball, the cup) are local and roughly translation-invariant, which a plain fully-connected network doesn't assume and would need far more data/parameters to learn on its own.

To your question — **where does the low-dimensionality actually come from?** Not from "having fewer parameters." A CNN's convolutional layers mostly do two things: extract local features, and progressively downsample the *spatial* resolution (64×64 → 32×32 → 16×16 → 8×8, say), while the *channel* count (number of feature maps) typically grows. That alone doesn't produce a low-dimensional vector — you'd still have, e.g., 8×8×64 = 4096 numbers at that point, not obviously smaller than the ~12,288 raw pixel values (64×64×3) you started with.

The actual bottleneck is a deliberate architectural choice at the very end: after the last convolutional layer, flatten (or global-average-pool) down to a single vector, then pass through one fully-connected layer to a chosen output size D — say D=32 or D=64. That final linear layer's output width *is* the latent dimension, and we pick it directly; it's a design decision, not a byproduct of the conv stack. Everything before it is feature extraction; that last projection is the actual compression.

Sketch, for a stacked input of shape `(k·3, 64, 64)` (k frames, RGB):

```
    Conv(32, 4x4, stride 2) + ReLU     → 32 x 32 x 32
    Conv(64, 4x4, stride 2) + ReLU     → 16 x 16 x 64
    Conv(128, 4x4, stride 2) + ReLU    →  8 x  8 x 128
    Conv(256, 4x4, stride 2) + ReLU    →  4 x  4 x 256
    Flatten                            → 4096
    Linear(4096 → D)                   → D          ← the actual bottleneck
```

### 3.2 Predictor P — a small MLP

P doesn't need to see pixels at all, and doesn't need convolution — it operates entirely on already-compressed vectors, so a small fully-connected network is enough:

```
    input:  concat(z_t, a_t)      shape (D + action_dim,)
    Linear(D+action_dim → H) + ReLU
    Linear(H → H) + ReLU
    Linear(H → D)                 → ẑ_{t+1}
```

(H is a hidden width, e.g. 256.) `action_dim = 2` here (the cup's x/z motors).

One thing worth flagging given the VICReg choice specifically: in the EMA/BYOL literature, P is *deliberately* kept low-capacity/asymmetric, since that asymmetry is part of what makes collapse dynamically unstable. Under VICReg we don't need that constraint — collapse is already excluded by the loss itself — so P's capacity is a free design choice, tuned purely for prediction quality.

### 3.3 Frame stacking

Tying back to the velocity discussion: E's input is not a single frame but a short stack of the k most recent frames (concatenated along the channel dimension), so the encoder has the information needed to infer motion, not just position. k=2 is the minimum that makes velocity inferable at all; k=3 gives a little more robustness to noise. We'll use k=2 to start, and it's easy to widen if the learned representation turns out to need it.

---

## 4. Full Training Objective

Putting it together, per minibatch of N transitions $(x_t, a_t, x_{t+1})$ (each x already a k-frame stack):

$$
    z_t      = E(x_t) \\
    z_{t+1}  = E(x_{t+1}) \\
    ẑ_{t+1}  = P(z_t, a_t) \\

    L = λ · (1/N)Σ‖ẑ_{t+1} − z_{t+1}‖²
      + μ · (v(Z_t) + v(Z_{t+1}))
      + ν · (c(Z_t) + c(Z_{t+1}))
$$

Both E and P are trained jointly by backpropagating through this single L — one optimizer, one network graph, no target network, no momentum schedule.

---

## 5. Design Choices to Fix

| Choice | Value | Notes |
|---|---|---|
| Latent dim D | 32–64 | Small enough to be a real bottleneck; can revisit |
| Frame stack k | 2 | Minimum for velocity to be inferable |
| Image size | 64×64 | Matches `collect_data.py` |
| Predictor hidden width H | 256 | Arbitrary reasonable default |
| Loss weights λ, μ, ν | 25, 25, 1 | VICReg paper's defaults, a reasonable start |
| Target std γ | 1 | VICReg paper's default |
| Batch size | 256+ | Needs to be large enough for meaningful variance/covariance stats |

None of these are set in stone — they're starting points we can revisit once we see how training behaves.

---

## 6. Deliberately Out of Scope Here

Nothing in this document trains a policy or plans anything. E and P are trained purely self-supervised, from $(x_t, a_t, x_{t+1})$ triples, with no reward signal at all. Planning (CEM/MPC in latent space) and the RL baseline comparison both come after this works — consistent with the "get JEPA running first" plan.

---

## 7. Revision: Single-Frame E, Windowed P, and the Norm-Form Loss

Sections 2–4 describe the scheme as first built and first trained: EMA was tried and dropped (Section 2.2/2.3 above already reflect landing on VICReg), and E took a `FRAME_STACK`-sized stack of frames, concatenated on the channel dimension, doing the temporal compression internally. Two things changed after actually training and evaluating that version, and this section states the current scheme precisely, with norms.

### 7.1 Architecture: E is single-frame, P owns the time window

$$
    z_t = E(x_t) \text{ in } R^D
$$

E no longer sees more than one frame. All temporal reasoning is P's job: given the K_CONTEXT most recent embeddings and the current action,

$$
    ẑ_{t+1} = P( [z_{t-k+1}; ... ; z_t] , a_t )  \text{ in } R^D, \quad k = 3
$$

where $[z_{t-k+1}; ... ; z_t]$ is the concatenation of k consecutive D-dimensional embeddings into one k·D-dimensional vector, fed alongside the action a_t in R^2 into P's MLP.

Why: ball_in_cup is governed by Newton's second law, a second-order ODE in position. A system like that is only Markovian — has a "state" from which the future is fully determined — once you augment it to phase space (position, velocity). A single frame (or a single vector compressed from a frame-stack) forces E to decide, once and for all, how to pack that phase-space information into one representation. Letting P see several raw single-frame embeddings directly lets it reconstruct velocity itself — the same operation any multistep/finite-difference integrator performs, except learned and nonlinear instead of Taylor-derived and linear.

k=3, not 2: with two points, velocity is estimated as $(x_t - x_{t-1})/dt$; with $dt = 0.02s$ (dm_control's control_timestep), dividing by `dt^2` amplifies position noise by roughly 2500x. Using the outer two of three points, $(x_t - x_{t-2})/(2dt)$, roughly halves the resulting noise's standard deviation for the same underlying position noise. No reason to go past 3: $(x, v)$ is a *complete* phase space for a second-order system — acceleration is a function of $(x, v, action)$ via F=ma, not an independent state component.

Confirmed empirically, not just argued: a linear probe from a single z to true position scored well under both schemes (test R^2 ~0.88–0.92), but a linear probe from a single z to true *velocity* scored ~0 under the old scheme. Probing the K_CONTEXT-window of z's against velocity under the new scheme scores ~0.6–0.7 (test R^2, held-out episodes) — the phase-space argument's predicted fix, confirmed on real data.

### 7.2 The anti-collapse penalty, precisely, with norms

For a batch of N embeddings $Z in R^{N x D}$ (D=32), define the sample covariance matrix

$$
    Cov(Z) = (1/(N-1)) (Z - Z̄)ᵀ(Z - Z̄)   in R^{D x D}          (Z̄ = batch mean, broadcast)
$$

Split $C = Cov(Z)$ into its diagonal and off-diagonal parts. The penalty is NOT the plain $||C - I||_F^2$ Section 2.3 originally described — that version collapsed during training and stayed collapsed (verified: its gradient w.r.t. z vanishes in proportion to ||z|| right as z approaches the collapsed point, because $C$'s diagonal entries are themselves already a squared quantity — $C_jj = σ_j^2$ — and squaring $(C_jj - 1)$ again compounds it). The diagonal is square-rooted before penalizing instead, which restores a non-vanishing gradient at collapse while staying fully squared and smooth (no hinge):

$$
    pen(Z) = || offdiag(C) ||_F^2   +   || sqrt(diag(C)) - 1 ||_2^2
$$

$offdiag(C)$ is $C$ with the diagonal zeroed out, $||·||_F^2$ the squared Frobenius norm (sum of squared entries). $diag(C)$ is the length-D vector of C's diagonal entries (the per-dimension variances); $sqrt(diag(C))$ is that vector's elementwise square root (the per-dimension standard deviations); $1$ is the all-ones vector in R^D; $||·||_2^2$ the ordinary squared Euclidean norm. So the off-diagonal (decorrelation) target is penalized in its natural, already-correctly-scaled form, while the diagonal (spread) target is penalized in standard-deviation units rather than variance units, specifically to keep its restoring gradient alive all the way down to collapse.

This penalty is applied to two branches: $Z_t$, the batch of embeddings at the most recent context frame ($z_t$ from each transition in the batch), and $Z_{t+1}$, the batch of target-frame embeddings.

### 7.3 Full loss

    L = (1/2) || ẑ_{t+1} - z_{t+1} ||_2^2
        + (κ/2) [ pen(Z_t) + pen(Z_{t+1}) ]

The first term is the ordinary squared Euclidean-norm prediction error (this is what the training log calls `avg_inv_loss`, averaged over the batch). The second term is the anti-collapse penalty from 7.2, applied symmetrically to both branches and weighted by $κ$ (this is what the log calls `avg_cov_penalty`, again pre-$κ$-weighting in the log). `avg_loss` is the full sum. E and P are trained jointly on this one objective — one optimizer, no target network, no stop-gradient.