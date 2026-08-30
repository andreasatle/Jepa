# Project brief: JEPA + planner on dm_control's ball_in_cup-catch

## Purpose

Learning JEPA (Joint-Embedding Predictive Architecture) hands-on by building it end-to-end on dm_control's `ball_in_cup-catch` task. The task itself is a deliberately chosen but ultimately unimportant proxy — the goal is understanding the algorithms (JEPA pretraining, world-model planning, CEM), not solving ball-in-cup specifically. An RL baseline (SAC) was started early and deliberately parked until JEPA pretraining worked; it's still parked.

## Architecture

- **E (encoder)**: single frame (64×64×3) → 32-dim latent `z`. Purely spatial, no temporal info in any one `z`.
- **P (predictor)**: takes a window of `K_CONTEXT=3` consecutive embeddings `[z_{t-2}, z_{t-1}, z_t]` plus an action `a_t`, predicts `ẑ_{t+1}`. All temporal/velocity reasoning lives here, not in E.
- Earlier version had E take a stacked multi-frame input directly; switched to single-frame-E + windowed-P after a linear probe showed the old scheme couldn't linearly recover velocity (R²≈0). Reasoning: ball_in_cup is a second-order ODE (Newton's law), so position alone isn't Markovian — phase space needs (position, velocity), and forcing E to pre-compress that into one vector per frame was the wrong place to do it. K_CONTEXT=3 rather than 2: with dm_control's `control_timestep=0.02s`, a 2-point finite difference amplifies position noise by `1/dt²≈2500×`; using the outer two of three points roughly halves that.
- **Loss**: VICReg-style, shared encoder (no EMA/target network/stop-gradient). `L = ½‖ẑ_{t+1}−z_{t+1}‖² + (κ/2)[pen(z_t) + pen(z_{t+1})]`, where `pen(z) = ‖offdiag(Cov(z))‖_F² + ‖√diag(Cov(z)) − 1‖²` — diagonal term is square-rooted before penalizing (fixes a vanishing-gradient-at-collapse bug found via numerical gradient checks; plain `(C_jj−1)²` has gradient ~500× weaker near collapse).
- Confirmed via primary-source reading of LeCun's JEPA paper: this is a plain single-level JEPA, not H-JEPA (no natural coarse/sub-goal decomposition was found for this task — though this was revisited once the planning-horizon problem below came up).

## Status

JEPA pretraining works. Real-data linear probes (after the K_CONTEXT restructuring, ~200+ epochs trained):
- Single-frame `z` → position: test R² ≈ 0.92–0.94 (as expected, velocity not recoverable from one frame — that's the point).
- K_CONTEXT window → position + velocity: position R² ≈ 0.92–0.94, velocity R² ≈ 0.59–0.72.

## Built so far (all Python scripts in `~/Projects/Jepa/ball_in_cup/`, run from the repo root)

- `collect_data.py` / `collect_eval_data.py` — random-policy rollout collection (pixels+actions; eval version also logs ground-truth position/velocity from `physics.position()`/`physics.velocity()`).
- `jepa_train.py` — E, P, VICReg loss, training loop, checkpointing/resume. `VERSION` string only bumped when explicitly asked.
- `eval_linear_probe.py` — fits affine probes (`np.linalg.lstsq`) from frozen `z` to true state, held out by episode. **Does not yet persist the fitted `W` to disk** — needed before a planner can use it without refitting every run.
- `jepa_evaluator.py` — loads both E and P, rolls `P` forward autoregressively given a starting window + action sequence (batched, for CEM later). Smoke-tested against the real checkpoint.
- `ball_in_cup_demo.py` — headless GIF / interactive viewer / teleop (arrow keys, bang-bang full-magnitude force via `gear=5` MuJoCo actuators) modes. `keytest.py` is a standalone pynput diagnostic (macOS Accessibility permission issue was the recurring gotcha).
- Verified geometry (read directly from dm_control's XML/source, not memory): cup is 5 capsule segments (radius .008), opening ≈0.084–0.10m wide vs ball diameter 0.05m (real clearance, not a razor's-edge fit); target zone sits at the cup's half-depth, `in_target()` tolerance is `±0.025` around it; `ball_to_target` verified numerically against `physics.ball_to_target()`.

## Planner: design status (nothing built yet beyond a cost-function sketch)

- Approach: CEM (Cross-Entropy Method) / MPC — sample candidate action sequences, roll each through P, score via a cost function, refit the sampling distribution toward the elites, repeat; execute only the first action, replan every real step.
- Cost function (sketched, not yet implemented as real code beyond a draft): `C = min_τ [Δx_τ² + Δz_τ² + λ·max(0, v_rel_τ)²]` — min-over-horizon (not terminal-only) squared position distance, decoded via the position probe, plus a penalty for the ball moving *upward relative to the cup* at any step (rather than settling downward into it). `v_rel = ball_z_vel − cup_z_vel`, chosen over absolute velocity specifically because the cup itself moves.
- **Explicitly decided against**: an eikonal-field / geometry-aware obstacle-avoidance cost. Reasoning: it would require hand-injecting privileged, exact knowledge of the cup's geometry (from the XML) into the cost function — a task-specific hack that wouldn't generalize to a new environment, and philosophically wrong per JEPA's own design: obstacle/collision physics is supposed to be something P learns from data, not something engineers hand-code into C.
- **Explicitly decided against (for now)**: warm-starting CEM from a recorded human teleop demonstration. Reasoning discussed: real risk of premature convergence to a mediocre local optimum near the demo, even though in principle CEM's population retains enough spread to escape a bad seed if given enough iterations/budget.

## The core open difficulty (unresolved)

Ball-in-cup requires a "pump the swing" strategy (lateral pumping builds pendulum amplitude over several cycles before the ball can arc over the rim and drop in) — confirmed empirically via teleop, where horizontal force alone was enough to solve it. This makes the true cost trajectory **non-monotonic**: distance-to-target genuinely increases before it decreases along the correct solution. Consequences worked through in detail:
- Local/greedy search (short-horizon CEM, or gradient descent through the differentiable rollout) can't find this — it needs a horizon long enough to see the whole payoff, not just smoothness.
- CEM's own mechanism (score only *complete* H-step rollouts, refit toward elites) is structurally fine with this in principle — it doesn't prune partial paths — but is only as good as its ability to *sample* a working strategy in the first place. The set of successful action sequences is a small fraction of the full space (this is, precisely, the "rare event" character the Cross-Entropy Method was originally built for), and without warm-starting, CEM has to find that rare region by population diversity and iteration alone.
- Horizon length needed (plausibly 100+ raw steps, at `dt=0.02s`, to span several pump cycles) may exceed how far the learned predictor P can be trusted before compounding autoregressive error makes its rollout meaningless. This was never empirically measured — the planned diagnostic (roll P forward on a real recorded episode, compare predicted-vs-true position/velocity via the probes) has been proposed multiple times and not yet built.
- A collision-modeling concern was also raised and left unresolved: the cost function has no explicit notion of the cup's solid geometry, so CEM's search — which only ever sees the world through P's predictions — could in principle be misled by P's likely-poor modeling of contact/rim-collision dynamics (rare in random training data, and a near-discontinuity a smooth network trained on averaged one-step MSE would tend to blur across, not represent sharply).

## Considered alternative: swap the physics to Push-T

Discussed at length, not decided. `gym-pusht` (pip-installable, Hugging Face LeRobot ecosystem, adapted from the Diffusion Policy paper) would let the algorithmic pipeline (E, P, loss, K_CONTEXT windowing, CEM design) carry over largely unchanged while replacing the environment. Trade-offs surfaced:
- Removes the non-monotonic-cost problem: pushing gives fairly immediate, correctly-signed local progress from most configurations (dense coverage-percentage reward, not ball-in-cup's sparse binary), so short-horizon CEM should work over most of the state space.
- Does *not* remove difficulty in general — trades it for a different kind: Push-T's dynamics are contact/friction-dominated as the *core*, constant mechanism (not a rare edge case like the cup's rim), and the task is famously multimodal (approach the block from the left or right — Diffusion Policy's whole motivation), which a deterministic, one-step-MSE-trained P would likely handle by blending/averaging modes rather than committing to one — the same failure mode discussed for pixel-space video prediction (LeCun's blurry-highway-video illustration) and for ball-in-cup's collision uncertainty, just showing up constantly instead of near one boundary.
- No solid obstacle blocking the path in Push-T (unlike the cup's rim), so that specific concern goes away regardless of which task is used.

## Standing working preferences

- Only bump `VERSION` in `jepa_train.py` when explicitly asked.
- Verify claims concretely (run code, read primary sources/actual XML/source, check numerically) rather than assert from memory — this has caught real errors along the way (an incorrect exponential-vs-linear H-JEPA receptive-field claim, an incorrect Adam-protects-against-loss-crowding claim).
- Never decide architecture/algorithm changes unilaterally — always propose and get explicit confirmation before implementing.
- No warm-starting the CEM planner (explicitly rejected).
- The desktop app is connected for this project — direct read/write access to `~/Projects/Jepa/` is available in-session (re-grant needed each new session).

## Immediate next steps (not yet done, no particular order implied)

1. Persist the position (and ideally position+velocity) probe's fitted `W` to disk.
2. Build and run the autoregressive drift diagnostic (P rolled forward on real data vs. ground truth) to get a real number for how far P can be trusted.
3. Decide: stick with ball_in_cup and build the full CEM planner around the sketched cost, or switch to Push-T first.
4. (Floated, not started) An RCDM-style diffusion decoder trained on frozen `z` to visualize what the representation captures, and — more usefully — to visually inspect whether P's *rolled-out* predictions go blurry/incoherent past some horizon, as a richer version of the drift diagnostic.
