# JEPA + CEM/MPC on `ball_in_cup-catch`

A from-scratch, hands-on implementation of a **Joint-Embedding Predictive Architecture (JEPA)** world model, trained on pixels from dm_control's `ball_in_cup-catch` task, driven by a **Cross-Entropy Method (CEM)** planner inside a **Model Predictive Control (MPC)** loop.

The task itself is a deliberately unimportant proxy. The actual goal of this project was to learn JEPA end to end — self-supervised world-model pretraining, evaluating what a learned representation actually captures, and planning through a learned model — not to solve ball-in-cup for its own sake. It ended up solved anyway (97% success rate from random starting positions), which is a nice bonus.

This document assumes no prior JEPA knowledge. For the full math derivation of the loss function and architecture, see [`jepa-theory-and-design.md`](jepa-theory-and-design.md).

## Contents

- [What is JEPA?](#what-is-jepa)
- [The task](#the-task)
- [Architecture](#architecture)
- [Two pipelines: fine-grained and strided](#two-pipelines-fine-grained-and-strided)
- [Reading physical state out of z: probes](#reading-physical-state-out-of-z-probes)
- [Planning: CEM inside MPC](#planning-cem-inside-mpc)
- [Results: the journey from 60% to 97%](#results-the-journey-from-60-to-97)
- [Lessons](#lessons)
- [Repo map](#repo-map)
- [Reproducing this from scratch](#reproducing-this-from-scratch)
- [Status and what's next](#status-and-whats-next)

## What is JEPA?

The obvious way to build a "world model" — something that predicts what happens next given the current situation and an action — is to predict the next *frame*, pixel by pixel. This has a real problem: most of a pixel frame is irrelevant detail (lighting, texture, background), and a network trained to minimize pixel error tends to hedge its bets by predicting a blurry average over the plausible futures, rather than committing to one. Predicting pixels also just wastes model capacity reconstructing things nobody asked about.

JEPA's fix: **predict in an abstract latent space instead of pixel space.** Two networks:

- **E, the encoder.** Maps a raw frame to a compact vector `z = E(x)` — in this project, a 64×64×3 image down to 32 numbers.
- **P, the predictor.** Given the current embedding(s) and an action, predicts the *next* embedding: `ẑ_{t+1} = P(z_t, a_t)`.

Training compares the predicted embedding `ẑ_{t+1}` to the real embedding `z_{t+1} = E(x_{t+1})` — nothing ever touches pixels after `E` runs. There's no decoder anywhere in this project; `z` is never turned back into an image during training or planning.

**The central danger, and why most of the design choices below exist:** representation collapse. If `E` maps every image to the same constant vector, `P` can predict the "next" embedding perfectly (it's always the same vector) and the prediction loss goes to exactly zero — while the network has learned nothing whatsoever about the world. Nothing about the plain prediction objective prevents this on its own. Guarding against it is the whole reason this project's loss function (below) has a second term besides prediction error, and it's why some other self-supervised approaches (BYOL-style) instead use a momentum/EMA target network or a stop-gradient trick — this project uses neither; collapse is prevented by an explicit anti-collapse penalty (VICReg-style) added directly to the loss.

## The task

`ball_in_cup-catch` is one of dm_control's built-in `suite` tasks. Verified directly from its source (not assumed):

- The cup sits on two slide joints (x and z) driven by two continuous actions in `[-1, 1]`, each `gear=5` — this is the entire action space. The cup is **not** a free body: its joints have a real spring-damper suspension (stiffness=20, damping=3), so with zero action it settles to a fixed hanging equilibrium rather than drifting freely.
- The ball is a genuinely free mass, connected to the cup by an **inextensible spatial tendon** ("string"), max length 0.3m — not a rigid rod, and the string can go slack.
- Success (`physics.in_target()`) means the ball is within `0.025m` of a target point sitting a fixed offset above the cup's own origin — derived from the XML (target site half-extent `.05` minus ball radius `.025`) and checked numerically against `physics.ball_to_target()`.
- The default initial condition is **randomized** each episode (`ball_x ~ U(-.2, .2)`, `ball_z ~ U(.2, .5)`), not fixed.

The task is genuinely non-trivial to control: getting the ball from a resting hang into the cup requires *pumping* — lateral motion that builds pendulum amplitude over multiple swings before the ball can arc over the rim and drop in. Confirmed by hand via teleoperation. This matters a lot for the planner design below: the correct solution's cost trajectory is **non-monotonic** — distance-to-target has to get worse before it gets better — which rules out any purely greedy/short-horizon control strategy.

## Architecture

- **E**: one 64×64×3 frame → 32-dim `z`. Purely spatial — no temporal information lives in a single `z`.
- **P**: a window of `K_CONTEXT=3` consecutive embeddings `[z_{t-2}, z_{t-1}, z_t]` plus the 2-dimensional action `a_t`, predicting `ẑ_{t+1}`. All velocity/temporal reasoning lives here, not in `E`.

Why split it this way: `ball_in_cup` is governed by Newton's second law, so its true state is `(position, velocity)`, not position alone — position by itself isn't Markovian, and a single frame genuinely cannot show velocity. An earlier version fed `E` several stacked frames directly; a linear probe showed that scheme couldn't recover velocity at all (R² ≈ 0), which is what motivated moving to a single-frame `E` plus a windowed `P` instead. `K_CONTEXT=3` rather than the more obvious `2`: with `control_timestep=0.02s`, a plain 2-point finite difference amplifies position noise by roughly `1/dt² ≈ 2500×`; using the two outer points of a 3-point window roughly halves that.

**Loss** (VICReg-style, shared encoder, no EMA/target network, no stop-gradient):

$$
L \;=\; \frac{1}{2} \left\lVert \hat{z}_{t+1} - z_{t+1} \right\rVert^2
\;+\; \frac{\kappa}{2} \Big[ \operatorname{pen}(z_t) + \operatorname{pen}(z_{t+1}) \Big]
$$

$$
\operatorname{pen}(z) \;=\; \left\lVert \operatorname{offdiag}\big(\operatorname{Cov}(z)\big) \right\rVert_F^2
\;+\; \left\lVert \sqrt{\operatorname{diag}\big(\operatorname{Cov}(z)\big)} - \mathbf{1} \right\rVert^2
$$

The first term is plain prediction error. The second is the anti-collapse penalty: it pushes each embedding dimension's standard deviation toward 1 and pushes different dimensions to be uncorrelated with each other, both of which are impossible to satisfy if the encoder is collapsing everything to a constant. The diagonal term is square-rooted before being penalized — a plain $(\operatorname{Var}(z_j) - 1)^2$ term has a gradient that vanishes near collapse (roughly 500× weaker there than away from it, confirmed via numerical gradient checks), which lets the encoder sit right at the edge of collapsing without being pushed back out. Square-rooting first fixes that.

## Two pipelines: fine-grained and strided

The first full closed-loop planning attempt exposed a serious problem: **autoregressive drift**. `P` was trained purely via teacher forcing — every training example gave it a real, ground-truth context window — so it had never once seen its own predicted output fed back in as context. During actual planning, `P`'s own predictions get fed back in every step (autoregressive rollout), which turned out to be wildly off-distribution from what `P` was trained on. Measured directly (`drift_diagnostic.py`, rolling `P` forward on real recorded actions and comparing the decoded position against ground truth): early in training, rollout error **exploded**, reaching millions of meters by 140 steps. More training bounded the explosion (~1.5–1.8m peak) but didn't fix the underlying problem — the near-term error floor stayed around 0.10m, already above the 0.025m success tolerance.

Two independent, stackable fixes:

1. **`UNROLL_STEPS=5` (multi-step unrolled training)** — `jepa_train.py` now trains `P` on 5 successive self-fed predictions per example, not just one. The *predicted* (not real) embedding is what gets slid into the context window between calls, while the loss still supervises every step against the real target. This directly targets exposure bias by actually training `P` on the kind of input it'll see at deployment time.
2. **`PREDICTOR_STRIDE=5` (decoupling `P`'s own step from the physics)** — one call to `P` now spans 5 real `env.step()` calls, with the action held constant across them (action-repeat, the standard RL "frame-skip" technique). This means far fewer autoregressive `P`-calls are needed to plan a given real-time horizon, and it's implemented as a wrapper around the environment rather than by changing dm_control's own `control_timestep` — the environment's real definition, and every other tool built against it, stays untouched and directly comparable.

These required a fully separate, action-repeat data collection (`collect_data_strided.py`), since `PREDICTOR_STRIDE`-style training examples ("held constant for 5 steps") can't be constructed from data collected with a fresh i.i.d. random action every step. The result is two parallel, independently checkpointed pipelines — every `*_strided.py` script mirrors a non-strided one, reusing shared code (`jepa_train.py`'s `E`/`P`/loss, `jepa_evaluator.py`'s rollout) rather than duplicating it.

**Combined result** (`drift_diagnostic_strided.py`, run twice, consistent): ball position error stays **flat at ~0.08–0.11m across the entire tested 3-second / 30-call horizon** — no compounding blowup. The exposure-bias/horizon problem was solved. What remained was a flat precision floor, a different and much more tractable kind of problem (see next section).

## Reading physical state out of `z`: probes

`z` is never given any explicit meaning — nothing in the loss above ever tells the encoder "this dimension should mean ball height." Whether `z` actually captures anything physically meaningful is a separate, empirical question, answered by **probing**: freezing the trained encoder and fitting a small supervised model from `z` to the true (simulator-only) ground-truth state, on held-out episodes. A good probe score is real evidence about what the representation captures; it doesn't change the representation itself (the probe is always fit *after* `E`/`P` are already trained and frozen — see [Lessons](#lessons) for why that separation matters).

- **Linear probe** (`eval_linear_probe*.py`): single matrix multiply, `position = [z | 1] · W`, fit by least squares. Single-frame `z → position` test R² ≈ 0.92–0.94 (fine-grained model); the strided model's own encoder did slightly better (~0.96 train / ~0.94 test). The K_CONTEXT-window variant (testing whether the window `P` itself sees can also recover velocity) got R² ≈ 0.92–0.94 for position and ≈ 0.59–0.72 for velocity.
- **Nonlinear (MLP) probe** (`eval_mlp_probe_strided.py`): same task, same train/test split, but a small 2-hidden-layer network instead of one matrix multiply. Mean ball position error on held-out data dropped from 0.0827m (linear) to **0.0067m** — about 12× better — confirming the position information was present in `z` all along, just not *linearly* extractable.

That 12× more accurate probe turned out to make **no measurable difference** to closed-loop planning performance (see the results table below) — a genuinely useful, if initially surprising, negative result. The likely explanation: the planner's cost function only ever uses the *difference* between the decoded cup and ball positions, and since both go through the same probe, a systematic bias in the probe largely cancels out of that difference even when each absolute position is off by several centimeters. Whether an episode counts as solved is also decided by the simulator's own ground truth (`physics.in_target()`), never by the probe — the probe only ever shapes what the planner's search *believes* mid-search, not the measured outcome.

## Planning: CEM inside MPC

**Cross-Entropy Method**, the search algorithm: sample a population of candidate action sequences, score every one, keep the best fraction ("elites"), refit the sampling distribution toward those elites, repeat for a fixed number of iterations. It's a general-purpose, gradient-free optimizer — nothing about it is specific to JEPA or to this task.

**Model Predictive Control**, the deployment loop around it: solve a full CEM search from the *current real state*, execute only the first action of the best sequence found, observe the real result, and repeat from scratch. This is what lets a genuinely imperfect world model still work in practice — every decision is re-grounded against reality (the real frame is re-encoded into a real `z` every step), so errors in any one plan never get to compound past a single real step's worth of consequences.

**Cost function**, scored on `P`'s rolled-out predictions, decoded to position via the probe:

$$
\operatorname{cost}(a_{1:H}) \;=\; \min_{\tau \,\in\, [1,\,H]} \Big[\,
(x^{\text{cup}}_\tau - x^{\text{ball}}_\tau)^2 \;+\;
(z^{\text{cup}}_\tau + \Delta z - z^{\text{ball}}_\tau)^2 \;+\;
\lambda \max\!\big(0,\, v^{\text{rel}}_\tau\big)^2 \Big]
$$

where $\Delta z$ is the fixed cup-to-target vertical offset (`CUP_TO_TARGET_Z_OFFSET`) and $v^{\text{rel}}_\tau$ (ball's vertical velocity relative to the cup's) is penalized only when positive — the ball closing in too fast to actually settle rather than bounce through. Taking the **minimum over the whole horizon**, rather than scoring only the final step, is what lets CEM find "pump the swing" strategies at all: a candidate sequence is judged by its single best moment anywhere in the lookahead window, so a trajectory that's far away for most of the horizon and only gets close near the end still scores well.

Two real bugs/design gaps were found and fixed along the way, both in `cem_planner.py`/`cem_planner_strided.py`:

- **No running-best tracking (elitism).** Vanilla CEM refits its sampling distribution from scratch each iteration; without deliberately carrying the best-found candidate forward, the reported "best in population" can regress between iterations from pure resampling variance — caught by noticing a non-monotonic `cost_history`. Fixed by injecting the best-so-far candidate as a guaranteed member of every iteration's population, and returning that tracked best rather than the final iteration's mean.
- **No memory across real decision points (warm-starting).** Every call to `plan_step()` was solving CEM completely from scratch, discarding whatever trajectory the previous real step had found — even a genuinely good, still-in-progress damping strategy got randomly reinitialized at the very next replan. Fixed with a **shift-forward-plan warm start** (distinct from — and not a reversal of — the project's separate, standing decision against seeding CEM from a recorded human demo, which still holds): each `plan_step()` call keeps a slice of its *own* top elites from every one of its CEM iterations (not just the final, converged one — a converged final iteration's elites are all near-duplicates of each other, which would defeat the purpose), shifted forward by one step and injected into the very first iteration of the *next* real decision point's search. The per-timestep sampling `std` always resets fresh regardless, so exploration is never at risk of collapsing even if the carried-forward candidates happen to be similar.

## Results: the journey from 60% to 97%

All numbers below are `n=30` random-initial-condition trials (`solve_strided_batch_random.py`) unless noted, real closed-loop episodes against the live simulator, not offline metrics.

| Stage | What changed | Success rate | Mean closest approach |
|---|---|---|---|
| Steady-state IC, no warm-start, linear probe | baseline, `n=10` | 9/10 (90%) | 0.0220m |
| Random IC, no warm-start, linear probe | baseline, `n=10` | 6/10 (60%) | — |
| Random IC, no warm-start, MLP probe | swap probe only | 20/30 (67%) | 0.0421m |
| Random IC, no warm-start, linear probe (re-run at `n=30`) | confirms the probe swap is a null result | 19/30 (63%) | — |
| Random IC, **warm-start**, MLP probe | stratified-elite warm start added | 22/30 (73%) | 0.0363m |
| Random IC, warm-start, **24s time budget** (was 12s) | `MAX_REAL_STEPS` 600→1200 | **29/30 (97%)** | **0.0240m** |

The MLP-probe swap and its confirmation are a genuine (if initially surprising) null result — see [above](#reading-physical-state-out-of-z-probes).

The last jump is the biggest, and came from a methodological correction, not a model or algorithm change. Visually inspecting failed episodes (`solve_strided_inspect.py` renders labeled filmstrip PNGs of real episodes) showed every single failure had the *same* signature: the ball still visibly mid-swing — not close to settling — at the 12-second cutoff, even in cases that had swung right past the cup opening earlier in the episode without being caught. That's a "can't stick the landing yet" pattern, not a near-miss/precision problem. The fix followed from a simple but easy-to-miss distinction: a real robot executing this policy can never discard its current state and reroll a fresh, easier starting position if it's struggling — the only physically meaningful notion of "retry" is *continuing to act longer from wherever it actually is*. Doubling the real-time budget (with the search itself completely unchanged) let almost all of the previously-failing, still-actively-converging trajectories finish settling; several successes only resolved well past the old 12-second cutoff (up to 21.7s), confirming most "failures" were slow, not stuck.

## Lessons

A few things worth taking away, useful beyond this specific project:

- **Verify by direct measurement, not by assumption.** Nearly every real finding here came from building a diagnostic and actually running it — the exposure-bias diagnosis, the MLP probe's null result (confirmed at `n=30` specifically because `n=10` wasn't conclusive either way), and the visual filmstrip inspection that reframed the whole final-stretch problem.
- **A more accurate encoder readout isn't automatically a better policy.** The MLP probe was 12× more accurate offline and made no measurable difference to the actual task. Evaluate the whole closed loop, not representation quality in isolation.
- **Self-supervised representations can capture real physical structure with no supervision at all.** `z` was never told what "position" means; a simple linear probe recovers it with R² > 0.9 anyway, purely as a side effect of the prediction + anti-collapse objective.
- **A probe is a readout, not part of the model.** It's fit *after* `E`/`P` are frozen, from ground truth that's cheap here (it's a simulator) but wouldn't be on a real robot. Training `E` jointly with a position-supervised loss from the start would likely have closed the precision floor faster — but would also have stopped testing what self-supervision alone can recover, which was the actual point of doing this as JEPA rather than as ordinary supervised learning.
- **Closed-loop replanning (MPC) can absorb model imperfections that look severe in open-loop diagnostics.** The fine-grained model's early drift diagnostic looked catastrophic in isolation, but MPC's constant re-grounding against real state is a large part of why the final system works at all despite `P` never being perfectly accurate.

## Repo map

This whole project is self-contained in `ball_in_cup/` — scripts, datasets, checkpoints, probes, logs and generated media all live here. Run everything from this directory (e.g. `cd ball_in_cup && uv run python jepa_train_strided.py`); all paths in the code are relative to the current directory. Every `*_strided.py` script mirrors a non-strided sibling and operates on the strided (`PREDICTOR_STRIDE=5`, action-repeat) model and data — the two pipelines never share checkpoints, only code.

**Data collection**
- `collect_data.py` / `collect_data_strided.py` — training rollouts (i.i.d. random actions / action-repeat respectively).
- `collect_eval_data.py` / `collect_eval_data_strided.py` — held-out rollouts with logged ground-truth position/velocity, for probes and drift diagnostics.

**Training**
- `jepa_train.py` — `E`, `P`, the VICReg-style loss, multi-step unrolled training (`UNROLL_STEPS`), checkpointing/resume. The fine-grained model's own trainer; also imported (not duplicated) by the strided one.
- `jepa_train_strided.py` — trains the strided model, reusing `jepa_train.py`'s `E`/`P`/loss with a `PREDICTOR_STRIDE`-spaced dataset.

**Evaluation**
- `eval_linear_probe.py` / `eval_linear_probe_strided.py` — fit and persist the linear position/state probes.
- `eval_mlp_probe_strided.py` — fits and persists the nonlinear position probe.
- `drift_diagnostic.py` / `drift_diagnostic_strided.py` — roll `P` forward autoregressively on real recorded actions, compare decoded position against ground truth.
- `jepa_evaluator.py` — shared loading/rollout helpers used by everything above and by the planner.

**Planning and solving**
- `cem_planner.py` / `cem_planner_strided.py` — the CEM/MPC planner: `CEMConfig`, `rollout_cost`, `plan_step`, the position-probe decode (supports both linear and MLP probes transparently).
- `solve_strided.py` — one full closed-loop episode against the live simulator, with a saved GIF.
- `solve_strided_batch.py` / `solve_strided_batch_random.py` — batches of episodes (steady-state / random initial condition) for success-rate statistics; accept an optional probe-path argument for A/B comparisons.
- `solve_strided_inspect.py` — collects a handful of failures and successes with visual filmstrip PNGs, for qualitative inspection.

**Other**
- `ball_in_cup_demo.py` — GIF/viewer/teleop modes, useful for sanity checks and for confirming a strategy by hand.
- `jepa-theory-and-design.md` — the full math derivation behind the loss and architecture choices.

## Reproducing this from scratch

```bash
cd ball_in_cup   # all scripts and data are relative to this directory

# 1. Collect action-repeat training and eval data
uv run python collect_data_strided.py
uv run python collect_eval_data_strided.py

# 2. Train the world model (E, P)
uv run python jepa_train_strided.py

# 3. Fit probes (linear, then optionally the more accurate nonlinear one)
uv run python eval_linear_probe_strided.py
uv run python eval_mlp_probe_strided.py

# 4. Sanity-check the world model's rollout accuracy
uv run python drift_diagnostic_strided.py

# 5. Smoke-test the planner
uv run python cem_planner_strided.py

# 6. Solve — a single episode with a GIF, or a batch for success-rate stats
MUJOCO_GL=egl uv run python solve_strided.py
uv run python solve_strided_batch_random.py
```

## Status and what's next

The ball-in-cup phase is considered complete: 29/30 (97%) from random initial conditions clears the bar for this project's purposes, and the core JEPA-learning goals — world-model pretraining, evaluating the representation, and planning through the learned model — were already covered end to end by that point. Squeezing out the last failure via further hyperparameter tuning is a real, available option, just not the current priority; the project's next step is the discrete-action follow-up below rather than further tuning here. The fine-grained (non-strided) pipeline is kept for comparison/reference, not actively developed further.

A follow-up project is being considered: a **discrete** action-space domain (as opposed to `ball_in_cup`'s continuous one) — most likely a grid maze with pickup-key/unlock mechanics. What's driving that choice for this project isn't which search algorithm to swap CEM for, but what the *latent space* needs to capture once state includes discrete, combinatorial facts (has a key: yes/no; door: locked/unlocked) rather than purely continuous physical quantities — and confirming hands-on, the way `ball_in_cup` did for velocity, that a single frame is already Markovian for those facts (very likely true, and not expected to be a novel finding — this project's value is in verifying it directly rather than taking it on faith).
