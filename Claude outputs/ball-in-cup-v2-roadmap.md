# Ball-in-Cup v2: Hyperbolic Embedding Roadmap

Fresh project directory, built on what the diagnostic run on the current (v1, Euclidean) encoder found: `rel_delta` in the 0.15–0.27 range on held-out data, meaning the existing latent space is already emergently tree-like without anything forcing it to be. This roadmap moves from "measured incidentally" to "designed and exploited on purpose."

## What carries over unchanged, and why

The dm_control simulation, the action space, and the already-collected episode data (`jepa_data_strided/`, `jepa_eval_data_strided/`) have nothing to do with the embedding geometry — they're raw pixels and actions, recorded before any encoder ever sees them. Copy `collect_data_strided.py`, `collect_eval_data_strided.py`, and the existing `.npz` episode files straight into the new directory rather than recollecting. Everything downstream of "pixels in, embedding out" is what actually changes.

## Phase 0 — New project skeleton

**Do:** create the new directory, copy over `collect_data_strided.py`, `collect_eval_data_strided.py`, and the episode data directories as-is. Copy `jepa_train.py` and `jepa_train_strided.py` as a *starting point* to edit, not as final code.

**Different from before:** nothing yet — this phase is pure setup, no geometry decisions made.

## Phase 1 — Hyperbolic projection head on the encoder

**Do:** add one new module (`hyperbolic.py`) implementing the closed-form Poincaré-ball primitives: `exp0(v)` (exponential map at the origin, projects an ordinary Euclidean vector onto the ball), `dist(x, y)` (the closed-form hyperbolic distance formula), and a norm-clipping helper to keep points off the boundary where the metric blows up.

**Different from before, and from the original plan above:** the original plan was to bake `exp0` into `Encoder.forward()` so `z` *is* a ball point everywhere downstream. Implementing Phase 2 surfaced a real conflict with that: the anti-collapse `cov_penalty_term` (VICReg-style) pushes every latent dimension's variance toward 1, but a point confined to the open unit ball can never have all 32 dimensions at variance ≈1 simultaneously — the sum of squared coordinates has to stay under 1. Baking `exp0` into the encoder would put the anti-collapse term at permanent war with the ball's own geometry, likely driving everything toward the boundary where `λ(x)` blows up. Resolution: `Encoder` keeps returning a plain Euclidean vector in `R^32`, exactly as v1 did, untouched by this phase. `exp0` is applied only transiently, at the one place that actually needs a point on the manifold — see Phase 2.

## Phase 2 — Swap the loss's distance metric

**Do:** in `jepa_train_strided.py`'s training loop, wrap `z_pred` and each `z_target` with `hyp.exp0(...)` right where the loss compares them, and replace the old squared-Euclidean term with `0.5 * hyp.dist(exp0(z_pred), exp0(z_target)) ** 2`.

**Different from before:** this is the actual mechanism that makes the embedding *become* hyperbolic — the network is explicitly optimized against hyperbolic distance between ball-projected points, not Euclidean distance between raw vectors. `cov_penalty_term` keeps operating on the raw, pre-`exp0` vectors exactly as before, so the anti-collapse term never sees the ball at all (see Phase 1's note above).

Squared distance, not raw distance: `hyperbolic.dist` is built from `arccosh`, which has a sqrt-like singularity in its gradient as its argument approaches the branch point (distance → 0) — the same reason the very first derivation in this project dropped the square root from the length functional in favor of the energy functional. Squaring keeps the same minimizer (zero is still zero either way) but removes that blowup from the training signal.

## Phase 3 — Predictor: simplest viable option first

**Do:** nothing beyond Phase 1/2 — this phase turned out to already be satisfied by the Phase 1 resolution above.

**Different from before, and from the original plan above:** the original plan was to re-apply `exp0` to `Predictor`'s output before it re-enters the sliding context window. Once Phase 1 settled on keeping the whole `z_context` pipeline — encoder output, predictor output, and the window they both feed — raw and Euclidean throughout, there's nothing left for this phase to do: `Predictor`'s raw output already matches what `Encoder` produces, with no reprojection needed to keep them consistent. A fully Riemannian predictor remains a legitimate later upgrade, just not one this rebuild needs.

## Phase 4 — Training loop and optimizer

**Do:** run training essentially as `jepa_train_strided.py` already does, pointed at the new `Encoder`/`Predictor`/loss.

**Different from before:** nothing, on the optimizer side — this is the one place v2 explicitly does *not* need new machinery. Because the network's actual learnable weights are still ordinary Euclidean numbers (the manifold only enters via the fixed `exp0` map and the loss), plain Adam works unchanged. No Riemannian SGD/Adam, no parallel transport. This is the "wrapped parametrization" trick standard in the hyperbolic-embedding literature specifically to avoid that machinery.

## Phase 5 — Evaluation: add the hyperbolicity diagnostic to the existing probe suite

**Do:** add the δ-hyperbolicity script (already written and run once, this session) as a regular step alongside the existing `eval_linear_probe_strided.py` / `eval_mlp_probe_strided.py` probes — run it every N epochs on held-out episodes, same role as `probe_out_v3`'s confusion matrices.

**Different from before:** v1 never measured this at all during training; it was a one-off post-hoc check. v2 tracks `rel_delta` as a training-time diagnostic curve, the same way probe accuracy is already tracked — expect it to trend down as training proceeds, and treat a rise as a warning sign the same way a probe-accuracy drop would be.

One deliberate choice worth recording: `hyperbolicity_eval.py` measures Euclidean distance on the *raw*, pre-`exp0` embeddings, not `hyperbolic.dist` on `exp0`-projected points. The true Poincaré ball is already maximally δ-hyperbolic by construction — it has a fixed δ, a constant of the model space itself — so testing with `hyperbolic.dist` as the metric would just re-derive that built-in number and say nothing about what the encoder learned. Testing the raw Euclidean latent is what's actually informative, and it's directly comparable to v1's incidental `0.15–0.27` baseline using the identical test methodology.

## Phase 6 — Planning and inference

**Do:** nothing to the cost function. Copy `cem_planner.py`, `cem_planner_strided.py`, `jepa_evaluator.py`, `eval_linear_probe.py`, `eval_linear_probe_strided.py`, `eval_mlp_probe_strided.py`, and `drift_diagnostic_strided.py` into the new directory unchanged (Phase 0 only copied the data-collection and training scripts, so these were still missing). Re-run `drift_diagnostic_strided.py` against the new checkpoint, once trained, to see whether compounding rollout error behaves differently under the new geometry.

**Different from before, and from the original plan above:** the original plan assumed `cem_planner_strided.py`'s cost function scores candidates by some Euclidean distance between latent embeddings ("distance to goal embedding, etc.") and needed that swapped for `hyperbolic.dist`. Reading the actual `rollout_cost` function shows that assumption doesn't hold — it never compares raw `z` vectors at all. It decodes every predicted `z` through the trained position probe (`decode_position`, `W`) into physical `[cup_x, cup_z, ball_x, ball_z]` coordinates, in meters, *first*, and every piece of the cost — the cup-to-ball distance term, the relative-velocity term — is computed entirely on those decoded physical coordinates. Physical position in meters is genuinely Euclidean space; routing it through `hyperbolic.dist` would be a category error, not an upgrade. The cost function is correct as-is and needs no change.

One real prerequisite this phase surfaced, for later: `decode_position`'s probe `W` was fit against v1's Euclidean encoder, so it has no relationship to v2's differently-trained latent space — `eval_mlp_probe_strided.py` needs to be re-run to refit a new `W` against a v2 checkpoint before this planner can produce anything meaningful. That's a data/retraining step, already covered by Phase 5's unchanged probe scripts, not a code change here.

## Explicitly out of scope for this rebuild

Multi-timescale / hierarchical predictors (a coarse long-stride `P` plus the existing fine-grained one) are a well-motivated follow-on — they'd directly address the compounding-error concern already flagged in `jepa_evaluator.py`'s own docstring, and they pair naturally with hyperbolic geometry's capacity for branching structure. But it's a separate architectural project with its own training target, not a Phase 7 of this rebuild. Get the single-timescale hyperbolic version working and measured first.

There's a second, more concrete reason to eventually pick this follow-on up, worth recording now while it's fresh: **CEM planning cost is roughly `O(T)` per candidate sequence** — one predictor call and one cost evaluation per planning step, `T` steps to cover whatever real-time horizon the planner cares about. The per-step cost formula itself (Euclidean distance today, `hyperbolic.dist` after this rebuild) is essentially the same order of cheapness either way, so this rebuild alone doesn't change `T` or CEM's total compute. What *would* cut `T` is planning with a bigger stride — fewer, longer predictor jumps covering the same horizon. A bigger stride means each jump has to represent a more divergent set of possible futures in one step, and in Euclidean space that divergence blurs together unless you add dimensions (the same dimension-scales-with-tree-size problem the δ-hyperbolicity test is about); in hyperbolic space that divergence stays cleanly separated at a fixed, even small, dimension. So the hyperbolic rebuild is what *permits* a larger predictor stride without losing resolution — meaning a future coarse-stride planning predictor, built on top of this rebuild, wouldn't just fight compounding error, it would make CEM itself measurably cheaper by needing fewer planning steps for the same horizon. This rebuild doesn't cash that benefit in on its own — nothing here changes `predictor_stride` — but it's the reason the hierarchical follow-on is worth prioritizing once v2 is validated, not just a nice-to-have.

**Design note for whenever the hierarchical follow-on gets built — the "nearsighted encoder" risk.** The plan discussed for it: keep `Encoder` exactly as v2 trained it (frozen), and just add new `Predictor` heads `P_k` at doubling strides (`k=1,2,4,8,...`), each taking `k` future actions and trained against `k`-step windows pulled from `jepa_data/`. Reusing the same `Encoder`/`hyperbolic.py` machinery for every level, untouched, is the clean, isolate-one-variable-at-a-time way to test the hierarchy — but it carries a real risk worth remembering: `Encoder` was only ever pressured to retain what matters for `PREDICTOR_STRIDE=5` accuracy. For a physics-based system, the true state (cup/ball position + velocity) sufficient to predict 5 steps ahead is, in principle, also sufficient to predict 50 steps ahead — dynamics don't need different state variables at different horizons. But the bottleneck isn't guaranteed to learn that *ideal* minimal state; something negligible at 5 steps but compounding into dominant error by step 40 (the same compounding-error phenomenon that made the original stride=1 rollout unusable) could get discarded, since nothing in the stride=5 loss ever punished losing it. If that happens, no amount of `P_k`-head training on top can recover information the frozen encoder never kept.

Mitigation, not a reason to abandon the frozen-encoder starting point: treat the frozen-encoder version as a **diagnostic**, not a final architectural commitment. Train `P_1, P_2, P_4, ...` against the frozen v2 encoder first, then check whether the coarse levels' rollout accuracy is actually bottlenecked by representation (plateaus well below what seems achievable) or holds up fine. If it holds up, the harder joint-training problem was never needed. If it's clearly capped, that's real evidence — not a guess — that unfreezing `Encoder` and jointly fine-tuning it against all the stride levels' losses together is the next, deliberate experiment, with a clean before/after to compare against.

**Status update — the diagnostic ran.** `jepa_train_multistride.py` built exactly this: `P_1, P_2, P_4, P_8, P_16`, five fully independent `Predictor` instances, each trained against the frozen v2 encoder on `jepa_data/`. The held-out drift diagnostic (`drift_diagnostic_multistride.py`, run against `jepa_eval_data/`) came back flat across all five — no blowup at any stride out to a 150-real-step/3s horizon, matching v1/v2's own convention — and, past stride=2, ball position error actually *improved* with stride (0.086m at stride 1 down to 0.039m at stride 8, ticking back up slightly to 0.051m at stride 16), consistent with the original fewer-hops-less-compounding rationale rather than the nearsighted-encoder failure mode. That's evidence the frozen encoder is not badly bottlenecked, at least through stride 8 — the case for unfreezing wasn't forced by a clear cap, but a joint fine-tuning pass (`jepa_train_joint.py`, unfreezing `Encoder` and training it together with all five predictors against a shared per-anchor-point loss, warm-started from the frozen-run checkpoints) is in progress anyway, both to push further and because it structurally closes the cross-stride-consistency gap noted below.

**Design note: coupling the stride levels into a cascade, instead of training them independently (not yet built).** All five predictors above are, and remain even after the joint fine-tune, structurally independent networks — nothing but a shared loss ties `P_16` to `P_8` to `P_4`, etc. Andreas raised a different structure worth keeping for whichever comes next: couple the levels into an explicit cascade, where level `i`'s predictor doesn't take raw context — it takes level `i-1`'s *output* (the stride `2^(i-2)` prediction) plus the next block of new actions, and produces the stride `2^(i-1)` result in one call. Reaching stride 16 then costs 5 cascade evaluations (one per doubling), not 16 — genuinely `O(log(stride))` compute for `O(stride)` reach, unlike literally chaining a single fine predictor stride-many times (already rejected earlier in this project's discussion for defeating the entire point of building coarse levels: that costs `O(stride)` calls, the same cost as native resolution). It also gets cross-level consistency for free, structurally, rather than via an added telescoping-consistency loss — level `i` can't help but agree with level `i-1`, because it's built directly from `i-1`'s own output. Two open design questions before this is buildable: (1) what "context" means at level `i` — presumably built from `i-1`'s recent output(s) rather than `k_context` real frames spaced `2^(i-1)` apart, and the "3 points for a robust velocity estimate" reasoning that originally set `K_CONTEXT=3` needs to be re-derived for this setting; (2) training regime — teacher-forcing level `i` on the real intermediate frame during training reopens the same exposure-bias problem the multi-step unroll fix (v7) solved for a single flat predictor, now once per cascade level, and needs the same kind of fix reapplied at each level.

## Reference: every file that changes, in one place

```
collect_data_strided.py          unchanged, copy as-is
collect_eval_data_strided.py     unchanged, copy as-is
jepa_data_strided/                unchanged, copy as-is (raw episodes)
jepa_eval_data_strided/           unchanged, copy as-is (raw episodes)

hyperbolic.py                     NEW — exp0(v), dist(x,y), boundary clipping

jepa_train_strided.py             edited:
  Encoder                           unchanged — still returns a raw Euclidean vector (Phase 1)
  Predictor                         unchanged — still returns a raw Euclidean vector (Phase 3)
  loss function                     0.5*||z_pred-z_target||^2  ->  0.5*hyp.dist(exp0(z_pred), exp0(z_target))^2
  cov_penalty_term                  unchanged — operates on raw z, never sees the ball (Phase 1)
  optimizer                         unchanged (plain Adam)

eval_linear_probe.py              unchanged, copy as-is
eval_linear_probe_strided.py      unchanged, copy as-is
eval_mlp_probe_strided.py         unchanged, copy as-is — re-run against a v2 checkpoint to refit W (Phase 6)
hyperbolicity_eval.py             NEW — delta-hyperbolicity check on raw z (comparable to v1's 0.15–0.27 baseline), run periodically

jepa_evaluator.py                 unchanged, copy as-is
cem_planner.py                    unchanged, copy as-is
cem_planner_strided.py            unchanged, copy as-is — cost operates on decoded physical positions, not latent z, so no hyperbolic swap applies (Phase 6)
drift_diagnostic_strided.py       unchanged, copy as-is; re-run against the new checkpoint
```
