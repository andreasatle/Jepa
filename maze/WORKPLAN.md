# Maze JEPA — workplan (draft, for review)

Nothing here is implemented yet. This is a proposal to work from — flag anything you'd change before we start on Phase 0.

## Goal

Understand whether hyperbolic-geometry latent prediction earns its keep on problems with genuinely branching futures, in general — not just whether it happens to work on one hand-picked task. This follows directly from ball_in_cup_hyperbolic, where the "radial hierarchy" hypothesis was cleanly disconfirmed and where we concluded the architecture never exploited hyperbolic space's real advantage — representing several simultaneously-plausible, mutually-exclusive futures as distinguishable points — because the predictor only ever outputs a single point, and ball-in-cup's dynamics are a single smooth basin with no genuine forks anyway. The maze is a strong candidate domain for testing this, not the object of study itself — Phase 0 and the domain-fit check below decide whether it's actually a good fit before we build it out.

## Fundamental vs. convenience, restated for this project

Fundamental (what we're actually trying to test here): hyperbolic space's volume grows exponentially with radius, so it can hold an exponentially-branching tree of distinguishable futures at low dimension without them colliding — something Euclidean space can't do at the same dimension. This only matters if the predictor's output is genuinely allowed to represent more than one point.
Convenience (what ball_in_cup_hyperbolic actually did): using `hyp.dist` as the loss function for an otherwise-ordinary deterministic point predictor. Same mechanism as Euclidean MSE, different distance function — this is the part we now know doesn't do anything special on its own.
This project's whole point is to move from the convenience use to a design that can actually exercise the fundamental one, and to check honestly whether that exercise pays off.

## Core hypothesis under test

When the true future is genuinely multimodal (a fork — two or more mutually exclusive, equally-valid reachable outcomes), a predictor whose loss/geometry is hyperbolic represents and separates those outcomes better than an otherwise-identical predictor using Euclidean MSE, and this shows up as a measurable difference, not just a different-looking loss curve.
Falsifiable directly: if the hyperbolic predictor's behavior at a fork is not meaningfully different from the Euclidean one's, the hypothesis is disconfirmed for this task too, the same way the radial-hierarchy one was.

## Why a maze

A grid maze with discrete actions gives real forks by construction. A junction cell has two or more reachable, mutually exclusive future paths — unlike ball-in-cup, where nearby trajectories blend into a continuum. This is the property ball-in-cup was missing.

## Phased plan — cheapest, most decisive experiment first

Ordered so the cheapest, most decisive checks come first — cheap enough to reconsider or adjust course early if something doesn't hold up, rather than building the full maze + planner stack on faith (JAS Gripen caution, still applies).

### Phase 0 — parametric sweep on synthetic multimodal targets (no maze, no dm_control, no images) — done, see "Phase 0 result" below

Isolates the one mechanical claim everything else depends on, and does it in a way that generalizes beyond the maze: does a point predictor trained with a hyperbolic distance loss on a genuinely multimodal target land relatively closer to a real mode than one trained with Euclidean MSE, which is known to collapse toward the middle of its targets — and under what conditions?
Two design traps were found and fixed before writing the script, both confirmed numerically rather than assumed — see the script's own docstring (`multimodal_sanity_check.py`) for the full detail:
Two modes, any separation — the squared-distance minimizer is forced to the exact metric midpoint in BOTH geometries, by symmetry. A two-mode test is guaranteed to show no difference, independent of whether hyperbolic geometry helps anywhere. Not usable.
Three or more modes arranged symmetrically (a regular polygon) — same flaw: the configuration's own rotational symmetry forces both geometries to the centroid. Also not usable.
What actually works: three or more modes placed asymmetrically (random, not on a regular polygon), swept over mode count, separation, and embedding dimension, averaged over several random placements per setting — confirmed this lets the two geometries diverge before committing to the full sweep.
Metric: how close the optimum lands to the nearest actual mode, relative to the typical spacing between modes, for both geometries side by side — lower is better (closer to something real, not lost in the empty middle).
Still cheap: ~400 small optimizations of a single point each, seconds to run in total, no maze environment touched at all. The result of this phase is what lets us check any candidate domain — the maze or otherwise — against a real reference before committing to building it.

### Domain-fit check — before building Phase 1

Before writing any maze-specific code: check the maze's actual structure against the regime Phase 0 found, using arithmetic on the maze design, not a simulation — how many genuine forks does a typical episode pass through, how separated are the branch outcomes in coordinate space, does the branching factor fall inside the range Phase 0 showed an effect for. This is the step meant to catch a bad-fit domain before any investment, the same way ball-in-cup's smooth-basin problem should have been caught before four training runs rather than after.
Considered and rejected for this role: Gromov δ-hyperbolicity (`hyperbolicity_eval.py` from ball_in_cup_hyperbolic). It looked like a ready-made domain-fit signal, but checking it against controls showed it doesn't discriminate — ball-in-cup's raw physics state scored rel_delta ≈ 0.20, and so did pure i.i.d. random noise with no structure at all (≈ 0.21–0.27), while an actual tree metric scores 0.0000 under the same method. It measures something real — tree-likeness from a shared-origin/radiating structure that's close to generic for any rollout-shaped dataset — just not the specific thing this project needs (genuine discrete, mutually-exclusive branching). Not worth reaching for here without pairing it against those same controls every time.

### Phase 1 — minimal maze environment + data collection

Only if Phase 0 doesn't disconfirm the idea outright, and the domain-fit check above says the maze actually sits in the regime where Phase 0 showed an effect.
Small procedurally-generated grid maze, discrete action space (up/down/left/right; key pickup + unlock deferred — see open decisions).
Data collection needs to deliberately include junctions: episodes/rollouts that pass through cells with 2+ valid onward paths, so the dataset actually contains multimodal futures rather than mostly-unimodal corridors.

### Phase 2 — Euclidean vs. hyperbolic predictor comparison

Same E/P architecture split as ball_in_cup_hyperbolic, two otherwise-identical predictors differing only in loss geometry (Euclidean MSE vs. `hyp.dist`), trained on the same junction-containing dataset.
Needs a concrete metric decided before training, not chosen after looking at results — candidates: can a downstream probe recover which branch was actually taken from the embedding; is there a measurable separation margin between branch-conditioned embeddings; does prediction error at junctions differ from the Phase 0 midpoint-collapse pattern.

### Phase 3 — planner integration (only if Phase 1–2 show a real, not marginal, effect)

Wire the winning predictor into a discrete planner (tree search or CEM-style over discrete actions) and check whether the representational difference actually changes maze-solving behavior or success rate, not just an offline metric — the same bar the ball-in-cup CEM planner work applied throughout.

## Evaluation stages

Not a pass/fail gate — each phase is a checkpoint with three possible outcomes: continue as planned, adjust something specific and re-test, or set the direction aside for now. "Set aside" is scoped to that specific direction, not a verdict on hyperbolic geometry as a whole, the same way ball-in-cup's disconfirmation didn't say anything about the maze.

### After Phase 0

Clear, consistent separation across the swept regime (hyperbolic avoids the collapse Euclidean shows) → continue to Phase 1 with the regime in hand.
Effect present but only in part of the swept regime → adjust: narrow Phase 1's maze design to sit inside the regime where Phase 0 actually showed something, rather than building blind.
No meaningful difference anywhere in the sweep → set aside the maze/geometry direction for now; the mechanism this whole project depends on isn't showing up even in the easiest synthetic setting, so building the maze on top of it wouldn't tell us anything new.

### After Phase 2

Separation advantage roughly matches what Phase 0 predicted → continue to Phase 3.
Some effect, but smaller or narrower than Phase 0 suggested → adjust: revisit maze design (more junctions, tighter separation) or the metric itself before deciding either way.
No effect, or it contradicts Phase 0's prediction → worth understanding why before concluding anything — could mean the maze doesn't actually sit in the regime we thought, which is itself useful information — but likely set aside before investing in a planner.

### After Phase 3

Representational advantage translates into better planner behavior → the finding actually pays off, worth keeping and building on.
Representational advantage exists but doesn't change planner behavior → a real, useful result in its own right (the same lesson the ball-in-cup CEM ranking check already taught us: a metric difference isn't automatically decision-relevant), but a reason to stop investing further here, not a failure.
No representational advantage → already resolved at Phase 2.

## Open design decisions (need your input, not decided yet)

Key/lock mechanic: memory has this as "a maze with one or two locks" — include from the start, or start key-less and add it once the core branching test is working?
Discrete vs. continuous latent for E: should the encoder output a small discrete/codebook-style latent (natural fit for a discrete maze, arguably a fairer test of "does the geometry help represent discreteness") or stay continuous and rely only on the loss geometry, matching ball-in-cup's setup for comparability?
How multimodality gets constructed for training: rollouts under a stochastic/random policy that sometimes takes each branch (simplest), vs. predicting several steps ahead without observing the action sequence (closer to genuine partial-observability uncertainty, more work to set up).
Metric for Phase 2: pick one of the candidates above (or another) before training starts, so we're not choosing the metric after seeing which one looks better.

## Planned files (none exist yet — sketch only, for reference)

`maze_gen.py` — perfect-maze generation (Prim's + DFS), extracted from domain_fit_check.py, shared with env.py. Also has `bfs_distance()` (exact shortest-path distance, added for maze_planner.py's horizon sizing and success grading). Written.
`domain_fit_check.py` — written and run: k>=8 reached by H=5 (Prim's) / H=8 (DFS) at every grid size tested (6/10/16); Prim's has 2-3x the junction density of DFS. Recommendation: Prim's, size 10, horizon ~12 for margin against per-instance variance. Separation between reached endpoints not yet measured, only count.
`env.py` — grid maze (MazeEnv class), step/reset/valid_actions, fixed 4-action space with wall-bump-as-no-op, ASCII self-test render. Written and self-tested; defaults GRID_SIZE=10, ALGO="prim" per the domain-fit numbers above.
`collect_data.py` — rollout collection: ONE fixed maze, every rollout starts at a junction cell, horizon 20, saved as maze_data.json. P is designed to predict from (start, horizon) WITHOUT the action sequence -- deliberately, so genuine target multimodality survives into training (see the script's own docstring for why conditioning on actions would remove it). Written and smoke-tested.
`multimodal_sanity_check.py` — Phase 0, the parametric sweep (>=3 asymmetric modes), standalone, no dependency on `env.py`. Written; not yet run.
`hyperbolic.py` — copied in from ball_in_cup_hyperbolic/, unmodified (this workspace's convention: each project directory is self-contained).
`domain_fit_check.py` — the domain-fit check, arithmetic over the maze design against Phase 0's swept regime, no training involved.
`jepa_model.py` — the model itself: Encoder (E, per-cell embedding table), Predictor (P, Phase 2's horizon-conditioned one), OneStepPredictor (Phase 3's action-conditioned one), to_geometry/squared_dist, and the VICReg anti-collapse pieces. Split out of jepa_train.py so other scripts (the planner, a diagnostic) can import the model without the training/CLI code.
`jepa_train.py` — data loading, train/holdout split, training loop, evaluation, `--geometry euclidean|hyperbolic|both` CLI. Imports the model from jepa_model.py. Now also saves a checkpoint per geometry (`jepa_checkpoint_<geometry>.pt`: E, P, geometry, num_cells, size, max_horizon, dim, hidden) -- needed for Phase 3. See "Phase 2 architecture" below.
`jepa_train_step.py` — Phase 3, step 1: trains OneStepPredictor (action-conditioned, one real step) on top of a FROZEN E loaded from jepa_train.py's checkpoint, using the actions already sitting unused in maze_data.json. Plain regression, no VICReg needed (deterministic target). Saves `jepa_step_checkpoint_<geometry>.pt`. Written and run.
`maze_planner.py` — Phase 3, step 2: CEM planner over the frozen E + OneStepPredictor, discrete/categorical version of ball_in_cup_hyperbolic's CEM (per-timestep action-frequency refit instead of a Gaussian). Horizon per trial = bfs_distance(start,target) + margin, using maze_gen.py's new `bfs_distance()`. Plans purely in latent space, then executes the plan for real on MazeEnv to check success rate and path efficiency vs BFS-optimal, Euclidean vs hyperbolic. Written and run twice, 0/40 both geometries both times. See "Phase 3: planner, built and run" below.
`branch_diagnostic.py` — Phase 2 evaluation, the decided-in-advance metric.
`step_drift_diagnostic.py` — tests whether P_step's autoregressive rollout error explodes with chain length (exposure bias), by chaining it along the TRUE BFS-optimal action sequence (via maze_gen.py's new `bfs_path_cells()`) both teacher-forced and free-running, and comparing the two error curves. Written, not yet run.
`README.md` — project summary, once the shape of the thing is actually known.

## Phase 0 result

Run: k in {3, 5, 8} modes, dim in {2, 4, 8}, separation in {0.3, 0.8, 1.5, 2.5}, 20 random asymmetric mode placements per cell, paired t-test + Wilcoxon on the per-trial (Euclidean − hyperbolic) difference in nearest-mode/spacing ratio.
Clean crossover by branching factor (k), not a uniform effect: at k=3, Euclidean wins significantly wherever there's a significant result at all (dim 2 and 4, every separation tested). At k=5, mostly no significant effect either way. At k=8, hyperbolic wins increasingly strongly and consistently as dimension and separation grow — most clearly at dim=8, where separation 0.8/1.5/2.5 are all p<0.001, and the effect size grows monotonically with separation (mean diff 0.0225 → 0.0386 → 0.0526).
This matches the theoretical story rather than just looking like noise: hyperbolic's advantage is about representing several distinguishable futures without collision, so it should show up specifically as branching factor grows, not everywhere uniformly — and that's exactly the shape the result has. At low branching factor there's nothing to gain and the curvature costs a little; at high branching factor it starts paying off, increasingly so with more separation.
Per the evaluation-stage criteria above, this is "effect present but only in part of the swept regime," not a clean continue — the maze needs to specifically reach the regime where the effect showed up (roughly k >= 8, with dimension and separation helping) for Phase 1-2 to be worth building on this basis.
Practical implication for the maze: a typical grid maze's per-step branching factor is small (2-3 directions at a junction) — nowhere near k=8 on its own. But predicting several steps ahead through a junction compounds this combinatorially (3 steps at branching factor 3 already reaches up to 27 distinct reachable end-states), so the domain-fit check and Phase 1's data collection should target a multi-step-ahead prediction horizon through junctions, not single-step transitions, to land in the regime this phase actually found favorable.

## Phase 2 architecture

E: a learned embedding table, one free vector per grid cell -- no CNN/MLP needed, unlike ball-in-cup's pixel input, since the maze's "observation" already IS the state (a discrete cell index). E always outputs the raw/tangent-space vector; a separate `to_geometry()` step maps it onto the manifold (identity for Euclidean, `exp0` for hyperbolic) only where a distance is actually computed.
P: `(E(start_cell), horizon) -> predicted end latent`, same raw/tangent convention. horizon is embedded and concatenated with the start latent, then a small MLP. Deliberately NOT conditioned on the action sequence -- see collect_data.py's docstring: conditioning on actions in a deterministic maze removes all genuine ambiguity, defeating the point.
Loss: squared distance between P's (mapped) prediction and E(actual end cell) (mapped) -- Euclidean MSE vs. `hyp.dist`^2 -- averaged over sampled (start, horizon, end) triples.
Anti-collapse (VICReg, ported from ball_in_cup_hyperbolic unchanged): the first run of jepa_train.py collapsed -- both geometries' training loss went to ~0.00000 within 500 steps, which is only possible if E stopped being informative (a genuinely multimodal target can't legitimately hit near-zero expected squared error against a live encoder). Fixed by adding the same VICReg covariance penalty ball-in-cup uses, applied to E's raw start/end embeddings each batch (never to P's prediction): `loss = inv_loss + kappa/2 * (cov_penalty(E(start)) + cov_penalty(E(end)))`, `kappa=1.0`. Watch `avg_cov_penalty` in the printed log on future runs -- this is the mechanism that keeps the Phase 0 comparison meaningful at all here.
Evaluation: same ratio metric as Phase 0 (nearest true endpoint distance / mean pairwise endpoint spacing), computed per (start, horizon) on held-out rollouts (80/20 split per start), paired t-test + Wilcoxon across junction starts, swept over horizons {3, 5, 8, 12, 20} to see whether the Phase 0 k>=8 crossover shows up as horizon grows (domain_fit_check.py found mean k crosses 8 by H=5 with Prim's).

## Phase 2 result

First VICReg-fixed run (960 episodes, 40 rollouts/junction, 8 holdout/junction after the 80/20 split): Euclidean won significantly at every tested horizon. Traced with eval_k_diagnostic.py (new, pure stdlib -- counts distinct true endpoints actually present in the exact holdout split jepa_train.py evaluates against) to an artifact, not a real result: only 8 holdout rollouts/junction structurally caps the OBSERVABLE distinct-endpoint count at 8, and in practice it averaged 4.4-5.75 across all horizons -- we were never actually inside the k>=8 regime Phase 0 found an effect in, regardless of what domain_fit_check.py's own larger-sample (300 rollouts) measurement said about the maze's true branching.
Fixed by bumping collect_data.py's ROLLOUTS_PER_JUNCTION 40 -> 200 (3840 train / 960 holdout episodes, 40 holdout rollouts/junction) and re-running. eval_k_diagnostic.py confirmed this dataset genuinely reaches the target regime: mean observed k in holdout goes 7.0 (H=3) -> 9.0 (H=5) -> 10.8 (H=8) -> 12.1 (H=12) -> 14.9 (H=20), with ALL 24 junctions at k>=8 by H=20 (min k=9 there).
Re-run result, now genuinely inside the regime: Euclidean still wins significantly at H=3 (p=0.005) and H=5 (p=0.002), matching Phase 0's low-k finding. The gap narrows as horizon (and k) grows -- H=8 borderline (p=0.037, Wilcoxon not significant), H=12 still significant (p=0.012) -- and by H=20, deep in the regime (mean k=14.9, min k=9, all junctions k>=8), there is NO significant difference (t_p=0.505, w_p=0.726): a statistical tie, not a hyperbolic win.
Honest read: this does NOT reproduce Phase 0's predicted crossover (hyperbolic increasingly WINNING as k grows past 8) -- it shows Euclidean's edge eroding to a tie, which is a different and weaker outcome. Leading hypothesis, not yet tested: Phase 0 fit one FREE point per independent setting, with full control over separation and no interaction between settings. Phase 2's E is one single embedding shared across all 100 cells and every (start, horizon) pair simultaneously -- the "modes" for any one (start, horizon) key are real maze cells that also have to serve as targets/starts for many other keys throughout the whole dataset. This entangled, jointly-constrained setup is a meaningfully harder problem than Phase 0's isolated per-setting fit, and nothing in Phase 0 guarantees the advantage survives that additional constraint.
Per the evaluation-stage criteria above, this is squarely "some effect, but smaller/narrower than Phase 0 suggested, or contradicts Phase 0's prediction -- worth understanding why before concluding." Not a clean continue to Phase 3; not yet a clean set-aside either.

## Latent dimension sweep (8 / 16 / 32), same dataset

Tested whether more room for E (LATENT_DIM 8 -> 16 -> 32) would strengthen a hyperbolic advantage at high horizon, given Phase 0's dim=8 was tuned for one isolated free k-point set, not Phase 2's 100-cell entangled shared embedding.
H=20 ratio (eucl / hyp) by dimension: dim=8: 0.185 / 0.202 (eucl slightly ahead, n.s.). dim=16: 0.250 / 0.213 (hyp ahead for the first time, n.s., p=0.12 -- but eucl got WORSE, not hyp much better). dim=32: 0.215 / 0.223 (eucl slightly ahead again, n.s.) -- the dim=16 crossover reversed rather than strengthening.
No case at any dimension reached significance in hyperbolic's favor at H>=12. At low horizon (H=3,5,8) Euclidean wins clearly and significantly at every dimension tested, and the gap there got WIDER (more significant, not less) going from dim=16 to dim=32.
Honest read: this disconfirms the "needs more dimension" hypothesis specifically -- three settings, no monotonic trend, the one hyperbolic-favoring data point didn't reproduce. Across the whole dimension sweep, high-horizon behavior is best described as a statistical tie between geometries, not a crossover, and low-horizon behavior clearly favors Euclidean. This is now real, repeated evidence against Phase 2's original hypothesis (hyperbolic should increasingly win once k>=8) in this specific entangled shared-embedding architecture -- not just an underpowered single run.
Remaining honest levers, not yet tried, before treating this as fully set aside: n=24 is a hard ceiling from one fixed 10x10 maze's junction count -- a bigger maze (more junctions) or pooling across several independently-generated mazes would give more statistical power to catch a real but small effect this design is currently too underpowered to see. Dimension alone is not that lever.

## Bigger maze (size 16, 70 junctions), LATENT_DIM=16

Dimension sweep found no monotonic trend at n=24, so switched levers: GRID_SIZE 10 -> 16 (confirmed via maze_gen.py: 70 junctions vs 24, domain_fit_check.py already showed k>=8 reachable by H=5 at size 16 too), collect_data.py re-run (11200 train / 2800 holdout episodes, 40 holdout rollouts/junction), jepa_train.py re-run at LATENT_DIM=16 (reverted from the inconclusive 32).
First real crossover, matching Phase 0's predicted SHAPE: H=3 eucl=0.3835/hyp=0.4518 (EUCL wins, t_p=0.024 significant, w_p=0.154 not). H=5 eucl=0.3283/hyp=0.3737 (EUCL, not significant). H=8 eucl=0.3213/hyp=0.3145 (HYP ahead numerically, not significant). H=12 eucl=0.2733/hyp=0.2594 (HYP ahead numerically, not significant). H=20 eucl=0.2330/hyp=0.2152 (HYP ahead, w_p=0.043 SIGNIFICANT on Wilcoxon, t_p=0.370 not significant on the paired t-test).
This is the first time in the whole session (maze or ball-in-cup) that hyperbolic has shown a significant advantage on anything other than Phase 0's synthetic setting -- worth taking seriously, but the t_p/w_p split at H=20 is a real caveat: Wilcoxon (rank-based, robust to outliers/skew) hit significance while the paired t-test (mean-based) didn't, which usually means the effect isn't uniform across all 70 junctions -- either a consistent-but-skewed small effect, or a subset of junctions driving it while others are flat or reversed. Not yet checked which.
Not yet tried: dumping the per-junction signed differences at H=20 to see the actual distribution (how many of the 70 favor hyperbolic vs Euclidean, and by how much) before treating this as a settled positive result.

## Dimension re-sweep on the bigger maze (n=70) -- real result, and a course correction

dim=8 (added the hyp-better/eucl-better sign count to the printed table first): HYPERBOLIC WINS CLEARLY at every horizon. H=3: eucl=0.4816/hyp=0.4000, t_p=0.0017 (46 junctions hyp-better vs 24 eucl-better). H=5: eucl=0.4011/hyp=0.3602, w_p=0.0197 (41 vs 29). H=8: eucl=0.3478/hyp=0.2779, t_p<0.0001 (51 vs 19). H=12: eucl=0.3079/hyp=0.2331, t_p<0.0001 (55 vs 15). H=20: eucl=0.2606/hyp=0.2061, t_p<0.0001 (51 vs 19). Broad and consistent across the 70 junctions, not a handful of outliers -- this is a genuinely strong, significant, robust result, the first of its kind anywhere in this session (maze or ball-in-cup).
This CONTRADICTS the direction I (Claude) was pushing dimension in after the n=24 sweep -- I suggested more dimension might help hyperbolic ("needs more room for the entangled embedding"), but dim=16 on this same maze showed only a weak/mixed result (Euclidean winning at low H, hyperbolic barely ahead and not clearly significant at high H -- see "Bigger maze" run above), and dim=8 now shows the strongest, cleanest win of anything tried. On reflection this matches the actual theoretical claim better than my dimension-up instinct did: hyperbolic space's advantage over Euclidean is specifically that it can hold an exponentially-branching structure at LOW dimension where Euclidean can't -- see WORKPLAN's own "Fundamental vs convenience" section above. At high enough dimension, Euclidean space also has plenty of room to spread out the same number of distinguishable points, so the relative advantage should SHRINK as dimension grows, not grow. dim=8 -> strong win, dim=16 -> weak/mixed is consistent with that; predicts dim=32 (not yet re-run on this maze, only on the old n=24 one) should show little to no advantage here too.
Not yet done: re-running dim=32 specifically on this size-16 maze (n=70) to confirm the predicted shrink-at-high-dimension pattern and complete the story.

## Dimension re-sweep, completed -- headline finding of the project

dim=32 result (same size-16 maze, n=70, same dataset as dim=8/16): EUCLIDEAN WINS CLEARLY at every horizon, reversing all the way back. H=3: eucl=0.2801/hyp=0.5065, t_p<0.0001 (56 of 70 junctions eucl-better). H=5: eucl=0.2397/hyp=0.4187, t_p<0.0001 (56 eucl-better). H=8: eucl=0.2342/hyp=0.3545, t_p<0.0001 (46 eucl-better). H=12: eucl=0.2159/hyp=0.3300, t_p=0.0001 (48 eucl-better). H=20: eucl=0.2033/hyp=0.2805, t_p=0.0045 (41 eucl-better).
Full three-point dimension story on the identical size-16 maze/dataset (n=70 throughout): dim=8 -> hyperbolic wins broadly and strongly at every horizon (46-55/70 junctions, mostly p<0.0001). dim=16 -> weak/mixed (Euclidean wins at low H, hyperbolic barely ahead and inconsistently significant at high H). dim=32 -> Euclidean wins broadly and strongly at every horizon (41-56/70 junctions, mostly p<0.0001) -- a full reversal, not just a fade to zero.
This is a clean, monotonic, strongly-evidenced crossover directly confirming this project's original theoretical claim ("Fundamental vs convenience" section above): hyperbolic's advantage over Euclidean is specifically a LOW-DIMENSION phenomenon -- at high enough dimension Euclidean has plenty of room too, and here it doesn't just catch up, it wins outright. This is the first unambiguous, broad, statistically strong confirmation of the hyperbolic-geometry hypothesis anywhere in this project (maze or ball_in_cup_hyperbolic) outside Phase 0's synthetic setting.
Open question: dim=8 was the smallest value tried -- unknown whether the hyperbolic advantage keeps growing below 8 (e.g. dim=4) or dim=8 is already near a peak for this maze's actual branching structure. Also unknown whether this specific dimension-crossover point (advantage somewhere between 8 and 16) is a property of the maze's branching factor specifically, or would shift with a different maze/grid size.

## Dimension sweep, summary

Phase 0 (synthetic, real signal at k>=8) -> domain-fit check -> env.py -> collect_data.py -> jepa_train.py collapsed -> VICReg fix -> eval_k_diagnostic.py caught an undersized-holdout artifact -> re-run with more rollouts/junction -> dimension sweep at n=24 (8/16/32), underpowered/no clear trend -> bigger maze (size 16, n=70) -> full dimension re-sweep on n=70: dim=8 strong hyperbolic win, dim=16 weak/mixed, dim=32 strong Euclidean win. Clean, monotonic, well-evidenced crossover. See "Dimension re-sweep, completed" above.
This is a real, positive, well-supported finding for the core hypothesis (hyperbolic geometry helps represent branching futures, specifically at low latent dimension) -- the strongest and clearest result in the whole project.
Chose to move to Phase 3 (planner) at dim=8, the setting that won, rather than narrowing the dimension curve further first. Still open for later: narrowing the crossover (dim=4, 10/12), and checking robustness on a second maze seed.

## Phase 3: planner, built and run -- 0/40 both geometries, root cause still open

Chose (b) from the open choices above: build the planner at dim=8, the setting that actually won, rather than narrowing the dimension curve further first.

Key realization before building: Phase 2's P cannot be reused for planning -- it deliberately has no action input (see collect_data.py's docstring), so there is nothing to optimize over. A planner needs a genuinely different, action-conditioned model. Built the version WORKPLAN originally sketched (not the cheaper "use E as a heuristic over real simulated rollouts" alternative) -- see the conversation for why both were considered.

New pieces: `OneStepPredictor` in jepa_model.py -- (E(cell), action) -> E(next cell), trained with E FROZEN so it tests whether the already-learned dim=8 representation supports one-step prediction, not a new representation. Plain regression, no VICReg (the maze's per-step transitions are deterministic, so there's no multimodality to guard against here, unlike Phase 2's P). `jepa_train_step.py` trains it, reusing the actions already sitting unused in maze_data.json -- no new data collection needed. `maze_planner.py` runs discrete CEM (categorical per-timestep action distribution, refit from elites each iteration -- the discrete analog of ball_in_cup_hyperbolic's continuous Gaussian CEM) purely in latent space, then executes the chosen plan for real on MazeEnv to check success and path efficiency vs BFS-optimal, comparing geometries on the same sampled (start, target) pairs. `bfs_distance()` added to maze_gen.py, used both to size each trial's planning horizon fairly and to grade the resulting path.

jepa_train.py now also saves a checkpoint per geometry (E, P, and enough metadata to reconstruct them) -- this didn't exist before and was a real gap, since nothing was previously saved anywhere after training.

Deliberately testing the narrower, harder question WORKPLAN's own Phase 3 bar sets: this maze is small enough that plain BFS solves it exactly and instantly, so the point isn't "can a planner solve the maze" -- it's whether the hyperbolic embedding's distance function is a genuinely better cost signal for CEM than the Euclidean one's, in a way that shows up in success rate and path efficiency, not just an offline embedding-distance number.

Run, in order (all torch, run by Andreas):
1. `uv run python jepa_train.py --geometry both` (produces jepa_checkpoint_euclidean.pt / jepa_checkpoint_hyperbolic.pt at dim=8 -- old runs never saved anything)
2. `uv run python jepa_train_step.py --geometry euclidean` and `--geometry hyperbolic`
3. `uv run python maze_planner.py --geometry both`

### First run: 0/40 success, both geometries

`maze_planner.py --geometry both` gave `euclidean: success_rate=0.00 (0/40)` and `hyperbolic: success_rate=0.00 (0/40)`. Too uniform a failure across both geometries to be "the search is just hard" -- that pattern says something structural in the planner itself, not a fair comparison that happened to go badly.

Found one real bug: `cem_plan` was decoding the chosen action sequence as the ARGMAX of the final iteration's per-timestep marginal distribution. That's a fine shortcut for continuous CEM (ball_in_cup_hyperbolic's planner does this -- nearby-the-mean is still reasonable in a smooth space), but wrong for a discrete maze: different elite sequences can represent genuinely incompatible paths through different junctions, so independently-decoded per-timestep marginals can stitch together a hybrid sequence that was never actually sampled or scored as a whole. Fixed: `cem_plan` now tracks and returns the single best ACTUALLY-SAMPLED sequence seen across all iterations (`best_cost`/`best_actions`, updated from `costs.min(dim=0)` each iteration), not a marginal decode.

### Second run, after the fix: still 0/40, both geometries

Same command, same result: `euclidean: success_rate=0.00 (0/40)`, `hyperbolic: success_rate=0.00 (0/40)`. The marginal-decoding bug was real but evidently not the (only) cause.

Leading hypothesis, not yet confirmed: **exposure bias / autoregressive drift** in `P_step`. `jepa_train_step.py` trains `OneStepPredictor` one step at a time, always fed the REAL previous E(cell) as input (teacher forcing) -- it never sees its own predicted output as input during training. But `maze_planner.py`'s CEM chains `P_step` autoregressively for up to ~45 steps (its own prediction becomes next step's input) when it rolls candidates forward in latent space. This is exactly the failure mode `ball_in_cup_hyperbolic/jepa_train.py` already hit and documented a fix for (see that project's "Multi-step (unrolled) training" section): a one-step-only-trained predictor's autoregressive rollout error can explode well before the horizon actually needed, even though its one-step error looks fine.

Wrote `step_drift_diagnostic.py` to test this directly before assuming it and rebuilding `P_step`'s training around multi-step unrolling: for a handful of long (start, target) pairs, takes the TRUE BFS-optimal action sequence (via `maze_gen.bfs_path_cells`, added for this) and chains `P_step` along it two ways -- "teacher-forced" (always predicting from the real E(cell), matching how it was trained) vs "autoregressive" (predicting from its own last prediction, matching what the planner actually does). If autoregressive error grows sharply with step count while teacher-forced error stays flat, that confirms exposure bias as the cause. Not yet run:
4. `uv run python step_drift_diagnostic.py --geometry euclidean` and `--geometry hyperbolic`

Confirmed: ran `step_drift_diagnostic.py` for both geometries, 5 pairs each (BFS distances 17-44). Teacher-forced error stayed low and flat regardless of step count in both geometries (euclidean ~0.01-0.3, hyperbolic ~0.01-1.5, no upward trend). Autoregressive error grew with chain length in every pair, reaching 10-26 by 30-45 steps -- roughly 100-1000x the teacher-forced error at the same step. Exposure bias confirmed as (at least a major part of) the root cause.

Implemented the fix in `jepa_train_step.py`: `UNROLL_STEPS = 5` (matching ball_in_cup_hyperbolic's choice, tunable later -- our chains only need to survive ~45 steps vs their ~150). `build_transitions` now cuts each episode into 5-step windows (start cell + 5 actions + 5 real next-cells) instead of single triples. Training loop calls `P_step` 5 times per example: the first call takes the real `E(start_cell)`; every later call takes `P_step`'s own predicted embedding from the previous call as input, never the real one; the target at every one of the 5 calls stays the real `E(next_cell)`. The 5 per-step losses are averaged into one scalar and backpropagated through the whole chain in one pass. U=1 would reproduce the original single-step loss exactly, kept as an implicit regression check. No architecture change -- `OneStepPredictor` and `jepa_model.py` untouched, only `jepa_train_step.py`'s data assembly and training loop.

Not yet run. Needs, in order:
1. `uv run python jepa_train_step.py --geometry euclidean` and `--geometry hyperbolic` (retrain P_step with unrolled loss)
2. `uv run python maze_planner.py --geometry both` (re-run the planner against the retrained P_step)

## Phase 3: unroll fix insufficient alone -- cost correlation is real but weak, oracle ablation added

Retrained `jepa_train_step.py` with `UNROLL_STEPS=5` for both geometries, re-ran `maze_planner.py --geometry both`: still 0/40, both geometries. Re-ran `step_drift_diagnostic.py` against the retrained checkpoints -- autoregressive error curves essentially unchanged (still 10-30 by 30-45 steps). U=5 wasn't deep enough relative to the ~45-step deployment horizon: unrolled training only teaches recovery from up to 5 hops of drift, and real trials go 3-9x deeper than that, so P_step is still extrapolating into territory training never covered.

Before pushing U higher, tested a sharper question (Andreas's framing: "if the embedding is a good representation, accumulated errors in latent space should correspond to accumulated errors in the world"): does CEM's predicted cost for the chosen plan actually track how close the plan lands in reality, even under drift? Added logging of predicted cost + real leftover BFS distance (0 if the plan succeeded) per trial, plus their Spearman rank correlation, to `maze_planner.py`. Result: rho=0.637 (euclidean), rho=0.597 (hyperbolic), n=40 -- a real, moderate signal, not noise, but real leftover distance itself is nowhere close: min=2-3, mean~19, max~38-40 across both geometries, essentially as far as a fresh random position, not a near-miss. So the correlation being "not zero" doesn't mean the plans are nearly working.

Raised a further, separate question: is this even solvable via discrete CEM the way it's set up, independent of the representation? Unlike ball_in_cup_hyperbolic's continuous action space (a slightly-wrong action still lands near the right outcome), the maze's 4 categorical actions mean one wrong choice at a junction sends the rollout down an entirely different, unrelated corridor -- and since a perfect maze has exactly one correct path between any two cells, success requires CEM's per-timestep distribution to concentrate on one exact sequence, not a good-enough neighborhood. The per-timestep-independent elite refit also doesn't model path-dependence between timesteps (which action is right at step 10 depends on the branch taken at step 3).

Added `--geometry oracle` to `maze_planner.py`: identical CEM scaffold (same hyperparameters, same trial pairs), but scores candidates via a real simulation on the maze graph and the TRUE BFS distance to target -- no torch, no E, no P_step. If oracle-CEM also fails to land exactly on target often, discrete search itself is the bottleneck, independent of the representation; if it succeeds close to 100%, the search mechanism is fine and the remaining gap is about what E/P_step predict. Not yet run:
5. `uv run python maze_planner.py --geometry oracle`

Ran it: oracle-CEM (perfect cost signal, real BFS distance) got success_rate=0.20 (8/40), rho=0.999 (sanity check -- cost is literally the true distance here), real leftover distance min=0 mean=8.1 max=23, within<=5:18/40. So even with a PERFECT cost signal, this CEM search design only solves the maze exactly 20% of the time. That settles the question from the correlation check: the dominant bottleneck is the discrete search itself, not the representation -- CEM's per-timestep-independent marginal refit is a poor fit for a tree maze, where one wrong turn at a junction sends the rollout down an entirely different, unrelated corridor, and elites that took different branches get averaged into an incoherent distribution.

Added `--search beam` to `maze_planner.py`: deterministic beam search (BEAM_WIDTH=50), replacing CEM's population/refit scaffold. At each step, every surviving candidate PREFIX (a genuine, coherent, complete-so-far action sequence, never an average) is expanded by all 4 actions and scored by predicted (or true, in oracle mode) distance to target; only the best BEAM_WIDTH survive. This can't blend incompatible branches the way CEM's marginal refit could. The oracle variant (`oracle_beam_plan`) also deduplicates same-cell children each step (valid there since it sees real cells), making it close to a proper graph search. Both variants track the best prefix seen at any depth, not just the final one. Not yet run:
Ran both. Oracle: success_rate=1.00 (40/40), exactly BFS-optimal every time (mean actual/bfs-optimal steps=1.00), rho=1.000 -- beam search fully solves the discrete-search problem given a perfect cost signal, confirming the diagnosis. Against the real trained models: euclidean barely moved (0/40 -> 1/40, rho 0.637 -> 0.728); hyperbolic got WORSE (0/40 -> 0/40, rho 0.597 -> 0.189). Diagnosis: beam search's greedy, irrevocable commitment (a pruned branch never comes back) is fine under a perfect signal but brittle under a noisy one, and hyperbolic distance (arcosh-based, steep near the Poincare ball's boundary) amplifies small P_step drift into large, spiky cost swings in a way beam's no-recovery search is specifically bad at handling -- CEM's floor-protected resampling (MIN_PROB) tolerates that better despite being structurally worse-suited to a tree maze on paper.

**Decision (Andreas): discard beam search as the planner's search method.** `beam_plan`/`oracle_beam_plan` are kept in `maze_planner.py`, fully working and documented, but unused -- not wired into `run_trials`/`run_oracle_trials`/`main()`, which reverted to CEM-only (the `--search` CLI flag was removed). Kept for reference in case a future reduction in P_step's noise level (see below) makes beam search worth revisiting.

## Does E itself have the local-smoothness property everything else assumed?

Separately, tested a more basic question directly (Andreas: "are we just assuming this is emergent?") -- independent of P_step and the planner entirely, does E alone put real-world-close cells close together in latent space? Wrote `embedding_metric_check.py`: for every pair of cells, compares real BFS distance (via a new `maze_gen.bfs_distances_from`, one BFS per source cell) to E's latent distance, both as a global Spearman rho across all C(256,2)=32640 pairs and as mean latent distance broken out per real-BFS-distance bucket.

Hyperbolic: clean and monotonic throughout, rho=0.741, bucket means climbing smoothly from 0.22 (direct neighbors) to 72.3 (max distance, 46 steps). E's own geometry is sound -- no problem here.

Euclidean: rho=0.698 (looks reasonable in aggregate) but the per-bucket table reveals a sharp odd/even oscillation -- direct neighbors (bucket 1) mean 13.53, but bucket 2 mean 0.11, bucket 3 mean 13.26, bucket 4 mean 0.46, alternating all the way out. Direct neighbors are consistently embedded FARTHER apart than cells two steps away. Root cause: a perfect maze is a tree, and every grid move flips (row+col) parity -- literally a chessboard coloring, nothing to do with the specific maze layout. Since collect_data.py's rollouts only take valid (non-wall-bump) moves, after exactly H steps you land on a cell whose color is deterministically fixed by parity of H alone -- a perfectly predictable, zero-effort piece of Phase 2's prediction target, sitting next to the genuinely-hard part (which specific room). At dim=8, Euclidean's training apparently found it worth carving out a dedicated cluster split for this free signal, at the cost of local smoothness between real neighbors -- since a direct neighbor is always the opposite color, that's exactly the pairs it happens to disrupt most. Hyperbolic's exponential-volume-growth-at-low-dimension apparently gave it enough room to represent the color split without this tradeoff.

This decomposes the whole Phase 3 investigation cleanly by geometry: euclidean's planner failure = (E's own checkerboard pathology) + (P_step autoregressive drift) + (search brittleness). Hyperbolic's = (P_step autoregressive drift) + (search brittleness) only -- E itself is fine. Not yet tested: whether the checkerboard is dim=8-specific (Euclidean was squeezed for capacity there) or also appears at dim=32 (where Euclidean won Phase 2 outright, suggesting more room to represent both signals without conflict) -- a natural next check if useful.

## Scope decision: hyperbolic-first going forward

After the checkerboard finding (see "Does E itself have the local-smoothness property" above), decided to stop spending effort diagnosing or fixing anything Euclidean-specific. This isn't excluding Euclidean from the comparison -- existing results and checkpoints stay as the baseline, and Euclidean is still fine to run at whatever dimension it naturally works at (including higher dims like 32, where it has no such pathology and wins outright per the original dimension sweep). The checkerboard breaking down specifically as dimension is pushed low is itself evidence for the project's core hypothesis, not a bug to patch. All new work from here (P_step stabilization, planner iteration) targets hyperbolic at dim=8 specifically -- that's the setting the headline result was won at, and the one worth making actually work end to end.

## P_step lever check: unroll depth and training duration both ruled out, dim=12 next

Bumped UNROLL_STEPS 5 -> 20 for hyperbolic (see "Scope decision" above -- hyperbolic only from here): step_drift_diagnostic.py's raw numbers looked marginally different (some trajectories now show a self-correcting dip mid-sequence that U=5 never showed), but the metric that matters -- maze_planner.py --geometry hyperbolic -- barely moved: rho 0.597 -> 0.604, mean real leftover distance ~19.3 -> ~19.8, still 0/40. Unroll depth alone isn't the fix.

Then tested training duration: N_STEPS 2000 -> 10000 (UNROLL_STEPS=20 kept). Loss curve dropped to ~0.55 by step 2000 (matching the old run's endpoint), then oscillated noisily in a 0.4-1.0 band for the remaining 8000 steps with no further net decline -- a genuine plateau, not a still-descending curve. Planner result confirmed it: rho actually dropped slightly to 0.408, still 0/40, mean leftover ~18.9. Training duration is ruled out as the bottleneck; 2000 steps was already enough to reach whatever this architecture can reach.

That leaves capacity. P_step's training loss plateauing around 0.4-1.0 rather than continuing toward zero, even with enough unroll depth and training time, suggests LATENT_DIM=8 (or HIDDEN_DIM=32) may not have enough room to represent one-step transitions precisely -- even though dim=8 was enough for Phase 2's coarser multi-step reachability target (winning that comparison against Euclidean isn't the same as having enough absolute capacity for this harder, finer-grained task). Decided to bump LATENT_DIM 8 -> 12 for hyperbolic specifically (still short of dim=16, where the original sweep found hyperbolic's relative advantage fading -- this is a capacity experiment, not a re-litigation of the geometry comparison). Needs a full retrain from E up: jepa_train.py (produces a new E+P at dim=12), then jepa_train_step.py, then step_drift_diagnostic.py and maze_planner.py again, all --geometry hyperbolic. Note: jepa_train.py's checkpoint filename isn't dim-specific (jepa_checkpoint_hyperbolic.pt), so this retrain overwrites the dim=8 checkpoint the headline dimension-sweep finding was measured on -- the sweep's numbers are already recorded above, but the checkpoint file itself won't be reproducible afterward unless backed up first.

## dim=12 result: capacity bump also ruled out

Ran the full retrain sequence at LATENT_DIM=12, hyperbolic only: jepa_train.py -> jepa_train_step.py (UNROLL_STEPS=20, N_STEPS=10000 kept) -> step_drift_diagnostic.py -> maze_planner.py.

Phase 2 (E) still converges fine at dim=12 (ratio-by-horizon numbers reasonable, e.g. H=20 ratio=0.215, similar shape to the dim=8 sweep). P_step's unrolled training loss plateaus in the same 0.45-0.6 band it plateaued in at dim=8 -- no meaningful drop from the extra dimensions. step_drift_diagnostic.py shows the same qualitative signature as before: teacher-forced error stays low and bounded (mostly <1, occasional spikes to ~1) regardless of chain length, while autoregressive error stays flat/small for roughly the first 10-12 steps of every chain, then explodes -- reaching 10-30 by steps 17-44, independent of how long the total chain is (the explosion onset is consistently around step 12-17 across all 5 pairs, not tied to the pair's total BFS distance). This is the same shape seen at dim=8, just re-confirmed at dim=12.

Planner result: rho=0.585 (down slightly from dim=8/N=10000's 0.408, roughly back to the original dim=8/U=5 baseline of 0.597 -- no real improvement), mean real leftover distance ~19.8 (same ballpark as every prior run), still 0/40.

So all three P_step levers tried so far -- unroll depth (5->20), training duration (2000->10000), and capacity (dim 8->12) -- have each been ruled out individually as the fix. The autoregressive error's consistent onset around step ~12-17 regardless of total chain length is a new, sharper observation: it suggests the explosion isn't simply "runs out of capacity over a long horizon" (which would predict later onset for longer chains) but something that kicks in at a roughly fixed number of compounding steps -- consistent with geometric/exponential error growth from a per-step contraction/expansion factor slightly above 1, rather than a capacity ceiling. Wrote `pstep_manifold_drift_check.py` to test this directly, no retrain needed -- two checks against whatever checkpoint is already on disk: (A) tracks ||raw|| (the raw/tangent-space embedding norm, pre to_geometry) along the same 5 real autoregressive rollouts used above, alongside the true path's ||raw|| at the same step, to see whether the rollout's norm drifts away from what's typical; (B) an independent local-sensitivity test -- perturbs E(cell) by a small random raw-space delta, applies P_step once, and measures the expansion factor (post-P_step geometry distance / pre-P_step geometry distance), bucketed by the input's raw norm. If the rollout's norm climbs toward the range where exp0's tanh saturates, and the expansion factor in that same range is > 1 and climbing, that's a mechanistic explanation for the fixed-step-count blowup: hyp.dist's arcosh (dist(0,x) = 2*artanh(|x|)) is hypersensitive right where exp0 saturates, so any residual per-step error gets amplified once the rollout's raw norm reaches that zone -- independent of capacity, unroll depth, or training duration, all three of which are already ruled out. Also runs against --geometry euclidean as a control (no manifold boundary there, so a flat expansion factor across norm buckets would confirm the effect is boundary-specific, not just generic MLP sensitivity). Not yet run:
    uv run python pstep_manifold_drift_check.py --geometry hyperbolic
    uv run python pstep_manifold_drift_check.py --geometry euclidean

## pstep_manifold_drift_check.py result: edge-of-ball hypothesis REFUTED

Ran both geometries. Result contradicts the hypothesis:

Part B (local expansion factor, one P_step call): hyperbolic is CONTRACTIVE (factor < 1) at every norm bucket tested, and gets MORE contractive at higher raw norm -- [0-1)=0.45, [1-2)=0.14, [2-3)=0.03 -- the opposite of what boundary hypersensitivity predicts (which would need the factor to climb toward/above 1 near the edge). Euclidean (the control -- no manifold boundary at all) shows a similar, flatter contraction -- [1-2)=0.79, [2-3)=0.78, [3-4)=0.75 -- and Part A's real autoregressive rollouts show essentially the SAME error-explosion shape and magnitude in both geometries (e.g. the (0,4)->(1,3) d=44 pair: hyperbolic reaches ~29 by step 41, euclidean reaches ~30 by step 43). A geometry with no boundary produces the same qualitative failure, so this isn't a hyperbolic-specific numerical artifact.

Revised diagnosis: P_step is locally stable (contractive) in both geometries -- small perturbations shrink, not grow, after one step. So the growing autoregressive error isn't amplified noise/instability near a boundary; it's P_step's own one-step prediction being systematically biased away from the true transition, and that bias compounds additively step over step regardless of geometry. A contraction mapping doesn't fix a map whose typical output is itself wrong.

This also sharpens a question about the U=20 unrolled training itself: real rollouts blow up starting around step 12-17 (see prior pstep_manifold_drift_check.py Part A run), well INSIDE the 20-step window P_step was explicitly trained to handle autoregressively. jepa_train_step.py's logged loss is a single scalar averaged over all 20 unroll positions each step -- that average is consistent with training having genuinely stabilized all positions, or with it never having fixed the late positions at all (their high loss just gets diluted by the easy early ones in the average). Wrote `pstep_unroll_position_loss_check.py` to tell these apart: re-runs the same autoregressive-unroll forward pass jepa_train_step.py's training loop uses (imports build_transitions from it directly), against the existing checkpoint, no retrain, but reports mean loss AT EACH unroll position separately instead of one averaged number. Not yet run:
    uv run python pstep_unroll_position_loss_check.py --geometry hyperbolic
    uv run python pstep_unroll_position_loss_check.py --geometry euclidean

## pstep_unroll_position_loss_check.py result: smooth in-distribution, not exploding -- points to a training-distribution mismatch, not exponential blowup

Ran both geometries against 4000 real transitions sampled from maze_data.json (collect_data.py's actual training data). Hyperbolic's per-position loss grows smoothly and modestly across all 20 unroll positions: 0.23 (position 1) -> 0.94 (position 20), roughly linear, steady ~0.02-0.07 increments, no acceleration. Euclidean (unroll_steps=5 there): 0.05 -> 0.14, same shape. Nothing like the 10-30x catastrophic jumps seen in step_drift_diagnostic.py/pstep_manifold_drift_check.py's 5 hand-picked long pairs, on the SAME checkpoints. Andreas asked whether this linear-ish growth is itself natural: yes -- steady per-step increments are the signature of a stable system accumulating small, roughly independent per-step errors (like a random walk's variance growing ~linearly with step count), not runaway instability. So in-distribution, P_step is well-behaved; the catastrophic blowup is something else.

The difference between this eval and the failing diagnostic pairs is WHAT KIND of action sequence is unrolled, not how long. Checked collect_data.py directly: every training episode is `env.random_rollout(...)` -- actions sampled uniformly among valid moves, never steered toward a target. A perfect maze is a tree, so a random walk backtracks/wanders constantly; 15+ consecutive non-backtracking, monotonically-progressing steps toward one specific far-off cell (exactly what a BFS-optimal path is, and exactly what the CEM planner needs P_step to handle) essentially never occurs in that data. `MAX_HORIZON=20` in collect_data.py exactly matches `UNROLL_STEPS=20`, so P_step's unrolled training has literally only ever seen 20-step random-walk windows -- never a directed one.

Andreas raised a weighted-loss idea (amplify later unroll positions) as an alternative to chase. Reasonable lever on its own (the unweighted mean currently lets Adam undervalue late-position accuracy), but it doesn't address the distribution gap -- reweighting only changes how hard the optimizer tries on the SAME random-walk examples; it can't manufacture examples of directed sequences the model has never seen.

Wrote `pstep_distribution_mismatch_check.py` to test the distribution-mismatch hypothesis directly, no retrain: same 5 start cells and path lengths (d) as step_drift_diagnostic.py's failing pairs, but instead of the BFS-optimal path to that pair's target, generates a RANDOM valid rollout of the same length d from the same start cell (env.random_rollout, same policy collect_data.py used) and runs the identical teacher-forced-vs-autoregressive comparison. If random rollouts of the same length stay smooth (matching the per-position check above) while the BFS-directed path from the same start still blows up, that confirms the mismatch directly and points toward getting directed/goal-seeking rollouts into P_step's training data -- not reweighting the existing loss -- as the real fix. Not yet run:
    uv run python pstep_distribution_mismatch_check.py --geometry hyperbolic
    uv run python pstep_distribution_mismatch_check.py --geometry euclidean

## pstep_distribution_mismatch_check.py result: CONFIRMED -- distribution mismatch, not chain length, is the cause

Ran both geometries: same 5 start cells and path lengths d as the failing BFS-directed pairs (18, 30, 17, 44, 26), but each one a RANDOM valid rollout of length d from that start instead. Result is decisive. Hyperbolic errors stayed bounded across every pair, including the d=44 one that goes twice as deep as anything P_step was ever unrolled through in training: max ~2.4 (vs ~29 for the BFS-directed path from the same start, same length). Euclidean: max ~0.37 (vs ~30 for BFS-directed). No pair showed the 10-30x catastrophic blowup pattern -- values stayed in the same rough 0.01-2.4 range the per-position check already found for in-distribution data, with only a mild upward drift, not runaway growth, even past step 20 (collect_data.py's MAX_HORIZON, the longest chain P_step was ever unrolled through during training).

Same checkpoint, same start cell, same chain length, only the ACTION SEQUENCE differs -- random valid moves vs. the unique BFS-optimal path to a specific target -- and that alone is the difference between bounded and catastrophic. This rules out chain length (the d=44 random rollout goes further than any BFS-directed pair and stays fine) and confirms it isn't a hyperbolic-specific effect (euclidean shows the identical qualitative pattern). The cause is specifically that P_step has never been unrolled through a long, non-backtracking, goal-directed sequence during training -- collect_data.py's episodes are pure random walks (uniformly among valid moves, never steered toward a target), so this exact TYPE of trajectory (the type the CEM planner actually needs) is essentially absent from what jepa_train_step.py trained on, regardless of unroll depth, training duration, or capacity -- all three already ruled out separately.

Implication: the fix is getting directed/goal-seeking rollouts into P_step's unroll training data -- not reweighting the existing loss (which only changes emphasis on the same random-walk examples, addressed to Andreas directly when he raised it) and not any further lever on the existing random-walk-only data. Not yet decided: how to generate that data (pure BFS-optimal paths to random targets vs. a biased-random policy that's more likely to make net progress, giving a spectrum between pure-random and pure-directed closer to what CEM's own candidate sequences look like during planning) -- open question for Andreas before implementing.

## Fix: collect_directed_data.py, goal-directed rollouts for P_step's training

Andreas raised a fair concern before implementing: doesn't training P_step on goal-directed data defeat the point of an "intrinsic" representation? Answered directly (and recorded in collect_directed_data.py's own docstring): this does NOT touch E -- E stays exactly as trained by jepa_train.py on collect_data.py's random rollouts alone, so the encoder's own geometry and the whole hyperbolic-vs-euclidean comparison built on it is untouched. P_step is a different kind of object: a supervised approximator of a known, deterministic ground-truth function (the maze's actual transition rule), not a representation-learning objective with a stay-task-agnostic purity requirement the way E and Phase 2's P deliberately have. Training P_step on inputs resembling what it's actually asked to predict along is ordinary train/test distribution matching, not smuggling in the planner's answer -- CEM still has to find its own path.

Andreas chose option 2 (biased-random walk over pure BFS-optimal), then asked to mix both. Both are the same function with one parameter: `directed_rollout(..., bias_prob)` -- at each step, with probability bias_prob takes the unique action that reduces BFS distance to target (found via a precomputed bfs_distances_from from the target); otherwise picks uniformly among all valid moves. bias_prob=1.0 IS option 1 (pure BFS-optimal path) exactly; bias_prob<1 is option 2. "Mixing 1 and 2" is generating episodes across a spread of bias_prob values (BIAS_PROBS=[1.0, 0.85, 0.7, 0.5]) rather than committing to one.

Wrote `collect_directed_data.py`: for each bias_prob, samples EPISODES_PER_BIAS=500 random (start, target) pairs with bfs_distance >= MIN_DISTANCE=UNROLL_STEPS(=20) (so every episode yields at least one full unroll window), generates a directed_rollout of exactly that pair's true bfs_distance in length (not a fixed horizon -- naturally covers depths beyond collect_data.py's MAX_HORIZON=20, which is what the planner actually needs), writes to maze_data_directed.json in the same {"start","actions","states"} schema as maze_data.json (plus bias_prob/target/bfs_distance for provenance). Uses DIRECTED_DATA_SEED=21, kept separate from every other seed in the project (collect_data.py's DATA_SEED=1, the diagnostics' DIAG_SEED=3, maze_planner.py's PLANNER_SEED=2) so a planner improvement after this still reflects genuine generalization, not memorization of the eval pairs.

Updated `jepa_train_step.py` to load and concatenate episodes from BOTH maze_data.json (existing random-walk coverage) and maze_data_directed.json (new directed coverage) before building unroll windows -- not a replacement, an addition. Fails loudly (FileNotFoundError) if maze_data_directed.json is missing, rather than silently training on random-walk data alone again.

E is untouched, so no need to re-run jepa_train.py. Full sequence, hyperbolic (per the scope decision):
    uv run python collect_directed_data.py
    uv run python jepa_train_step.py --geometry hyperbolic
    uv run python step_drift_diagnostic.py --geometry hyperbolic
    uv run python pstep_distribution_mismatch_check.py --geometry hyperbolic
    uv run python maze_planner.py --geometry hyperbolic
Not yet run.

## Directed-data result: P_step measurably fixed, planner still 0/40 -- points at the CEM search ceiling, not representation

Ran the full sequence. step_drift_diagnostic.py's 5 BFS-directed pairs, retrained with directed data mixed in: 4 of 5 went from the earlier catastrophic 10-30x blowups to bounded errors (roughly 0.5-5 throughout, vs ~12-30 before) -- a real, substantial improvement, confirming the distribution-mismatch diagnosis and fix. One pair (start=(12,4) target=(12,5), bfs_distance=17 -- physically adjacent cells but a long, winding tree path between them) still blows up badly (up to 15.5), so the fix is real but incomplete -- not every region of the maze got adequate directed-trajectory coverage from 500 episodes/bias level.

But maze_planner.py: still success_rate=0.00 (0/40), rho=0.486 (down slightly from 0.604), mean leftover distance ~19.6 (essentially unchanged), within<=5: 1/40. A substantial reduction in P_step's raw autoregressive error did NOT translate into planner success at all.

Read on this: the maze demands EXACT final-cell correctness, and CEM's own structural ceiling was already measured independently of representation quality -- the oracle-CEM ablation (perfect cost signal, ground truth) only reached 20% success (see "Phase 3: unroll fix insufficient alone" above). So even a much-improved but still-imperfect P_step feeding into the same CEM scaffold can't be expected to land above that 20% ceiling, and apparently isn't yet precise enough to get close to it either. The remaining bottleneck looks like it's shifted from "P_step's accuracy" (where we just made real progress) to "CEM's search mechanism" (already known to be weak even under perfect information).

This directly revisits the earlier beam-search decision: beam search fully solved the oracle case (100%, exactly BFS-optimal) but was explicitly shelved for the real models because it made hyperbolic WORSE (rho 0.597 -> 0.189) -- diagnosed as beam's greedy, no-recovery search being too brittle under P_step's noisy cost signal, particularly given hyperbolic's arcosh-driven sensitivity. That diagnosis was conditioned on P_step's noise level at the time. P_step's noise on the diagnostic pairs just dropped by roughly an order of magnitude for 4/5 pairs. The condition WORKPLAN.md's beam-search section flagged for revisiting it ("kept for reference... in case a future reduction in P_step's noise level makes beam search worth revisiting") may now be met. Wired back in: maze_planner.py gained a `--search {cem,beam}` flag (default cem, so nothing changes unless explicitly requested), run_trials/run_oracle_trials dispatch to cem_plan/beam_plan (or the oracle variants) accordingly.

Ran it: hyperbolic+beam against the new (directed-data-trained) checkpoint -- still success_rate=0.00 (0/40), rho=0.306 (WORSE than CEM's 0.486 on the same checkpoint), mean leftover distance ~20.2 (slightly worse than CEM's ~19.6), max leftover 41. So the revisit hypothesis doesn't hold: even with P_step's noise cut by roughly an order of magnitude on 4/5 diagnostic pairs, beam's greedy, no-recovery search is still too brittle for hyperbolic's boundary-sensitive distance, apparently because the residual noise (still present, especially on the one pair that never got fixed) is enough to break it. CEM remains the better search of the two. Decision: don't pursue beam further; the gap is still on the P_step-accuracy side, not the search-algorithm side, since real CEM is still well below oracle-CEM's already-low 20% ceiling -- there's room to close before search-algorithm limits would even start to bind. Bumped collect_directed_data.py's EPISODES_PER_BIAS 500 -> 1500 (broader coverage of harder/less-common maze regions, directly targeting the one diagnostic pair that never got fixed) and regenerating/retraining. Not yet run.

## EPISODES_PER_BIAS=1500 regression, then fixed with balanced-ratio batch sampling

Bumped collect_directed_data.py's EPISODES_PER_BIAS 500 -> 1500 (per the prior section's plan, targeting broader coverage of harder maze regions including the one diagnostic pair that never got fixed). Result was a severe REGRESSION, not an improvement: jepa_train_step.py's loss oscillated in a 2.4-5.9 band and plateaued at ~2.96 (vs ~0.68-1.2 before). step_drift_diagnostic.py showed TEACHER-FORCED error -- which had stayed reliably low (<1) throughout every single prior run in this entire investigation, without exception -- now catastrophically high from step 1 onward (values like 9-22+ recurring across pairs). pstep_distribution_mismatch_check.py confirmed general degradation even on random rollouts, not just the BFS-directed ones. maze_planner.py: rho=0.375 (worse than the 500-episode run's 0.486), still 0/40. This is a fundamentally different, worse failure mode than anything seen before: the one-step model itself degraded, not just autoregressive compounding.

Diagnosis: build_transitions cuts every episode into UNROLL_STEPS=20-length sliding windows, and directed episodes are much longer than random-walk episodes (up to ~45 steps vs collect_data.py's fixed 20), so each directed episode yields many more windows than a random-walk episode does. At EPISODES_PER_BIAS=500 this happened to net out roughly balanced (~53% random / ~47% directed windows, 14000 vs ~15965). At 1500 it didn't (~22% random / ~78% directed, 14000 vs 48511 -- directed windows outnumbered random ones ~3.5:1). jepa_train_step.py's training loop sampled batches UNIFORMLY over the full concatenated window pool (`torch.randint(0, n, (BATCH_SIZE,))`), so batch composition silently mirrored that imbalance -- training became dominated by long-distance directed trajectories at the expense of the broad, dense, general-purpose one-step coverage random-walk data provides. Plausible explanation for why even teacher-forced (single-step) accuracy collapsed, not just autoregressive drift.

Fix: jepa_train_step.py now builds the random-walk and directed window pools SEPARATELY (two calls to build_transitions instead of one on the concatenated episode list: `r_starts, r_action_seqs, r_target_seqs = build_transitions(random_data["episodes"], UNROLL_STEPS)` and the equivalent `d_*` for directed_data) and samples a FIXED split per batch -- RANDOM_BATCH = BATCH_SIZE // 2, DIRECTED_BATCH = BATCH_SIZE - RANDOM_BATCH, an even split, not tuned -- regardless of how large either pool is (`r_batch = torch.randint(0, n_r, (RANDOM_BATCH,))`, `d_batch = torch.randint(0, n_d, (DIRECTED_BATCH,))`, concatenated into one BATCH_SIZE=256 batch per step). This makes growing the directed pool for broader coverage pure upside from here on -- it can never again crowd out the random-walk signal by sheer window count. Data (maze_data_directed.json at EPISODES_PER_BIAS=1500) does not need to be regenerated; only jepa_train_step.py changed. Full re-run sequence, hyperbolic:
    uv run python jepa_train_step.py --geometry hyperbolic
    uv run python step_drift_diagnostic.py --geometry hyperbolic
    uv run python pstep_distribution_mismatch_check.py --geometry hyperbolic
    uv run python maze_planner.py --geometry hyperbolic --search cem
Not yet run.

## Balanced-sampling fix result: regression resolved, but planner still glued to 0/40

Re-ran the full sequence with the balanced-ratio fix in place (data unchanged, still EPISODES_PER_BIAS=1500). Teacher-forced error is back to the historical pattern -- bounded, almost always <1 across all 5 diagnostic pairs (one outlier at 0.9566) -- confirming the regression is resolved. Training loss plateaued ~0.9-1.2 (vs the broken run's ~2.96; close to but slightly above the original 500-episode run's ~0.68-1.2).

Planner: rho=0.510 (best yet -- above the 500-episode run's 0.486 and well above the 1500-broken run's 0.375), mean leftover=19.4 (best yet, marginally), within<=5: 1/40. Still success_rate=0.00 (0/40). Note rho is still BELOW the pre-directed-data baseline of 0.604 despite P_step now being numerically far more accurate on the diagnostic pairs than the original checkpoint ever was -- absolute autoregressive accuracy and planner cost-signal quality are evidently not the same thing.

Two of the five diagnostic pairs are still clearly broken in autoregressive mode despite bounded teacher-forced error and 3x the directed-data volume: (12,4)->(12,5) (d=17, physically-adjacent cells via a long winding path) still reaches 21.6 by step 17; the (0,4)->(1,3) d=44 pair still reaches ~46.6 by step 43. Tripling EPISODES_PER_BIAS didn't touch either -- coverage volume alone isn't fixing these specific regions, suggesting whatever's wrong there isn't simply "not enough directed examples nearby."

Cross-run pattern, now with 5 distinct P_step configurations tried (U=5 baseline; U=20/N=10000; dim=12; +directed@500; +directed@1500-balanced): rho has moved only within a 0.38-0.60 band, mean leftover has stayed in a 19-21 band, and success has stayed at 0 or 1 out of 40 every single time, regardless of how much P_step's own diagnostic-pair accuracy improved. This is a much stronger signal than before that the bottleneck has shifted off P_step's raw accuracy and onto either (a) CEM's search mechanism itself (independently already shown to cap at 20% even under a perfect oracle cost signal) or (b) a mismatch between what P_step is accurate ON (the hand-authored directed/biased-random distribution) and what CEM's actual candidate sequences look like mid-search, which was flagged as a real possibility when the directed-data fix was first proposed (see "Fix: collect_directed_data.py" section) but not yet tested directly.

Not yet done: inspecting what CEM's own best candidate actually looks like on a few real (non-oracle) failing pairs -- does it get close then diverge, wander from the start, or converge to a plausible-looking but wrong path -- as a cheap diagnostic before committing to a bigger fix (e.g. on-policy/DAgger-style data collection: run real CEM, capture its actual candidate rollouts including bad ones, add those as P_step training data, directly targeting the true query distribution instead of the hand-authored approximation).

## cem_candidate_diagnostic.py result: found the actual mechanism -- P_step never learned "invalid move -> no movement"

Wrote cem_candidate_diagnostic.py (reuses cem_plan and every maze_planner.py constant directly, so it replays exactly what the real 0/40 planner run already did, not an approximation) to inspect, step by step, what CEM's actual chosen plan does on real failing trials: the real cell reached, the real BFS leftover distance, and P_step's predicted cost, all at every step of the SAME chosen action sequence.

Result was decisive and much sharper than "search vs representation": across all 40 failing trials, mean common-prefix-with-the-true-path length was just 3% of bfs_distance, and 36/40 trials diverged from the correct path within the very first action. The 5 detailed traces all show the same mechanism: the real agent gets stuck almost immediately -- frozen at one cell for many consecutive steps (every neighbor CEM tries from there is a wall) or oscillating between two adjacent cells -- while CEM's PREDICTED cost keeps sliding smoothly toward 0 the whole time, as if real progress were happening. E.g. trial 1: real cell frozen at (12,8) for 6 straight steps (both UP and RIGHT from there are walls), while predicted cost drops 5.26 -> 3.20 -> 1.27 -> 0.34 -> 0.08.

Root cause: both P_step data sources -- collect_data.py's random_rollout(valid_only=True) and collect_directed_data.py's directed_rollout (previously fell back to rng.choice(valid) on the "not heading toward target" branch) -- sampled ONLY among valid moves. P_step has therefore never once seen a (cell, invalid-action) -> same-cell (wall-bump/no-op) training example in its entire history. But cem_plan has zero notion of maze structure -- it samples candidate actions from a plain categorical distribution over all 4 actions every time, valid or not. So real CEM search constantly queries P_step on exactly the input it was never trained on, and P_step extrapolates there in a way that (apparently, by whatever accident of the learned function) tends to read as low cost -- and CEM, which only ever sees that predicted cost, immediately finds and exploits the hole in nearly every single trial. This is a much more specific and directly fixable finding than the earlier (a)-vs-(b) framing: it's not that CEM's search is fundamentally too weak for a discrete tree, or that P_step's accuracy on genuine candidate-like trajectories is insufficient -- it's a straightforward gap in what P_step was ever asked to model at all.

Fix: collect_directed_data.py's directed_rollout fallback branch (the existing "not heading toward target" wobble branch, used whenever bias_prob<1.0) now samples uniformly among ALL 4 actions (`rng.choice(ACTIONS)`) instead of only the valid ones (`rng.choice(valid)`), so wall-bump (cell, invalid-action) -> same-cell examples appear naturally inside otherwise-ordinary directed episodes. bias_prob=1.0 episodes are untouched (that branch is never taken there), so the pure BFS-optimal reference data stays exactly valid-only as before. Regenerating maze_data_directed.json and retraining is required (the data itself changes, unlike the balanced-sampling fix). Full sequence, hyperbolic:
    uv run python collect_directed_data.py
    uv run python jepa_train_step.py --geometry hyperbolic
    uv run python step_drift_diagnostic.py --geometry hyperbolic
    uv run python cem_candidate_diagnostic.py --geometry hyperbolic
    uv run python maze_planner.py --geometry hyperbolic --search cem
Not yet run.

## Status

Phase 0 -> domain-fit check -> env.py -> collect_data.py -> jepa_train.py (collapsed, VICReg fix, eval_k_diagnostic.py artifact caught and fixed, bigger maze for power, full dimension re-sweep) -> headline result: dim=8 hyperbolic wins broadly, dim=16 mixed, dim=32 Euclidean wins broadly, on the size-16/n=70 maze. See "Dimension re-sweep, completed" above.
Phase 3 (planner) built on top of that dim=8 result: jepa_model.py gained OneStepPredictor, jepa_train.py now checkpoints, jepa_train_step.py and maze_planner.py are new. Still 0/40 both geometries after: fixing a real CEM marginal-decoding bug; retraining P_step with UNROLL_STEPS=5 unrolled loss (step_drift_diagnostic.py had confirmed exposure bias, precedented by ball_in_cup_hyperbolic's v7 fix, but U=5 turned out insufficient for the ~45-step deployment horizon); and trying beam search as a replacement for CEM (fully solved the oracle case, 40/40, but was brittle under the real models' noisy cost signal and made hyperbolic worse -- discarded, kept in maze_planner.py for reference only). embedding_metric_check.py then found the deeper split: hyperbolic's E has a clean, locally-smooth geometry on its own (the remaining gap there is P_step drift + search brittleness), while euclidean's E additionally has a "checkerboard" pathology at dim=8 -- direct neighbors embedded farther apart than 2-hops-away cells, from a free/deterministic parity signal in Phase 2's training target competing with real distance for limited dimensions. See "Phase 3: unroll fix insufficient alone" and "Does E itself have the local-smoothness property" above. Tried bumping LATENT_DIM 8->12 for hyperbolic next: no improvement, still 0/40, rho unchanged within noise. Unroll depth, training duration, and capacity have now all been individually ruled out as the fix for P_step's autoregressive drift. Tested the edge-of-ball-sensitivity hypothesis directly (pstep_manifold_drift_check.py) and it's REFUTED: P_step is contractive, not expansive, at every norm tested in both geometries, and euclidean (no boundary) shows the same explosion pattern as hyperbolic. Then pstep_unroll_position_loss_check.py showed per-position loss on real (random-walk) training data grows smoothly/linearly, not catastrophically -- pointing away from instability and toward a training-DISTRIBUTION mismatch. pstep_distribution_mismatch_check.py CONFIRMED it directly: random rollouts of the same start/length as the failing BFS-directed pairs stay bounded (max ~2.4 hyperbolic, ~0.37 euclidean) while the BFS-directed path from the same start explodes (~20-30) -- same checkpoint, same length, only the action-sequence type differs. Root cause: collect_data.py's training rollouts are pure random walks, so P_step has never been unrolled through the long, non-backtracking, goal-directed sequences the CEM planner actually needs, regardless of unroll depth/training duration/capacity (all separately ruled out). Fix implemented and run: collect_directed_data.py + jepa_train_step.py retrained on combined data measurably fixed P_step's autoregressive drift on 4/5 diagnostic pairs (bounded ~0.5-5 vs ~12-30 before), but maze_planner.py stayed at 0/40 -- the bottleneck looks like it's shifted to CEM's search mechanism itself (independently shown to cap at 20% success even under a perfect oracle cost signal), not P_step's accuracy. Re-tested beam search against the directed-data-improved checkpoint (--search beam flag added to maze_planner.py): still 0/40 and WORSE than CEM (rho 0.306 vs 0.486) -- confirmed beam's greedy, no-recovery search remains too brittle for hyperbolic even at the reduced noise level; staying with CEM. Bumped collect_directed_data.py's EPISODES_PER_BIAS 500 -> 1500 for broader coverage, which caused a severe REGRESSION (teacher-forced error, previously always <1, collapsed to 9-22+; rho fell to 0.375) -- diagnosed as a batch-composition imbalance (directed windows outnumbering random-walk windows ~3.5:1, up from roughly parity at 500/bias, while training sampled batches uniformly over the concatenated pool). Fixed by rewriting jepa_train_step.py to build the random-walk and directed window pools separately and sample a fixed 50/50 split per batch regardless of pool size -- see "EPISODES_PER_BIAS=1500 regression, then fixed with balanced-ratio batch sampling" above. Re-ran with the balanced-sampling fix: regression resolved (teacher-forced error back under ~1), rho improved to 0.510 (best yet, but still below the pre-directed-data baseline of 0.604), mean leftover ~19.4, still 0/40. Ran the cheap diagnostic (cem_candidate_diagnostic.py) next and it found the actual mechanism: P_step was never trained on a single invalid-action (wall-bump/no-op) example, since both data sources sampled only valid moves, but CEM's search has no notion of maze structure and samples all 4 actions regardless -- so it constantly queried P_step on untrained inputs and found/exploited a blind spot there in 36/40 trials (diverged from the true path within the first action, real cell often frozen for many steps while predicted cost kept sliding toward 0 anyway). Fixed collect_directed_data.py's wobble branch to sample among all 4 actions instead of only valid ones, so wall-bump examples now appear naturally in the directed training data -- see "cem_candidate_diagnostic.py result" above. Not yet re-run. Not yet decided: whether the checkerboard is dim=8-specific. embedding_tree_metric_check.py then independently reconfirmed and precisely quantified the checkerboard pathology through the parent-closer-than-sibling lens: hyperbolic passes 63.3% of checks, euclidean inverts 100% of sibling pairs -- see "embedding_tree_metric_check.py result" above; directly optimizing for this property was considered and declined as it would defeat the project's purpose. Decided to go DAgger next instead of just re-verifying the wall-bump fix alone: collect_dagger_data.py (on-policy, real-simulator-labeled data from the CURRENT planner's actual choices) and jepa_train_step.py's generalization to an optional 3rd balanced source are both implemented and syntax-verified on-device -- see "DAgger-style on-policy data collection for P_step" above. Not yet run.

## embedding_tree_metric_check.py result: hyperbolic substantially respects parent-closer-than-sibling, euclidean totally inverts it

Follow-up to the "Does E itself have the local-smoothness property" line of investigation, prompted by the question of whether hyperbolic embeddings should satisfy |c1-parent| < |c1-c2| for two children c1,c2 of the same parent (the standard mechanism behind "hyperbolic suits trees": place parent near the origin, fan children apart angularly). Wrote embedding_tree_metric_check.py: roots the maze at an arbitrary (0,0) via BFS, and for every junction cell with >=2 children, checks both children's embedded distance to the parent against their distance to each other.

Result: hyperbolic passes 63.3% of individual child-side checks (119/188), with 42/94 sibling pairs fully consistent (both children closer to parent than to each other) and 17/94 (18%) fully inverted. Euclidean: 0.0% pass (0/188) -- **94/94 (100%) of sibling pairs are fully inverted**, i.e. every single pair of siblings embeds closer to each other than either does to their shared parent. This is the same "checkerboard" pathology already documented above (euclidean's dim=8 E embeds direct neighbors farther apart than 2-hop cells, from a parity signal competing with real distance) independently reconfirmed and precisely quantified through a different, more targeted lens: not just "some local distortion" but a total, 100%-consistent inversion of the one property hyperbolic geometry is specifically supposed to provide for tree data. (Note: the script's "mean margin" statistic is unreliable, dominated by outliers from near-zero-distance denominators -- e.g. euclidean's -1468.944 is not a meaningful number; the pass-rate/count metrics above are the trustworthy ones.)

Considered and declined: directly optimizing E to enforce this property (or the root-independent generalization, matching pairwise graph distance for all cell pairs, measurable via Gromov hyperbolicity -- see the original "Domain-fit check" section). This would guarantee "hyperbolic wins" by construction, which is already settled math (Poincare embeddings, Sarkar's construction) -- it would stop testing whether hyperbolic's advantage emerges naturally from solving the actual one-step-prediction task, which is the entire point of this project. Left as a diagnostic only, not a training objective.

## DAgger-style on-policy data collection for P_step

The wall-bump fix above (collect_directed_data.py's wobble branch now sampling all 4 actions) is a targeted patch for one specific hole cem_candidate_diagnostic.py found. The deeper issue it's a symptom of: P_step's directed-data training distribution is still a hand-authored APPROXIMATION of what CEM's real candidate sequences look like, not the real thing -- flagged as a real possibility back when directed data was first proposed, never tested directly until the diagnostic above.

Fix: DAgger (Dataset Aggregation, Ross/Gordon/Bagnell 2011) -- iteratively run the CURRENT planner for real, record what it actually does, get the TRUE outcome from the real simulator (never from any model's own prediction, so this can't be circular the way training on a model's own guesses would be), add those transitions to P_step's training data, retrain, repeat. Mapped onto this project: "model"=CEM+P_step, "states"=candidate action sequences CEM evaluates, "expert"=the maze simulator itself (trivial, deterministic, no human needed).

Implemented:
- collect_dagger_data.py (new): one ROUND of on-policy collection. Loads whichever jepa_step_checkpoint is currently on disk for the given geometry, samples 300 fresh start/target pairs (DAGGER_SEED=37, kept separate from every other seed in the project -- collect_data.py's 1, collect_directed_data.py's 21, diagnostics' 3, maze_planner.py's 2), runs real cem_plan for each, executes the CHOSEN action sequence on the real simulator for its full horizon (wall bumps included, same as collect_directed_data.py's convention), and appends the resulting (start, actions, real states, target, bfs_distance, round) episodes to maze_data_dagger_<geometry>.json. Geometry-specific (unlike maze_data.json/maze_data_directed.json, which are pure maze physics) since it depends on what THIS geometry's P_step actually gets wrong. Each run APPENDS a new round rather than overwriting, so data accumulates across rounds ("Aggregation").
- jepa_train_step.py (modified): generalized from the fixed 2-source (random-walk + directed) balanced split to an N-source split that also picks up maze_data_dagger_<geometry>.json when present (optional -- absent on the very first training run, before any DAgger round has been collected). Each present source gets its own separately-built window pool and an even 1/N share of every training batch, same balanced-ratio principle as the EPISODES_PER_BIAS=1500 fix above, generalized so a growing DAgger pool can never crowd out the other two sources either.

Workflow per round: collect_dagger_data.py -> jepa_train_step.py -> maze_planner.py (evaluate) -> collect_dagger_data.py again (now against the improved checkpoint) -> repeat. Full sequence, hyperbolic, starting from the wall-bump-fix data already generated but not yet trained on:
    uv run python collect_directed_data.py
    uv run python jepa_train_step.py --geometry hyperbolic
    uv run python maze_planner.py --geometry hyperbolic --search cem
    uv run python collect_dagger_data.py --geometry hyperbolic
    uv run python jepa_train_step.py --geometry hyperbolic
    uv run python maze_planner.py --geometry hyperbolic --search cem
Not yet run. (First three lines re-verify the wall-bump fix alone; the last three add DAgger round 1 on top. Repeat the last three for further rounds.)

## Wall-bump fix + DAgger round 1 result: first-ever success (1/40)

Ran the full verification sequence. Wall-bump fix alone (collect_directed_data.py regenerated with the all-4-actions wobble branch, retrained, re-evaluated): rho=0.598 (best yet, above the balanced-sampling fix's 0.510 and the pre-directed-data baseline's 0.604 -- essentially back to baseline), mean leftover=19.2, still success_rate=0.00 (0/40), within<=5: 1/40. Confirms the wall-bump fix helped the cost signal's overall correlation with real distance without yet producing a single exact success.

DAgger round 1 (collect_dagger_data.py against that checkpoint -> retrain with the 3-way balanced split -> re-evaluate): **success_rate=0.03 (1/40) -- the first success anywhere in this entire project, across every geometry/config tried.** rho=0.563 (down slightly from the wall-bump-only 0.598), mean leftover=18.7 (marginally better), min leftover=0 (new), within<=1:1/40, <=2:1/40, <=3:2/40, <=5:4/40 (up from 1/40 pre-DAgger). Training loss rose from ~0.84 (wall-bump-only) to ~1.72 with DAgger data mixed in -- read as DAgger data being intrinsically harder to fit (real on-policy trajectories close to actual failure, not clean random/directed rollouts), not as a regression, given the planner outcome improved on every count-based metric (success, min leftover, within<=1/2/3/5) despite rho dropping slightly. This is the first empirical evidence that rho (global rank correlation) and actual exact-success count can move in different directions -- consistent with DAgger fixing specific failure regions rather than uniformly improving cost-signal quality everywhere.

Next: run DAgger round 2 (collect_dagger_data.py again, now against the round-1-improved checkpoint -- it appends to maze_data_dagger_hyperbolic.json, so round 2's data accumulates on top of round 1's), retrain, re-evaluate. Same three commands as round 1:
    uv run python collect_dagger_data.py --geometry hyperbolic
    uv run python jepa_train_step.py --geometry hyperbolic
    uv run python maze_planner.py --geometry hyperbolic --search cem
Not yet run.

## DAgger round 2 crash: gradient explosion, fixed with grad clipping

DAgger round 2 (collect_dagger_data.py appended round 2's 300 episodes on top of round 1's, 600 total) then jepa_train_step.py: loss tracked normally (14.3 -> ~1.9-3.0) through step 8000, then spiked to 166.8 at step 8500 and never recovered (stayed 135-166 through step 9999). maze_planner.py then crashed: `TypeError: 'NoneType' object is not iterable` in cem_plan -- best_actions never got assigned because best_cost was initialized to float('inf') and every candidate's cost came back NaN, and `NaN < inf` is always False in Python, so the "keep best seen" comparison silently never fires across any of the N_ITERS iterations.

Root cause: jepa_train_step.py's training loop never clipped gradients. One bad DAgger example, chained through the U=20 unrolled/BPTT-style loss, produced a large enough gradient to corrupt the weights in a single Adam step -- plausibly triggered by hyperbolic.py's own documented numerical caveat (torch.acosh has an infinite derivative at arg==1, i.e. predicted almost exactly equal to target -- easy to hit once DAgger data includes near-successful trajectories). The corrupted checkpoint then got chained by cem_plan for ~45-50 steps (more than 2x the 20-step training horizon), overflowing to Inf/NaN raw embeddings by the time squared_dist was computed.

Fix: added `torch.nn.utils.clip_grad_norm_(P_step.parameters(), max_norm=1.0)` between loss.backward() and opt.step() in jepa_train_step.py -- standard safeguard for exactly this failure mode in unrolled training, previously missing. max_norm=1.0 is an untuned starting point. Retraining is required (the round-2 checkpoint is corrupted); DAgger data itself (maze_data_dagger_hyperbolic.json, 600 episodes/2 rounds) is unaffected and does not need to be regenerated. Re-run:
    uv run python jepa_train_step.py --geometry hyperbolic
    uv run python maze_planner.py --geometry hyperbolic --search cem
Not yet run.

## Grad-clipping fix confirmed; DAgger round 2 result: no clear gain, same band as everything else

Re-ran jepa_train_step.py with clip_grad_norm_(max_norm=1.0) in place, same round-2 DAgger data (no regeneration needed). Loss stayed bounded and smooth throughout (1.6-2.9, no spike) -- confirms the fix works.

Planner: rho=0.566 (essentially matching round 1's 0.563), mean leftover=19.4 (vs round 1's 18.7), min leftover=3 (worse than round 1's 0), success_rate=0.00 (0/40, DOWN from round 1's 1/40 -- that single success did not persist), within<=5:5/40 (up from round 1's 4/40), within<=3:1/40 (down from 2/40), within<=1/2:0/40 (down from 1/40 each). Net read: mixed/noisy, not a clear improvement or regression over round 1.

Updated cross-run tally: 7 distinct P_step configurations now tried (U=5 baseline; U=20/N=10000; dim=12; +directed@500; +directed@1500-balanced; +wall-bump-fix; +DAgger round 1; +DAgger round 2), and every single one lands in the same rho~0.4-0.6 / leftover~19-21 / success~0-1-of-40 band, regardless of which specific fix was applied or how much P_step's own accuracy/coverage improved. This is now a strong pattern, not noise. Reinforces the standing hypothesis flagged after the balanced-sampling-fix result and again after the wall-bump diagnostic: the bottleneck looks increasingly like CEM's search mechanism itself, independently already shown to cap at 20% success even under a PERFECT oracle cost signal (see oracle-CEM ablation, "Phase 3: unroll fix insufficient alone" above) -- not P_step's remaining inaccuracy, which has now been attacked from several different angles (unroll depth, capacity, directed data, wall-bump coverage, on-policy DAgger data) without moving the planner's success rate off its 0-1/40 floor.

Not yet decided: whether to run further DAgger rounds (cheap, but rounds 1-2 show no clear trend) or pivot to investigating/tuning CEM's own search mechanism directly (N_SAMPLES/N_ITERS/ELITE_FRAC, or a fundamentally different search strategy) given the oracle ceiling evidence. Flagged for discussion, not yet run either direction.

## DAgger rounds 1-5 trend: rho climbing, round 5 doubles the best success count yet

Ran 5 real DAgger rounds total (round 2's first attempt crashed from the gradient explosion documented above and was retrained with grad clipping in place before continuing; one extra retrain occurred on round 4's data due to a shell paste glitch that silently skipped that round's collect_dagger_data.py call -- harmless, and confirms training is reproducible: re-running jepa_train_step.py on identical data gave rho=0.644 both times, byte-for-byte-consistent metrics). maze_data_dagger_hyperbolic.json: 5 rounds, 1500 episodes total.

| round | success | rho   | mean leftover | min leftover | within<=5 |
|-------|---------|-------|----------------|---------------|-----------|
| 1     | 1/40    | 0.563 | 18.7           | 0             | 4/40      |
| 2     | 0/40    | 0.566 | 19.4           | 3             | 5/40      |
| 3     | 0/40    | 0.619 | 19.1           | 2             | 4/40      |
| 4     | 0/40    | 0.644 | 18.9           | 2             | 2/40      |
| 5     | 2/40    | 0.513 | 19.2           | 0             | 2/40      |

Read: rho climbed steadily across rounds 1-4 (0.563 -> 0.566 -> 0.619 -> 0.644, the best rho seen in the whole project, above even the pre-directed-data baseline's 0.604), then dropped at round 5 even as success count hit 2/40 -- the best success count anywhere in this project so far, double round 1's single success. First time the planner has cleared the 0-1/40 floor that held across every earlier configuration (see "DAgger round 2... same band as everything else" above -- that read is now superseded by round 5). Rho and success-count aren't moving together round to round, consistent with the round-1-vs-2 observation already made: DAgger rounds look like they're patching specific failure regions rather than uniformly improving the global cost signal, so a temporary rho dip alongside a real success-count gain isn't necessarily a contradiction.

Not yet decided: continue with rounds 6+ (cheap, ~1 min/round, and the trend has now produced genuine improvement for the first time) vs. stop here and investigate CEM's search mechanism. Leaning toward a few more rounds given round 5 just broke new ground.

## DAgger rounds 6-8: the round 1-4 "climbing rho" trend was noise, not progress

Ran 3 more rounds. Round 6: success=0/40, rho=0.707 (new best, briefly). Round 7: success=0/40, rho=0.429 (new WORST of the whole DAgger run), mean leftover=20.1 (worst mean leftover of the whole run). Round 8: success=0/40, rho=0.509.

Full 8-round table:

| round | success | rho   | mean leftover |
|-------|---------|-------|----------------|
| 1     | 1/40    | 0.563 | 18.7           |
| 2     | 0/40    | 0.566 | 19.4           |
| 3     | 0/40    | 0.619 | 19.1           |
| 4     | 0/40    | 0.644 | 18.9           |
| 5     | 2/40    | 0.513 | 19.2           |
| 6     | 0/40    | 0.707 | 19.1           |
| 7     | 0/40    | 0.429 | 20.1           |
| 8     | 0/40    | 0.509 | 19.4           |

Corrected read: the "rho climbing 0.563->0.644 across rounds 1-4" pattern flagged earlier does NOT continue -- round 6 briefly beat it (0.707) then round 7 immediately gave the single worst rho of the entire DAgger run (0.429). rho is bouncing in a roughly 0.43-0.71 band with no visible trend, i.e. noise around a fixed operating point, not sustained improvement -- same character as the pre-DAgger cross-config pattern, just a wider noise band. Success has been 0/40 for 6 of the last 7 rounds; across all 8 rounds combined, only 3 successes out of 320 total trials run (~0.9%), despite the DAgger pool growing 8x (300 -> 2400 episodes, 4278 -> 33923 windows). Mean leftover distance has stayed essentially flat in an 18.7-20.1 band for all 8 rounds -- no visible trend there either.

Decision: stop iterating pure DAgger rounds here. 8 rounds (2400 accumulated on-policy episodes) have not produced sustained improvement in any planner metric -- rho, success rate, and mean leftover are all consistent with noise around the same operating point this whole investigation has circled since the directed-data fix. This reinforces the standing hypothesis (see "Balanced-sampling fix result" and "cem_candidate_diagnostic.py result" above): the bottleneck is very likely CEM's search mechanism itself, independently already capped at 20% success even under a perfect oracle cost signal, not P_step's training data or accuracy -- which has now been attacked from unroll depth, capacity, directed data, wall-bump coverage, AND 8 rounds of on-policy DAgger data, without moving the needle. Next: investigate CEM's search mechanism directly (N_SAMPLES/N_ITERS/ELITE_FRAC tuning, or why real CEM is so far below its own oracle ceiling) rather than further P_step/data-side fixes.

## Closed-loop replanning wired into the real planner: helps, but doesn't repeat oracle's jump -- cleanly isolates the remaining bottleneck

Added --replan to maze_planner.py (run_closed_loop/run_closed_loop_oracle, closed_loop_max_steps -- same d0*3+20 cap validated by oracle_receding_horizon_check.py). Ran against the current (DAgger round 8) hyperbolic checkpoint:

hyperbolic+cem+replan: success_rate=0.05 (2/40), rho=0.554, mean leftover=16.1, min=0, within<=3:3/40, within<=5:5/40, mean actual/bfs-optimal steps among successes=5.10.

Compare to open-loop (same round-8 checkpoint, no replan): success=0/40, rho=0.509, mean leftover=19.4. So replan: mean leftover dropped 19.4 -> 16.1 (best of the whole project), within<=3 improved 1->3/40, within<=5 tied the project best (5/40), success matched the project's best-ever (2/40, tying DAgger round 5) rather than exceeding it. Successful trials now take ~5x longer than bfs-optimal (vs ~1.2x under open-loop) -- consistent with replanning doing real work correcting course, not just getting lucky on an easy trial.

Read: replanning is a real, unambiguous improvement on every "closeness" metric and should be kept as the standard execution mode going forward -- there's no reason to plan blind now that this is available. But it did NOT repeat oracle-CEM's dramatic 20%->100% jump. That gap is itself the diagnostic: with a PERFECT cost signal, replanning alone was sufficient for near-total success (the search/execution strategy was fully fixed). With P_step's real (still noisy) cost signal, replanning helps a lot but success stays low -- meaning the remaining bottleneck has now been cleanly isolated to P_step's residual prediction noise, not the search/execution strategy, which is no longer implicated at all.

Open question this raises: the DAgger rounds 1-8 showed no clear trend in success rate under the OLD open-loop harness (see "DAgger rounds 6-8" above) -- but open-loop success in a zero-tolerance tree is close to all-or-nothing (one wrong step early dooms the whole trial), so it may have been too brittle a yardstick to detect real, incremental P_step accuracy gains. --replan is a more forgiving, more sensitive measure (mean leftover distance moved cleanly even though exact success didn't). Worth re-examining whether further P_step improvements (more DAgger rounds, etc.) show a clearer signal now that they're measured under --replan instead of the old open-loop harness. Not yet tested (would need checkpoints from multiple DAgger rounds, which aren't separately saved -- jepa_step_checkpoint_hyperbolic.pt is overwritten each run).

## Beam + replan: fixed the crash, but beam still doesn't benefit from replanning

Fixed a real bug beam_plan/oracle_beam_plan shared: both only kept a candidate if it beat the cost measured BEFORE any move, a stricter bar than cem_plan's "just track the best ever seen" convention. That rarely bound under open-loop (long horizon, easy to find something better eventually), but --replan calls it repeatedly with much shorter horizons near the end of a trip, where nothing sometimes clears that bar -- best_actions stayed [] and run_closed_loop crashed on actions[0] (IndexError). Fixed both functions to initialize best_cost=inf/best_actions=None, matching cem_plan -- horizon>=1 guarantees the first iteration always assigns a valid action.

Result, hyperbolic+beam+replan (same round-8 checkpoint): success_rate=0.00 (0/40), rho=0.270, mean leftover=19.8, within<=5:3/40. Compare to hyperbolic+cem+replan: success=2/40, rho=0.554, mean leftover=16.1. Beam does NOT get the boost replanning gave CEM -- it's essentially unchanged from (or slightly worse than) its own pre-replan open-loop numbers (rho=0.306, mean leftover~20.2, from the earlier beam-vs-CEM comparison). Read: beam's search itself (deterministic, single greedy path per replan call, no randomness to escape a bad local choice) is still the weaker search method even with replanning available -- confirms the earlier decision to stick with CEM, now under both open-loop and closed-loop conditions. CEM+replan remains the best configuration found in the whole project.

## Checkpoint history: jepa_train_step.py now keeps a round-tagged copy per run

Added: alongside the usual jepa_step_checkpoint_<geometry>.pt (still overwritten every run, unchanged default behavior), jepa_train_step.py now ALSO saves jepa_step_checkpoint_<geometry>_dagger<N>.pt, where N = how many DAgger rounds existed in maze_data_dagger_<geometry>.json at training time (0 = pre-DAgger baseline). This is why rounds 1-8's checkpoints weren't separately comparable earlier -- going forward every round's checkpoint is preserved automatically, no extra step needed, so future rounds can be evaluated under --replan individually rather than only checking the latest one.

## DAgger round 10 + replan: no clean trend even under the more sensitive harness

Round 10 (3000 episodes total) + replan: success=0/40, rho=0.530, mean leftover=18.3, within<=3:1/40, within<=5:3/40. Compare to round 8 + replan (the only other --replan datapoint so far): success=2/40, rho=0.554, mean leftover=16.1, within<=3:3/40, within<=5:5/40. Round 10 is WORSE than round 8 on every metric under the same --replan evaluation.

This weakens (doesn't yet refute, only 2 datapoints) the hypothesis raised in the "Closed-loop replanning" section above -- that --replan might reveal a DAgger improvement trend the old brittle open-loop harness couldn't see. So far it looks like the same noise-around-a-fixed-point pattern persists even under the more sensitive measure. Round-tagged checkpoints (jepa_step_checkpoint_hyperbolic_dagger{8,10}.pt) are now both saved, so this comparison can be extended with more rounds, or revisited later, without re-collecting anything.

## DAgger + replan rounds 8-12: confirmed noise, not a trend

Round 12 (3600 episodes) + replan: success=0/40, rho=0.525, mean leftover=17.8, within<=3:1/40, within<=5:6/40 (new best on this one metric). Full --replan table so far:

| round | success | rho   | mean leftover | within<=5 |
|-------|---------|-------|----------------|-----------|
| 8     | 2/40    | 0.554 | 16.1           | 5/40      |
| 10    | 0/40    | 0.530 | 18.3           | 3/40      |
| 11    | 3/40    | 0.506 | 17.0           | 5/40      |
| 12    | 0/40    | 0.525 | 17.8           | 6/40      |

Round 11's success=3/40 (the project's best-ever) did not hold -- round 12 immediately reverted to 0/40. Same bouncing-around-a-fixed-point character as the open-loop DAgger rounds 1-8, just with a somewhat better baseline (mean leftover mid-16-18 vs the open-loop's 18.7-20.1) thanks to replanning. Conclusion: --replan did NOT reveal a hidden DAgger improvement trend as hypothesized -- 4 datapoints now (rounds 8, 10, 11, 12) show the same noise character the open-loop harness showed across 8 rounds. Further DAgger rounds look like diminishing-returns grinding rather than a promising lever; the more informative next moves are probably elsewhere (P_step architecture/capacity, CEM's own hyperparameters under closed-loop conditions which haven't been tuned since --replan was added, or accepting the current ~16-18 mean-leftover / 0-3-of-40 operating point as close to this setup's ceiling).

## cem_hparam_search.py: automated CEM hyperparameter search under --replan

Written since N_SAMPLES/N_ITERS/ELITE_FRAC/MIN_PROB were all chosen once, early on, for the original open-loop planning regime and never retuned since --replan was added -- closed-loop changes the cost/benefit math substantially (many more, shorter-horizon plan_fn calls; each individual call's mistake matters less since the next replan corrects it), so the old values may no longer be well-suited.

Grid: N_SAMPLES x N_ITERS x ELITE_FRAC x MIN_PROB = [50,100,200,400] x [2,4,6,10] x [0.05,0.1,0.2,0.3] x [0.01,0.02,0.05] = 192 combos. Reuses run_trials(..., replan=True) unchanged by monkey-patching maze_planner.N_SAMPLES etc. as module globals before each call (cem_plan reads them as globals, not parameters, so this works without touching cem_plan itself). Combos visited cheapest-first (est. cost = N_SAMPLES*N_ITERS) so results bank up early regardless of total runtime. Per Andreas: no artificial time cap -- the goal is completing the full 192-combo grid correctly, however long that takes (~40-90+ min estimated); --budget-seconds is only a 6h safety net against a genuinely pathological hang, not a real target. Results written incrementally to cem_hparam_search_results_<geometry>.json after every combo, safe to interrupt regardless.

Run:
    uv run python cem_hparam_search.py
Not yet run.

## cem_hparam_search.py sweep result: fewer CEM iterations wins under replanning, new defaults set

Full 192-combo grid completed in 2642s (~44 min) against the current (round-12) hyperbolic checkpoint. Top result: success=3/40, TYING the project's best-ever (DAgger round 11) rather than beating it -- three combos tied at 3/40: (N=100,iters=2,elite=0.1,min_prob=0.05), (N=400,iters=2,elite=0.1,min_prob=0.01), (N=400,iters=2,elite=0.1,min_prob=0.02).

The clearer, more useful finding was structural, not a single best point: N_ITERS=2 (the cheapest setting tried) dominates the top of the leaderboard, while N_ITERS=10 (the most expensive) clusters heavily toward success=0/40 across nearly every N_SAMPLES/ELITE_FRAC/MIN_PROB combination it appeared in. N_SAMPLES showed little consistent effect either way (50 through 400 all appear among both good and mediocre results). Read: more CEM refinement per replan call looks actively counterproductive once replanning is in the loop -- plausibly because grinding harder against P_step's own noisy predicted cost overfits to that noise within a single call, a cost that outweighs whatever precision it buys, since constant replanning already provides correction that extra iterations were trying to substitute for.

Updated maze_planner.py's defaults to the cheapest of the tied-best combos: N_SAMPLES 200->100, N_ITERS 6->2, MIN_PROB 0.02->0.05, ELITE_FRAC unchanged at 0.1. This also makes every future --replan run ~6x cheaper per real step (N_SAMPLES*N_ITERS: 200*6=1200 -> 100*2=200), which matters since replanning already multiplies plan_fn calls per trial by roughly the real step count.

Full results (all 192 combos) saved to cem_hparam_search_results_hyperbolic.json for later reference.

## DAgger round 13 (first round with new tuned CEM defaults): still within the established noise band

Round 13 (3900 episodes total) was the first round where the new CEM defaults (N_SAMPLES=100, N_ITERS=2, ELITE_FRAC=0.1, MIN_PROB=0.05) were used for BOTH steps: collect_dagger_data.py's on-policy trajectory generation AND the final --replan eval. Result: success=1/40, rho=0.568, mean leftover=16.9, within<=3:2/40, within<=5:3/40, mean actual/bfs-optimal steps among successes=7.00.

Full --replan table, updated:

| round | success | rho   | mean leftover | within<=5 | CEM config          |
|-------|---------|-------|----------------|-----------|----------------------|
| 8     | 2/40    | 0.554 | 16.1           | 5/40      | old (N=200,iters=6)  |
| 10    | 0/40    | 0.530 | 18.3           | 3/40      | old                  |
| 11    | 3/40    | 0.506 | 17.0           | 5/40      | old                  |
| 12    | 0/40    | 0.525 | 17.8           | 6/40      | old                  |
| 13    | 1/40    | 0.568 | 16.9           | 3/40      | new (N=100,iters=2)  |

Read: round 13 sits inside the range rounds 8-12 already established on every metric -- rho=0.568 is actually the best of any --replan round so far (previous best 0.554), mean leftover=16.9 is second-best (behind only round 8's 16.1), but success=1/40 is mid-pack, not a new high. No clean signal that combining the new CEM defaults with fresh DAgger data compounds -- still consistent with noise around the same operating point. One confound worth flagging: round 13 changed two things at once (the eval-time CEM config AND the data-collection-time CEM config, since collect_dagger_data.py also uses maze_planner's module defaults), so even a real effect here would be hard to attribute cleanly to one or the other.

5 datapoints now (rounds 8, 10, 11, 12, 13) with no sustained trend under any CEM configuration. Consistent with the round-12 read: further DAgger rounds look like diminishing-returns grinding on this axis. The current ~16-18 mean-leftover / 0-3-of-40 operating point looks close to this setup's ceiling under --replan; the more promising remaining levers are probably elsewhere (P_step architecture/capacity, or accepting this as the project's current best result).

## branch_point_diagnostic.py: isolated the mechanism -- E under-resolves fine local structure, worst exactly near the target

Andreas pushed back that the last several rounds (DAgger 8-13, the 192-combo CEM sweep) were empirical grinding on aggregate metrics without a mechanistic diagnosis ("we have to find something that is wrong... it looks like we are fumbling in the dark"). Agreed, and went back to the one concrete unresolved lead from weeks-old work: embedding_tree_metric_check.py found hyperbolic's E passes only 63.3% of "parent closer than sibling" checks, not 100% -- noted at the time, never chased down.

Wrote branch_point_diagnostic.py: a one-step lookahead test that isolates the cost signal entirely from CEM's stochastic search and from multi-step unroll error. For every cell b with degree>=2 in the maze's spanning tree, and every other cell t as a candidate target, there is exactly one correct first action (removing b splits the maze into deg(b) disjoint components -- no cycles -- and t falls in exactly one). Tests whether `argmin_a squared_dist(to_geometry(P_step(E(b),a),geom), to_geometry(E(t),geom), geom)` -- the identical formula CEM's own cost function uses, here at horizon=1 with zero sampling noise -- picks the correct action, against the round-13 hyperbolic checkpoint.

Results:
- By degree: corridor (degree==2) 71.9% (18714/26010), branch (degree>=3) 65.5% (11684/17850), overall 69.3%. Branches are worse but only by ~6pts -- too small a gap to explain a 30% base error rate as "sibling confusion at forks" alone.
- By bfs_distance(b,t): accuracy climbs monotonically WITH distance -- d=1-5: 58.1%, d=6-10: 62.3%, d=11-15: 66.5%, d=16-20: 71.2%, d=21-30: 73.5%, d=31+: 74.9%. Backwards from "noise accumulates over range" -- means E resolves coarse/global structure fine but fine/local structure poorly. This is the same regime (short range) as embedding_tree_metric_check.py's sibling-pair test (distance 2, via the shared parent) -- two independently-built diagnostics converge on the same conclusion.
- Margin analysis (cost gap between correct and best-wrong action, normalized by current squared_dist-to-target): when right, mean margin=0.138, median=0.070. When wrong, mean margin=0.206, median=0.054 -- median close to the right-case median (near-tie territory), but mean pulled well above it. 47.3% of wrong picks are near-ties (<5% margin, plausibly embedding-space granularity noise, hard to fully eliminate), but the remainder are confidently, substantially wrong -- a real structural defect, not just noise near a decision boundary.

Read: this is a mechanistic finding, not another aggregate metric. E (frozen since Phase 2, trained only to predict where H random steps land + an anti-collapse term) was never asked to preserve fine-grained tree-metric ordering, and it doesn't -- worst exactly in the near-target regime, which is exactly where the zero-tolerance success criterion is decided during --replan (replanning corrects course fine when there's slack/distance, 70-75% accurate there, but the final approach into the target is the worst-resolved region at 58%). This plausibly explains why oracle-CEM+replan hits 100% but the real planner caps at 0-3/40: not that planning is broken in general, but that the cost signal is specifically unreliable in the region that decides success/failure. None of unroll depth, capacity (dim 8->12), directed data, wall-bump coverage, 13 rounds of DAgger, or CEM hyperparameter tuning could have fixed this -- it's upstream of all of them, in E itself.

Not yet decided: what to do about it. Options flagged, not yet chosen: (a) revisit training E with some form of explicit local/tree-metric-aware objective (previously considered and declined as "cheating" re: the hyperbolic-vs-euclidean research question -- but now with actual causal evidence this is the bottleneck, worth reconsidering as a deliberate, labeled experiment rather than folding it into the main comparison); (b) a distance-weighted or near-target-weighted training objective for E, emphasizing exactly the regime this diagnostic shows is weakest; (c) accept this as the project's core finding and write it up rather than continuing to chase the success-rate metric further.

## Design discussion: direct tree-additivity training objective for E, and CEM-as-degenerate-case

Following the branch_point_diagnostic.py finding above, worked through (with Andreas) what a direct fix to E would look like, rather than continuing to attack the symptom via more data/CEM tuning:

- Minimal proposed training signal: every real maze EDGE (a,b) should map to unit embedded distance, `squared_dist(to_geometry(E(a)),to_geometry(E(b))) ~= 1`. This is deliberately minimal -- not the full (x,b,y) triple loss discussed earlier, just per-edge. Rationale: this is specifically well-founded in hyperbolic space (not Euclidean) -- Sarkar's construction shows that unit-edge placement plus small angular separation between siblings at each branch point is PROVABLY sufficient to separate entire subtrees, because hyperbolic space's volume grows exponentially with radius. That's the actual mathematical reason hyperbolic is the right geometry for trees, not just a heuristic choice.
- Caveat flagged: that guarantee is for an explicit combinatorial construction (Sarkar's algorithm places the whole tree at once, using full knowledge of the branching structure). Our E is trained by gradient descent on local/sampled losses with no global view -- not guaranteed to find that same good arrangement from a unit-edge loss alone. May need the wider (x,b,y) triple signal too (or some other longer-range term) to reliably steer training away from arrangements that satisfy local edges but still muddle distant branches. Not yet decided which to implement first -- cheapest (unit-edge only) is the natural first experiment given it's simplest.
- Corollary if E gets good enough: CEM's own machinery (multi-sample, multi-iteration, multi-step horizon search) becomes unnecessary and can collapse to its own degenerate/minimal case -- horizon=1, N_ITERS=1, N_SAMPLES covering all 4 actions exhaustively -- which is mathematically just "evaluate all 4 neighbors once, take the best." Not a new algorithm to write; CEM already contains this as a special case at minimal hyperparameters. This reframes the whole CEM-tuning history (including the 192-combo sweep's "fewer iterations wins" finding) as CEM being pushed toward this degenerate limit precisely because replanning was already compensating for embedding noise -- consistent, not a new finding, just a mechanistic explanation for an already-observed trend.
- Flagged for later, explicitly NOT now: generalizing E to arbitrary/unseen mazes (not just this one). Current E is a plain per-cell lookup table (nn.Embedding, num_cells x dim) -- memorized coordinates for THIS maze only, with zero ability to handle a new maze. Generalizing would require E to become a function of local graph structure (e.g. a GNN) rather than a fixed table, trained across many random mazes. Distinct, larger architectural change from fixing this maze's embedding; parked for a future project phase.

Not yet implemented: the unit-edge-length loss itself, retraining E with it, or testing the degenerate/minimal-CEM planner against it.

---

## Addendum: mixed-curvature geometry for general graphs (NOT maze-specific -- future/general interest, parked here for reference)

Long conceptual side-discussion, triggered by trying to build intuition for why hyperbolic geometry specifically suits THIS maze's tree structure, that generalized into the broader question of which geometry suits which kind of graph structure in general. Not applicable to the current maze (an exact spanning tree, zero cycles by construction) -- captured here because it's a real, well-founded direction worth returning to for future, non-tree-shaped problems.

**The three curvature types and what each is for**, characterized by Gromov's δ-hyperbolicity and by "triangle degeneracy": a tree has every geodesic triangle degenerate (`dist(x,y) = dist(x,b) + dist(b,y)` exactly, whenever b sits on the unique path between x and y) -- hyperbolic space is the geometry whose triangles stay close to that same degenerate/"thin" shape, which is why it's the natural fit for trees, not just a heuristic choice (formalized by Sarkar's construction: unit-edge placement + small angular separation between siblings is provably sufficient to separate whole subtrees, because hyperbolic volume grows exponentially with radius). Euclidean space (curvature 0) is the flat middle point on the same curvature dial -- the right fit for grid/lattice-like structure (polynomial, not exponential or periodic, volume growth; the underlying 16x16 grid this maze's spanning tree is carved out of is exactly this case). Spherical space (positive curvature) is the right fit for closed loops/cycles -- geodesics reconverge (like lines of longitude), matching a cycle's "short way around," the exact opposite behavior from hyperbolic's ever-diverging geodesics.

**Trees vs. cycles need their curvature "amount" tuned differently.** For a cycle of length n, there's an EXACT, closed-form, zero-distortion radius: 2*pi*R = n, so R = n/(2*pi) -- both too big and too small a radius give distortion, because a cycle has to close up on itself (an equality constraint). A tree has no such closing-up constraint -- distortion only ever monotonically decreases as the scale/curvature-magnitude increases, asymptotically approaching (never exactly reaching, at any finite value) zero as radius -> infinity. So there's no single "correct" value to solve for on the tree side, just "push as far as your numerical precision budget allows" -- directly the same tradeoff already seen in this project (LATENT_DIM/HIDDEN_DIM capacity, and the arcosh-derivative gradient-explosion crash from pushing points too close to the Poincare ball boundary).

**Product spaces**: a graph with a MIX of tree-like, grid-like, and cyclic structure doesn't have to be forced into one constant-curvature geometry. A product space (Cartesian product of several constant-curvature factors, e.g. H^a x E^b x S^c) gives every node coordinates in ALL factors simultaneously, with distance combined as a Pythagorean-style sum across factors (`dist^2 = dist_hyp^2 + dist_euc^2 + dist_sph^2`). A direct distortion-minimizing loss on this combined distance is self-regulating: whichever factor best explains a given relationship naturally absorbs it, and a factor with nothing real to represent naturally collapses toward a degenerate/uninformative state (contributing ~0 to every distance) -- this is a DESIRABLE, loss-driven self-pruning, not the same pathological collapse VICReg was added to Phase 2's Predictor to prevent (that collapse was a cheap shortcut around an indirect predictive objective that didn't reward preserving information; a genuine distortion-minimizing objective doesn't have that shortcut, so blanket anti-collapse regularization across all factors would be actively counterproductive here -- it would force meaningless variance into a factor that has nothing real to encode).

A graph with cyclic structure at genuinely DIFFERENT characteristic scales (e.g. a 4-cycle and a 20-cycle in different regions) can't be exactly fit by one shared spherical factor (constant curvature is constant everywhere within one factor) -- the standard fix (per the actual mixed-curvature-embedding literature) is to use SEVERAL factors of the same sign, each with its own separately-tuned/learned scale, rather than one shared compromise value.

**Concrete cost estimate** (worked through at this project's own scale, LATENT_DIM=12/HIDDEN_DIM=32/256 cells, for a naive equal 3-way split to dim_total=36, purely illustrative -- not something to build for the maze): E's embedding table scales close to the naive 3x (256x12=3072 params -> 256x36=9216), since a pure per-cell lookup table has no shared structure to amortize across factors. P_step's parameter count grows sub-linearly, under 2x (~1964 -> ~3500 params), because its hidden layer (32) is a shared bottleneck whose width doesn't need to scale with the raw embedding dimension -- only the input/output layers widen. Distance computation cost grows by only a small constant factor (three cheap closed-form formulas instead of one, summed). At this project's actual scale, all of this is computationally free regardless; the real cost only starts to matter at production scale (millions of nodes).

**References** for anyone picking this up later: Sarkar (2011) for the explicit combinatorial tree-into-hyperbolic-plane construction; Nickel & Kiela (2017, "Poincare Embeddings for Learning Hierarchical Representations") for the learned/gradient-descent version of hyperbolic tree embeddings; Gu, Sala, Sawhney, Re (2019, "Learning Mixed-Curvature Representations in Product Spaces") for the product-space framework this whole addendum reconstructs from first principles.

Not applicable to the current maze project (no cycles exist in a spanning tree by construction) -- parked here purely as a reference for a future, genuinely mixed-structure problem.

## jepa_train_treeloss.py result: unit-edge-only loss converges perfectly locally, fails globally (confirms the flagged risk)

Ran jepa_train_treeloss.py --geometry hyperbolic: converged almost immediately (loss ~0 by step 500 of 3000, mean edge cost 1.000, min 0.994/max 1.006 -- every real edge ends up essentially exactly unit distance apart, as designed).

Built treeloss_global_check.py to test the property that actually matters (do FAR-apart cells stay far apart in the embedding, not just adjacent ones) across all ~32,640 pairs in the maze, exhaustively. First run had a real bug in the script (an `E`-output tensor indexed by position-in-`adj.keys()` instead of by cell_index, silently comparing the wrong cells' embeddings for almost every pair) -- caught because the result was inconsistent with what training itself had already shown, fixed, and reconfirmed correct (true_dist=1 bucket now reproduces training's own converged numbers exactly).

The corrected result confirms the risk flagged when this loss was designed, empirically: Spearman rho(true bfs_distance, embedded distance) = 0.14 (WORSE than the original indirect-objective E's ~0.5-0.6). 64% of far pairs (true distance >=15) collapsed to embedded distance <=3. Mean embedded distance by true-distance bucket: 1.00 (d=1) -> 1.69 (d=2-5) -> 1.89 (d=6-10) -> 1.92 (d=11-20) -> 1.99 (d=21-30) -> 2.02 (d=31+) -- true distance grows 31x across that range, embedded distance barely doubles. The embedding saturates almost immediately: every edge is locally perfect, but distant, unrelated branches of the tree are NOT kept apart.

Read: unit-edge-length alone under-constrains the embedding. A huge family of solutions satisfies "every edge is unit length" while still letting far-apart subtrees curl back close together, since nothing in that loss penalizes non-adjacent cells being close. This is exactly the caveat already flagged in the "Design discussion" section above (Sarkar's guarantee requires unit edges PLUS explicit angular separation between siblings at each branch; plain gradient descent on edges alone has no incentive to find that separated arrangement, and empirically didn't).

Next: implement the wider (x,b,y) triple loss originally proposed (minimize |d(x,b)+d(b,y)-d(x,y)|^2 for x,y on different sides of branch cell b, sampled via branch_point_diagnostic.py's existing components_excluding()), which explicitly constrains cross-subtree distances rather than only adjacent edges. Not yet implemented.

## jepa_train_treeloss.py, continued: all-pairs loss, distance-weighting iterations, and the real fix -- radius saturation, not capacity

Following on from the section above: generalized the loss from edges-only to ALL pairs, per Andreas's simplification -- bfs_distance(x,y) already IS the additive sum along the tree's one unique path, so training directly against it for every pair covers the same constraint as the originally-proposed explicit (x,b,y) triple decomposition, more simply. Fixed the global correlation immediately: rho 0.14 -> 0.835, far-pair collapse (true_dist>=15, embedded<=3) 64% -> 0/32640.

That surfaced a new problem: short-range precision regressed (true_dist=1 bucket mean drifted from ~1.0 to 2.294, wide min/max spread), because unweighted averaging over ~32,640 mostly-far pairs starves the rare short-range ones (this maze's true-distance histogram is bell-shaped, peaking ~1268 pairs around d=19, vs. only 255 at d=1). Iterated on per-pair weighting to fix this:
- First attempt weighted by 1/(exact-integer-distance count) -- Andreas flagged this as risky before running it. Confirmed via the real histogram: some distances near the tree's diameter (46) have only 1-2 pairs, which would get ~600x the weight of a typical pair. Not used.
- Switched to a smooth importance weight keyed off distance itself (not a population count) -- the actual goal Andreas wanted was "make small distances matter more, regardless of how many samples exist at each distance," a different thing from population-balancing across buckets. Tried log-based (1/log2(d+1), ~5.5x spread d=1 to d=46) -- too mild, true_dist=1 bucket barely moved (2.294 -> 1.986). Switched to 1/d (~46x spread) -- still barely moved (1.986 -> 1.781), and a handful of specific true-edge pairs stayed stuck at embedded distance ~7-11 regardless of how hard the weight pushed.

Wrote treeloss_global_check.py's complementary "near-but-stretched" check (true_dist<=1, embedded>=6) to name the actual offenders instead of just watching the bucket mean drift: 4-5 specific edges, reproducing near-identically across different random seeds (e.g. (5,7)<->(4,7): 7.404 at seed 0, 7.399 at seed 1) -- different seeds converging to nearly the same failure value rules out "SGD got unlucky," pointing to a real constraint instead of a local-minimum accident.

Tested the capacity hypothesis (dim 12 -> 24, same seed): changed almost nothing (same edges, nearly identical values, same loss floor ~44.0). Ruled out dimension as the bottleneck.

Root cause, confirmed via treeloss_norm_check.py (new diagnostic): hyperbolic.py's exp0 maps raw embedding vectors through tanh(|v|), which saturates hard (vanishing gradient) as |v| grows -- at dim=24, 65% of cells already had raw norm > 4 (tanh already ~flat there), and the implied max dist-from-origin (12.2) sat right at this EPS's theoretical ceiling. The loss was asking the embedding to represent true bfs_distance directly (range 1-46), but the unit Poincare ball's *practically reachable* radius tops out far short of that -- a fixed-scale numerical artifact of this implementation (EPS=1e-5 clip), not a real limit of hyperbolic geometry (Sarkar's construction gets separation by scaling radius *up* with tree size; this code fixes radius at 1). Far pairs were saturating against that ceiling, and short-range edges were collateral damage from every point being pushed toward the boundary to chase unreachable far-distance targets.

Fix: added --dist-scale to jepa_train_treeloss.py -- train against bfs_distance / c instead of raw bfs_distance (c=4 tried). Only relative order matters for CEM's argmin, so this is functionally free. Result (dim=12, seed=0, dist-scale=4): rho 0.83 -> 0.9988, far-pair collapse 0/32640, near-but-stretched 0/32640, all six distance buckets increase cleanly and monotonically (0.326 -> 0.990 -> 2.067 -> 3.925 -> 6.248 -> 8.632), max raw embedding norm 3.12 (0% of cells near saturation, vs. 65% before). Confirms the diagnosis fully -- rescaling the target, not adding capacity, was the fix.

Next: train P_step against this fixed treeloss E and evaluate with the real planner (maze_planner.py CLI overrides for --encoder-checkpoint/--step-checkpoint still need to be added -- queued from earlier, not yet done).

## Weighting ablation: removed, was never necessary

Tested whether the per-pair distance-based loss weighting (added while chasing the short-range regression, before the real cause -- radius saturation -- was found) still did anything now that --dist-scale addresses that cause directly. Ran the identical --dist-scale 4 setup with --weight-power 0 (uniform weight) vs. 1 (the short-range-favoring weight): results were the same to 3-4 decimal places (rho 0.9988 both; mean|embed-true| 0.026 unweighted vs. 0.039 weighted -- if anything slightly better without it). Confirms the weighting was only ever compensating for the saturation problem, not fixing anything of its own -- it's been removed from jepa_train_treeloss.py. Andreas's point going in: something added mid-troubleshooting tends to get kept by default just because it was present when things started working, whether or not it's actually load-bearing -- worth checking, not assuming.

## Checkpoint now self-describing: dist_scale saved and consumed downstream

Added dist_scale to jepa_train_treeloss.py's saved checkpoint dict, and updated treeloss_global_check.py to read it back and scale its own absolute embedded-distance thresholds (CLOSE_THRESHOLD, FAR_EMBED_THRESHOLD) accordingly, rather than assuming whatever scale happened to be used at training time. Prompted by asking "does dist_scale change anything downstream" -- it did: those two thresholds were calibrated when embedded distances ranged up to ~14 (the pre-fix, saturating regime), so at dist_scale=4 the near-but-stretched threshold (6) was ~3-4x looser than intended and would only catch a much worse outlier than originally meant, though it happened to still report 0/0 correctly by coincidence, not because it was doing real discriminating work. Retrained once more to pick up the new field; rechecked -- same 0/0 result, now for the right reason (thresholds auto-scaled to 0.75/1.5).

Also confirmed the scope of "upstream" here: hyperbolic.py's curvature-1 formulas and the baseline (non-treeloss) jepa_checkpoint_{geometry}.pt are untouched by any of this -- dist_scale only changes what jepa_train_treeloss.py regresses its own separate checkpoint toward. Mathematically, dist_scale and curvature are the same knob (dist_scale = 1/sqrt(kappa), i.e. kappa = 1/dist_scale^2) -- dist_scale=4 is exactly equivalent to training at curvature -1/16 instead of -1. Scaling the target rather than changing hyperbolic.py's fixed curvature=-1 convention was a practical choice (that file is shared with the ball_in_cup project), not a mathematical necessity.

## DAgger data made opt-in for jepa_train_step.py, off by default

Andreas caught something during the treeloss-P_step troubleshooting: jepa_train_step.py was silently pulling in the existing 13-round DAgger dataset (maze_data_dagger_hyperbolic.json), which was collected on-policy against the ORIGINAL E -- a completely different embedding space from the new treeloss E. DAgger's value comes from matching the training data to the CURRENT policy's visited-state distribution; reusing those rounds for a new E/P_step pair isn't genuinely on-policy for it, just historical (still-valid) transitions that may not resemble what this new pair actually visits. Added --use-dagger (default off) so this data is only included when deliberately asked for, not by default just because it was part of the setup before. Matches the same principle behind removing the LR/hidden ablation flags: prove something is load-bearing before keeping it around.

Next: retrain P_step against the treeloss E WITHOUT dagger data (the new default) and rerun pstep_onestep_error_check.py -- this also directly answers whether the stale dagger data was helping, hurting, or neutral for the one-step error (currently 2.3-2.5x the true edge length, unaffected by LR or hidden-width changes).

## First real movement on P_step: residual parameterization, 2.35x -> 1.49x

After E was fixed (rho 0.9988) but branch_point_diagnostic.py's decision accuracy barely moved (69.3% -> 70.8%), isolated P_step's own one-step regression error directly (pstep_onestep_error_check.py, new diagnostic): P_step's error averaged 2.3-2.45x the true edge length, and that ratio was UNCHANGED across four separate ablations -- LR (0.01 -> 0.003), hidden width (32 -> 64), and removing stale DAgger data from training (collected on-policy against the ORIGINAL E, made opt-in via --use-dagger, off by default -- see "DAgger data made opt-in" section above). All four came back flat, ruling out tuning/capacity/stale-data as the cause.

Next hypothesis, informed by the actual numbers: true edges are short (mean ~0.34 in the current treeloss E) relative to a typical embedding's own scale (raw norm mean ~1.5, from treeloss_norm_check.py). OneStepPredictor was asking the network to output the WHOLE next absolute position from scratch, when the actual answer is always "close to the input, shifted by a small amount." Added an additive --residual option (jepa_model.py's OneStepPredictor: predicts start_raw + a small delta instead of a whole new vector, off by default so old checkpoints load unchanged). Result: error/true-edge-length ratio dropped from 2.35x to 1.49x -- the first ablation in this whole investigation that actually moved the number, not noise.

Still >1.0x (P_step's error is still bigger than a true edge), so this isn't fully solved, but it's real progress and confirms the parameterization mismatch was part of the problem. Next: check whether this shows up in branch_point_diagnostic.py's decision accuracy and the real planner's success rate, not just the isolated regression error.

## --force-horizon 1 test: degenerate CEM hypothesis REFUTED -- multi-step lookahead is necessary, not just noisy

Tested whether CEM's full multi-step search (horizon = bfs_distance + margin, often 15-20+ steps of autoregressive P_step rollout) was actively hurting the planner given P_step's known multi-step compounding-error problem, now that one-step accuracy is much improved (75.8% decision accuracy, residual parameterization). Added --force-horizon to maze_planner.py to pin the horizon at every replanning step instead of sizing it to the true remaining distance.

--force-horizon 1 (pure one-step lookahead, replanned every real step) result: success_rate 0/40 (WORSE than the full-horizon result, 4/40), real leftover distance mean=20.1 (worse than 13.5). But predicted-cost-vs-real-leftover-distance rho=0.984 -- the one-step cost signal is now excellent at tracking local progress. That combination (great local signal, zero convergence) is the signature of local oscillation/looping, not miscalibration: a purely greedy one-step policy with no memory of visited cells can ping-pong between two cells that each look better from the other's perspective, or wander into a dead end invisible to one-step lookahead.

Conclusion: multi-step lookahead is doing real, necessary work despite its own known compounding-error problems -- removing it makes things categorically worse, not just neutral. The residual-P_step one-step improvement (see section above) hasn't yet translated into better real-world success (3/40 -> 4/40) because the planner still depends on P_step's MULTI-step (autoregressive) quality, which hasn't been re-examined since the treeloss E + residual P_step changes. Next: rerun pstep_manifold_drift_check.py / step_drift_diagnostic.py (both need the same --encoder-checkpoint/--step-checkpoint override plumbing as everything else) against the current pair to see if the same "raw norm drifts into tanh's saturated zone over repeated autoregressive steps" mechanism still applies, now that both E's and P_step's own parameterization have changed.

## Drift diagnostics rerun against treeloss E + residual P_step: old mechanism refuted, distribution mismatch is the live suspect again

Reran step_drift_diagnostic.py and pstep_manifold_drift_check.py (both given the same --encoder-checkpoint/--step-checkpoint override plumbing as everything else) against the current pair. Part B (local expansion factor of one P_step call, bucketed by input raw norm) is now CONTRACTIVE in every bucket tested (0-1: 0.98, 1-2: 0.93, 2-3: 0.92) -- the OLD explanation (raw norm drifts toward the Poincare ball boundary, into tanh's high-sensitivity zone, amplifying error) does not hold for this pair; if anything P_step damps small perturbations everywhere tested.

Yet Part A still shows severe autoregressive blowups on specific rollouts: one pair's error climbs from 0.62 to 14.7 over 17 steps, another's tail climbs from 0.22 to 6.16 in its last dozen steps -- locally contractive everywhere but exploding somewhere globally is the signature of a DIFFERENT, previously-confirmed mechanism: distribution mismatch (the autoregressive chain drifting off the manifold of states P_step actually saw in training -- see the earlier "pstep_distribution_mismatch_check.py result: CONFIRMED" section). That was originally fixed by collecting on-policy DAgger data. We just made DAgger opt-in and off by default (correctly, since the 13 existing rounds are stale for this new E) -- but that also means the actual fix for this mechanism is currently absent for the treeloss E/residual P_step pair, and no fresh on-policy data has been collected for it yet.

Added --output-path to collect_dagger_data.py (so fresh rounds for a non-standard E/P_step pair don't get appended into the same file as the old, now-invalid rounds -- would silently mix on-policy and off-policy data with no way to separate them later) and --encoder-checkpoint/--step-checkpoint overrides (already had load_models, just needed CLI exposure). Added matching --dagger-path to jepa_train_step.py so --use-dagger can point at that new file instead of the standard one.

Next: collect_dagger_data.py --geometry hyperbolic --encoder-checkpoint jepa_checkpoint_treeloss_hyperbolic.pt --step-checkpoint jepa_step_checkpoint_treeloss_hyperbolic.pt --output-path maze_data_dagger_treeloss_{geometry}.json, then retrain jepa_train_step.py with --use-dagger --dagger-path maze_data_dagger_treeloss_{geometry}.json, then rerun the drift diagnostics and the real planner to see if genuinely-on-policy data fixes the blowup this time.

## Fresh on-policy DAgger round against treeloss E + residual P_step: made things WORSE, not better -- second confirmation that DAgger accumulation is unreliable for this task

Ran the queued test: collect_dagger_data.py --output-path maze_data_dagger_treeloss_{geometry}.json (300 episodes, 4278 windows, genuinely on-policy against the current treeloss E + residual P_step pair, kept separate from the old 13-round file), then jepa_train_step.py --residual --use-dagger --dagger-path maze_data_dagger_treeloss_{geometry}.json (residual parameterization kept, fresh dagger data added), then reran step_drift_diagnostic.py and maze_planner.py --replan.

Result: worse across the board, not better.
- Real planner success_rate: 0.00 (0/40), down from 0.10 (4/40) without this dagger round.
- Real leftover distance mean: 15.0, up from 13.5 (worse).
- Predicted-cost-vs-real-leftover rho: 0.722, up slightly from 0.623 (better correlation, but on a worse-performing policy -- calibration isn't the bottleneck here).
- step_drift_diagnostic.py Part A (autoregressive blowup): most rollouts got WORSE, not better -- the bfs_distance=44 pair's tail error climbed to 17.17 (worse than the previous worst, 6.16, which was a different pair); the bfs_distance=30 pair climbed to 3.84 (worse than 1.02). One pair did improve: the bfs_distance=17 pair, previously the worst offender at 14.7, dropped to 3.97. Net effect: negative -- fixing one rollout's blowup while creating/worsening others elsewhere.

This is the second time in this project's history that adding DAgger rounds has failed to reliably fix autoregressive drift -- the ORIGINAL E went through 13 rounds of exactly this pattern, and WORKPLAN's own "DAgger rounds 6-8" and "DAgger + replan rounds 8-12" sections concluded the apparent improvement trend across those rounds was noise, not real progress. One fresh round against the new treeloss E + residual P_step pair reproduces the same shape of result: some rollouts improve, others get worse, and the net real-world outcome is negative. A likely contributing mechanism: the balanced-ratio sampling used across pools gives this tiny 4278-window dagger pool equal per-batch weight against much larger pools (14000 and 48119 windows) -- over 10,000 training steps those 4278 windows get resampled roughly 200 times, which risks narrow overfitting to a small, on-policy-from-an-imperfect-planner sample rather than genuine generalization improvement, especially with only one round collected (the original E's own rounds 1-4 also looked like they were "climbing" before more rounds revealed it was noise).

Conclusion: reverting to the best-known-good configuration for now -- treeloss E (--dist-scale 4) + residual P_step (--residual), WITHOUT dagger data. That's success_rate=4/40, rho=0.623, leftover mean=13.5 -- the best result in this entire investigation, and the DAgger angle has now been tested twice (once historically, once freshly) with the same negative-or-noisy outcome both times. Not ruling out DAgger forever -- multiple rounds might behave differently, as data volume grows, as it did partially for the original E -- but it is not a quick fix, and chasing more rounds one at a time has a track record of looking like signal when it's noise.

The real open problem remains unresolved: E and P_step's one-step quality are both now good (rho 0.9988 for E, 1.49x error ratio for P_step, 75.8% decision accuracy), but the real closed-loop planner still only succeeds 4/40 (10%) -- far from the near-100% this toy tree problem should allow. The bottleneck is P_step's MULTI-step/autoregressive behavior specifically: locally contractive everywhere tested (pstep_manifold_drift_check.py Part B), yet still exploding on specific rollouts under autoregressive rollout (step_drift_diagnostic.py Part A) in a way that isn't fixed by one round of on-policy data. No new hypothesis is queued yet for this beyond DAgger (tested, unreliable) and --force-horizon (tested, refuted in the other direction -- shorter horizon is worse). Worth considering next: whether the blowups correlate with something structural (which node pairs, which tree branches) rather than being uniformly distributed, which the current diagnostics haven't checked directly.

## Maze-size-down experiment: size 10 (UNROLL_STEPS=8, fixing the data-coverage confound) -- best result yet, but rho stayed uninformative

Tested whether shrinking the maze (isolating horizon length as a variable) helps, per Andreas's suggestion. GRID_SIZE 16 -> 10 (diameter 46 -> 27, the project's original size, directly vetted by domain_fit_check.py unlike the untested size 8). Retrained treeloss E fresh (--dist-scale 2, chosen proportionally to the new diameter via a direct BFS diameter computation rather than guessing): rho=0.9950, 0/4950 failure-mode pairs, 0% near-saturated -- clean fit, same quality as the size-16 result.

First P_step attempt (UNROLL_STEPS left at 20, unchanged) came back flat-to-worse: 3/40, rho=0.082 (worse than size-16's 0.623). Diagnosed why before concluding anything: collect_directed_data.py's MIN_DISTANCE == UNROLL_STEPS == 20 requirement meant directed (goal-directed) training data only covered the near-diameter sliver (3.7% of all pairs), while real eval trials (MIN_TRIAL_DISTANCE=5, no upper bound) draw 84.2% of their pairs from bfs_distance in [5,20) -- a range directed data never touched at all. Not a fair test of the horizon-length hypothesis; a confound from keeping an absolute constant fixed while shrinking the maze around it.

Fixed by dropping UNROLL_STEPS 20 -> 8 (comfortably above MIN_TRIAL_DISTANCE=5, well under the new diameter 27) -- this automatically fixes collect_directed_data.py's coverage too (same import). Regenerated directed data (6000 episodes, lengths 8-27, covering the real eval range this time) and retrained P_step (--residual, no dagger, same as the best-known size-16 config).

Result: success_rate=0.175 (7/40) -- the best result in this entire investigation, beating the prior best (4/40 at size 16) both in absolute count and proportionally (real leftover distance mean 6.0 / diameter 27 = 22.2%, vs 13.5/46 = 29.3% at size 16). Real leftover distance distribution also tightened (7/40 exact successes, 18/40 within <=5, vs 13/40 within <=5 at size 16).

But: predicted-cost vs real-leftover-distance rho dropped to 0.003 -- essentially zero, worse than both the confounded size-10 attempt (0.082) and the size-16 result (0.623). So the outcome-level metrics improved while the calibration/ranking signal that CEM's own cost estimate provides got worse, not better -- an odd combination worth treating as a real, unresolved anomaly rather than noise (n=40 trials, so some of this could be sampling variance, but a 0.623 -> 0.003 swing is large).

Interpretation so far: partial confirmation of the horizon-length hypothesis (shorter horizons + adequate mid-range directed-data coverage measurably helped, in the best-result-yet sense) but not a full fix -- 17.5% success is still far from the project's own target for this toy tree problem ("should be almost 100%"). The collapsed rho suggests something about P_step's cost signal specifically at this new (UNROLL_STEPS=8, size-10) configuration doesn't track real drift well even where outcomes are better on average, which doesn't fit a simple "horizon length was the whole problem" story either.

Note for later: UNROLL_STEPS=8 and GRID_SIZE=10 are now both size-10-specific values, changed together as part of this isolating experiment -- revert both (UNROLL_STEPS->20, GRID_SIZE->16) before retraining against the original maze. The old size-16 artifacts (maze_data_size16.json, maze_data_directed_size16.json, jepa_checkpoint_treeloss_hyperbolic.pt, jepa_step_checkpoint_treeloss_hyperbolic.pt) are all still on disk, untouched.

Next: rerun step_drift_diagnostic.py / pstep_manifold_drift_check.py against this new size-10 pair to see whether the autoregressive blowup pattern is actually smaller now (the mechanism the horizon-length hypothesis targets directly), independent of what the aggregate rho number says.

## step_drift/pstep_manifold_drift rerun on size-10 (UNROLL_STEPS=8): horizon length REFUTED as the mechanism, new lead -- instability near the embedding origin

Reran both diagnostics against the size-10, UNROLL_STEPS=8, --residual pair (the one that gave the best-yet 7/40 planner result). Two findings, both against the horizon-length story:

1. step_drift_diagnostic.py: teacher-forced error stays low and bounded for all 5 pairs (bfs_distance 15-20), confirming the one-step model itself is sound -- but autoregressive error still explodes in 4/5 pairs, reaching 22-54 by the end of the (much shorter, 15-20 step) rollout. That's comparable to or WORSE in absolute magnitude than the worst blowups seen at size 16 (max ~17, over 30-44 step rollouts). Shrinking the maze did not shrink the blowup's magnitude -- it refutes the simple version of the horizon-length hypothesis (more steps = proportionally more compounding). One pair (bfs_distance=15) stayed comparatively bounded (max err 2.93), a useful negative case.

2. pstep_manifold_drift_check.py Part B, new result: the [0-1) raw-norm bucket -- previously CONTRACTIVE at size 16 (0.98) -- is now EXPANSIVE here (mean expansion factor 1.19, P_step amplifies small perturbations there). Buckets further out stay contractive ([1-2): 1.01, [2-3): 0.91, [3-4): 0.90). This bucket is also sparse: only 3 of the 40 sampled cells fell in it. Cross-referencing Part A's raw-norm trajectories: in the 4 pairs that blew up, the autoregressive chain's raw norm visibly dips down toward/into this same near-origin bucket a few steps before the error starts compounding (e.g. one chain dips to 0.20 at step 12-13, then error goes 2.14 -> 5.33 -> 10.48 -> ... -> 54.46 over the remaining 7 steps); the one pair that stayed bounded also dipped near there but didn't linger.

New hypothesis, more specific than "distribution mismatch" in general: P_step is locally unstable specifically near the embedding origin (likely where the tree's central/hub cell(s) sit), and this may be a training-coverage gap rather than an intrinsic property of the geometry -- worth checking directly rather than assuming. Wrote pstep_origin_coverage_check.py: identifies which real cells have raw norm < 1, their degree, and how often they actually appear in the random-walk and directed training pools vs a uniform base rate. Not yet run.

Also worth flagging: this reopens a question about the original "boundary saturation" explanation (high-norm/near-boundary instability, fixed via dist_scale) -- the mechanism now observed is at the OPPOSITE end of the norm range (near-origin, not near-boundary). These are two distinct failure modes of the same underlying architecture, not the same bug reappearing.

## pstep_origin_coverage_check.py result: sparse-data hypothesis REFUTED

Ran the coverage check against the size-10 pair. 14 of 100 cells have raw norm < 1 (more than the ~7.5% Part B's 40-cell sample suggested, but consistent with sampling noise), spatially clustered (rows ~1-6, cols ~3-7), degree 1-4 (not conspicuously above the maze's mean degree of 1.98 -- not simply "high-degree hub cells").

Checked their visit frequency in both training pools against a uniform 1/100 base rate: NOT under-represented. Random-walk rates 0.6x-2.6x uniform, directed rates 0.4x-5.6x uniform -- most cells at 2-5x uniform, especially in the directed pool. If anything these cells are OVER-visited. This refutes the "P_step is unstable near the origin because it rarely trains there" hypothesis directly -- it's not a data-count problem.

New hypothesis, not yet tested: representational crowding. 14% of cells packed into raw norm < 1 (a small fraction of E's 0-3.486 range) may mean these particular cells sit unusually close together in embedded distance relative to their true bfs_distance from each other -- a resolution problem, not a coverage problem. Wrote pstep_origin_crowding_check.py to check this directly (embedded distance vs. same-true-distance bucket mean, for all pairs among the 14 cells). Not yet run.

## pstep_origin_crowding_check.py result: crowding hypothesis also REFUTED -- new lead points at the --residual parameterization itself

Mean ratio (embedded distance / same-true-distance bucket mean) across all 91 pairs among the 14 near-origin cells: 1.013 -- essentially exactly 1.0, no compression. These cells are embedded completely normally relative to their true bfs_distance from each other. E's local fidelity there is fine (consistent with its overall rho=0.9950). So representational crowding is refuted, same as sparse training data before it -- E itself is not the problem near the origin.

New candidate, this time about P_step's --residual parameterization specifically, not E or training data: the size-10 checkpoint's true edge length (true_dist=1 bucket mean_embedded) is 0.608. Of the 14 near-origin cells, 7 have raw norm BELOW that (as low as 0.114) -- meaning for these cells, a typical action's target displacement (~0.6) is comparable to or larger than the cell's own position norm. --residual predicts start_raw + delta, implicitly assuming delta is small relative to start -- an assumption that only holds when |start| is large enough relative to typical edge length. Near the origin, |start| shrinks toward zero while delta stays roughly edge-length-sized, so the "small correction" premise breaks down exactly there. This would be a property of the residual formulation itself (would apply in euclidean too, wherever position magnitude approaches zero), not a hyperbolic-curvature effect like the original boundary-saturation problem was.

Next: retrain P_step at size 10 WITHOUT --residual (same E, same data) and rerun pstep_manifold_drift_check.py's Part B, specifically the [0-1) bucket. If the expansion factor there drops back to <=1 without --residual, that confirms the mechanism directly -- a clean ablation, not a guess.

## --residual ablation at size 10: CONFIRMED as the mechanism -- and actively harmful at this scale

Retrained P_step at size 10 WITHOUT --residual (same E, same data) and reran pstep_manifold_drift_check.py + step_drift_diagnostic.py's same 5 pairs. Both checks confirm the hypothesis decisively:

Part B, [0-1) bucket: 1.19 (expansive) with --residual -> 0.78 (contractive) without it. Every other bucket also dropped (was 1.01/0.91/0.90, now 0.54/0.31/0.27) -- without residual, P_step damps perturbations everywhere, increasingly so at higher norm.

Part A, final-step autoregressive error, residual vs. non-residual, same 5 pairs: bfs=18: 35.31 -> 4.37. bfs=15: 22.33 -> 8.25. bfs=15: 2.93 -> 0.29. bfs=19: 39.07 -> 0.97. bfs=20: 54.46 -> 0.39. Every pair improved 3x to >100x -- not a marginal difference.

Conclusion: --residual, which measurably helped at size 16 (one-step error ratio 2.35x -> 1.49x, decision accuracy 70.8% -> 75.8%), is actively HARMFUL at size 10. Mechanism: the size-10 E's typical edge length (0.608) is comparable to or larger than many cells' own raw position norm (14 of 100 cells have raw norm < 1, 7 of those < 0.608) -- "start + small delta" isn't a small correction there, so residual's implicit assumption breaks. This is scale-dependent, not a universal property of the parameterization -- exactly the "don't keep what happened to work once" trap Andreas has flagged repeatedly this project. --residual needs to be re-ablated per maze size/scale going forward, not assumed.

Next: rerun the real closed-loop planner (maze_planner.py --replan) against this non-residual size-10 checkpoint -- if Part A/B's dramatic improvement translates to real success rate, this could be the biggest jump in the whole investigation.

## Non-residual size-10 planner result: diagnostics improved sharply, real success rate did NOT move

success_rate stayed at 0.175 (7/40), identical count to the --residual run. But everything else improved: rho 0.003 -> 0.245 (still weak, but no longer near-zero), real leftover distance mean 6.0 -> 5.9, and the near-miss distribution tightened meaningfully (within<=3: 9/40 -> 14/40, within<=5: 18/40 -> 20/40). So non-residual is strictly better or equal on every metric except the raw success count, which stayed flat.

This matches a pattern that has now recurred several times this project: a real, verified improvement to one component (E's fit, P_step's one-step error, now P_step's autoregressive stability) fails to move the real closed-loop success rate by nearly as much as the component-level diagnostics would suggest. Recommend adopting non-residual as the new default at size 10 (better on every other axis, no worse on success), but the flat success count means something else is now the binding constraint -- most likely CEM's own search process (sample count, elite fraction, iteration count -- none tuned since --replan was added, per the size-16 WORKPLAN note from much earlier), not remaining P_step/E quality.

## cem_hparam_search.py sweep, size 10 (non-residual pair): ELITE_FRAC 0.1 -> 0.05 nearly doubles success

Added --encoder-checkpoint/--step-checkpoint/--output-path overrides to cem_hparam_search.py (previously hardcoded to the default checkpoint paths, never updated since the original size-16 sweep that chose the current N_SAMPLES=100/N_ITERS=2/ELITE_FRAC=0.1/MIN_PROB=0.05 defaults). Ran the full 192-combo grid against the size-10, non-residual, --dist-scale 2 pair -- finished in 1116s (~19 min), much faster than the original 40-90 min estimate (shorter horizons at this maze size).

Best combo: N=100, iters=2, elite=0.05, min_prob=0.05 -> success=13/40 (32.5%), rho=0.532, mean_leftover=5.9, within5=21/40. The exact current defaults (N=100, iters=2, elite=0.1, min_prob=0.05) reproduced the already-known 7/40 result inside the sweep too, confirming the sweep script's evaluate_combo() matches real maze_planner.py behavior exactly. So the ONLY change from the working config is ELITE_FRAC 0.1 -> 0.05 -- nearly doubling success. Several other elite=0.05 combos also clustered at 10-11/40, suggesting this is a real regional effect, not a single lucky draw (n=40 trials means +/-1 trial is easily noise, but the elite=0.05 cluster is consistently ahead of elite=0.1 across multiple N_SAMPLES/N_ITERS/MIN_PROB combinations).

No combo in the grid got anywhere close to the project's target (near 100% for this toy tree problem) -- best is 32.5%. So CEM tuning alone doesn't solve the underlying problem, but it's a real, cheap, verified gain stacked on top of the treeloss/residual-ablation work. Updated maze_planner.py's ELITE_FRAC default to 0.05 with the same documented-history comment convention as N_SAMPLES/N_ITERS/MIN_PROB, and noted this knob should be re-swept whenever GRID_SIZE/E/P_step changes meaningfully, not assumed stable.

Next: confirm with a direct maze_planner.py --replan run (now using the new default, no CLI override needed) that this reproduces 13/40 outside the sweep script too.

## Confirmed: ELITE_FRAC=0.05 reproduces 13/40 outside the sweep script

Direct maze_planner.py --replan run (new default, no override needed) reproduced the sweep's result exactly: success_rate=0.33 (13/40), rho=0.532, mean_leftover=5.9, within5=21/40. Matches evaluate_combo()'s number precisely -- confirms the sweep script's monkey-patching approach is faithful to real maze_planner.py behavior.

This is the best result of the entire investigation: 13/40 (32.5%), vs. 7/40 (17.5%) before this sweep, vs. 4/40 (10%) at size 16 (the previous best before switching to size 10). Cumulative stack of fixes to get here: dist_scale fix for E's radius saturation, GRID_SIZE 16->10 + UNROLL_STEPS 20->8 (fixing the horizon/data-coverage confound), dropping --residual (harmful at this scale, confirmed via ablation), ELITE_FRAC 0.1->0.05 (this sweep). Still well short of the project's own target (near 100% for this toy tree problem) -- real work remains, but each step has been a verified, real gain, not a lucky guess kept around.

## branch_point_diagnostic.py rerun (correct checkpoints this time): 79.6% one-step decision accuracy, best yet

First attempt ran with no checkpoint overrides -- silently used the stale default (jepa_checkpoint_hyperbolic.pt / jepa_step_checkpoint_hyperbolic.pt, the original pre-treeloss baseline), which this script doesn't assert-check against GRID_SIZE the way maze_planner.py does (no ckpt_size==size assertion) -- gave a meaningless 49.2% comparing a 100-cell maze's cells against what's very likely a size-16-trained E. Disregarded; not a real result.

Rerun with the correct size-10/non-residual checkpoints: overall one-step decision accuracy 79.6% (5435/6831), the best in this project's history (previous best 75.8% at size 16 with --residual). Consistent across bfs_distance buckets (70.1%-82.0%, no clear degradation with distance) and degree buckets (corridor 81.4%, branch 76.1%). Margin analysis: of the wrong decisions, 53.7% are near-ties (margin < 5% of current distance) -- close calls, not gross misjudgments; the remaining 46.3% are more confidently wrong.

Gap to reconcile: 79.6% one-step accuracy vs. 32.5% real closed-loop success. Not necessarily contradictory -- with ~10-27 real steps per trial and replanning after each one, even a well-calibrated ~80% per-step accuracy compounds substantially over a full trajectory (independent 80% correct across 15 steps would be ~3.5%; replanning's self-correction pulls the real number up from that naive floor to the observed 32.5%, which is plausible without implying a large untapped one-step-accuracy gain remains). Open question, not yet resolved: does the ~46% "confidently wrong" (non-near-tie) share of errors cluster on specific structural patterns (particular branches, particular degree/distance combinations) rather than being uniform -- that would point at a further fixable gap distinct from compounding.

## branch_error_clustering_check.py result: sharp clustering found -- 2 of 69 branch cells account for a hugely disproportionate share of confidently-wrong decisions

Mean confidently-wrong rate across 69 branch cells: 9.5%. But b=(9,6) and b=(3,0) sit at 98.0% and 97.0% -- essentially always wrong (correct only 2.0% for both), and confidently so, regardless of which of the 99 possible targets is asked. That's ~10x the mean, concentrated in 2.9% of branch cells. Neither is near-origin (raw norms 1.99/2.05, close to the overall mean 1.73) -- this is a DIFFERENT mechanism from the origin-instability finding, not a recurrence of it. A handful of other cells are moderately elevated (4,6)=57.6%, (5,6)=44.4%, (2,5)=40.4% -- three of these ARE near-origin, so that mechanism may still contribute at a smaller scale -- but the two extreme outliers dominate the picture.

Secondary, milder pattern: per-correct-action breakdown shows DOWN is the worst direction (76.2% correct, 15.7% confidently wrong) vs UP (85.9% correct, 3.9% confidently wrong) -- real but much smaller than the two-cell outlier (roughly 4x spread vs ~10x).

Target-cell breakdown showed no comparably sharp clustering -- worst targets sit around 13-15% confidently-wrong vs a presumably similar background rate, nothing like the branch-cell outliers.

Near-total, target-independent failure at one specific cell (regardless of the 99 different targets tried) points at a specific, mechanistic bug rather than a general embedding-quality issue -- most likely P_step badly mispredicting ONE of that cell's real outgoing actions in a way that lands unusually close to most targets (an "attractor" failure), winning the argmin almost universally. Next: inspect (9,6) and (3,0) directly -- their real neighbors/actions, what P_step actually predicts for each of the 2-4 actions from these cells, and where those predictions land relative to a sample of real targets, to find the specific mechanism.

## Trajectory-level cycles confirmed as the mechanism (Andreas's hypothesis) -- random walks bounce into dead ends and immediately reverse

Andreas asked directly whether the two broken branch cells' problem was really about hyperbolic geometry being bad at "cycles." The MAZE graph has no cycles (exact spanning tree, as always), and E's all-pairs training doesn't depend on trajectories at all -- but P_step's training data absolutely can and does contain trajectory-level cycles even on an acyclic graph. Checked directly (no torch needed, pure data): at b=(9,6), the dead-end-entering action (UP) appears 627 times in the random-walk pool (vs 626 for the correct majority action, RIGHT -- NOT under-sampled), but 606/627 (97%) are immediately reversed the very next step. Same at b=(3,0): 469 appearances, 432 (92%) immediately reversed. The directed pool's reversal rate is much lower (31-36%), since goal-biased rollouts wander into dead ends far less.

This is a sharper, more specific mechanism than my earlier "regression to the mean" hand-wave: most of what P_step ever sees following the dead-end-entering action is "and then immediately undo it" -- exactly the kind of self-canceling multi-step window (UNROLL_STEPS=8) that would corrupt the one-step prediction for that specific action, biasing it toward something blurred/central rather than confidently landing at the true dead-end cell.

Root cause: env.py's random_rollout samples uniformly among all valid moves every step, with no bias against immediately reversing the previous action -- so at any cell adjacent to a dead end, there's a real chance of walking in and being forced straight back out, over and over across many episodes.

Fix (not yet tested, opt-in): added --backtrack-weight to collect_data.py / MazeEnv.random_rollout (env.py) -- downweights (does not forbid) choosing the exact reverse of the previous action when another valid option exists, still allows it when it's the only valid move (forced dead-end reversal). Off by default (preserves original uniform policy) -- needs ablation before being trusted, same as everything else in this project.

Next: regenerate random-walk data with --backtrack-weight (e.g. 0.1) into a NEW file (don't overwrite maze_data_size10.json), retrain P_step (same E, same directed data, no --residual), rerun branch_cell_inspect.py on (9,6)/(3,0) specifically to see if the mechanism is fixed directly, then the full planner to see if it helps broadly (this changes data generation everywhere a small side-branch exists, not just these 2 cells, so it could plausibly move the overall numbers too, not just the two known offenders).

## Next commands to run: testing --backtrack-weight (downweight immediate reversal in random-walk data)

New files throughout, so the current best (13/40, non-residual + ELITE_FRAC=0.05) is never overwritten:

```
uv run python collect_data.py --output-path maze_data_size10_antibt.json --backtrack-weight 0.1

uv run python jepa_train_step.py --geometry hyperbolic \
    --encoder-checkpoint jepa_checkpoint_treeloss_size10_hyperbolic.pt \
    --step-checkpoint-out jepa_step_checkpoint_treeloss_size10_antibt_{geometry}.pt \
    --data-path maze_data_size10_antibt.json \
    --directed-data-path maze_data_directed_size10.json

uv run python branch_cell_inspect.py --geometry hyperbolic \
    --encoder-checkpoint jepa_checkpoint_treeloss_size10_hyperbolic.pt \
    --step-checkpoint jepa_step_checkpoint_treeloss_size10_antibt_hyperbolic.pt \
    --cells "(9,6)" "(3,0)"

uv run python maze_planner.py --geometry hyperbolic --replan \
    --encoder-checkpoint jepa_checkpoint_treeloss_size10_hyperbolic.pt \
    --step-checkpoint jepa_step_checkpoint_treeloss_size10_antibt_hyperbolic.pt
```

Note: --residual is intentionally omitted (matches current best config, non-residual).
backtrack_weight=0.1 means an immediate reversal is 10x less likely to be chosen than
before, when a real alternative move exists (never prevented when it's the only option,
so forced dead-end backtracking is untouched). If this doesn't move the two broken
cells enough, worth trying a lower value (e.g. 0.02) before concluding the mechanism
needs a different fix (e.g. discounting/downweighting these transitions in the P_step
loss directly, rather than at data-collection time).

## --backtrack-weight=0.1 result: did NOT fix the attractor, and made closed-loop WORSE (negative result)

Ran the full antibt sequence (new data/checkpoint files, non-residual, ELITE_FRAC=0.05 default unchanged):
- branch_cell_inspect on (9,6): dead-end action UP win-rate 98%->82.8% (some softening, but now
  split across UP 82.8% + LEFT 13.1%, still leaving correct action RIGHT at only 4.0% -- no
  practical improvement, correct action still loses almost always)
- branch_cell_inspect on (3,0): dead-end action UP win-rate unchanged at 97.0% (correct action
  DOWN still at 3.0%)
- maze_planner --replan: success_rate dropped to 6/40 (0.15), WORSE than the 13/40 best and even
  worse than the flat 7/40 baseline. rho=0.515 (similar to before).

Root cause of why the fix couldn't work here: checked adj directly -- (8,6) [the dead-end reached
by (9,6)'s UP action] and (2,0) [reached by (3,0)'s UP action] are both literal degree-1 leaves,
connected ONLY back to b. So `random_rollout`'s `backtrack_weight` (which only reweights the
choice when `len(options) > 1`) is a no-op at exactly the cells that matter: once inside a true
leaf dead-end, reversal isn't a choice being made among alternatives, it's the ONLY valid move.
The corrupting "b -> dead_end -> b" round trip is forced by topology, not by the random-walk
policy's preference -- so nothing about how we weight choices *inside* the dead end can reduce
how often that self-canceling window gets generated.

Implication: the real lever is upstream -- how often the walk chooses to ENTER a degree-1 dead
end from b in the first place, not what it does once inside. Next candidate fix: extend
backtrack_weight (or add a new flag) to also downweight, at b itself, choosing an action whose
destination is a degree-1 cell (a knowable structural fact, not something P_step has to learn) --
this reduces how many of these self-canceling windows exist in the training data at all, rather
than trying (impossibly) to prevent the forced bounce-back once entered. Reverting
maze_data_size10_antibt.json / the antibt checkpoint from further use until this is retested --
current best config remains non-residual + ELITE_FRAC=0.05 on the ORIGINAL maze_data_size10.json
(13/40).

## attractor_geometry_check.py: is the attractor a hyperbolic-curvature artifact? (Andreas's question)

Same raw predicted vectors, argmin recomputed under both metrics, no retrain:

  b=(9,6): hyperbolic UP=97.0%  |  euclidean UP=76.8%
  b=(3,0): hyperbolic UP=99.0%  |  euclidean UP=85.9%

Answer: both. The SAME wrong action wins under plain Euclidean distance on the
identical raw vectors -- so the attractor is NOT a hyperbolic-curvature artifact,
it's already present in what P_step predicts, before any geometry is applied.
But hyperbolic distance clearly AMPLIFIES it substantially (~20pp worse at both
cells) -- consistent with curvature's steep growth with radius giving a more
"central" point an extra advantage on top of the base effect. So: root cause is
geometry-independent (a learned-prediction/data-imbalance bias, confirms the
prior claim), but curvature is a real, measured aggravating factor, not nothing.
Distinguish from --residual (a scale bug that was excluded from this checkpoint
entirely) and from the dist_scale/boundary-saturation fix (a target-range
problem) -- neither of those explanations is what's happening here.

## --deadend-weight added: disallow bouncing INTO dead ends, not out of them (Andreas's question)

--backtrack-weight targets the wrong choice point: at the two known-broken cells, the dead end
one step away is a literal degree-1 leaf, so the bounce back out is forced (no alternative to
downweight). Andreas asked directly whether we can just disallow entering dead ends in the first
place -- that IS a real choice (every branch cell tested has degree >= 2, so an alternative always
exists at the point of entry). Added `deadend_weight` to env.py's random_rollout: for each
candidate action, if its destination has degree 1 in the maze graph, its relative sampling weight
is multiplied by deadend_weight (0.0 = fully excluded whenever an alternative exists). Composes
multiplicatively with backtrack_weight. Threaded --deadend-weight through collect_data.py (default
None/off). Directed data (much lower reversal rate there already) still teaches P_step the
b->dead_end edge even at deadend_weight=0.0, so this isn't zeroing out that edge's training signal
entirely, just removing the corrupted majority of it from the random-walk pool. Not yet tested --
next: regenerate data with --deadend-weight 0.0 (start with the hard version, since it's cheap to
soften later if it overcorrects), retrain P_step (non-residual, same as current best), rerun
attractor_geometry_check.py + branch_cell_inspect.py on (9,6)/(3,0), then the full planner.

## --deadend-weight=0.0 result: ALSO didn't fix the attractor, and made closed-loop even WORSE

  b=(9,6): hyperbolic UP=83.8% (vs 97.0% baseline, 82.8% with backtrack_weight=0.1) -- barely moved
  b=(3,0): hyperbolic UP=97.0% (literally unchanged from baseline)
  planner: success_rate=4/40 (0.10) -- worse than backtrack_weight's 6/40, worse than the 7/40
  flat baseline, and far below the 13/40 best. rho=0.480, similar range.

Collateral damage, caught via collect_data.py's own sanity-check line: distinct endpoints at H=12
dropped to 21/23/18 (vs 41/37/33 with the much milder backtrack_weight=0.1 run) -- i.e. forbidding
entry into ANY of the maze's degree-1 cells from ANY step of ANY random walk (not just at the two
known-broken branch cells) roughly halves the walk's exploration diversity throughout the ENTIRE
maze. This is a much blunter intervention than intended: there isn't just one dead end near (9,6)/
(3,0), there are many degree-1 cells scattered through a 100-cell spanning tree, and suppressing
entry into all of them reshapes/narrows the whole random-walk data distribution, likely degrading
many OTHER transitions' training signal too -- plausibly explains why closed-loop got worse overall
despite barely helping the two cells this was aimed at.

Conclusion: two different environment-level (data-generation) interventions have now both failed,
each with real, measurable side effects beyond their target. Pivoting the hypothesis: the dead-end
window's single-step TARGET (state after the first action) is always correct ground truth (it's a
real environment transition, not corrupted data) -- what may actually be corrupting P_step(b,
UP)'s learned representation is that essentially every UNROLL_STEPS=8 training WINDOW containing
that action is a short, degenerate "in and immediately back out" trajectory, so the recursive/
autoregressive multi-step loss's gradient for that action is dominated by matching a trivial
bounce-back continuation, at the plausible expense of precisely grounding the single-step (t=0)
prediction that branch_point_diagnostic.py / the planner actually rely on. Next candidate: a
TRAINING-time (not data-generation-time) fix in jepa_train_step.py -- filter or downweight
windows whose first step is immediately reversed (states[i] == states[i+2]) when building
UNROLL_STEPS windows, leaving collect_data.py and the random walk's exploration coverage
completely untouched. Not yet implemented -- proposing to Andreas before writing it, given two
misses in a row on the data-generation side.
