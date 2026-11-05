"""
maze_planner.py

Maze JEPA -- Phase 3: a CEM (cross-entropy method) planner that searches
for an action sequence from a start cell to a target cell PURELY in
latent space -- using only the frozen encoder E and the frozen one-step
predictor P_step from jepa_train_step.py, never the real maze -- then
executes the chosen sequence on the real MazeEnv to check whether it
actually works. See WORKPLAN.md's Phase 3 section.

WHY THIS TEST, NOT "CAN THE MAZE BE SOLVED": this 16x16 maze is small and
exactly known, so plain BFS solves it instantly and perfectly (see
maze_gen.py's bfs_distance) -- a planner isn't needed to solve the maze
itself. The actual question is narrower: does searching through the
HYPERBOLIC latent space (the one Phase 2 found to win at dim=8) produce a
better cost signal for CEM than the EUCLIDEAN one does, in a way that
shows up as behavior -- success rate and path efficiency -- not just an
offline embedding-distance metric. This directly follows WORKPLAN's own
Phase 3 evaluation bar (does the representational difference change
planner behavior, the same lesson ball_in_cup_hyperbolic's CEM ranking
check taught).

CEM FOR DISCRETE ACTIONS: unlike ball_in_cup_hyperbolic's continuous CEM
(a diagonal Gaussian per timestep over continuous action dims), the
maze's 4 actions are categorical, so the per-timestep distribution being
refit each iteration is a categorical distribution over
{UP,DOWN,LEFT,RIGHT}, not a Gaussian. Elites vote on each timestep's
action frequency; that becomes the next iteration's sampling
distribution (with a floor so it never fully collapses to one action
before n_iters is up). Otherwise the same CEM structure as
cem_planner.py: sample candidates, roll them forward through the learned
one-step model, score by latent distance to the goal, keep elites, refit,
repeat.

HORIZON, PER TRIAL: sized as bfs_distance(start, target) + a margin, not
one fixed H for every (start, target) pair -- this maze's shortest paths
range from 0 to 45 steps (measured directly, see WORKPLAN.md), so a
single fixed horizon would be wildly wrong for most pairs. Using the true
BFS distance only to SIZE the search budget is fair: the planner still
has to find the right actions itself via latent-space CEM, it's simply
given a reasonable amount of "time" to do it in, the same way you'd size
a chess clock without that telling you the winning moves.

PREDICTED-COST-VS-REAL-LEFTOVER-DISTANCE CHECK (added after the unroll
fix (jepa_train_step.py, UNROLL_STEPS=5) still left the planner at 0/40
both geometries): step_drift_diagnostic.py showed autoregressive latent
error still grows to 10-30 by 30-45 steps even after the fix -- but CEM
only needs candidate RANKING to be right, not the absolute magnitude of
predicted cost, to still pick good plans. So the open question, raised
directly by Andreas: if the embedding is a good representation, drift in
latent space should correspond to drift in the real world -- a high
predicted cost should mean the plan actually lands far from the target,
not that the number is merely large. run_trials now also records, per
trial, CEM's predicted cost for the chosen plan (`best_cost` from
cem_plan) and the REAL leftover BFS distance from wherever the plan
actually left the agent to the target (0 if it succeeded). main() prints
the Spearman rank correlation between those two per geometry. A strong
positive correlation means the latent metric is doing its job under
drift (and the planner's own hyperparameters -- horizon margin, CEM
sample/iteration counts, or a bigger UNROLL_STEPS -- are the next things
to tune); a weak/no correlation means predicted cost stops meaning
anything once drift sets in, regardless of how well P_step was trained,
and the embedding's local geometry itself is the bottleneck.

ORACLE ABLATION (added after rho~0.6 for both geometries turned out to
mean "moderately correlated but landing nowhere close" -- real leftover
distance min=2-3, mean~19, max~38-40, i.e. the chosen plans are ending
up roughly as far from the target as a fresh random position, not just
missing by a few cells): this raises a separate question from
representation quality -- is discrete CEM search itself even capable of
finding the single correct path through a maze, given a PERFECT cost
signal? Unlike ball_in_cup_hyperbolic's continuous action space, where a
slightly-wrong action still lands near the right outcome, the maze's 4
actions are categorical: one wrong choice at a junction doesn't nudge
you off course, it sends you down an entirely different, unrelated
corridor of the spanning tree. And since a perfect maze has exactly ONE
correct path between any two cells, success requires CEM's per-timestep
categorical distribution to concentrate on one exact discrete sequence,
not just a good-enough neighborhood -- and the per-timestep-independent
elite refit doesn't model the strong path-dependence between timesteps
(which action is right at step 10 depends on which branch was taken at
step 3), so elites that took genuinely different branches can pull the
marginal distribution toward something incoherent as a whole.

`--geometry oracle` runs the identical CEM scaffold (same N_SAMPLES,
ELITE_FRAC, N_ITERS, MIN_PROB, same trial pairs via PLANNER_SEED) but
scores every candidate by simulating it for REAL on the maze graph
(oracle_step, via the adjacency structure directly) and costing it by
the TRUE BFS distance from the resulting real cell to the target --
no torch, no E, no P_step at all. If oracle-CEM also fails to land
exactly on target often, discrete search itself is the bottleneck,
independent of the representation. If oracle-CEM succeeds close to
100%, the search mechanism is fine and the gap really is about what
E/P_step predict.

BEAM SEARCH -- TESTED AND REJECTED, KEPT FOR REFERENCE ONLY, NOT USED
(added after oracle-CEM -- CEM's search scaffold given a PERFECT cost
signal -- still only reached success_rate=0.20 (8/40), confirming the
search mechanism itself, not the representation, was a real bottleneck):
CEM's per-timestep-independent marginal refit is a poor fit for a tree
maze -- elites that actually took different branches get averaged into
a distribution that represents no coherent path. `beam_plan` /
`oracle_beam_plan` replace it with deterministic beam search: at each
step, every one of the current BEAM_WIDTH candidate PREFIXES (each a
genuine, concrete, fully coherent partial action sequence, never an
average) is expanded by all 4 actions, every child is scored by
predicted (or, in oracle mode, true) distance to target, and only the
best BEAM_WIDTH children survive. This can never blend incompatible
branches, because every survivor is always a real, complete, executable
path so far. On the oracle (perfect cost signal), this fully solved the
search problem: success_rate=1.00 (40/40), exactly BFS-optimal path
length every time. But run against the actual trained models, it barely
helped euclidean (0/40 -> 1/40, rho 0.637 -> 0.728) and made hyperbolic
WORSE (0/40 -> 0/40, rho 0.597 -> 0.189) -- verdict: beam search's
greedy, irrevocable commitment (once a branch is pruned from the beam,
it's gone for good) is fine under a perfect cost signal but brittle
under a noisy, imperfect one, and hyperbolic distance's arcosh-driven
sensitivity near the Poincare ball's boundary makes it a worse fit for
that kind of greedy search than CEM's more exploratory, floor-protected
resampling. DECISION (Andreas, after this result): discard beam search
as the planner's search method for the time being -- CEM stayed the one
in use -- but keep `beam_plan`/`oracle_beam_plan` in this file, unused,
as a documented, working reference for a genuinely different search
strategy, in case a future fix to P_step's noise level makes revisiting
it worthwhile.

REVISITED (after collect_directed_data.py's fix substantially reduced
P_step's autoregressive noise -- see WORKPLAN.md's "Directed-data
result" section): the condition above was met, so beam search is wired
back in via `--search {cem,beam}` (default cem, so nothing changes
unless explicitly requested). run_trials/run_oracle_trials now dispatch
to cem_plan/beam_plan (or oracle_cem_plan/oracle_beam_plan) based on
this flag, instead of always calling CEM.

Run (in Andreas's terminal, after jepa_train.py AND jepa_train_step.py
have both been run for the geometries being compared):
    uv run python maze_planner.py --geometry both
    uv run python maze_planner.py --geometry oracle
    uv run python maze_planner.py --geometry both --search beam
    uv run python maze_planner.py --geometry oracle --search beam
"""
import argparse
import random

import torch

from env import ACTIONS, ACTION_DELTA, GRID_SIZE, ALGO, MazeEnv
from maze_gen import GENERATORS, bfs_distance
from jepa_model import (
    cell_index,
    load_encoder_checkpoint,
    load_step_checkpoint,
    squared_dist,
    to_geometry,
)

MAZE_SEED = 0     # same fixed maze as collect_data.py / jepa_train.py
PLANNER_SEED = 2  # separate from MAZE_SEED and DATA_SEED -- fresh trial pairs

N_TRIALS = 40
MIN_TRIAL_DISTANCE = 5    # skip near-trivial (start, target) pairs
HORIZON_MARGIN = 6        # planning budget = bfs_distance + this

N_SAMPLES = 100           # was 200 -- cem_hparam_search.py's 192-combo sweep
                          # under --replan found N_SAMPLES has little effect either way,
                          # so the cheaper of the two tied-best values wins (see below)
ELITE_FRAC = 0.05         # was 0.1 -- re-swept against the size-10/non-residual pair
                          # (see WORKPLAN.md's "cem_hparam_search.py sweep, size 10"
                          # section): N=100/iters=2/elite=0.05/min_prob=0.05 gave
                          # success=13/40, the single best combo in that sweep, vs 7/40
                          # at elite=0.1 (everything else held fixed) -- a real jump, not
                          # noise (several nearby elite=0.05 combos also clustered at
                          # 10-11/40). Re-sweep any time GRID_SIZE/E/P_step changes
                          # meaningfully -- this knob is not assumed stable across scale.
N_ITERS = 2               # was 6 -- the ORIGINAL (size-16) sweep found N_ITERS=2
                          # dominates the top of the leaderboard and N_ITERS=10 clusters
                          # toward 0/40 -- more refinement per replan call looks like it
                          # overfits P_step's noisy cost signal, which constant replanning
                          # already corrects for without needing that extra grinding.
                          # Reconfirmed as still-best by the size-10 re-sweep above.
MIN_PROB = 0.05           # was 0.02 -- floor per action so CEM can't fully collapse
                          # early; bumped per the original sweep result above, reconfirmed
                          # by the size-10 re-sweep

BEAM_WIDTH = 50            # beam search: survivors kept per step -- chosen so
                           # BEAM_WIDTH * 4 (every survivor expanded by every
                           # action) is roughly N_SAMPLES, for a comparable
                           # per-step search budget to CEM

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


def load_models(geometry, encoder_checkpoint=None, step_checkpoint=None):
    """encoder_checkpoint/step_checkpoint: optional path TEMPLATES (containing
    "{geometry}", same convention as jepa_model.py's own CHECKPOINT_PATH/
    STEP_CHECKPOINT_PATH) to override the default checkpoint files -- lets this
    planner evaluate an alternative E/P_step pair (e.g. jepa_train_treeloss.py's
    jepa_checkpoint_treeloss_{geometry}.pt + a P_step trained against it via
    jepa_train_step.py --encoder-checkpoint/--step-checkpoint-out) without
    duplicating this file. Meant for single-geometry runs (--geometry hyperbolic,
    not "both"/"oracle") since e.g. the treeloss checkpoint only exists for
    hyperbolic."""
    e_kwargs = {"path": encoder_checkpoint} if encoder_checkpoint else {}
    s_kwargs = {"path": step_checkpoint} if step_checkpoint else {}
    E, e_ckpt = load_encoder_checkpoint(geometry, **e_kwargs)
    P_step, s_ckpt = load_step_checkpoint(geometry, **s_kwargs)
    assert e_ckpt["size"] == s_ckpt["size"]
    assert e_ckpt["dim"] == s_ckpt["dim"], "E and P_step were trained at different LATENT_DIM -- retrain jepa_train_step.py against the current E checkpoint"
    return E, P_step, e_ckpt["size"]


def cem_plan(E, P_step, geometry, start, target, size, horizon, rng_seed):
    """Returns (actions, best_cost): the `horizon`-length action sequence
    found by CEM search purely in latent space, and CEM's predicted cost
    (squared latent distance to the target) for that sequence. Never
    touches the real MazeEnv.

    Returns the single best ACTUALLY-SAMPLED sequence found across every
    iteration (tracked as best_actions/best_cost below) -- NOT the
    argmax of the final per-timestep marginal distribution. That
    marginal-decoding shortcut is fine for continuous CEM (ball_in_cup's
    planner uses it: nearby-the-mean is still a reasonable path in a
    smooth space), but it's wrong here: different elite sequences in a
    discrete maze can represent genuinely different, mutually
    incompatible paths through different junctions, so "the most popular
    single action at each independent timestep" can stitch together a
    path that was never coherent as a whole and was never actually
    scored by anything. First real run of this planner got 0/40 success
    on BOTH geometries simultaneously -- too uniform a failure to be
    "the search is just hard," and tracing it back, this was the bug."""
    torch.manual_seed(rng_seed)
    num_actions = len(ACTIONS)

    with torch.no_grad():
        start_idx = torch.tensor([cell_index(start, size)])
        target_idx = torch.tensor([cell_index(target, size)])
        start_raw = E(start_idx)                      # (1, dim)
        target_mapped = to_geometry(E(target_idx), geometry)  # (1, dim)

        probs = torch.full((horizon, num_actions), 1.0 / num_actions)
        best_cost = float("inf")
        best_actions = None

        for _ in range(N_ITERS):
            # sample N_SAMPLES candidate action sequences, one categorical
            # draw per timestep from that timestep's current distribution
            candidates = torch.stack([
                torch.multinomial(probs[t], N_SAMPLES, replacement=True)
                for t in range(horizon)
            ], dim=1)  # (N_SAMPLES, horizon)

            raw = start_raw.expand(N_SAMPLES, -1)
            for t in range(horizon):
                raw = P_step(raw, candidates[:, t])
            final_mapped = to_geometry(raw, geometry)

            costs = squared_dist(final_mapped, target_mapped.expand(N_SAMPLES, -1), geometry)
            n_elite = max(1, int(N_SAMPLES * ELITE_FRAC))
            elite_idx = torch.topk(costs, n_elite, largest=False).indices
            elites = candidates[elite_idx]  # (n_elite, horizon)

            top_cost, top_i = costs.min(dim=0)
            if top_cost.item() < best_cost:
                best_cost = top_cost.item()
                best_actions = candidates[top_i].tolist()

            counts = torch.zeros(horizon, num_actions)
            for t in range(horizon):
                counts[t] = torch.bincount(elites[:, t], minlength=num_actions).float()
            probs = counts / counts.sum(dim=1, keepdim=True)
            probs = probs.clamp_min(MIN_PROB)
            probs = probs / probs.sum(dim=1, keepdim=True)

        return [ACTIONS[a] for a in best_actions], best_cost


def beam_plan(E, P_step, geometry, start, target, size, horizon, rng_seed):
    """Deterministic beam search in latent space -- see the module
    docstring's "BEAM SEARCH" section for why, versus cem_plan. Returns
    (actions, best_cost), same shape as cem_plan, so it's a drop-in.
    rng_seed is accepted but unused (beam search here is deterministic)
    -- kept only so run_trials can call cem_plan and beam_plan
    interchangeably."""
    num_actions = len(ACTIONS)

    with torch.no_grad():
        start_idx = torch.tensor([cell_index(start, size)])
        target_idx = torch.tensor([cell_index(target, size)])
        start_raw = E(start_idx)                              # (1, dim)
        target_mapped = to_geometry(E(target_idx), geometry)  # (1, dim)

        beam_raw = start_raw          # (B, dim), B starts at 1
        beam_actions = [[]]           # length-B list of action-index lists

        # NOT squared_dist(start, target) as the initial "must beat this" threshold --
        # that was a real bug, only exposed once --replan started calling beam_plan
        # with short horizons: if literally no candidate ever beat the STARTING cost
        # within the whole horizon, best_actions stayed [] and callers crashed on
        # actions[0] (IndexError). float('inf')/None matches cem_plan's convention:
        # horizon >= 1 guarantees iteration 0 always sets best_actions, since
        # top_costs[0].item() < inf is unconditionally true there.
        best_cost = float("inf")
        best_actions = None

        for _ in range(horizon):
            b = beam_raw.shape[0]
            expanded_raw = beam_raw.repeat_interleave(num_actions, dim=0)  # (B*4, dim)
            action_col = torch.arange(num_actions).repeat(b)               # (B*4,)
            next_raw = P_step(expanded_raw, action_col)                    # (B*4, dim)

            next_mapped = to_geometry(next_raw, geometry)
            costs = squared_dist(next_mapped, target_mapped.expand(b * num_actions, -1), geometry)

            k = min(BEAM_WIDTH, b * num_actions)
            top_costs, top_idx = torch.topk(costs, k, largest=False)

            beam_raw = next_raw[top_idx]
            beam_actions = [
                beam_actions[i // num_actions] + [i % num_actions]
                for i in top_idx.tolist()
            ]

            if top_costs[0].item() < best_cost:
                best_cost = top_costs[0].item()
                best_actions = beam_actions[0]

        return [ACTIONS[a] for a in best_actions], best_cost


def oracle_step(adj, cell, action):
    """Same wall-bump-as-no-op semantics as MazeEnv.step, but operating
    directly on the adjacency graph -- no torch, no latent model."""
    dr, dc = ACTION_DELTA[action]
    target = (cell[0] + dr, cell[1] + dc)
    return target if target in adj[cell] else cell


def oracle_cem_plan(adj, start, target, horizon, rng_seed):
    """Same CEM scaffold as cem_plan (identical N_SAMPLES, ELITE_FRAC,
    N_ITERS, MIN_PROB), but every candidate is scored by simulating it
    for REAL on the maze graph and costing it by the TRUE BFS distance
    from the resulting cell to the target -- no torch, no E, no P_step.
    See the module docstring's "ORACLE ABLATION" section: this isolates
    whether discrete CEM search itself, given a perfect cost signal, can
    reliably find the exact path, separate from whether the latent
    model's cost signal is any good. Returns (actions, best_cost), same
    shape as cem_plan, so it's a drop-in for run_trials/summarize."""
    rng = random.Random(rng_seed)
    num_actions = len(ACTIONS)
    n_elite = max(1, int(N_SAMPLES * ELITE_FRAC))

    probs = [[1.0 / num_actions] * num_actions for _ in range(horizon)]
    best_cost = float("inf")
    best_actions = None

    for _ in range(N_ITERS):
        candidates = [
            [rng.choices(range(num_actions), weights=probs[t])[0] for t in range(horizon)]
            for _ in range(N_SAMPLES)
        ]

        costs = []
        for seq in candidates:
            cell = start
            for a in seq:
                cell = oracle_step(adj, cell, ACTIONS[a])
            costs.append(bfs_distance(adj, cell, target))

        ranked = sorted(range(N_SAMPLES), key=lambda i: costs[i])
        elite_idx = ranked[:n_elite]

        if costs[ranked[0]] < best_cost:
            best_cost = costs[ranked[0]]
            best_actions = candidates[ranked[0]]

        new_probs = []
        for t in range(horizon):
            counts = [0] * num_actions
            for i in elite_idx:
                counts[candidates[i][t]] += 1
            total = sum(counts)
            p = [max(c / total, MIN_PROB) for c in counts]
            s = sum(p)
            new_probs.append([x / s for x in p])
        probs = new_probs

    return [ACTIONS[a] for a in best_actions], best_cost


def oracle_beam_plan(adj, start, target, horizon, rng_seed):
    """Deterministic beam search on the real maze graph -- ground-truth
    counterpart to beam_plan, no torch/E/P_step involved. Additionally
    deduplicates children landing on the same real cell each step,
    keeping only the best-cost prefix per cell (valid here since this
    function is allowed to see real cells; beam_plan can't do this --
    see the module docstring). rng_seed is accepted but unused (kept
    for the same drop-in reason as beam_plan). Returns (actions,
    best_cost), same shape as oracle_cem_plan."""
    beam = [(start, [])]  # list of (cell, action_prefix)
    # Same fix as beam_plan's -- see its comment: float('inf')/[] via this same
    # pattern previously left best_actions == [] whenever nothing beat the
    # starting bfs_distance within the horizon, crashing --replan callers on
    # actions[0]. horizon >= 1 guarantees the loop's first iteration always
    # assigns best_actions.
    best_cost = float("inf")
    best_actions = []

    for _ in range(horizon):
        best_per_cell = {}
        for cell, prefix in beam:
            for a_idx, a in enumerate(ACTIONS):
                next_cell = oracle_step(adj, cell, a)
                cost = bfs_distance(adj, next_cell, target)
                next_prefix = prefix + [a_idx]
                if next_cell not in best_per_cell or cost < best_per_cell[next_cell][0]:
                    best_per_cell[next_cell] = (cost, next_prefix)

        ranked = sorted(best_per_cell.items(), key=lambda kv: kv[1][0])
        beam = [(cell, prefix) for cell, (cost, prefix) in ranked[:BEAM_WIDTH]]

        top_cell, (top_cost, top_prefix) = ranked[0]
        if top_cost < best_cost:
            best_cost = top_cost
            best_actions = top_prefix

    return [ACTIONS[a] for a in best_actions], best_cost


def execute_plan(env, start, target, actions):
    """Runs the plan for REAL on the maze (ground truth, deterministic).
    Returns (success, steps_to_reach, final_state) -- steps_to_reach is
    the first index at which the target was reached, or None if never
    reached within the plan; final_state is wherever the agent actually
    ended up (== target when success)."""
    env.reset(start=start)
    if env.state == target:
        return True, 0, env.state
    for i, a in enumerate(actions):
        env.step(a)
        if env.state == target:
            return True, i + 1, env.state
    return False, None, env.state


def closed_loop_max_steps(d0):
    """Generous per-trial safety cap for closed-loop replanning, scaled
    to the INITIAL bfs_distance rather than a flat constant -- same
    formula as oracle_receding_horizon_check.py's max_steps_for(), which
    validated this scale is generous enough not to bind in practice
    (mean actual/optimal steps ratio was 2.04 there, well under this cap
    of 3x+20)."""
    return d0 * 3 + 20


def run_closed_loop(E, P_step, geometry, plan_fn, adj, env, start, target, size, base_seed, d0,
                     force_horizon=None):
    """Receding-horizon (MPC-style) execution for the REAL geometry
    planner: replan (via plan_fn -- cem_plan or beam_plan) from the REAL
    cell at every step, executing only the first action of each plan for
    real, then replanning from wherever that actually landed. Motivated
    directly by oracle_receding_horizon_check.py, which found this alone
    takes oracle-CEM from 20% to 100% success on this maze -- see
    WORKPLAN.md's "DAgger rounds 6-8" and CEM-search-mechanism sections.
    Returns (success, steps, final_state, first_predicted_cost) --
    first_predicted_cost is the very first step's predicted cost, kept
    as the single representative per-trial scalar summarize_cost_
    correlation expects (later steps' costs aren't retained).

    force_horizon: if set, use this FIXED horizon at every replanning step
    instead of bfs_distance(cell, target) + HORIZON_MARGIN. Lets CEM's
    candidate-sequence scoring rely on much shorter (e.g. 1-step)
    P_step rollouts, avoiding the long-horizon autoregressive compounding
    error this project already found (pstep_manifold_drift_check.py,
    step_drift_diagnostic.py) -- testing whether that compounding, not
    one-step accuracy (recently much improved, see WORKPLAN.md's
    "residual parameterization" section), is what's actually capping
    real closed-loop success now."""
    env.reset(start=start)
    cell = env.state
    cap = closed_loop_max_steps(d0)
    first_cost = None
    for step in range(cap):
        if cell == target:
            return True, step, cell, first_cost if first_cost is not None else 0.0
        d = bfs_distance(adj, cell, target)
        horizon = force_horizon if force_horizon is not None else d + HORIZON_MARGIN
        actions, cost = plan_fn(E, P_step, geometry, cell, target, size, horizon, base_seed + step)
        if first_cost is None:
            first_cost = cost
        env.step(actions[0])
        cell = env.state
    return cell == target, cap, cell, first_cost if first_cost is not None else 0.0


def run_closed_loop_oracle(plan_fn, adj, env, start, target, base_seed, d0):
    """Same as run_closed_loop but for the oracle planner (oracle_cem_plan
    / oracle_beam_plan) -- no torch/E/P_step, ground-truth cost signal.
    Kept separate from run_closed_loop rather than unified, matching this
    file's existing convention of parallel real/oracle function pairs
    (cem_plan/oracle_cem_plan, beam_plan/oracle_beam_plan)."""
    env.reset(start=start)
    cell = env.state
    cap = closed_loop_max_steps(d0)
    first_cost = None
    for step in range(cap):
        if cell == target:
            return True, step, cell, first_cost if first_cost is not None else 0.0
        d = bfs_distance(adj, cell, target)
        horizon = d + HORIZON_MARGIN
        actions, cost = plan_fn(adj, cell, target, horizon, base_seed + step)
        if first_cost is None:
            first_cost = cost
        env.step(actions[0])
        cell = env.state
    return cell == target, cap, cell, first_cost if first_cost is not None else 0.0


def run_trials(geometry, adj, size, rng, search="cem", replan=False,
               encoder_checkpoint=None, step_checkpoint=None, force_horizon=None):
    E, P_step, ckpt_size = load_models(geometry, encoder_checkpoint, step_checkpoint)
    assert ckpt_size == size
    env = MazeEnv(size=size, algo=ALGO, seed=MAZE_SEED)  # same fixed maze
    cells = list(adj.keys())
    plan_fn = cem_plan if search == "cem" else beam_plan

    results = []
    trial_seed = 0
    while len(results) < N_TRIALS:
        start, target = rng.sample(cells, 2)
        d = bfs_distance(adj, start, target)
        if d < MIN_TRIAL_DISTANCE:
            continue
        horizon = force_horizon if force_horizon is not None else d + HORIZON_MARGIN
        if replan:
            success, steps, final_state, predicted_cost = run_closed_loop(
                E, P_step, geometry, plan_fn, adj, env, start, target, size, trial_seed, d,
                force_horizon=force_horizon)
        else:
            actions, predicted_cost = plan_fn(E, P_step, geometry, start, target, size, horizon, trial_seed)
            success, steps, final_state = execute_plan(env, start, target, actions)
        real_leftover_distance = 0 if success else bfs_distance(adj, final_state, target)
        results.append({
            "start": start, "target": target, "bfs_distance": d,
            "horizon": horizon, "success": success, "steps": steps,
            "predicted_cost": predicted_cost,
            "real_leftover_distance": real_leftover_distance,
        })
        trial_seed += 1
    return results


def run_oracle_trials(adj, size, rng, search="cem", replan=False):
    env = MazeEnv(size=size, algo=ALGO, seed=MAZE_SEED)  # same fixed maze
    cells = list(adj.keys())
    plan_fn = oracle_cem_plan if search == "cem" else oracle_beam_plan

    results = []
    trial_seed = 0
    while len(results) < N_TRIALS:
        start, target = rng.sample(cells, 2)
        d = bfs_distance(adj, start, target)
        if d < MIN_TRIAL_DISTANCE:
            continue
        horizon = d + HORIZON_MARGIN
        if replan:
            success, steps, final_state, predicted_cost = run_closed_loop_oracle(
                plan_fn, adj, env, start, target, trial_seed, d)
        else:
            actions, predicted_cost = plan_fn(adj, start, target, horizon, trial_seed)
            success, steps, final_state = execute_plan(env, start, target, actions)
        real_leftover_distance = 0 if success else bfs_distance(adj, final_state, target)
        results.append({
            "start": start, "target": target, "bfs_distance": d,
            "horizon": horizon, "success": success, "steps": steps,
            "predicted_cost": predicted_cost,
            "real_leftover_distance": real_leftover_distance,
        })
        trial_seed += 1
    return results


def summarize(geometry, results):
    n = len(results)
    successes = [r for r in results if r["success"]]
    success_rate = len(successes) / n
    if successes:
        ratios = [r["steps"] / r["bfs_distance"] for r in successes]
        mean_ratio = sum(ratios) / len(ratios)
    else:
        mean_ratio = float("nan")
    print(f"{geometry}: success_rate={success_rate:.2f} ({len(successes)}/{n})  "
          f"mean actual/bfs-optimal steps among successes={mean_ratio:.2f}")
    return success_rate, mean_ratio


def _ranks(xs):
    """Plain (no tie-averaging needed in practice -- these are floats)
    ascending ranks, 0-indexed."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0] * len(xs)
    for rank, i in enumerate(order):
        ranks[i] = rank
    return ranks


def spearman_correlation(xs, ys):
    """Rank correlation, not Pearson: predicted cost and real leftover
    distance aren't expected to be linearly related (squared latent
    distance can have a very different scale/curvature than a BFS step
    count, especially in hyperbolic space), but if the embedding's local
    geometry is sound under drift, low predicted cost should still tend
    to mean low real leftover distance -- a MONOTONIC relationship,
    which rank correlation tests directly. Plain, dependency-free
    implementation: Pearson correlation of the ranks."""
    n = len(xs)
    rx, ry = _ranks(xs), _ranks(ys)
    mean_rx, mean_ry = (n - 1) / 2, (n - 1) / 2
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / (var_x * var_y) ** 0.5


def summarize_cost_correlation(geometry, results):
    costs = [r["predicted_cost"] for r in results]
    leftovers = [r["real_leftover_distance"] for r in results]
    rho = spearman_correlation(costs, leftovers)
    print(f"{geometry}: predicted-cost vs real-leftover-distance, Spearman rho={rho:.3f} "
          f"(n={len(results)}; +1 = predicted cost perfectly tracks real drift, "
          f"0 = predicted cost is uninformative once drift sets in)")
    return rho


def summarize_leftover(geometry, results):
    """How close the CHOSEN plan actually lands, real leftover BFS
    distance to target (0 = exact success). Distinguishes two different
    failure shapes that both show up as success_rate=0.00: consistently
    landing a couple of cells short (a search-precision problem -- CEM
    needs more samples/iterations, or the horizon needs more slack) vs.
    landing nowhere near the target on the harder trials (still a
    representation-limited problem, only partly masked by the easier
    trials in the rho above)."""
    leftovers = [r["real_leftover_distance"] for r in results]
    n = len(leftovers)
    mean_l = sum(leftovers) / n
    within = {k: sum(1 for l in leftovers if l <= k) for k in (0, 1, 2, 3, 5)}
    print(f"{geometry}: real leftover distance -- min={min(leftovers)} "
          f"mean={mean_l:.1f} max={max(leftovers)}  "
          f"within<=0:{within[0]}/{n} <=1:{within[1]}/{n} <=2:{within[2]}/{n} "
          f"<=3:{within[3]}/{n} <=5:{within[5]}/{n}")
    return leftovers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic", "both", "oracle"], default="both")
    parser.add_argument("--search", choices=["cem", "beam"], default="cem")
    parser.add_argument("--replan", action="store_true",
                         help="Receding-horizon (MPC-style) execution: replan from the "
                              "REAL cell at every step instead of committing to one open-loop "
                              "plan for the whole horizon. See run_closed_loop()/"
                              "run_closed_loop_oracle() and oracle_receding_horizon_check.py "
                              "(took oracle-CEM 20%% -> 100%% on this maze).")
    parser.add_argument("--encoder-checkpoint", default=None,
                         help="Override E's checkpoint path template (containing {geometry}), "
                              "e.g. jepa_checkpoint_treeloss_{geometry}.pt, to evaluate an "
                              "alternative E. Use with --geometry hyperbolic (not both/oracle).")
    parser.add_argument("--step-checkpoint", default=None,
                         help="Override P_step's checkpoint path template, matching whichever "
                              "E it was trained against via --encoder-checkpoint.")
    parser.add_argument("--force-horizon", type=int, default=None,
                         help="Use this FIXED horizon at every (re)planning step instead of "
                              "bfs_distance + HORIZON_MARGIN. Use 1 to test the 'degenerate "
                              "CEM' idea -- with --replan, this makes CEM's cost rely only on "
                              "P_step's ONE-step prediction (recently much improved) instead "
                              "of long autoregressive rollouts (a known source of compounding "
                              "error -- see pstep_manifold_drift_check.py).")
    args = parser.parse_args()

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))

    if args.geometry == "both":
        geometries = ["euclidean", "hyperbolic"]
    else:
        geometries = [args.geometry]

    label = lambda g: f"{g}+{args.search}" + ("+replan" if args.replan else "")
    # tags output with the search method (and replan mode) too, so
    # beam vs cem and open-loop vs closed-loop runs are distinguishable

    all_results = {}
    for geometry in geometries:
        # same trial (start, target) pairs across geometries (and oracle),
        # for a fair comparison -- fresh rng from PLANNER_SEED each time
        rng = random.Random(PLANNER_SEED)
        if geometry == "oracle":
            results = run_oracle_trials(adj, GRID_SIZE, rng, search=args.search, replan=args.replan)
        else:
            results = run_trials(geometry, adj, GRID_SIZE, rng, search=args.search, replan=args.replan,
                                  encoder_checkpoint=args.encoder_checkpoint,
                                  step_checkpoint=args.step_checkpoint,
                                  force_horizon=args.force_horizon)
        summarize(label(geometry), results)
        all_results[geometry] = results

    for geometry in geometries:
        summarize_cost_correlation(label(geometry), all_results[geometry])
        summarize_leftover(label(geometry), all_results[geometry])


if __name__ == "__main__":
    main()
