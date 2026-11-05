"""
cem_hparam_search.py

Maze JEPA -- Phase 3, automated hyperparameter search over CEM's own
search parameters (N_SAMPLES, N_ITERS, ELITE_FRAC, MIN_PROB), evaluated
under closed-loop replanning (--replan). Motivated by: these four
constants were chosen once, early on, for the ORIGINAL open-loop
planning regime (commit to one long plan, execute blind) and have never
been retuned since closed-loop replanning was added -- see WORKPLAN.md's
"Closed-loop replanning" section. Closed-loop changes the math a lot:
many more total plan_fn calls (one per REAL step instead of one per
trial), each with a much shorter horizon near the end of a trip, and
each individual call's mistake matters less (the next replan can correct
it) -- so the old choices, tuned for a regime where ONE call had to get
the WHOLE plan right, may no longer be anywhere near optimal.

Reuses maze_planner.py's run_trials(..., replan=True) UNCHANGED --
cem_plan reads N_SAMPLES/ELITE_FRAC/N_ITERS/MIN_PROB as plain module-
level globals (not function parameters), so this script monkey-patches
those attributes on the maze_planner module directly before each call.
This works because Python functions resolve global names at CALL time,
not at def time, so cem_plan (and run_closed_loop, which calls it) sees
whatever value is currently set on the module -- no need to touch or
duplicate cem_plan itself.

Time-budgeted, not a fixed grid: combos are visited cheapest-first (by
an estimated cost = N_SAMPLES * N_ITERS, the dominant cost driver, since
that's the total number of P_step forward-passes per plan_fn call) so a
long run always finishes many combos early rather than risking getting
stuck on one huge slow combo near the end. Results are written
incrementally to cem_hparam_search_results_<geometry>.json after EVERY
combo (not just at the end), so the run can be interrupted (Ctrl-C, or
hitting the time budget) at any point without losing anything already
computed.

Grid: N_SAMPLES x N_ITERS x ELITE_FRAC x MIN_PROB = 4x4x4x3 = 192 combos.
HORIZON_MARGIN is left fixed -- it's a maze-physics/horizon-slack choice,
not a search-quality knob. Rough cost estimate put the full grid at
40-90+ minutes depending on this machine's actual per-call speed.

--budget-seconds is a SAFETY NET, not the real stopping condition -- the
goal is to complete the full 192-combo grid, however long that actually
takes; the default (6 hours) just guards against a genuinely pathological
runaway (e.g. a bug making one combo hang), not against a legitimately
long but finite search. Results are written incrementally either way, so
there's no cost to letting it run long except wall-clock time.

Run (defaults to a 6-hour safety-net budget -- just let it run):
    uv run python cem_hparam_search.py
    uv run python cem_hparam_search.py --geometry euclidean
"""
import argparse
import itertools
import json
import random
import time

from env import ALGO, GRID_SIZE
from maze_gen import GENERATORS
import maze_planner
from maze_planner import MAZE_SEED, PLANNER_SEED, run_trials, spearman_correlation

RESULTS_PATH = "cem_hparam_search_results_{geometry}.json"

N_SAMPLES_GRID = [50, 100, 200, 400]
N_ITERS_GRID = [2, 4, 6, 10]
ELITE_FRAC_GRID = [0.05, 0.1, 0.2, 0.3]
MIN_PROB_GRID = [0.01, 0.02, 0.05]


def estimated_cost(n_samples, n_iters, elite_frac, min_prob):
    """Rough relative cost, used only to order combos cheapest-first --
    dominated by total P_step forward passes per plan_fn call
    (N_SAMPLES * N_ITERS); elite_frac/min_prob don't change per-call
    tensor sizes."""
    return n_samples * n_iters


def evaluate_combo(geometry, adj, n_samples, n_iters, elite_frac, min_prob,
                   encoder_checkpoint=None, step_checkpoint=None):
    maze_planner.N_SAMPLES = n_samples
    maze_planner.N_ITERS = n_iters
    maze_planner.ELITE_FRAC = elite_frac
    maze_planner.MIN_PROB = min_prob

    rng = random.Random(PLANNER_SEED)
    results = run_trials(geometry, adj, GRID_SIZE, rng, search="cem", replan=True,
                          encoder_checkpoint=encoder_checkpoint, step_checkpoint=step_checkpoint)

    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    leftovers = [r["real_leftover_distance"] for r in results]
    mean_leftover = sum(leftovers) / n
    costs = [r["predicted_cost"] for r in results]
    rho = spearman_correlation(costs, leftovers)
    within5 = sum(1 for l in leftovers if l <= 5)

    return {
        "success_rate": n_success / n,
        "n_success": n_success,
        "rho": rho,
        "mean_leftover": mean_leftover,
        "within5": within5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic"], default="hyperbolic")
    parser.add_argument("--budget-seconds", type=float, default=21600.0)  # 6h safety net, not a real target -- see module docstring
    parser.add_argument("--encoder-checkpoint", default=None,
                         help="override CHECKPOINT_PATH template -- use this to sweep CEM "
                              "hyperparameters against a non-standard E (e.g. a different "
                              "GRID_SIZE or the treeloss checkpoints), instead of the default.")
    parser.add_argument("--step-checkpoint", default=None,
                         help="override STEP_CHECKPOINT_PATH template -- same reasoning as "
                              "--encoder-checkpoint.")
    parser.add_argument("--output-path", default=None,
                         help="override RESULTS_PATH -- use a distinct name when sweeping "
                              "against a non-standard checkpoint/GRID_SIZE, so this doesn't "
                              "overwrite results from a previous sweep.")
    args = parser.parse_args()
    geometry = args.geometry

    adj = GENERATORS[ALGO](GRID_SIZE, random.Random(MAZE_SEED))

    combos = list(itertools.product(N_SAMPLES_GRID, N_ITERS_GRID, ELITE_FRAC_GRID, MIN_PROB_GRID))
    combos.sort(key=lambda c: estimated_cost(*c))
    print(f"{len(combos)} combos queued, cheapest-first, budget={args.budget_seconds:.0f}s")

    results_path = (args.output_path or RESULTS_PATH).format(geometry=geometry)
    all_results = []
    t0 = time.time()

    for i, (n_samples, n_iters, elite_frac, min_prob) in enumerate(combos):
        elapsed = time.time() - t0
        if elapsed >= args.budget_seconds:
            print(f"time budget reached ({elapsed:.0f}s) -- stopping after {i}/{len(combos)} combos")
            break

        t_combo = time.time()
        metrics = evaluate_combo(geometry, adj, n_samples, n_iters, elite_frac, min_prob,
                                  encoder_checkpoint=args.encoder_checkpoint,
                                  step_checkpoint=args.step_checkpoint)
        combo_time = time.time() - t_combo

        row = {
            "n_samples": n_samples, "n_iters": n_iters,
            "elite_frac": elite_frac, "min_prob": min_prob,
            **metrics, "combo_seconds": combo_time,
        }
        all_results.append(row)

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"  [{i+1}/{len(combos)}] N={n_samples} iters={n_iters} elite={elite_frac} "
              f"min_prob={min_prob}: success={metrics['n_success']}/40 "
              f"rho={metrics['rho']:.3f} mean_leftover={metrics['mean_leftover']:.1f} "
              f"within5={metrics['within5']}/40  [{combo_time:.1f}s, {time.time()-t0:.0f}s elapsed]")

    all_results.sort(key=lambda r: (-r["n_success"], -r["within5"], r["mean_leftover"]))
    print("\ntop 10 combos (by success, then within<=5, then mean leftover):")
    for r in all_results[:10]:
        print(f"  N={r['n_samples']} iters={r['n_iters']} elite={r['elite_frac']} "
              f"min_prob={r['min_prob']}: success={r['n_success']}/40 rho={r['rho']:.3f} "
              f"mean_leftover={r['mean_leftover']:.1f} within5={r['within5']}/40")
    print(f"\nfull results written to {results_path}")


if __name__ == "__main__":
    main()
