"""
domain_fit_check.py

Maze JEPA -- domain-fit check, run before building env.py.

Question: does a maze of a given (grid size, generation algorithm), over a
given horizon H (number of random-policy steps), actually reach the
branching-factor regime Phase 0 found a real effect in (k roughly >= 8,
see WORKPLAN.md "Phase 0 result")? Answered by direct simulation, not
estimated from theory or maze-generation folklore.

k here is defined operationally the same way Phase 0 defined it, and the
same way Phase 1's data collection is planned to work (WORKPLAN.md, "How
multimodality gets constructed for training" -- rollouts under a
stochastic/random policy): from a fixed start cell, run many independent
H-step rollouts under a uniform-random policy (pick uniformly among the
moves the maze's walls currently allow, which includes backtracking --
there's no explicit no-backtrack rule), and count the number of DISTINCT
final cells reached. That count is the empirical k for that
(maze, start cell, H).

Perfect mazes only (spanning tree, no loops) -- see WORKPLAN.md for why:
every junction is then a genuine, permanent fork, not one that can
reconverge with another branch. Two generators compared since they're
known to differ in junction density: randomized Prim's (expected bushier,
more short branches) vs. randomized DFS / recursive backtracker (expected
longer corridors, fewer junctions) -- compared empirically here rather
than trusted from folklore.

No torch, no numpy -- pure stdlib, runs anywhere, no project dependencies.
Maze generation lives in maze_gen.py, shared with env.py.

Run (from inside maze/):
    uv run python domain_fit_check.py
"""
import random
import statistics

from maze_gen import GENERATORS, count_junctions

GRID_SIZES = [6, 10, 16]
HORIZONS = [3, 5, 8, 12, 20]
ALGORITHMS = ["prim", "dfs"]
N_MAZES = 5        # maze instances generated per (algorithm, size)
N_STARTS = 4        # random start cells per maze
N_ROLLOUTS = 300     # random rollouts per (maze, start, horizon)
SEED = 0


def random_rollout_endpoint(adj, start, horizon, rng):
    cell = start
    for _ in range(horizon):
        options = list(adj[cell])
        if not options:
            break
        cell = rng.choice(options)
    return cell


def empirical_k(adj, start, horizon, rng, n_rollouts=N_ROLLOUTS):
    endpoints = set()
    for _ in range(n_rollouts):
        endpoints.add(random_rollout_endpoint(adj, start, horizon, rng))
    return len(endpoints)


def main():
    rng = random.Random(SEED)
    print(f"{'algo':>5} {'size':>5} {'H':>3}  {'mean k':>8} {'std k':>7}  {'mean junctions/maze':>20}")
    for algo in ALGORITHMS:
        gen = GENERATORS[algo]
        for size in GRID_SIZES:
            mazes = [gen(size, rng) for _ in range(N_MAZES)]
            junction_counts = [count_junctions(adj) for adj in mazes]
            for H in HORIZONS:
                ks = []
                for adj in mazes:
                    cells = list(adj.keys())
                    starts = rng.sample(cells, min(N_STARTS, len(cells)))
                    for s in starts:
                        ks.append(empirical_k(adj, s, H, rng))
                mean_k = statistics.mean(ks)
                std_k = statistics.pstdev(ks)
                print(f"{algo:>5} {size:>5} {H:>3}  {mean_k:>8.2f} {std_k:>7.2f}  {statistics.mean(junction_counts):>20.1f}")


if __name__ == "__main__":
    main()
