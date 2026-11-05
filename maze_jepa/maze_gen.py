"""
maze_gen.py

Perfect-maze (spanning tree, no loops) generation, shared by
domain_fit_check.py and env.py -- see WORKPLAN.md for why a perfect maze
specifically (every junction a permanent, non-reconverging fork).

Two generators, since they differ meaningfully in junction density
(confirmed empirically in domain_fit_check.py, not just folklore):
randomized Prim's (bushier, more short branches -- the default, since it
reaches the branching-factor regime Phase 0 found real effect in, faster
and with more margin) and randomized DFS / recursive backtracker (longer
corridors, fewer junctions, kept for comparison).

No dependencies -- pure stdlib.
"""
import random
from collections import defaultdict, deque


def neighbors(r, c, n):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            yield nr, nc


def generate_maze_prim(n, rng):
    """Randomized Prim's algorithm. Returns adjacency dict: cell -> set of
    connected neighbor cells (the spanning-tree edges -- i.e. the valid
    moves)."""
    adj = defaultdict(set)
    in_maze = set()
    start = (rng.randrange(n), rng.randrange(n))
    in_maze.add(start)
    frontier = [(nb, start) for nb in neighbors(*start, n)]
    while frontier:
        idx = rng.randrange(len(frontier))
        cell, came_from = frontier.pop(idx)
        if cell in in_maze:
            continue
        adj[cell].add(came_from)
        adj[came_from].add(cell)
        in_maze.add(cell)
        for nb in neighbors(*cell, n):
            if nb not in in_maze:
                frontier.append((nb, cell))
    return adj


def generate_maze_dfs(n, rng):
    """Randomized recursive backtracker. Returns adjacency dict."""
    adj = defaultdict(set)
    start = (rng.randrange(n), rng.randrange(n))
    visited = {start}
    stack = [start]
    while stack:
        cur = stack[-1]
        unvisited_nb = [nb for nb in neighbors(*cur, n) if nb not in visited]
        if not unvisited_nb:
            stack.pop()
            continue
        nxt = rng.choice(unvisited_nb)
        adj[cur].add(nxt)
        adj[nxt].add(cur)
        visited.add(nxt)
        stack.append(nxt)
    return adj


def generate_maze(algo, n, rng):
    """Dispatches to the requested generator by name."""
    match algo:
        case "prim":
            return generate_maze_prim(n, rng)
        case "dfs":
            return generate_maze_dfs(n, rng)
        case _:
            raise ValueError(f"unknown maze algorithm: {algo!r}")


def count_junctions(adj):
    """Cells with >= 3 spanning-tree connections (a real fork: at least 2
    ways onward plus the way you came from)."""
    return sum(1 for nbs in adj.values() if len(nbs) >= 3)


def bfs_distance(adj, start, target):
    """Exact shortest-path distance (in steps) from start to target along
    the maze's spanning tree -- since a perfect maze has exactly one path
    between any two cells, this IS the path, not just a lower bound.
    Used by maze_planner.py to size a fair planning horizon per (start,
    target) pair, and as ground truth for judging how efficient a
    planned path actually is."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == target:
            return dist[cur]
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    raise ValueError(f"{target} unreachable from {start} -- should never happen in a connected maze")


def bfs_distances_from(adj, start):
    """Exact shortest-path distance from start to EVERY other reachable
    cell, in one BFS pass -- more efficient than calling bfs_distance
    once per target when ALL-PAIRS distances are needed (e.g.
    embedding_metric_check.py's local-geometry check). Since a perfect
    maze is fully connected, every cell in adj ends up in the result."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


def bfs_path_cells(adj, start, target):
    """Exact shortest-path cell sequence from start to target (inclusive
    of both ends) -- the unique tree path, same guarantee as
    bfs_distance. Used by step_drift_diagnostic.py to get a real,
    correct action sequence to test one-step-model drift against,
    independent of anything the planner itself does."""
    parent = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == target:
            break
        for nb in adj[cur]:
            if nb not in parent:
                parent[nb] = cur
                queue.append(nb)
    if target not in parent:
        raise ValueError(f"{target} unreachable from {start} -- should never happen in a connected maze")
    path = [target]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path
