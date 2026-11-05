"""
env.py

Maze JEPA -- the environment. Perfect maze (spanning tree, no loops),
generated with maze_gen.py. Default generator is Prim's (domain_fit_check.py,
in the original maze/ project, confirmed Prim's produces roughly 2-3x more
junctions per maze than DFS at every size tested).

GRID_SIZE=10: the size directly vetted by domain_fit_check.py's branching-
factor sweep (24 junctions, k>=8 reached by H=5).

ACTION SPACE: fixed at 4 discrete actions, always, regardless of position --
Action, an enum whose value is each action's (row, col) delta. Stepping
into a wall is a no-op: the agent stays in place.

DATA COLLECTION POLICY: random_rollout() samples uniformly among *valid*
moves only (not uniformly among all 4 raw actions with wall-bumps
included) -- matches domain_fit_check.py's k measurements.

Run (from inside maze_jepa/):
    uv run python env.py
"""
import random
from enum import Enum

from maze_gen import count_junctions, generate_maze

GRID_SIZE = 10
ALGO = "prim"


class Action(Enum):
    """Each member's value IS its (row, col) delta -- no separate lookup
    table needed. list(Action) iterates in definition order (UP, DOWN,
    LEFT, RIGHT); action.value gives the delta; action.name gives the
    string ("UP", ...) for anything that needs a plain/serializable form
    (e.g. writing actions out to JSON later)."""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


ACTIONS = list(Action)


class MazeEnv:
    def __init__(self, size=GRID_SIZE, algo=ALGO, seed=None):
        self.size = size
        self.rng = random.Random(seed)
        self.adj = generate_maze(algo, size, self.rng)
        self.state = None

    def reset(self, start=None):
        cells = list(self.adj.keys())
        self.state = start if start is not None else self.rng.choice(cells)
        return self.state

    def valid_actions(self, state=None):
        state = state if state is not None else self.state
        r, c = state
        valid = []
        for a in Action:
            dr, dc = a.value
            if (r + dr, c + dc) in self.adj[state]:
                valid.append(a)
        return valid

    def step(self, action):
        """Fixed 4-action space; stepping into a wall is a no-op (stay in
        place), matching the module docstring's ACTION SPACE note."""
        r, c = self.state
        dr, dc = action.value
        target = (r + dr, c + dc)
        if target in self.adj[self.state]:
            self.state = target
        return self.state

    def random_rollout(self, horizon, start=None, valid_only=True):
        """Returns (states, actions) for one rollout of length `horizon`
        -- actions is a list of Action members. valid_only=True (the
        default) samples uniformly among valid moves only -- see the
        module docstring's DATA COLLECTION POLICY note. valid_only=False
        samples uniformly among all 4 raw actions instead (wall-bumps
        included)."""
        self.reset(start)
        states = [self.state]
        actions = []
        for _ in range(horizon):
            if valid_only:
                options = self.valid_actions()
                a = self.rng.choice(options) if options else self.rng.choice(ACTIONS)
            else:
                a = self.rng.choice(ACTIONS)
            actions.append(a)
            states.append(self.step(a))
        return states, actions

    def render(self):
        """ASCII rendering of the maze (spanning-tree edges), current
        state marked with @."""
        n = self.size
        lines = []
        for r in range(n):
            top = ""
            mid = ""
            for c in range(n):
                mid += "@" if self.state == (r, c) else "."
                mid += " " if (r, c + 1) in self.adj.get((r, c), ()) else "|"
                top += "  " if (r - 1, c) in self.adj.get((r, c), ()) else "--"
            lines.append(top)
            lines.append(mid)
        lines.append("--" * n)
        return "\n".join(lines)


def _self_test():
    env = MazeEnv(size=GRID_SIZE, algo=ALGO, seed=0)
    env.reset()
    print(f"MazeEnv(size={GRID_SIZE}, algo={ALGO}), {len(env.adj)} cells, "
          f"{count_junctions(env.adj)} junctions")
    print(env.render())
    print()
    for horizon in [3, 8, 12]:
        states, actions = env.random_rollout(horizon, start=(0, 0))
        action_names = [a.name for a in actions]
        print(f"horizon={horizon}: start=(0,0) actions={action_names} -> end={states[-1]}")


if __name__ == "__main__":
    _self_test()
