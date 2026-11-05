"""
env.py

Maze JEPA -- the environment itself (Phase 1). See WORKPLAN.md for the
full design reasoning and domain_fit_check.py for how GRID_SIZE and the
generator were chosen.

Perfect maze (spanning tree, no loops), generated with maze_gen.py.
Default generator is Prim's, not DFS: domain_fit_check.py confirmed Prim's
produces roughly 2-3x more junctions per maze at every size tested, and
reaches the k>=8 branching-factor regime Phase 0 found a real effect in
faster and with more margin (mean k crosses 8 by H=5 with Prim's, vs H=8
for DFS).

ACTION SPACE: fixed at 4 discrete actions (UP/DOWN/LEFT/RIGHT), always,
regardless of position -- not a variable/masked space that changes shape
depending on which moves are valid from the current cell. This matches how
the rest of this project's predictors expect a fixed-size action encoding
every step (see jepa_train.py's ACTION_DIM), and keeps the maze's action
space directly analogous to ball-in-cup's. Stepping into a wall is a
no-op: the agent stays in place, exactly like bumping into a boundary.

DATA COLLECTION POLICY, DELIBERATELY DIFFERENT FROM THE RAW ACTION SPACE:
domain_fit_check.py's k measurements assumed a policy that samples
uniformly among VALID moves only (see its docstring) -- not uniformly
among all 4 raw actions with wall-bumps included. Sampling among all 4
raw actions would waste a fraction of every rollout on no-ops at
low-degree cells (dead ends most of all), silently lowering the effective
k reached in H steps relative to what was actually measured and tuned
against. So random_rollout() below defaults to sampling uniformly among
*valid* moves specifically, to stay faithful to the domain-fit numbers --
this is a policy choice for data collection, not a constraint on the
action space itself, which stays the full fixed 4 either way.

Run (from inside maze/):
    uv run python env.py
    (generates a maze with the settings below, prints it, and runs a few
    sample rollouts -- use this to eyeball different GRID_SIZE / ALGO /
    horizon choices before committing to a real data-collection run)
"""
import random

from maze_gen import GENERATORS, count_junctions

GRID_SIZE = 10  # dropped from 16 to isolate whether P_step's autoregressive
                # drift is a horizon-length effect -- size 10 was the
                # project's original maze size (24 junctions) and was
                # directly vetted by domain_fit_check.py (k>=8 reached by
                # H=5), unlike the untested size 8. Bumped to 16 earlier
                # for statistical power in Phase 2's paired test; that
                # reasoning doesn't apply here, so reverting is safe.
ALGO = "prim"

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTA = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
REVERSE_ACTION = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


class MazeEnv:
    def __init__(self, size=GRID_SIZE, algo=ALGO, seed=None):
        self.size = size
        self.rng = random.Random(seed)
        self.adj = GENERATORS[algo](size, self.rng)
        self.state = None

    def reset(self, start=None):
        cells = list(self.adj.keys())
        self.state = start if start is not None else self.rng.choice(cells)
        return self.state

    def valid_actions(self, state=None):
        state = state if state is not None else self.state
        r, c = state
        valid = []
        for a in ACTIONS:
            dr, dc = ACTION_DELTA[a]
            if (r + dr, c + dc) in self.adj[state]:
                valid.append(a)
        return valid

    def step(self, action):
        """Fixed 4-action space; stepping into a wall is a no-op (stay in
        place), matching the module docstring's ACTION SPACE note."""
        r, c = self.state
        dr, dc = ACTION_DELTA[action]
        target = (r + dr, c + dc)
        if target in self.adj[self.state]:
            self.state = target
        return self.state

    def random_rollout(self, horizon, start=None, valid_only=True, backtrack_weight=None,
                        deadend_weight=None):
        """Returns (states, actions) for one rollout of length `horizon`.
        valid_only=True (the default) samples uniformly among valid moves
        only, matching domain_fit_check.py's k measurements -- see the
        module docstring's DATA COLLECTION POLICY note. valid_only=False
        samples uniformly among all 4 raw actions instead (wall-bumps
        included), which reaches a lower effective k for the same
        horizon.

        backtrack_weight=None (default) preserves the original uniform-
        among-valid-moves policy, unchanged. When set to a float in [0,1),
        the action that exactly reverses the PREVIOUS step gets that
        relative weight instead of 1 (still chosen when it's the only
        valid option, e.g. at a forced dead-end reversal -- this only
        discourages a pointless immediate backtrack when another real
        option exists, never forbids it). Motivated by
        branch_split_imbalance_check.py / branch_cell_inspect.py finding
        that P_step's decision is specifically broken at branch cells
        with a tiny dead-end side, and that the random-walk data for the
        dead-end-entering action is 92-97% immediately reversed the very
        next step -- see WORKPLAN.md's "trajectory-level cycles" section.

        Confirmed NOT to help on its own (see WORKPLAN.md): the two known-
        broken cells' dead ends are literal degree-1 leaves, so the bounce
        back out is the ONLY valid move once inside -- backtrack_weight
        only reweights a choice among alternatives, and there is no
        alternative at that point, so it's a structural no-op exactly
        where it mattered.

        deadend_weight=None (default) preserves the original policy,
        unchanged. When set to a float in [0,1], any candidate action
        whose DESTINATION is a literal dead end (degree 1 -- an answer to
        Andreas's "can we disallow bouncing into walls") gets that
        relative weight instead of 1, still chosen when it's the only
        valid option. 0.0 fully excludes entering a dead end whenever a
        real alternative exists (every branch cell tested so far has
        degree >= 2, so an alternative always exists there). Unlike
        backtrack_weight, this discourages entering a dead end in the
        first place -- a real choice among alternatives -- rather than
        trying to discourage the forced move back out, which can't be
        discouraged at all. Directed data (which visits dead ends far
        less destructively -- 31-36% immediate-reversal rate vs 92-97% in
        random-walk, see WORKPLAN.md) still teaches P_step that edge, so
        setting this to 0.0 does not leave the edge with zero training
        signal. Composes multiplicatively with backtrack_weight when both
        are set. Not yet proven to help; test via ablation before
        adopting as the new default, same as every other flag in this
        project."""
        self.reset(start)
        states = [self.state]
        actions = []
        prev_action = None
        for _ in range(horizon):
            if valid_only:
                options = self.valid_actions()
                if not options:
                    a = self.rng.choice(ACTIONS)
                elif (backtrack_weight is not None or deadend_weight is not None) and len(options) > 1:
                    r, c = self.state
                    weights = []
                    for o in options:
                        w = 1.0
                        if backtrack_weight is not None and prev_action is not None \
                                and o == REVERSE_ACTION[prev_action]:
                            w *= backtrack_weight
                        if deadend_weight is not None:
                            dr, dc = ACTION_DELTA[o]
                            neighbor = (r + dr, c + dc)
                            if len(self.adj[neighbor]) == 1:
                                w *= deadend_weight
                        weights.append(w)
                    if sum(weights) <= 0:
                        # degenerate case (e.g. every option got weighted to 0) --
                        # fall back to uniform rather than crashing random.choices
                        a = self.rng.choice(options)
                    else:
                        a = self.rng.choices(options, weights=weights, k=1)[0]
                else:
                    a = self.rng.choice(options)
            else:
                a = self.rng.choice(ACTIONS)
            actions.append(a)
            states.append(self.step(a))
            prev_action = a
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
        print(f"horizon={horizon}: start=(0,0) actions={actions} -> end={states[-1]}")


if __name__ == "__main__":
    _self_test()
