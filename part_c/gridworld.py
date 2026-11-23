"""HW6 Part C: Grid World model and value iteration, written by hand.

The floor rules (reward, deterministic or slippery movement, terminal, wall)
come from data/maps.json. Nothing in this file names a specific floor letter,
so a new floor type is added by adding one entry to "tiles" in maps.json,
without touching value iteration.

Coordinates in printed output are (row, column), starting at 1.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "data" / "maps.json").read_text(encoding="utf-8"))

ACTIONS = DATA["actions"]  # ["up", "right", "down", "left"]: also the tie breaking order
MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
PERPENDICULAR = {"up": ("left", "right"), "down": ("left", "right"),
                 "left": ("up", "down"), "right": ("up", "down")}
ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
GAMMA = DATA["discount"]
TOLERANCE = DATA["solver"]["tolerance"]
MAX_ITERATIONS = DATA["solver"]["max_iterations"]
TIE_EPS = 1e-9  # two Q values closer than this count as a tie


class GridWorld:
    """A map plus the tile rules. Builds the known transition model P(s'|s,a) and R."""

    def __init__(self, rows, tiles=None):
        self.grid = [list(r) for r in rows]
        self.tiles = tiles or DATA["tiles"]
        self.n_rows, self.n_cols = len(self.grid), len(self.grid[0])
        for row in self.grid:
            for ch in row:
                if ch not in self.tiles:
                    raise ValueError(f"Tile {ch!r} has no rule in maps.json")

    # ----- tile rules, read from data -----
    def rule(self, cell):
        r, c = cell
        return self.tiles[self.grid[r][c]]

    def is_wall(self, cell):
        return self.rule(cell).get("traversable", True) is False

    def is_terminal(self, cell):
        return bool(self.rule(cell).get("terminal", False))

    def reward(self, cell):
        return self.rule(cell)["reward"]

    # ----- states -----
    def states(self):
        """Every non wall tile is a state."""
        return [(r, c) for r in range(self.n_rows) for c in range(self.n_cols) if not self.is_wall((r, c))]

    def nonterminal_states(self):
        return [s for s in self.states() if not self.is_terminal(s)]

    def start(self):
        for s in self.states():
            if self.rule(s).get("meaning") == "start":
                return s
        raise ValueError("Map has no start tile")

    # ----- transitions -----
    def move(self, cell, direction):
        """One step in a direction; a wall or the map edge leaves the agent in place."""
        dr, dc = MOVES[direction]
        nr, nc = cell[0] + dr, cell[1] + dc
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols) or self.is_wall((nr, nc)):
            return cell
        return (nr, nc)

    def outcomes(self, cell, action):
        """List of (probability, next_cell, reward) for one action.

        Slipping depends on the tile the move STARTS from. Each slip direction is
        resolved separately against walls and edges. The reward is the reward of
        the tile where the move ends, even when the agent stays in place.
        """
        if self.is_terminal(cell):
            return []  # no actions and no rewards after termination
        rule = self.rule(cell)
        if rule.get("movement") == "slippery":
            side_a, side_b = PERPENDICULAR[action]
            directions = [(action, rule["intended"]), (side_a, rule["perpendicular_each"]),
                          (side_b, rule["perpendicular_each"])]
        else:
            directions = [(action, 1.0)]
        result = []
        for direction, p in directions:
            nxt = self.move(cell, direction)
            result.append((p, nxt, self.reward(nxt)))
        return result

    def q_value(self, values, cell, action):
        """sum over s' of P(s'|s,a) * (R + gamma * V(s')); terminal values are always 0."""
        return sum(p * (r + GAMMA * values[nxt]) for p, nxt, r in self.outcomes(cell, action))


def bellman_update(world, values):
    """One full synchronous update: every state uses only the previous round's values."""
    new_values = {}
    for s in world.states():
        if world.is_terminal(s):
            new_values[s] = 0.0
        else:
            new_values[s] = max(world.q_value(values, s, a) for a in ACTIONS)
    return new_values


def value_iteration(world, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS):
    """Start from V0 = 0 and update until the largest absolute change is below tolerance.

    Returns (history, converged) where history[k] is Vk, so history[1] is V1.
    """
    values = {s: 0.0 for s in world.states()}
    history = [values]
    for _ in range(max_iterations):
        new_values = bellman_update(world, values)
        delta = max(abs(new_values[s] - values[s]) for s in values)
        history.append(new_values)
        values = new_values
        if delta < tolerance:
            return history, True
    return history, False


def max_change(history):
    return max(abs(history[-1][s] - history[-2][s]) for s in history[-1])


def greedy_policy(world, values):
    """Best action at every nonterminal state, comparing Q values computed from the given values.

    Tie rule: all actions within TIE_EPS of the best are listed as ties; the chosen
    action is the first of them in the order up, right, down, left.
    """
    policy, ties = {}, {}
    for s in world.nonterminal_states():
        q = {a: world.q_value(values, s, a) for a in ACTIONS}
        best = max(q.values())
        tied = [a for a in ACTIONS if best - q[a] < TIE_EPS]
        policy[s] = tied[0]
        ties[s] = tied
    return policy, ties


# ----- printing -----
def label(world, cell):
    return world.grid[cell[0]][cell[1]]


def value_map(world, values, decimals=3):
    width = 9 + decimals - 3
    lines = ["          " + "".join(f"{c + 1:^{width}}" for c in range(world.n_cols))]
    for r in range(world.n_rows):
        cells = []
        for c in range(world.n_cols):
            ch = label(world, (r, c))
            if world.is_wall((r, c)):
                cells.append(f"{'#':^{width}}")
            elif world.is_terminal((r, c)):
                cells.append(f"{ch:^{width}}")
            else:
                cells.append(f"{values[(r, c)]:^{width}.{decimals}f}")
        lines.append(f"Row {r + 1}   " + "".join(cells))
    return "\n".join(lines)


def policy_map(world, policy, ties=None):
    lines = ["          " + "".join(f"{c + 1:^7}" for c in range(world.n_cols))]
    for r in range(world.n_rows):
        cells = []
        for c in range(world.n_cols):
            cell = (r, c)
            ch = label(world, cell)
            if world.is_wall(cell) or world.is_terminal(cell):
                cells.append(f"{ch:^7}")
            else:
                mark = ARROWS[policy[cell]] + ("*" if ties and len(ties[cell]) > 1 else "")
                cells.append(f"{ch + ' ' + mark:^7}")
        lines.append(f"Row {r + 1}   " + "".join(cells))
    return "\n".join(lines)


def pos(cell):
    return f"({cell[0] + 1},{cell[1] + 1})"
