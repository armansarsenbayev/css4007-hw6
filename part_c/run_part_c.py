"""HW6 Part C: run every step of the custom solver and save the results.

1. Rule checks (walls and edges, ice probabilities, floor rewards, termination).
2. V1 and the three V2 values on the paper map, compared with my Part A pages.
3. Value iteration to convergence on the paper map and on the extended map.
4. Simulation of each computed policy: 3 batches x 50 episodes, Part B seeds.

Run from the repository root:
    python part_c/run_part_c.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gridworld import (ACTIONS, DATA, GAMMA, MAX_ITERATIONS, TOLERANCE, GridWorld,  # noqa: E402
                       greedy_policy, max_change, policy_map, pos, value_iteration, value_map)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SIM = DATA["simulation"]
BATCH_SEEDS = SIM["batch_seeds"]
EPISODES = SIM["episodes_per_batch"]
MAX_STEPS = SIM["max_steps"]

# My hand calculations from Part A (part_a/part_a_page1-3.jpg), 1 based (row, column).
PAPER_V1 = {(1, 1): -1, (1, 2): -1, (1, 3): 10, (2, 1): -1, (2, 3): -1,
            (3, 1): -1, (3, 2): -1, (3, 3): -1, (3, 4): -1}
PAPER_V2_Q = {(1, 2): {"up": -3.9, "right": 8, "down": -3.9, "left": -1.9},
              (1, 3): {"up": 8, "right": 10, "down": -1.9, "left": -3.9},
              (2, 3): {"up": 5.21, "right": -7.39, "down": -2.71, "left": -0.91}}


def zero_based(cell):
    return (cell[0] - 1, cell[1] - 1)


# ---------------------------------------------------------------- rule checks
def rule_checks():
    paper = GridWorld(DATA["paper_map"])
    ext = GridWorld(DATA["extended_map"])
    lines, failures = [], 0

    def check(name, ok, detail):
        nonlocal failures
        failures += 0 if ok else 1
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    lines.append("Walls and map edges (paper map)")
    for cell, action, world in [((1, 2), "up", paper), ((1, 2), "down", paper), ((2, 1), "right", paper),
                                ((3, 1), "left", paper), ((3, 1), "down", paper), ((1, 1), "up", paper)]:
        (p, nxt, r), = world.outcomes(zero_based(cell), action)
        stay = nxt == zero_based(cell)
        own = world.reward(zero_based(cell))
        check(f"{pos(zero_based(cell))} {action}", stay and r == own and p == 1.0,
              f"stays on {pos(nxt)} with probability {p}, reward {r} (reward of its own tile {own})")

    lines.append("\nIce probabilities (paper map, ice at (2,3))")
    ice = zero_based((2, 3))
    for a in ACTIONS:
        outs = paper.outcomes(ice, a)
        total = sum(p for p, _, _ in outs)
        detail = ", ".join(f"{p:.1f} to {pos(n)} (reward {r})" for p, n, r in outs)
        check(f"ice {a}", abs(total - 1) < 1e-12 and sorted(p for p, _, _ in outs) == [0.1, 0.1, 0.8],
              f"{detail}; sum = {total:.10f}")
    all_ok = all(abs(sum(p for p, _, _ in w.outcomes(s, a)) - 1) < 1e-12
                 for w in (paper, ext) for s in w.nonterminal_states() for a in ACTIONS)
    check("every action from every nonterminal state, both maps", all_ok, "probabilities sum to 1")
    slip_left = {n for _, n, _ in paper.outcomes(ice, "up")}
    check("ice up: the left slip hits the wall (2,2)", ice in slip_left,
          "the 0.1 slip to the left keeps the agent on the ice")

    lines.append("\nFloor rewards (reward of the tile where the move ends)")
    for name, world, cell, action, expected in [
            ("enter mud M", paper, (1, 1), "right", -3),
            ("enter sand D", ext, (3, 1), "right", -2),
            ("enter shallow water W", ext, (3, 3), "right", -5),
            ("enter goal G (no extra floor cost)", paper, (1, 3), "right", 10),
            ("enter hole H", paper, (3, 4), "up", -10),
            ("ordinary floor", paper, (3, 1), "right", -1)]:
        (p, nxt, r), = world.outcomes(zero_based(cell), action)
        check(name, r == expected, f"{pos(zero_based(cell))} {action} -> {pos(nxt)}, reward {r}, expected {expected}")
    for cell in [(3, 2), (3, 4)]:
        c = zero_based(cell)
        for a in ACTIONS:
            assert len(ext.outcomes(c, a)) == 1
    check("sand and water are deterministic", True, "every action from (3,2) D and (3,4) W has one outcome with probability 1")

    lines.append("\nTermination")
    for world, name in [(paper, "paper"), (ext, "extended")]:
        terminals = [s for s in world.states() if world.is_terminal(s)]
        no_actions = all(world.outcomes(s, a) == [] for s in terminals for a in ACTIONS)
        history, _ = value_iteration(world)
        stays_zero = all(v[s] == 0 for v in history for s in terminals)
        check(f"{name} map: G and H have no actions and no rewards", no_actions,
              f"terminal states {[pos(s) for s in terminals]} return no outcomes")
        check(f"{name} map: G and H values stay 0 in all {len(history) - 1} rounds", stays_zero, "value is 0 every round")
    trace = simulate_episode(paper, greedy_policy(paper, value_iteration(paper)[0][-1])[0], seed=11000, keep_trace=True)
    after_end = trace["trace"][-1]
    check("simulation stops on a terminal tile", trace["outcome"] in ("goal", "hole")
          and after_end["next"] in [s for s in paper.states() if paper.is_terminal(s)],
          f"last move ends on {pos(after_end['next'])}, outcome {trace['outcome']}, no further rewards are collected")

    lines.append(f"\n{failures} failed checks")
    return "\n".join(lines), failures


# --------------------------------------------------------- paper comparison
def paper_comparison():
    world = GridWorld(DATA["paper_map"])
    history, _ = value_iteration(world, max_iterations=2, tolerance=-1)
    v1, v2 = history[1], history[2]
    lines = ["V1 from the solver (paper map):", value_map(world, v1, decimals=2), "",
             "V1: solver vs my paper (Part A page 1)"]
    mismatches = 0
    for cell, paper_value in PAPER_V1.items():
        code_value = v1[zero_based(cell)]
        same = abs(code_value - paper_value) < 1e-9
        mismatches += 0 if same else 1
        lines.append(f"  {pos(zero_based(cell))}: solver {code_value:.4f}, paper {paper_value}, {'match' if same else 'DIFFERENT'}")
    lines.append("\nV2 at the three requested tiles: every action, solver vs paper (Part A pages 2 and 3)")
    for cell, paper_q in PAPER_V2_Q.items():
        c = zero_based(cell)
        lines.append(f"  {pos(c)} {world.grid[c[0]][c[1]]}:")
        for a in ACTIONS:
            q = world.q_value(v1, c, a)
            same = abs(q - paper_q[a]) < 1e-9
            mismatches += 0 if same else 1
            lines.append(f"    {a:>5}: solver {q:.4f}, paper {paper_q[a]}, {'match' if same else 'DIFFERENT'}")
        lines.append(f"    V2 = {v2[c]:.4f}")
    lines.append(f"\n{mismatches} differences between the solver and my paper calculations")
    return "\n".join(lines), mismatches


# ------------------------------------------------------------------- solve
def solve(world, title):
    history, converged = value_iteration(world)
    values = history[-1]
    policy, ties = greedy_policy(world, values)
    iterations = len(history) - 1
    change = max_change(history)
    start = world.start()
    text = [
        f"{title}",
        "Map:", *("  " + "".join(r) for r in world.grid), "",
        f"Value iteration: tolerance {TOLERANCE}, safeguard {MAX_ITERATIONS} iterations",
        f"Converged: {converged}, iterations: {iterations}, largest change in the last full update: {change:.3e}",
        "", "Final values:", value_map(world, values), "",
        "Policy (tie rule: first best action in the order up, right, down, left; * marks a tie):",
        policy_map(world, policy, ties), "",
        "Q values at every nonterminal tile (from the final values):",
    ]
    for s in world.nonterminal_states():
        q = {a: world.q_value(values, s, a) for a in ACTIONS}
        tie = f"  TIE: {', '.join(ties[s])}" if len(ties[s]) > 1 else ""
        text.append(f"  {pos(s)} {world.grid[s[0]][s[1]]}: " + ", ".join(f"{a} {q[a]:.4f}" for a in ACTIONS)
                    + f"  -> {policy[s]}{tie}")
    text.append(f"\nValue of the start {pos(start)}: {values[start]:.6f}")
    info = {"iterations": iterations, "converged": converged, "final_change": change,
            "start_value": values[start],
            "values": {pos(s): round(v, 6) for s, v in values.items()},
            "policy": {pos(s): a for s, a in policy.items()},
            "ties": {pos(s): t for s, t in ties.items() if len(t) > 1}}
    return "\n".join(text), values, policy, info


# ---------------------------------------------------------------- simulate
def simulate_episode(world, policy, seed, keep_trace=False):
    """Sample one episode from the known model with its own seeded generator."""
    rng = np.random.default_rng(seed)
    state = world.start()
    discounted, total, steps, trace = 0.0, 0.0, 0, []
    outcome = "time limit"
    while steps < MAX_STEPS:
        action = policy[state]
        outs = world.outcomes(state, action)
        k = rng.choice(len(outs), p=[p for p, _, _ in outs])
        _, nxt, reward = outs[k]
        discounted += (GAMMA ** steps) * reward  # r0 + 0.9 r1 + 0.9^2 r2 + ...
        total += reward
        steps += 1
        if keep_trace:
            trace.append({"step": steps, "state": state, "action": action, "next": nxt, "reward": reward})
        state = nxt
        if world.is_terminal(state):
            outcome = "goal" if world.reward(state) > 0 else "hole"
            break
    return {"steps": steps, "total_reward": total, "discounted_return": discounted,
            "outcome": outcome, "trace": trace}


def simulate(world, policy, map_name):
    rows = []
    for batch in BATCH_SEEDS:
        for i in range(EPISODES):
            seed = 1000 * batch + i
            r = simulate_episode(world, policy, seed)
            rows.append({"map": map_name, "batch_seed": batch, "episode_seed": seed, "steps": r["steps"],
                         "total_reward": r["total_reward"], "discounted_return": round(r["discounted_return"], 6),
                         "outcome": r["outcome"]})
    return rows


def main():
    OUT.mkdir(exist_ok=True)

    checks_text, failures = rule_checks()
    (OUT / "part_c_rule_checks.txt").write_text(checks_text + "\n", encoding="utf-8")
    print(checks_text, "\n")

    compare_text, mismatches = paper_comparison()
    (OUT / "part_c_paper_comparison.txt").write_text(compare_text + "\n", encoding="utf-8")
    print(compare_text, "\n")

    summary = {"rule_check_failures": failures, "paper_mismatches": mismatches, "maps": {}}
    all_rows, table = [], [
        "| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, title in [("paper_map", "Paper map (Part A)"), ("extended_map", "Extended map (sand D, shallow water W)")]:
        world = GridWorld(DATA[key])
        text, values, policy, info = solve(world, title)
        (OUT / f"part_c_{key}.txt").write_text(text + "\n", encoding="utf-8")
        print(text, "\n")
        rows = simulate(world, policy, key)
        all_rows += rows
        info["batches"] = []
        for batch in BATCH_SEEDS:
            b = [r for r in rows if r["batch_seed"] == batch]
            stats = {"batch_seed": batch,
                     "goals": sum(r["outcome"] == "goal" for r in b),
                     "holes": sum(r["outcome"] == "hole" for r in b),
                     "time_limits": sum(r["outcome"] == "time limit" for r in b),
                     "mean_steps": sum(r["steps"] for r in b) / len(b),
                     "mean_discounted_return": sum(r["discounted_return"] for r in b) / len(b)}
            info["batches"].append(stats)
            table.append(f"| {key.replace('_', ' ')} | {batch} | {stats['goals']} | {stats['holes']} | "
                         f"{stats['time_limits']} | {stats['mean_steps']:.2f} | {stats['mean_discounted_return']:.3f} |")
        info["mean_discounted_return_all_150"] = sum(r["discounted_return"] for r in rows) / len(rows)
        summary["maps"][key] = info

    with open(OUT / "part_c_episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / "part_c_simulation_summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    (OUT / "part_c_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(table))
    for key in ("paper_map", "extended_map"):
        m = summary["maps"][key]
        print(f"{key}: V(start) = {m['start_value']:.4f}, mean discounted return over 150 episodes = "
              f"{m['mean_discounted_return_all_150']:.4f}")


if __name__ == "__main__":
    main()
