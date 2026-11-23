"""HW6 Part B: repeated FrozenLake simulations with Gymnasium.

Runs two policies (uniformly random, and the fixed policy committed earlier in
part_b/fixed_policy.json) with is_slippery=False and is_slippery=True: four
conditions. Each condition runs three batches of 50 episodes with batch seeds
11, 22 and 33. Episode i of a batch is reset with seed 1000 * batch_seed + i,
so all four conditions use exactly the same episode seed lists.

Nothing here learns: both policies are fixed before the run and never change.

Run from the repository root:
    python part_b/frozenlake.py
"""

import csv
import json
import re
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
POLICY_FILE = ROOT / "part_b" / "fixed_policy.json"

MAP_NAME = "4x4"
MAX_STEPS = 100
BATCH_SEEDS = [11, 22, 33]
EPISODES_PER_BATCH = 50
ACTION_NAMES = {0: "left", 1: "down", 2: "right", 3: "up"}
GOAL_STATE = 15
HOLE_STATES = {5, 7, 11, 12}


def load_fixed_policy() -> dict:
    """The policy table committed before any experiment, read unchanged."""
    raw = json.loads(POLICY_FILE.read_text(encoding="utf-8"))["policy"]
    return {int(state): int(action) for state, action in raw.items()}


def make_env(slippery: bool, render: bool = False):
    """FrozenLake-v1 4x4 with default rewards and slip rules; only is_slippery changes."""
    return gym.make(
        "FrozenLake-v1",
        map_name=MAP_NAME,
        is_slippery=slippery,
        max_episode_steps=MAX_STEPS,
        render_mode="ansi" if render else None,
    )


def episode_seed(batch_seed: int, i: int) -> int:
    return 1000 * batch_seed + i


def outcome_of(final_state: int, terminated: bool) -> str:
    """Goal and hole both end the episode, so the final state tells them apart.
    A terminal state on the last allowed step still counts as goal or hole."""
    if terminated and final_state == GOAL_STATE:
        return "goal"
    if terminated and final_state in HOLE_STATES:
        return "hole"
    return "time limit"


def clean_render(text: str) -> str:
    """Gymnasium marks the agent with an ANSI red background; show it as [X] instead."""
    text = re.sub(r"\x1b\[41m(.)\x1b\[0m", r"[\1]", text)
    return re.sub(r"\x1b\[[0-9;]*m", "", text).strip("\n")


def run_episode(env, choose_action, seed: int, record: bool = False) -> dict:
    """Reset with the episode seed, then step until terminated, truncated or 100 steps."""
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    trace = []
    frames = [clean_render(env.render())] if record and env.render_mode else []

    while not (terminated or truncated) and steps < MAX_STEPS:
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total_reward += reward
        if record:
            trace.append(
                {
                    "step": steps,
                    "state": int(state),
                    "action": ACTION_NAMES[action],
                    "next_state": int(next_state),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            if env.render_mode:
                frames.append(clean_render(env.render()))
        state = next_state

    return {
        "steps": steps,
        "total_reward": total_reward,
        "final_state": int(state),
        "outcome": outcome_of(int(state), terminated),
        "trace": trace,
        "frames": frames,
    }


def policy_function(name: str, fixed: dict, rng):
    if name == "random":
        # Uniformly random action from the seeded generator of this episode.
        return lambda state: int(rng.integers(0, 4))
    return lambda state: fixed[state]


def format_trace(title: str, seed: int, result: dict) -> str:
    lines = [f"{title} (episode seed {seed}, {result['steps']} steps, outcome: {result['outcome']})"]
    for t in result["trace"]:
        flags = " terminated" if t["terminated"] else (" truncated" if t["truncated"] else "")
        lines.append(
            f"  step {t['step']:>3}: state {t['state']:>2} --{t['action']:>5}--> "
            f"state {t['next_state']:>2}, reward {t['reward']:.0f}{flags}"
        )
    return "\n".join(lines)


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    fixed = load_fixed_policy()
    rows = []
    examples = {}  # (condition, outcome) -> first episode with that outcome

    for slippery in (False, True):
        env = make_env(slippery)
        for policy_name in ("random", "fixed"):
            condition = f"{'slippery' if slippery else 'not slippery'}, {policy_name} policy"
            for batch_seed in BATCH_SEEDS:
                for i in range(EPISODES_PER_BATCH):
                    seed = episode_seed(batch_seed, i)
                    # The random policy gets its own generator, seeded with the episode seed.
                    rng = np.random.default_rng(seed)
                    result = run_episode(env, policy_function(policy_name, fixed, rng), seed, record=True)
                    rows.append(
                        {
                            "condition": condition,
                            "is_slippery": slippery,
                            "policy": policy_name,
                            "batch_seed": batch_seed,
                            "episode_seed": seed,
                            "steps": result["steps"],
                            "total_reward": result["total_reward"],
                            "final_state": result["final_state"],
                            "outcome": result["outcome"],
                        }
                    )
                    examples.setdefault((condition, result["outcome"]), (seed, result))
        env.close()

    # One row per episode.
    with open(OUTPUTS / "part_b_episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # One row per batch.
    table = [
        "| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    summary = []
    for slippery in (False, True):
        for policy_name in ("random", "fixed"):
            for batch_seed in BATCH_SEEDS:
                batch = [
                    r for r in rows
                    if r["is_slippery"] == slippery and r["policy"] == policy_name and r["batch_seed"] == batch_seed
                ]
                goals = sum(r["outcome"] == "goal" for r in batch)
                holes = sum(r["outcome"] == "hole" for r in batch)
                limits = sum(r["outcome"] == "time limit" for r in batch)
                mean_steps = sum(r["steps"] for r in batch) / len(batch)
                summary.append(
                    {"is_slippery": slippery, "policy": policy_name, "batch_seed": batch_seed,
                     "goals": goals, "holes": holes, "time_limits": limits,
                     "success_percent": 100 * goals / len(batch), "mean_steps": round(mean_steps, 2)}
                )
                table.append(
                    f"| {'yes' if slippery else 'no'} | {policy_name} | {batch_seed} | {goals} | {holes} | "
                    f"{limits} | {100 * goals / len(batch):.0f}% | {mean_steps:.2f} |"
                )
    (OUTPUTS / "part_b_summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    (OUTPUTS / "part_b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Example traces: first success and first failure in every condition, or a note.
    blocks = []
    for slippery in (False, True):
        for policy_name in ("random", "fixed"):
            condition = f"{'slippery' if slippery else 'not slippery'}, {policy_name} policy"
            blocks.append(f"=== {condition} ===")
            for outcome in ("goal", "hole", "time limit"):
                if (condition, outcome) in examples:
                    seed, result = examples[(condition, outcome)]
                    blocks.append(format_trace(f"First '{outcome}' episode", seed, result))
                else:
                    blocks.append(f"Outcome '{outcome}' never occurred in this condition (150 episodes).")
            blocks.append("")
    (OUTPUTS / "part_b_traces.txt").write_text("\n".join(blocks), encoding="utf-8")

    # One episode saved with Gymnasium's own text rendering.
    env = make_env(slippery=True, render=True)
    seed = episode_seed(BATCH_SEEDS[0], 0)
    result = run_episode(env, policy_function("fixed", fixed, None), seed, record=True)
    env.close()
    frames = [f"Step {k}:\n{frame}" for k, frame in enumerate(result["frames"])]
    header = (
        f"FrozenLake-v1 4x4, is_slippery=True, fixed policy, episode seed {seed}.\n"
        f"[X] marks the agent. Outcome: {result['outcome']} after {result['steps']} steps.\n"
    )
    (OUTPUTS / "part_b_example_episode.txt").write_text(header + "\n" + "\n\n".join(frames) + "\n", encoding="utf-8")

    print("\n".join(table))
    print(f"\nSaved {len(rows)} episode rows and example traces to {OUTPUTS}")


if __name__ == "__main__":
    main()
