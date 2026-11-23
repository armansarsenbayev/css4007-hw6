# HW6 submission

## AI-use declaration

TODO(you): check that this is true and add Parts B and C when they are done.

* **Claude (Anthropic), Part A:** explained value iteration on separate practice maps; after I finished my own hand calculations it checked my numbers (they matched); rotated and brightened the photos of my pages; drafted the text of items 1 and 4, which I reviewed and can explain.

## Setup and reproduction

Record dependency versions, installation commands and exact run commands.

## Part A — Paper calculations

Hand calculations, photographed and committed before any code was written.

* Page 1: map, rewards, update rule and V1 for every nonterminal tile, with the V1 map: [part_a/part_a_page1.jpg](part_a/part_a_page1.jpg)
* Page 2: V1 at the ice tile (2,3), round 2 setup and V2 at (1,2) and (1,3): [part_a/part_a_page2.jpg](part_a/part_a_page2.jpg)
* Page 3: V2 at the ice tile (2,3) with all four probability weighted sums, and the arrow map: [part_a/part_a_page3.jpg](part_a/part_a_page3.jpg)

![Part A page 1](part_a/part_a_page1.jpg)
![Part A page 2](part_a/part_a_page2.jpg)
![Part A page 3](part_a/part_a_page3.jpg)

### V1 (typed copy of page 1)

| Row | Col 1 | Col 2 | Col 3 | Col 4 |
| --- | --- | --- | --- | --- |
| 1 | -1 | -1 (M) | 10 | G |
| 2 | -1 | # | -1 (I) | H |
| 3 | -1 (S) | -1 | -1 | -1 |

### V2 at the three requested tiles (typed copy of pages 2 and 3)

| Tile | Up | Right | Down | Left | V2 | Best action |
| --- | --- | --- | --- | --- | --- | --- |
| (1,2) mud | -3.9 (edge, stay in mud) | 8 (to (1,3)) | -3.9 (wall, stay in mud) | -1.9 (to (1,1)) | 8 | right, no tie |
| (1,3) floor | 8 (edge, stay on (1,3)) | 10 (to G) | -1.9 (to ice) | -3.9 (to mud) | 10 | right, no tie |
| (2,3) ice | 0.8·8 + 0.1·(-1.9) + 0.1·(-10) = 5.21 | 0.8·(-10) + 0.1·8 + 0.1·(-1.9) = -7.39 | 0.8·(-1.9) + 0.1·(-1.9) + 0.1·(-10) = -2.71 | 0.8·(-1.9) + 0.1·8 + 0.1·(-1.9) = -0.91 | 5.21 | up, no tie |

### Item 1: the MDP in my own words

* **States:** every tile the agent can stand on. The map has 12 tiles; the wall (2,2) is not a state, so there are 11 states. Two of them are terminal: the goal G (1,4) and the hole H (2,4). The other 9 are nonterminal, and those are the tiles I calculated.
* **Actions:** up, right, down and left in every nonterminal state. Terminal states have no actions, because the episode is over.
* **Transitions:** from S, ordinary floor and mud the chosen move always happens. From ice the move is uncertain: 0.8 in the chosen direction and 0.1 to each perpendicular side, and never backwards. If a move (or a slip) would go into the wall or off the map, the agent stays where it is. Each slip outcome is checked against walls and edges separately, so the probabilities of one action always add up to 1.
* **Rewards:** the agent receives the reward of the tile where the move ends: -1 for S, floor and ice, -3 for mud, +10 for the goal and -10 for the hole. Bumping into a wall still costs the reward of the tile it stays on, so bumping from mud costs -3. After G or H there are no more rewards, so their values stay 0.
* **What the agent maximizes:** the expected discounted sum of rewards, r0 + 0.9·r1 + 0.9²·r2 + ... "Expected" matters because ice is random, so an action is judged by the probability weighted average of its outcomes. The discount 0.9 makes a reward now worth more than the same reward later, so the agent prefers to reach the goal by a short, cheap path while avoiding mud and the hole.

### Item 4: arrows, ties and why two rounds are not enough

Arrows at the requested tiles: (1,2) right, (1,3) right, (2,3) up. At these three tiles there are no ties: each maximum is reached by one action only.

Two rounds do not prove that this is the final optimal policy:

1. **V2 only looks two moves ahead.** V2 is the best expected return if the agent had exactly two moves left, but in this problem the episode continues until the agent reaches G or H, with no move limit.
2. **The goal has not reached most of the map yet.** Each round spreads information about the goal by about one tile. After two rounds only (1,2), (1,3) and the ice (2,3) have positive values. The start (3,1) and the whole third row still have V2 = -1.9, and at the start all four actions tie, so V2 gives no useful direction there yet.
3. **The best action already changed once, so it can change again.** At the ice tile the best action in round 1 was left (bump into the wall and stay, -1, the safest single move), but in round 2 it became up (5.21), because (1,3) now "knows" about the goal. Values and arrows keep moving in V3, V4 and later rounds.
4. **Optimality needs converged values.** The policy is optimal only after the values stop changing, which is what Part C checks: run value iteration until the largest change in a full update is below 0.000001, then choose the best action at every tile from those final values.

## Part B — FrozenLake

### Fixed policy, written before the experiments

State the action for each nonterminal state and your reasoning.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |

Link episode rows and example traces in outputs/. Explain slipping, variation across batches and why these runs do not train a policy.

## Part C — Custom solver

### Compare code with paper

Show V1 and the requested V2 values. Explain discrepancies.

### Values, policies and convergence

Link both value/policy maps. Record iteration counts, final value changes and tie-breaking rule.

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |

### Explain your results

Which values/actions changed with sand and water? Where did cost and risk matter? How do simulations relate to the computed values?

### Rule checks

Show boundary/wall behavior, slip probabilities, floor rewards and terminal handling.

