# HW6 submission

## AI-use declaration

Which tools did you use, and for what?

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

TODO(you): item 1 (states, actions, transitions, rewards and what the agent maximizes, in your own words) and item 4 (why two rounds alone do not prove the final optimal policy).

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

