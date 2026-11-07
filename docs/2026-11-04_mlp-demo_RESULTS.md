# Spambase MLP demo: results, 2026-11-04

**Setup:** NVIDIA L4, `gpt-6-luna`, arms A–D plus random search, 3 seeds, up to 20 attempts, about 22 s of
compute per run. Took 52 minutes and $0.53 in API cost. All 15 runs completed with no failed experiments.
Files are in `studies/mlp-demo/` (local copy) and `/home/ml-agent/studies/mlp-demo/` (VM, with full artifacts).
Paths recorded inside those files say `/root/ml-agent`, where the runs happened; they were copied to `/home` afterward.

| Arm | Best validation accuracy | Test accuracy of that model | Seconds to target, by seed |
|---|---|---|---|
| Baseline | 94.21% | 94.05% | |
| A: agent alone | 95.22% | 94.12% | 9.4, 16.3, 15.3 |
| B: + error reports | 95.15% | 94.27% | 7.1, 15.5, 7.2 |
| C: + reviewer | 95.01% | 94.23% | 13.9, 8.3, 14.7 |
| D: reports + reviewer | 95.01% | 94.27% | 9.4, 18.6, 10.2 |
| Random search | 94.68% | 93.94% | 16.9, 9.4, missed |

**Reading:** one row is about 0.11% accuracy on both splits. No design clearly wins, and test scores sit within
about 2 rows of the baseline, so most of the validation gain is selection on noise. Spambase leaves little room.

**Signals worth checking on harder tasks, none established with 3 seeds:**
- B reached the target sooner than A on all 3 seeds. D versus C was mixed.
- Reviewer arms were more cautious: 1.1 settings changed per decision versus 1.5–1.8, and never 3 or more at once
  (0 of 69 decisions, versus 12 of 71). All 6 reviewer runs ended at exactly 95.01% validation.
- LLM arms beat random search on validation in every arm mean, and random missed the target once.
- No LLM agent ever changed class balancing, even with error reports. Agents did shrink models on their own.

**Proved by this run:** the unattended pipeline end to end on a GPU, real provider cost capture, constraint-only
choices with only 2 recipe repairs in 140 decisions, and budget-driven differences in attempt count (10 to 19).
**Not proved:** CNN and BERT training at scale, and any claim about which design is better.
