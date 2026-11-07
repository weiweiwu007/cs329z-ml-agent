# Covertype MLP: results, 2026-11-04

**Setup:** UCI Covertype (581,012 rows, 7 classes), two L4 GPUs with one GPU per run, `gpt-6-luna`, arms A–D
plus random search, 3 seeds, up to 20 attempts, 168 s of compute per run and 79 s per attempt. Took 34 minutes
and $0.19 in API cost, and used 7.1 GB. All 15 runs completed with no failed experiments and no recipe repairs.
Settings are in the last section of `DECISION_RECORD.md`. Files are in `studies/mlp-covertype/` (local plot and
tables) and `/home/ml-agent/studies/mlp-covertype/` (VM, with full artifacts).

**The plot:** `studies/mlp-covertype/figures/fig_abcd_curves.png`. X is compute used, Y is best validation accuracy
so far, averaged over seeds, with one curve per design.

| Arm | Validation, mean | Validation, by seed | Test, mean | Test, by seed | Runs with a wider model trained for 30+ s |
|---|---|---|---|---|---|
| A: agent alone | **90.12%** | 90.92, 89.85, 89.59 | **90.12%** | 90.90, 89.86, 89.61 | 3 of 3 |
| C: + reviewer | 89.31% | 89.91, 86.96, 91.04 | 89.26% | 89.80, 87.06, 90.94 | 2 of 3 |
| D: reports + reviewer | 88.91% | 88.86, 91.20, 86.66 | 88.88% | 88.94, 91.11, 86.61 | 2 of 3 |
| B: + error reports | 86.87% | 87.14, 86.79, 86.66 | 86.87% | 87.08, 86.91, 86.61 | 0 of 3 |
| Random search | 82.78% | 83.33, 83.08, 81.93 | 82.74% | 83.29, 83.08, 81.86 | Not applicable |
| Starting model | 81.87% | 81.83, 81.85, 81.93 | Not scored | | |

Target (from the usual rule): 83.66%. One validation row is worth 0.0009%, so measurement resolution is not the
limit here, and validation and test agree closely. The variation is between whole agent runs.

## Reading

- **A ends highest, but this is not a proven winner.** A beat B on all 3 seeds, and beat C and D on 2 of 3. The two
  highest single runs came from reviewer designs (D 91.20%, C 91.04%). A's lead over C is smaller than C's swing
  between seeds, about 4 points.
- **One decision explains most of the result: whether the run trained a wider model while budget remained.** The 7
  runs that did scored 88.86–91.20%; the 5 that did not scored 86.66–87.14%, with no overlap. The budget fits about two
  long attempts, so spending both on longer training of the small model leaves no room to widen. This grouping
  was observed after the run, so it is an explanation to test, not a pre-declared analysis.
- **Contrasts on mean validation accuracy.** Report: B−A −3.25, D−C −0.40. Reviewer: C−A −0.81, D−B +2.04. Interaction:
  (D−B)−(C−A) +2.85. Three seeds cannot separate these from run-to-run variation.
- **Error reports were rarely used.** 5 of 27 decisions in B and D mentioned errors, classes, or features, and no LLM
  arm tried class balancing, although the report showed low recall on rare classes (Aspen 0.26). Reports cost only about
  0.5 s per run, so their compute charge does not explain B's score.
- **Agents far exceeded random search,** which averaged 82.74% on test and barely moved from the start. Its ranges were
  rescaled for the short start and it does not read timings, so this is a reference, not a pure comparison. Class weights
  combined with label smoothing also handicap random search; see `2026-11-04_covertype-600s_RESULTS.md`. A–D never used either.

## Limits

- Three seeds is thin. Seeds can be added to this study without repeating finished work.
- The 5-epoch start and 168 s budget mostly test one early decision rather than sustained tuning, and part of the gain
  is the easy step of training longer. The run took 34 minutes of the roughly 2 hours the user allowed, so more compute would fit.
- The best models stayed far below the published tuned MLP's 96.2%, which used far more training and a different split.
