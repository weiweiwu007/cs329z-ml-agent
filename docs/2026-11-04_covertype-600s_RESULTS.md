# Covertype MLP with 600 seconds per run: results, 2026-11-04

**Setup:** the same as the 168-second run (`2026-11-04_covertype_RESULTS.md`), with 600 s of compute per run and 267 s
per attempt, and random search scaled to match. Two L4 GPUs with one per run, `gpt-6-luna`, arms A–D plus random search,
3 seeds. Took 95 minutes and $0.39 in API cost, and used 17 GB. All 15 workflow runs completed, with no recipe repairs.
Within those runs, one B/seed-2 attempt exceeded its per-trial time limit, and six final attempts were cut off by the
remaining compute budget; these attempts remain in the recorded costs and outcomes. The starting models reproduced
the first run's scores exactly. Settings are in the last section of
`DECISION_RECORD.md`. Files are in `studies/mlp-covertype-600s/` (local plot and tables) and
`/home/ml-agent/studies/mlp-covertype-600s/` (VM, with full artifacts).

**The plot:** `studies/mlp-covertype-600s/figures/fig_abcd_curves.png`, drawn with `scripts/abcd_curves.py --x-max 600`.
X is compute used in seconds, Y is best validation accuracy so far, averaged over seeds. `fig_abcd_random_curves.png`, in the
same folder, adds random search as a fifth curve (`--with-random`), drawn from the same saved results with no new training.

| Arm | Validation, mean | Validation, by seed | Test, mean | Test, by seed | Attempts | Parameters of each best model |
|---|---|---|---|---|---|---|
| D: reports + reviewer | **94.13%** | 94.02, 94.18, 94.18 | **94.11%** | 94.05, 94.16, 94.14 | 11, 9, 15 | 161k, 454k, 340k |
| A: agent alone | 93.76% | 93.41, 94.83, 93.06 | 93.83% | 93.49, 94.96, 93.04 | 5, 7, 7 | 56k, 340k, 56k |
| C: + reviewer | 93.41% | 93.00, 93.74, 93.47 | 93.45% | 93.09, 93.81, 93.46 | 9, 9, 10 | 96k, 161k, 96k |
| B: + error reports | 92.64% | 90.46, 92.18, 95.28 | 92.68% | 90.58, 92.24, 95.22 | 6, 6, 10 | 16k, 48k, 454k |
| Random search | 82.31% | 83.15, 81.85, 81.93 | 82.30% | 83.10, 81.92, 81.86 | 9, 9, 5 | Not applicable |
| Starting model | 81.87% | 81.83, 81.85, 81.93 | Not scored | | | 16k |

## Reading

- **D ends highest, but its lead is small.** It beat C on all 3 seeds, and beat A and B on 2 of 3. Its 0.36-point lead over A
  is smaller than A's swing between seeds (1.77 points on validation). The best single run was B on seed 3, at 95.28%.
- **The ranking depends on the budget.** At 168 s it was A, C, D, B; at 600 s it is D, A, C, B. B is lowest at both budgets.
  This budget was chosen after seeing the first result, so the two runs are reported together, not as one winner.
- **Model size again separates the runs.** The 6 runs whose best model had 161k or more parameters scored 93.74–95.28%. The 6
  with under 100k scored 90.46–93.47%, with no overlap. All 3 D runs reached the larger group, against 1 of 3 for each of A, B,
  and C. As before, this grouping was observed after the run, so it is an explanation to test.
- **Reviewer designs were steadier here,** with a test spread of 0.11 points for D and 0.72 for C, against 1.92 for A and
  4.64 for B. They also ran more attempts in the same compute. This did not hold at 168 s, when C and D swung by about 4 points,
  so it is not an established effect.
- **Contrasts on mean validation accuracy.** Report: B−A −1.12, D−C +0.72. Reviewer: C−A −0.36, D−B +1.49. Interaction:
  (D−B)−(C−A) +1.84. Three seeds cannot separate these from run-to-run variation.
- **Error reports reached every B and D decision but were not visibly used.** None of the 51 planner decisions in B and D named
  errors, classes, or features by a keyword search, and no LLM arm tried class balancing. Reviewer text was not included in that count.
- **Agents barely used batch size.** Every best model used the starting batch size of 128, and larger batches appeared
  only in short tests, even though they train about 3× faster per epoch.

## Random search was handicapped by class weights combined with label smoothing

Random search stayed near the start and failed to beat the starting model on 2 of 3 seeds. Its 7 completed attempts using balanced
class weights scored 2–14%, and the three inspected labeled about 65–82% of rows as Cottonwood, which is 0.5% of the data.

**Cause:** `torch.nn.functional.cross_entropy(..., weight=, label_smoothing=)` also multiplies the smoothed part of the target by each
class's weight. Cottonwood's weight is about 30, against 0.3–0.4 for the two largest classes, so a little smoothing tells every row
to look like Cottonwood. Random search always draws some smoothing, so all its weighted attempts had both. The trainer calls PyTorch
as documented, and the weights themselves are correct. An independent script reproduced this (the same small model, 8 epochs):

| Loss | Accuracy | Balanced accuracy | Share predicted Cottonwood |
|---|---|---|---|
| No weights, no smoothing | 85.21% | 74.76% | 0.5% |
| Class weights, no smoothing | 74.70% | 84.59% | 1.0% |
| Class weights + smoothing 0.05 | 66.12% | 81.65% | 15.4% |
| Class weights + smoothing 0.1 | 13.00% | 63.06% | 79.5% |
| Class weights + smoothing 0.2 | 11.30% | 59.17% | 84.8% |
| Weight by true class only + smoothing 0.1 | 75.89% | 84.69% | 0.9% |

**Effect:** none of the 169 A–D attempts across both Covertype studies used class weights or label smoothing, so **A–D results are
unaffected**. Random search lost about a third of its compute at 600 s to these attempts (604 of 1800 s), so its score here understates it. The trainer is
unchanged so far. Weighting each row by its true class alone would avoid this, and is identical when smoothing is zero. The check
scripts and logs are `/home/vm-logs/weight_check*.py` and `.log` on the VM.

## Limits

- Three seeds is thin. Seeds can be added to this study without repeating finished work.
- The best models are still below the published tuned MLP's 96.2%, which used far more training and a different split.
