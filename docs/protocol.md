# What must be decided before research execution

V1 is an implementation, not a locked experimental protocol. The included synthetic
fixture is a software check. The proposal's Spambase/OpenML, CIFAR-100, and GLUE
directions are not replaced by that fixture, and no large training run has begun.

## A–D design decision — 2026-11-04

The user explicitly selected these independent variables:

- Detailed error reports: unavailable in A/C; automatic in B/D.
- Separate reviewer: absent in A/B; present in C/D.

This replaces the earlier design where A/C could request reports. Ordinary scores
and scalar training/validation loss summaries remain available in every condition.
Both roles obey their condition's access restriction. The request allowance is
configurable, but cannot enable reports in A/C; automatic delivery in B/D does not
require a positive allowance. Report analysis still consumes the compute budget.
The same raw validation artifacts remain retained for later analysis in all cases.

Only the software design is updated. Task recipes, training/API budgets, seeds,
data preparation, and artifact-retention choices remain draft. This change does
not authorize experiments, downloads, or API calls. Existing run artifacts are
preserved; changed source/access settings prevent resuming an older run under the
new design.

## Study execution — 2026-11-04

The settings proposed for the real study, and what still needs approval, are in
[DECISION_RECORD.md](DECISION_RECORD.md). The study runner adds:

- **Shared baselines.** Trial 0 is the same recipe, seed, data, and weights in every arm, so
  it trains once. Each arm is charged the measured seconds. Adoption requires a matching
  training identity (data, recipe, seed, device, encoder), training fingerprint (training
  code, library versions, GPU model), starting weights, and intact artifacts.
- **Matching across GPUs.** Run identity uses the GPU model, not the physical device.
- **Crash recovery.** `--recover-interrupted` reruns the interrupted node from the last
  committed state. A partial trial moves to `interrupted/`, never deleted. Its elapsed time
  and any API call lost with it are not charged, and the event records that they happened.
- **`stop_on_target`.** When false, runs spend their whole budget, and time to target is
  derived from the trajectory.

## Experiment identity

Each task file explicitly declares:

1. Dataset identity and revision, exact split file paths/SHA-256 values, row counts,
   numeric feature count/order, integer class mapping, and whether to standardize.
2. Starting model/training recipe, allowed changes, parameter/epoch limits, and training seed.
3. Validation metric (`accuracy`, `macro_f1`, or binary positive-class `f1`), target, and development/evaluation purpose.
4. CPU or CUDA execution, per-trial and total allocated execution time, trial limit,
   API call limit, context-character limit, and output-token limit.
5. Whether to retain each epoch's checkpoint and raw validation logits.

Preprocessing uses training-only mean and population standard deviation, calculated
in float64, then float32 inputs. Zero standard deviations become 1. No categorical
encoding, imputation, resizing, or implicit random split is performed. Such changes
need a new explicit adapter/protocol. CSV labels must already be class IDs 0..C-1.
Identical split files are rejected by path/hash; users still need to audit semantic
overlap and dataset leakage that file hashes cannot detect.

Every candidate starts from the saved initial weights when its parameter structure
is compatible. A changed architecture is initialized with the same declared seed;
it cannot literally share tensors of different shapes. The original and revised
agent receive the same baseline model and the same initialization rule. Every
attempt trains from its starting state; checkpoint continuation is not implemented.

## Outcomes and cost

The independent grader calculates accuracy, macro-F1 over all declared classes
(zero when a class's F1 denominator is zero), and cross-entropy. Candidate selection
uses **final-epoch validation scores**, not the best epoch or a training loss.
Epoch curves and retained raw logits permit later analysis without claiming that
an unselected epoch was the returned model.

Compute is charged from the moment the worker is ready to train: training, validation, and
checkpoint writing. Imports and device initialization are recorded as `startup_seconds` and not
charged, because they are a fixed cost that would otherwise dominate small tasks. CPU time is labeled
CPU; CUDA execution reports elapsed allocation time, not GPU kernel utilization or
FLOPs. `training_seconds` is separately recorded and includes in-loop validation
and checkpoint overhead. Initial dataset preparation, agent latency, and report
rendering are outside per-trial execution time. API calls/tokens/cost are separate.
Wall-time enforcement can overshoot slightly during process termination; actual
elapsed time is retained, never clipped to make a run look compliant.

The baseline model's training is a counted experiment. Failed/rejected attempts
consume the experiment count; actual execution time is charged even without a score.
The first final-epoch score reaching the target determines attempts/time to target.
Misses are explicitly reported. Policy selection uses declared-budget penalties
for misses, so early failure cannot masquerade as efficient success.

Counterbalanced paired runs share the same task config, data, initial tensors, seed,
model/backend identity, hardware/environment/source fingerprint, and prior evidence.
The workflow stays frozen throughout each run. Results from purpose `evaluation`
cannot produce or select workflow revisions. Use separate development cases to
propose/select policies and held-out settings for the final claim. Reusing exactly
the same solved task measures reuse; it does not establish generalization.

Candidate generation and validation have separate cost receipts. A policy is
accepted only by the declared code-based rule, and a small suite is exploratory.
Before a substantial study, settle repeat counts, quality targets, task settings,
the comparison method, and the total budget. Report variability and failure modes;
do not require every proposed change to improve the model.

## Stored objects and information loss

| Object | Representation | What it preserves |
|---|---|---|
| Source data | Original CSV remains at its supplied path; source hashes recorded | Original records and exact split identity. Keep these source files. |
| Staged data | Raw numeric float32 features, normalized float32 features, int64 labels/row IDs | Re-score and change preprocessing from staged numeric features. Original CSV precision/formatting remains only in the source. |
| Normalization | Train-only float64 mean/scale, feature names | Exact declared transform and its training provenance. |
| Initial model | Exact float32 state dictionary and canonical tensor hash | Matched starting tensors. |
| Validation outputs | Unnormalized float32 logits, labels, row IDs, epoch | Alternative probability transforms, thresholds, and classification metrics without retraining. Argmax-only predictions are not substituted. |
| Checkpoint | Model/optimizer state, epoch/step, recipe, Python/NumPy/Torch/shuffle/CUDA RNG state | Later model evaluation and a basis for future continuation tooling. V1 does not automatically resume training. |
| Epoch records | Loss/metrics, LR endpoints, elapsed time | Training trajectory at epoch resolution. |
| Agent records | Structured hypotheses, predictions, decisions, typed outputs, usage, policy versions | Audit the model-development and workflow-improvement loops separately. |

The final checkpoint and raw final validation logits are always retained. With
`retain_epoch_checkpoints=true` and `retain_epoch_logits=true`, every completed epoch
is also retained. Hardlinks avoid duplicating the latest epoch where supported.
Turning either option off loses earlier weights/predictions; metrics alone cannot
recover them. Per-step gradients, activations, and optimizer trajectories between
epochs are not stored. Decide explicitly if a research question needs those.

Approximate tensor-only storage: a float32 MLP has `4P` bytes of weights for `P`
parameters; AdamW adds roughly `8P` bytes of moment tensors per checkpoint, plus
step counters and metadata. Raw validation logits need `4NC` bytes for `N` examples
and `C` classes, plus labels/row IDs. Multiply retained epochs by these quantities
and include every trial, staged datasets, logs, and serialization overhead.
These are analytic estimates, not measured disk use. CNN and text runs must also include staged raw inputs and per-trial token arrays
in their storage estimate before an expensive run.

## Scientific recipe boundaries

The implemented WSD variant uses step-based linear warmup, a stable LR, and terminal
linear decay. Rounding and endpoint conventions are defined in `learning_rate_factor`.
It is not WSD-S and does not implement checkpoint branch reuse. WSD on these MLP
tasks is an exploratory transfer, not an established advantage.

μP, EDM2, and BOHB appear as sourced knowledge cards, with applicability limits.
They are not implemented algorithms in this v1. An agent mentioning a paper does
not establish that a candidate faithfully implements it.

## A–D additions

See [tasks and comparisons](tasks-and-comparisons.md) for supported adapters and
exact preprocessing/model conventions. CIFAR-100 replaces ImageNet for this study.
The JSON templates keep budgets, targets, seeds, splitting choices, and retention
unresolved. Architecture fields are proposed starting points, not measured baselines.

Automatic/requested diagnostic analysis is charged to the shared execution-time
budget. Curves and time-to-target include analysis performed before each training
result; reports computed after reaching the target remain in total overhead.
Initial preparation time is recorded separately. API costs and calls include the
reviewer and diagnostic-request decisions. Comparison figures do not show API
latency as GPU execution time. Run order is explicitly listed per case.

CIFAR staging preserves uint8 [N,3,32,32] images, int64 labels, and source IDs.
Text staging preserves full input strings and source IDs; per-candidate token IDs,
attention masks, and type IDs are stored. The local pretrained source checkpoint
must remain available and match pinned hashes. No quantization is performed.
The compact diagnostic report is derived from full logits; it does not replace them.
An output-only report cannot recover images/text, weights, or prior epoch logits;
those objects are retained separately according to the declared artifact policy.

For storage planning: raw CIFAR images require 3072 bytes each; token arrays use
up to 3 × 8 × N × sequence_length bytes per candidate. Weight/optimizer estimates
above apply to total/trainable parameter counts respectively. Include every A–D
run, trial, retained epoch, raw source, and serialization overhead. Throughput and
ETA remain unmeasured until a separately approved pilot; do not infer authorization
for that pilot or for a full comparison from implementation permission.
