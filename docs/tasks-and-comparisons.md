# Tasks and the A–D comparison

This implementation follows the agreed class-project scope. It contains no
research results. All four task configurations and the comparison suite remain
**draft**: missing settings must be finalized before execution.

| Task | Version-zero architecture | Supported recipe edits | Metric |
|---|---|---|---|
| Covertype (replaced Spambase, 2026-11-04) | input → 128 → 64 → classes; ReLU; random initialization | Depth/width, activation, LayerNorm, dropout, raw/standardized/signed-log-standardized features, class weighting, sampling | Accuracy; macro-F1 also reported |
| CIFAR-100 | ResNet-18: stages [64,128,256,512], blocks [2,2,2,2], 100 outputs; random initialization | Stage widths/block counts, optional classifier hidden layers, dropout, augmentation, label smoothing, class weighting | Top-1 accuracy; top-5 also reported |
| SST-2 | Local pinned BERT-Tiny encoder, 2 layers / hidden size 128, linear classifier | Classifier hidden layers, CLS/mean pooling, dropout, freezing layers/embeddings, sequence length, sampling, class weighting | Accuracy |
| MRPC | Same BERT-Tiny initialization; separate task fine-tuning | Same text edits | Positive-class F1 (label 1); accuracy also reported |

Every family also supports AdamW/SGD, SGD momentum, learning rate, constant/cosine/
linear-WSD schedules, batch size, epochs, and weight decay. **The study sets only
constraints (`recipe_policy: constraints`):** agents choose any value the recipe schema
allows, within the parameter, time, and attempt limits. Data, labels, graders, targets,
starting checkpoint identity, and resource limits are outside agent control. A declared
`search_space` menu remains only for software demos and tests.
Changing to another pretrained encoder is a separate protocol, not a recipe edit.

Version zero is an architecture **and training recipe**. The first counted
experiment measures that recipe. MLP/CNN weights start randomly; only the BERT
encoder is pretrained. Each candidate restarts from exact saved initial weights
when tensor shapes match. Changed MLP/CNN shapes get seeded fresh initialization;
a changed text head gets the same pinned pretrained encoder plus a seeded head.
There is no migration of trained weights or automatic checkpoint continuation.

## Two independent workflow choices

| Variant | Proposing agent | Separate reviewer | Detailed error report |
|---|---|---|---|
| A | One agent proposes a complete validated recipe | No | Unavailable |
| B | Same | No | Supplied automatically |
| C | Same | Yes | Unavailable to both agents |
| D | Same | Yes | Supplied automatically to both agents |

The trusted recipe builder is code, not another LLM agent. All variants see the
same ordinary scores and training/validation curves (up to ten evenly spaced epochs:
loss, running train accuracy, validation metrics, learning rate). Agents are blind to
the design: identical prompts, no arm label, and no other role's instructions.

B/D automatically also receive a report calculated from raw validation logits: named
classes with the lowest recall, largest named confusions, calibration on errors, and up
to ten most confident mistakes. Text tasks show the mistaken text; tabular tasks show the
features that most distinguish errors from correct rows. The complete report on disk
adds the confusion matrix and per-class table. Error causes remain hypotheses.

**A/C cannot access detailed reports.** Their context contains no diagnostic tool
or report, including cached reports. Planner/reviewer requests are rejected before
analysis, even if the configured request allowance is positive. The diagnostic
graph node enforces the same restriction. Prior evidence from report-enabled runs
or older runs that requested reports is rejected for A/C before data preparation;
its prose could otherwise disclose the excluded evidence.

**B/D receive a report after each completed experiment**, before the next agent
decision, subject to the remaining compute budget. Automatic delivery does not
consume the optional `max_diagnostic_requests` allowance, which may be zero. The
bounded request tool is offered only when that allowance is positive, so an impossible
tool cannot lure B/D into a rejected plan that A/C could never make. Requests consume the
allowance and API calls; all report analysis time is charged to the compute budget. Missing/changed outputs produce a logged tool failure,
never invented diagnostics. Component ablation training experiments are not
automatically inserted by any variant.

Raw logits, labels, IDs, and curves are retained under the same artifact policy in
every configuration for later human analysis. Storage does not grant agent access:
agents receive typed context and cannot browse trial files or execute arbitrary code.

Same policy, LLM/backend settings, initial state, data, training tools, search space,
GPU/CPU allocation, API limits, and prior history are required across A–D.
Detailed report access and the reviewer are the two intentional differences.
The suite explicitly specifies each task/seed and each A–D execution order.
No automatic workflow optimization occurs inside this comparison.

An unknown citation in a plan is logged and removed, not treated as a failed experiment.
With `stop_on_target: false`, a plan's request to stop is logged and its recipe still runs.
Machine and provider faults (a lost GPU, a full disk, a killed worker, an outage) stop the run
at a resumable point and consume no trial. Malformed role output, a rejected recipe, a nonfinite
loss, and out-of-memory each count as that attempt's failure, with the reason recorded.

## What the comparison exports

- B−A and D−C: effect of access to automatically supplied detailed error reports.
- C−A and D−B: effect of a separate reviewer.
- For every task/seed: score versus cumulative execution plus diagnostic seconds,
  seconds to target (misses hatched at the cap), and experiment-outcome counts.
- JSON receipts include attempts to target, API usage/cost, actual costs,
  initialization/data hashes, failures, and exact repeated-recipe counts.
- Standalone SVG and CSV files accompany each task/seed. There is no pooled
  average mixing different datasets, metrics, targets, or devices.

An improvement/regression is measured against the best preceding completed
candidate. The first successful candidate is labeled baseline. An exact recipe
repeat is a separate flag. These are descriptive measurements, not claims about
causality or statistical significance. Final test scoring is a separate supervisor
command; test results never enter agent context.

## Implementation conventions to review before a run

These conventions are implemented explicitly and recorded by source hashes. They
are not validated research choices until the settings are finalized:

- CIFAR uses a 3×3, stride-1 stem and no initial max-pool. Residual stages use
  BatchNorm; stage transitions downsample by stride 2. Global average pooling
  feeds the classifier. Raw storage is uint8 NCHW at 32×32; normalization mean/std
  must be supplied explicitly. No resizing is performed.
- `crop_flip`: reflect-pad by 4, uniformly choose a 32×32 crop, horizontal flip
  with probability 0.5. `crop_flip_color` also multiplies brightness by a sampled
  factor in [0.8,1.2], clipping to [0,1]. Evaluation has no augmentation.
- Text uses the pinned local tokenizer, fixed-length padding and longest-first
  truncation. `cls` takes the first token directly; `mean` averages all unmasked
  token states. BERT's original pooler is unused. Encoder dropout follows the
  recipe. Freezing prevents parameter updates; normal train-mode dropout remains.
- Compute is FP32 with deterministic Torch algorithms and one CPU thread per
  training worker; CUDA has no silent CPU fallback. Batches, including the final
  short batch, are retained. Models and optimizer/RNG state are saved each epoch.
- Balanced weights use training-label counts only. Validation always uses plain,
  unsmoothed cross-entropy and the original class labels, independent of the
  training loss. No other preprocessing is silently applied.

## Sources

- [Torchvision ResNet topology](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html).
- [BERT-Tiny model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2).
- [BERT implementation interface](https://huggingface.co/docs/transformers/model_doc/bert).
- [CIFAR data formats](https://www.cs.toronto.edu/~kriz/cifar.html).

The CIFAR stem and bounded recipe controls are project adaptations. They are
separate from a claim of reproducing an official training recipe or result.
