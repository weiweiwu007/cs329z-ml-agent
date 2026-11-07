# ML development agents — v1

One shared harness supports MLP, CIFAR-100 CNN, and BERT text classification.
A–D independently toggle detailed error-report access and a separate reviewer.
A/C cannot access reports; B/D receive them automatically. A/B have no reviewer;
C/D do. Training and grading remain trusted code; DSPy supplies typed decisions
and LangGraph controls the loop.

**Current state:** the complete pipeline has run on Spambase and Covertype with real GPU
training and model API calls. Covertype results at 168-second and 600-second budgets are in
`docs/2026-11-04_covertype_RESULTS.md` and `docs/2026-11-04_covertype-600s_RESULTS.md`.
CNN and BERT adapters exist; their real-data study runs remain future work.

## Midway report

[Read the report](reports/midway/main.pdf) or edit [the LaTeX source](reports/midway/main.tex).
The report has three main-text pages plus references and disclosure. It includes the
A–D comparison, saved Covertype results, failure examples, and remaining work.
See [build and review notes](reports/midway/README.txt).

## Install and open the demo

Use Python 3.12 or newer. From a standalone clone:

```bash
git clone https://github.com/weiweiwu007/cs329z-ml-agent.git
cd cs329z-ml-agent
python3 -m venv .venv
.venv/bin/python -m pip install -c requirements-lock.txt -e '.[all,dev]'
mkdir -p ~/.config/rsi_ml
cp .env.example ~/.config/rsi_ml/api.env
chmod 600 ~/.config/rsi_ml/api.env
```

Edit `~/.config/rsi_ml/api.env` and replace `REPLACE_WITH_YOUR_OPENROUTER_KEY` with your
own OpenRouter API key. Do not commit this file. The demo uses the model configured in
`configs/study/mlp-covertype-600s.json`, currently `openrouter/openai/gpt-6-luna`.
The provider account must have access to that model. Setup replies incur API charges.

```bash
.venv/bin/python -m rsi_ml.intake --port 8766 --env-file ~/.config/rsi_ml/api.env
```

The server reads `OPENROUTER_API_KEY` from the specified file; the browser never receives
the key. This screen prepares a configuration without training or downloading a dataset.
Visit `http://127.0.0.1:8766`, click **New experiment**,
and type these six messages when prompted. There are no recommended-reply buttons.

1. “Improve an MLP that predicts forest cover type from terrain and soil features.”
2. “Use UCI Covertype.”
3. “Predict one of the seven forest cover types, measured by accuracy.”
4. “Use the existing 60/20/20 split.”
5. “600 seconds of compute, at most 500,000 parameters.”
6. Review the mission and say **Start**.

The chat saves a new `configs/study/intake-*.json` in the same protocol format used by
`autopilot`, plus a transcript and config hash in `intake_sessions/`. It ends with
**“Starting…”** at the handoff boundary: **optimization is not launched by this screen**.
The reviewed mission is D with seed 2, the existing starting recipe, and your resource limits.
The config can also be inspected or downloaded from the page. Existing studies are untouched.

The model interprets your messages and writes its replies within a fixed six-step Covertype
setup. The server validates structured replies and resource limits, then builds the config from the existing demo protocol.
Source metadata comes from the registered task definition; the GPU pipeline subsequently
fetches or reuses the pinned dataset and constructs its splits. This screen does not accept
dataset uploads. Unknown tasks, alternative metrics and unsupported splits require clarification.
The browser setup is safe on the Mac; the downstream optimizer and training tests run on the VM.

To hand off later, sync the new config with the same project checkout and use the existing
`autopilot --protocol configs/study/intake-<id>.json ... --execute` entry point on the VM,
with a new output directory. Credentials and hardware setup remain outside the conversation.
See [the demo setup notes](docs/demo-input.md) and [the six-message script](docs/demo-input-script.txt).
A six-exchange browser walkthrough completed with six real API calls and a saved config.
The saved settings were read back: D, seed 2, 600 seconds, and 500,000 parameters.
The walkthrough artifact is marked verification-only, not permission to run training.

## Reproduce the saved Covertype plots

The repository includes the exported CSV tables from the completed 168-second and
600-second studies. These commands regenerate the A–D curves from those measurements;
they need no API key, dataset, GPU, or training run:

```bash
.venv/bin/python scripts/abcd_curves.py studies/mlp-covertype/figures --x-max 168 --out output/reproduced-covertype-168s
.venv/bin/python scripts/abcd_curves.py studies/mlp-covertype-600s/figures --x-max 600 --out output/reproduced-covertype-600s
```

Each output directory contains `fig_abcd_curves.png`, `fig_abcd_curves.svg`, and
`abcd_curves.csv`. The input is `quality_vs_compute.csv`: X is cumulative training,
evaluation, and diagnostic compute; Y is the best validation accuracy so far, averaged
over three seeds. `final_test.csv` contains final held-out scores, and `decisions.csv`
contains each proposed change and its measured outcome.

These are plots of existing experiments, not new evaluations. The full checkpoints,
raw validation outputs, and agent traces remain outside this repository. To collect new
results, use the GPU workflow below. The random-search results have a documented
class-weighting/label-smoothing limitation; see the [600-second results](docs/2026-11-04_covertype-600s_RESULTS.md).

## Run the study on a GPU VM

**Never train on a laptop, including this test suite: it trains tiny models for real.**
Everything below runs on the VM, unattended. The [decision record](docs/DECISION_RECORD.md) lists the
settings, automatic rules, and guardrails.

Clone this repository on the GPU VM and create `~/.config/rsi_ml/api.env` there using
the instructions above. Keep the project, data, and output on persistent VM storage.

```bash
bash scripts/vm_setup.sh    # Installs, runs the test suite, checks GPU determinism.
# Explicitly start a new 600-second Covertype study (15 runs: A-D + random, 3 seeds).
.venv/bin/python -m rsi_ml autopilot --protocol configs/study/mlp-covertype-600s.json --data data --out studies/new-covertype-600s --env-file ~/.config/rsi_ml/api.env --execute
```

That one command downloads data, makes one real API call to prove the key and model work, trains
baselines, sets targets and budgets from them, runs every agent, and writes figures and test scores. It recovers from crashes and provider outages by itself.
Rerun the same command to continue after anything stops it. `study-status` and `study-figures` are
safe to run at any time. `study-run` runs one stage or an exact subset by hand.

To reproduce the shorter study, use `configs/study/mlp-covertype.json` and a separate
output directory. `configs/study/v1.json` contains the broader CNN/text plan; those
real-data studies have not produced results yet. Exact training times can vary with
hardware, so new runs need not match the saved scores. Record the GPU and software versions.

**Results:** the new study's `figures/` directory holds each figure as SVG and PNG, with a CSV twin.

| Figure | X-axis | Y-axis |
|---|---|---|
| `fig1_quality_vs_compute` | Cumulative training + analysis time | Best validation score so far |
| `fig2_quality_vs_experiments` | Experiment number | Best validation score so far |
| `fig3_cost_to_target` | Arm | Experiments and time to reach the target; misses shown at the budget |
| `fig4_experiment_outcomes` | Arm | Share of decisions that improved, tied, regressed, or failed |
| `fig5_agent_behavior` | Kind of setting; arm | Share of decisions changing it; weakest-class recall change |
| `fig6_score_vs_parameters` | Model parameters | Validation score of every completed experiment |
| `timeline_<task>_seed-<s>` | Experiment number | Each arm's sequence of changes and outcomes |

Color shows the error report (blue none, orange present). Line and marker show the reviewer
(solid circle none, dashed diamond present). `contrasts.csv` has the paired differences by
seed, and `decisions.csv` has one row per decision, including the agent's stated reasoning.

## Inspect the plan without running anything

```bash
.venv/bin/python -m rsi_ml compare-designs --suite configs/abcd-suite.template.json
.venv/bin/python -m rsi_ml preflight --task configs/cifar100.template.json --backend dspy
```

The first command only prints the proposed cases, variants, budgets, and unresolved
settings. Preflight exits nonzero for draft/incomplete tasks. Neither command
constructs an API backend, trains, stages data, or downloads anything.

| Configuration | Task | Starting model |
|---|---|---|
| `configs/spambase.template.json` | Numeric classification; selected OpenML tasks use separate copies | MLP 128→64, random weights |
| `configs/cifar100.template.json` | CIFAR-100 image classification | CIFAR-adapted ResNet-18, random weights |
| `configs/sst2.template.json` | Sentiment | Pinned pretrained BERT-Tiny + new head |
| `configs/mrpc.template.json` | Paraphrase detection | Same encoder; separate fine-tuning |

[Tasks, allowed changes, A–D, metrics, and exact implementation conventions](docs/tasks-and-comparisons.md).
[Protocol decisions and retained artifacts](docs/protocol.md).

## Software checks (GPU VM only)

```bash
# After installation on the VM:
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
```

Tests use tiny synthetic CPU fixtures, a locally generated toy BERT checkpoint,
stubbed execution receipts, and a local stand-in for the API. They are software validation,
not project results. No API key or downloaded dataset/checkpoint is needed. `-m "not slow"`
skips the 80-second whole-pipeline test. The retained
MLP `demo` command trains a tiny synthetic fixture with scripted decisions; it is
not part of A–D research evaluation and has not been run for this implementation update.

## Data preparation after settings are finalized

The local importer supports the official CIFAR-100 **binary** files, headerless
Spambase numeric CSV, numeric OpenML exports, and labeled SST-2/MRPC TSV files.
It does not download sources, invent train/validation/test splits, fit on test
labels, or modify the task configuration automatically.

Copy the relevant `configs/data/*-import.template.json`, supply source hashes,
revision, selection rule/seed, and explicit row-index lists (or paths to JSON
index arrays). `all` is allowed only when the entire source is intentionally one
split. Source IDs cannot overlap between splits. A finalized import is explicit:

```bash
.venv/bin/python -m rsi_ml import-data --plan finalized-import.json --out data/selected --execute
```

This produces raw split files and a manifest with hashes and source indices.
Use those identities/paths/row counts in a task config. CIFAR stays uint8; text
stays unmodified JSONL; numeric CSV retains the selected feature values. Keep the
original source files as well. Public GLUE test files without labels cannot be
used as a scored test set; select an explicitly held-out labeled partition.

For text, obtain the chosen encoder/tokenizer separately, pin its immutable
revision and every local file SHA-256 in `text_model`, and use safetensors weights.
All model/tokenizer loading is local-only. Missing files are an error, never an
implicit hub download. Checkpoint selection/download has not been performed.

## Running one task by hand

Fill every required setting: baseline training recipe, search space, splits,
targets, seeds, GPU/CPU choice, trial/parameter/epoch/time limits, API and diagnostic
request budgets, and artifact retention. The optional request allowance may be
zero: B/D still receive automatic reports, while A/C never gain report access.
Mark the task and suite `ready` only
after review. The suite also requires an explicit A–D order for every task/seed.
Set `RSI_MODEL` and `RSI_API_KEY` in the shell when provided; see [API setup](docs/api.md).

Future authorized commands:

```bash
.venv/bin/python -m rsi_ml run --task finalized-task.json --policy configs/policy.json --backend dspy --variant A --out runs/new-case-A --execute
.venv/bin/python -m rsi_ml compare-designs --suite finalized-suite.json --policy configs/policy.json --backend dspy --out runs/new-comparison --execute
.venv/bin/python -m rsi_ml resume runs/new-case-A --execute
.venv/bin/python -m rsi_ml score-test runs/new-case-A --trial trial-0002 --out final-test.json --execute
```

Without `--execute`, the suite command remains read-only and the other execution
commands refuse to start. Missing API credentials fail before training preparation.
No `.env` file is automatically loaded. Budget changes require a new run; resume
requires unchanged protocol, variant, backend, code/environment, and prior memory.
It resumes completed graph boundaries, not interrupted optimizer steps.

## Results and costs

Every run retains `task.json`, `policy.json`, `manifest.json`, `events.jsonl`, raw
validation logits/labels/IDs, checkpoints, recipes, learning curves, and source
fingerprints. Raw text/images are staged once per run; token IDs are also retained
for text candidates. Epoch artifact retention is an explicit research setting.

`comparison.json` contains the four contrasts and all costs/outcomes. Per-case SVGs
show score versus execution time, target-attainment cost, and experiment outcomes;
CSVs retain plot data. Failed/rejected/timeout attempts are counted, target misses
are explicit, and diagnostics consume the shared execution budget. Preparation,
API usage, and diagnostic time are separately visible. Execution seconds are wall
time, not measured GPU kernel utilization; tasks/devices are never pooled blindly.

The original `improve` and policy `compare` commands remain available behind
`--execute`, using development evidence only. They use reflective instruction
proposals, not GEPA, and do not run as part of A–D comparison.
