# Decision record: A–D study (`abcd-v1`)

**Status: approved by the user on 2026-11-04.** The Spambase demo has run
(`docs/2026-11-04_mlp-demo_RESULTS.md`), and Covertype now replaces it as the MLP task (see the last section).
The whole study runs unattended from one command, using `configs/study/v1.json`. That approval was
the only one: there are no later human steps. The full four-task study has not been run.
Provenance: **User** = decided by the user in
chat; **Codex** = Codex's recommendation; **Claude** = default added on 2026-11-04.

## Goal

Which agent design turns a limited experiment budget into better models? Two switches, from
the TA's request to study design decisions rather than build another MLE system:

| Arm | Detailed error report | Separate reviewer | Decides with |
|---|---|---|---|
| A | No | No | LLM |
| B | Automatic | No | LLM |
| C | No | Yes | LLM |
| D | Automatic | Yes | LLM |
| Random (reference) | No | No | Random draws from its own ranges |

Report effect: B−A, D−C. Reviewer effect: C−A, D−B. Interaction: (D−B)−(C−A). **User**, 2026-11-04.

## Tasks and constraints

**User decision, 2026-11-04: we set only constraints. Agents get no menu.** As in real MLE work, the
rule setter fixes the resource budget and the agent decides everything inside it.

| Constraint | Spambase (MLP) | CIFAR-100 (CNN) | SST-2 and MRPC (BERT-Tiny) |
|---|---|---|---|
| **Maximum parameters** | 0.5M | 20M | 6M |
| **Experiments per run** | Up to 20, including the baseline. A guard, not the real limit | same | same |
| **Time per experiment** | 4 × the slowest measured baseline | same | same |
| **Total compute** | 9 × the mean measured baseline. **This is the real budget** | same | same |
| **Metric, data, grader** | Fixed | Fixed | Fixed |
| **Starting point only** | MLP 128→64, AdamW, lr 1e-3, batch 128, 50 epochs | ResNet-18, SGD, lr 0.1, batch 128, 100 epochs | BERT-Tiny, linear head, AdamW, lr 1e-4, batch 32, 4 epochs |

**The agent decides:** depth and width (any number of layers and units under the ceiling), training
length, optimizer, learning rate, batch size, momentum, weight decay, dropout, schedule and warmup,
label smoothing, class weighting, sampling, and input handling. Numbers can be any value the schema
allows. A pretrained encoder cannot change shape, so on text tasks the agent shapes the head, what
to freeze, and sequence length.

**Lists that remain, because they are the algorithms the code implements:** activation, optimizer,
schedule, normalization, pooling, feature preprocessing, augmentation, and class balancing.
Schema bounds are only there to keep values meaningful, such as dropout below 1 or at most 16 hidden
layers. Fields a model family does not use stay inactive.

**What compute counts.** Training, validation, and checkpoints. Process startup is recorded but not
charged, because on a small task it would otherwise be most of the budget and hide the choice between
long and short runs. A run ends when under 10% of a baseline's compute remains.

**Compute is the real budget (user decision, 2026-11-04).** An agent can spend it on a few long runs
or on many short ones to screen ideas, which is the tradeoff a human engineer faces. Attempts past the
first few are shown in a compact table of every attempt, so none are forgotten.

**Over a limit is not an experiment.** Checking a model against the ceiling is free in real work,
so a recipe that breaks a rule goes back to the agent, up to twice per decision, without using an
attempt. After that the attempt is lost. A trial that runs past its time limit is stopped and counts.

**Efficiency.** Agents are told to raise the score and, among models scoring about the same, prefer
fewer parameters and less compute. Results report score and size side by side (`fig6`, `runs.csv`)
rather than combining them into one score, because any weighting would be an arbitrary choice.
The selected model is still the one with the best validation score.

**Random search needs ranges to draw from.** These belong only to that reference arm, are never shown
to agents, and sit in the protocol as `random_search`. They cover conventional ranges centered on the
same starting points, with learning rates that suit each optimizer (AdamW 1e-4 to 1e-2 and SGD 1e-3
to 0.3 on Spambase). Draws are redrawn if over a limit, or if they need over 2.5 × the baseline's
training steps, because this arm cannot read timings as agents do. Status: Claude's default.

## Data

| Task | Train / validation / test | Source and pin | Status |
|---|---|---|---|
| CIFAR-100 | 45,000 / 5,000 / 10,000 | Official binary; stratified split of official train, seed 20261104 | User chose the dataset; split is Claude's |
| Spambase | 60 / 20 / 20% of 4,601 | UCI 94; stratified, same seed. Duplicate rows kept | Claude |
| MRPC | 3,668 / 408 / 1,725 | `nyu-mll/glue@bcdcba7`, as published | Claude |
| SST-2 | 67,349 / 872 / 1,821 | GLUE; **test labels from `SetFit/sst2@00ea8cc`** by exact text | Claude; **needs decision** |
| BERT-Tiny | | `google/bert_uncased_L-2_H-128_A-2@30b0a37`, safetensors SHA-256 pinned | Claude |

GLUE withholds SST-2 test labels. Preparation stops if any sentence fails to match exactly.
Alternative if unwanted: split the 872 validation rows in half (a much noisier score).
Validation splits are small on text (408 and 872 rows), so one row moves accuracy 0.1–0.25%.

## Execution

| Setting | Value | Status |
|---|---|---|
| Attempts per run | Up to 20, including baseline; compute usually ends a run first | User |
| Seeds | 1, 2, 3, same across arms | Codex count; values Claude |
| API calls per run | 80 limit, to leave room for 20 attempts, reviews, and recipe repairs | Claude |
| Optional report requests | 0; B/D still get reports automatically | Codex |
| Stop at target | **No**: keeps complete curves; time to target is still derived. An agent's request to stop is logged and ignored, since random search never stops | Claude; changes Codex's default |
| Shared baseline | Trained once per task/seed; each arm charged its measured cost | Claude |
| Random-search arm | Included; about 20% more training | Claude; in the original proposal |
| Devices | All tasks on GPU, one job per GPU | User, 2026-11-04 |
| Precision | FP32, deterministic algorithms | Codex |

## What agents are told

**User decision, 2026-11-04: agents should think like human engineers.** `configs/policy.json` tells
them to diagnose what limits the model, judge which settings matter most, try values far apart until
they know where a setting works, refine only when measurements justify it, and spend the budget
deliberately. It says how to think and gives no values, so it stays consistent with constraints only.

All arms get identical instructions. Beyond the policy file, they are told facts that affect their
choices: the constraints above; attempts continue until the cap or until compute runs out; all attempts share one seed, and initial weights when the shape is unchanged, so
an unchanged recipe reproduces its result and adds nothing; and each task's general knowledge cards
(experiment discipline, WSD, BOHB), which originally reached only Spambase through a naming gap.

## What is stored

| Object | Kept | Not kept |
|---|---|---|
| Model weights | Initial and final-epoch, with optimizer and RNG state | **Weights at earlier epochs** |
| Validation outputs | Raw logits, labels, and IDs at every epoch | |
| Text inputs | Token IDs per trial; raw text once per run | |
| Agent records | Every context, output, plan, review, report, usage | |
| Training telemetry | Per-epoch loss and metrics, steps, examples, peak GPU memory | Per-step gradients, activations |

Later analysis without rerunning: any metric, threshold, or calibration at any epoch; all behavior
analysis. Not possible without rerunning: per-class analysis of intermediate weights, or continuing
training from an earlier epoch. Because agents now choose training length and sequence length, disk
use depends on their choices. At the starting settings the full study needs about 80–95 GB; if agents
often train much longer, it could reach a few hundred GB. Provision 300 GB for the full study. The
20 GB disk guardrail stops new jobs before the disk fills. The Spambase demo needs under 5 GB.

## Cost and time

**Training runs now vary,** because the agent chooses how to spend its compute. At most 20 per run,
though normal-length experiments use the budget after about 9. GPU time stays bounded by the budget:
9 × the baseline per run, across 5 arms and 3 seeds, for each task.

**Time is unmeasured until the baselines run.** Assuming 20–40 minutes per CIFAR baseline on a current
GPU, the full study needs roughly 45–90 GPU-hours, almost all CIFAR: about 2–4 days on one GPU, or
12–24 hours on four. The Spambase demo should take about 1–2 hours, mostly waiting on the model.

**API.** Measured on 2026-11-04: 5 calls on `gpt-6-luna` cost $0.0022, about 3,300 input and 900 output
tokens each. Later prompts are larger, and costs below are estimates. Key credit: about $199.

| Model | Price per 1M in / out | Spambase demo | Full study |
|---|---|---|---|
| `openai/gpt-6-luna` (chosen) | $0.10 / $0.50 | under $1 | $1–3 |
| `openai/gpt-6-sol` | $2 / $10 | $3–10 | $15–40 |
| `anthropic/claude-opus-5.5` | $4 / $20 | $6–20 | $30–80 |

## How it runs: no human in the loop

```bash
.venv/bin/python -m rsi_ml autopilot --protocol configs/study/v1.json --data data --out studies/abcd-v1 --env-file ~/.config/rsi_ml/api.env --execute
```

Data → baselines → targets and budgets → all agent runs → figures → test scores. Rerunning the same
command continues where it stopped and never repeats finished work. Progress is in
`studies/abcd-v1/autopilot.json`.

| Decision | Automatic rule, fixed before any agent runs | Status |
|---|---|---|
| Target | Remove 10% of the baseline's remaining error: median + 0.10 × (1 − median) | Claude |
| Per-trial time limit | 4 × the slowest baseline | Claude |
| Total compute limit | 9 × the mean baseline, so slow choices have a real cost | Claude |
| Model | `openrouter/openai/gpt-6-luna`, provider-default reasoning | User, 2026-11-04 |
| API check | One real call shaped like the largest prompt, before any training; about $0.05 | Claude |
| A job fails | Recover and retry. It gives up only after 5 passes in a row finish nothing, waiting 10 to 60 minutes between them | Claude |
| Test scoring | After all runs, on the validation-selected model; agents never see it | Claude |

**Guardrails, the only reasons it stops by itself.** Running jobs finish, nothing new starts, and
`autopilot.json` says why. Figures are still written.

| Guardrail | Limit |
|---|---|
| Free disk | Under 20 GB |
| API spending | $100, far above the estimated $1–3 for the full study |
| The API check fails 3 times | Key, credit, or model is unusable, or answers are cut off. Stops before any GPU time |
| A baseline cannot finish | That task has no valid starting point |

**A study's settings are frozen at its first run.** The policy and task templates are copied in.
Rerunning with different settings is refused, so one study cannot mix versions; use a new `--out`.
Adding seeds is the exception: new seeds train and keep the targets already fixed. Guardrail limits
can also be changed.

**Machine faults are not agent decisions.** A lost GPU, a full disk, a killed worker, or a provider
outage stops that run at a resumable point and consumes no attempt. A broken GPU slot stops taking
work so healthy ones finish the queue. Out-of-memory counts as that attempt's failure, because it
depends on the model and batch size chosen, on hardware every arm shares.

**What agents can do to the machine: nothing.** They return schema-validated recipe values. They
have no shell, file, or network tools, and training is trusted code, so an agent cannot delete data
or touch the VM. The pipeline never deletes run artifacts; partial work is moved aside.

## The go-ahead covered

- The starting points above. The constraint design itself is the user's own later decision.
- SST-2 test labels from `SetFit/sst2`. Alternative: half of validation, a much noisier score.
- The random-search arm, and running past the target.
- The automatic rules and the guardrails above. The model was changed afterward by user decision.

**Scientific risks to accept or address**

- **The model choice is a stated hypothesis (user decision, 2026-11-04, before any result).** A cheaper
  model has more room to benefit from reports and a reviewer, so A–D may separate more clearly. It is
  not guaranteed: a model too weak to use the extra evidence would flatten the differences instead.
  Findings apply to this model tier. To stay honest, do not switch models after seeing results and
  report only the clearest one; if a second model is run, report both.
- **It may still read as LLM-guided hyperparameter search.** Agents now choose numbers and model
  shape freely, but they still cannot write code. Random search is the honest check: if it ties A–D,
  that is the finding.
- **Attempt counts will differ between runs.** Compute ends most runs, so an arm that screens with
  short runs tries more ideas. That is intended: read results against compute, not experiment number.
- **Three seeds is thin.** The TA named variance directly, and a whole agent run is one sample. Seeds
  can be added later without rerunning anything, and the cheap tasks can take far more.
- **Agents versus random search is not a pure comparison.** Agents are told to use little compute,
  and random search ignores that. The A–D contrasts all share the wording and are unaffected.
- **This study does not test self-improvement,** the proposal's original central question. That is
  a later stage using the frozen best workflow. The existing `improve` command is a reflective
  revision loop, not GEPA.

## Covertype MLP run (`mlp-covertype`), 2026-11-04

**Why:** on Spambase the starting model left about 0.2 test points to gain, and one validation row was
worth 0.11%, so the designs could not separate. **User** decided to replace it with a harder MLP task and to
cap the run at about 2 hours. `configs/study/v1.json` still lists Spambase; update it if the full study runs.

| Decision | Setting | Source |
|---|---|---|
| Dataset | UCI Covertype, dataset 31, `covertype.zip` sha256 `89a975c2…105f6d`; 581,012 rows, 54 features, 7 classes | **User** chose the task; **Claude** chose the UCI copy |
| Split | Stratified 60/20/20, split seed 20261104, duplicates kept; labels renumbered 1–7 to 0–6 | Same rule as Spambase |
| Preprocessing | Standardized from the training split unless an agent chooses otherwise | Same as Spambase |
| Starting model | 54→128→64→7 (15,751 parameters), same recipe as the demo except epochs | Same as Spambase |
| Starting epochs | **5, down from 50** | **User** lowered compute, the limit that actually binds |
| Compute budget | Same rules: 9× the starting run per run, 4× per attempt, set from measured times | Same as demo |
| Random search | Same ranges, with epochs 3–10 instead of 25–100, the same 0.5×–2× ratio | Follows starting epochs |
| Other limits | 500k parameters, 20 attempts, 1000-epoch limit, 3 seeds, A–D plus random, `gpt-6-luna`, accuracy | Same as demo |
| Starting run ceiling | 1800 s, a safety limit for the starting run only | **Claude** |
| Stored | Each attempt's validation logits per epoch (float32), labels, row IDs, final checkpoint, curves, all agent inputs and outputs | Same as demo |
| Not stored | Per-epoch model weights and training-set outputs | Same as demo |
| Location | `/home/ml-agent`, because the VM's root disk is wiped on pause | **User** |

The published tuned MLP reaches 96.2% on this data ([Gorishniy et al. 2021](https://arxiv.org/abs/2106.11959)),
using a different split and far more training, so it is context and not a target. The run's target comes
from the usual rule. A 5-epoch start is deliberately cheap, so the first gains will come from simply
training longer; gains after that are what could separate the designs. Measured on two L4s, one per run:
34 minutes, 7.1 GB, and $0.19. Results are in `docs/2026-11-04_covertype_RESULTS.md`.

## Covertype with more compute (`mlp-covertype-600s`), 2026-11-04

**Why:** the 168-second budget fit only about two long attempts, so one early decision settled most runs. After
seeing that result, **User** judged it too tight and asked for 600 seconds per run. This is a separate study; the
168-second run stays as it is, and both should be reported. Compute was widened after seeing results, so this run
is a follow-up to the first, not a replication of it.

| Decision | Setting | Source |
|---|---|---|
| Compute per run | **600 s, up from 168 s**, set directly instead of derived | **User** |
| Compute per attempt | 267 s, up from 79 s, keeping the existing 4:9 ratio to the run budget | Same rule |
| Starting recipe | Unchanged, including 5 epochs, so the budget is the only change for A–D | Same as the 168-second run |
| Random search | Epochs 9–36 and a step cap of 9× the starting recipe, up from 3–10 and 2.5× | Same rule: 600 s is 9× what about 18 epochs take, so 0.5×–2× of 18 epochs and a cap of 2.5× those 18 |
| Target | Derived by the usual rule from the retrained starting models | Same rule |
| Everything else | Dataset, split, 500k parameters, 20 attempts, 3 seeds, model, storage, two L4s with one per run | Same as the 168-second run |

Measured: 95 minutes, 17 GB, and $0.39, with no run reaching the 20-attempt limit. Results are in
`docs/2026-11-04_covertype-600s_RESULTS.md`.
