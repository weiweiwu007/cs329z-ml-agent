# Architecture and course connections

The model-development loop and workflow-improvement loop have separate records and
execution commands. Only the latter changes the planner's instructions.

| Course topic | Implementation | Course background |
|---|---|---|
| Typed agent roles | Strict Pydantic contracts and DSPy signatures | Lecture 5, p51 |
| Workflow orchestration | LangGraph nodes and conditional transitions | Lecture 5, p76 |
| Reliability/observability | Timeouts, call limits, append-only traces, failure records | Lecture 5, p125 |
| Context engineering | Rebuild focused context from immutable task constraints and selected evidence | Lecture 6, pp59–63 |
| Memory | Working context, full trial history, and versioned procedural instructions | Lecture 7, pp17–20 |
| Multiagent coordination | One proposing agent plus an optional reviewer sharing a task ledger | Lecture 8, p54 and p89 |
| Agent optimization | Candidate instruction revisions selected only through measured development evaluation | Lecture 9, p10 and p121 |

The lecture PDFs are in the course workspace's `Slides/` directory. The proposal
remains unchanged. This v1 now implements all three selected task families.

## Data flow

`TaskSpec` and `Policy` enter a `Workflow`. The trusted adapter stages training and
validation tensors and an exact starting model. The baseline is trained first and
counted. After each training attempt, B/D receive diagnostics and C/D call a reviewer. A guard node
checks target attainment and remaining resources before admitting the next plan.

The context builder includes immutable requirements, bounded current history,
best/recent results, relevant scientific method cards, and shared prior development
evidence. Full history remains on disk when context is compacted. Prior trials use
qualified citations (`run_id:trial_id`) to avoid confusing different runs' trial IDs.
Neither the context builder nor a role tool exposes final-test labels.

`improve` consumes development records and outputs a new policy file with parent
hash and cited evidence. `compare` gives both policies identical prior evidence,
counterbalances execution order, restarts the initial model, and applies a fixed
independent acceptance rule. A winning policy becomes a possible parent for a later
round; failures remain recorded. Mock mode exercises these paths without reasoning
or improvement claims.

## Extending model families

The task adapter contract is in `tasks/base.py`. The registry dispatches MLP, CIFAR ResNet, and BERT classification adapters.
Adding another family requires explicit data, model, and grading contracts. The shared role/history/budget/policy-comparison design can
remain. A different metric, objective, tokenizer, noise distribution, or sampler
must never silently replace the old one.

## Deliberate v1 boundaries

- No API keys, live model quality, or research benchmark results are bundled.
- No unrestricted generated-code execution, parallel GPU trials, vector database,
  automatic dataset downloads, GEPA optimization, or mid-epoch restart.
- Reflection proposes instructions; it does not approve its own performance.
- Conventional random search is implemented; BOHB remains a future stronger baseline.
- Scientific claims require live-agent experiments on approved research tasks,
  multiple paired runs, and independent final evaluation.

Implementation references: [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/quickstart),
[DSPy source](https://github.com/stanfordnlp/dspy),
[PyTorch](https://pytorch.org/docs/stable/index.html).

## A–D graph

`execute baseline → optional automatic diagnostics → optional reviewer → guard →
proposing agent → trusted recipe validator/builder → execute candidate`.

A/C have no diagnostic tool and receive no detailed reports. This is enforced by
the context builder, request handler, and diagnostic graph node. B/D automatically
receive reports after completed training experiments, within the compute budget;
their optional request tool can revisit a report without another training trial.
Automatic delivery does not require a positive request allowance.

The generic builder is deterministic code and never calls the LLM. Instructions,
context bounds, training tools, and budget settings are shared across A–D; detailed
report access and the reviewer are the declared experimental differences. Prior
evidence that used reports cannot be imported into A/C. Raw validation outputs
remain stored for later analysis in every configuration, outside agent access.
The variant and its explicit access settings are recorded in run identity and
cannot change during resume.
