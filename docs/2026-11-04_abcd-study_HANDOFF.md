# Handoff: from Codex chat to Claude, 2026-11-04

Source: Codex chat "Develop agent engineering proposal" (`01a11fde-c242-7830-ba24-2187019214c3`),
through 2026-11-04 10:32 UTC. Codex's reasoning and compaction summaries are encrypted and unread.

## Where the project landed

**Question:** which agent design turns a limited experiment budget into better models?
A 2×2 over MLP/Spambase, CNN/CIFAR-100, and BERT-Tiny/SST-2 and MRPC: detailed error report
(off in A/C, automatic in B/D) × separate reviewer (off in A/B, on in C/D).

## How it got here

| When (UTC) | Who | Decision |
|---|---|---|
| Oct 26 | TA | MLE agents are saturated. Study design decisions, not a basic system. Mind variance. Characterize behavior beyond aggregates. |
| Oct 26 | User | Variance is routine: run more seeds. Focus on what the agentic system does. |
| Oct 30 | User | **Rejected** the checkpoint-migration idea as unnatural and hacky. Reset to proposal + TA comment. |
| Oct 30 | User | "This is a class research project." Wants clear results and plots; borrowing MLE-STAR's intuition is fine. |
| Nov 1 | User | CIFAR-100 replaces ImageNet. Version zero is not necessarily pretrained. |
| Nov 4 | User | A/C cannot access reports at all; B/D receive them automatically. |
| Nov 4 | User, to Claude | The pipeline is fully automated: minimal approval, guardrails only. Never train on the Mac. |
| Nov 4 | User, to Claude | Approved all defaults in the decision record ("Yes, use defaults"). |
| Nov 4 | User, to Claude | Up to 20 attempts per run, with compute as the real budget. |
| Nov 4 | User, to Claude | Use the cheaper `gpt-6-luna`, expecting harness differences to show more clearly. Declared before any result. |
| Nov 4 | User, to Claude | Agents must show an engineer's judgment: find what matters, tune coarse to fine, make educated choices. The system prompt should encourage this. |
| Nov 4 | User, to Claude | Replaced fixed menus with constraints only: "we as rule setter should not give the agent multiple choice." Agents decide depth, width, and all numeric settings under a parameter ceiling and compute budget. |

Abandoned, do not revive: learned experiment selector (overlaps Bayesian optimization), verification
as the research question, checkpoint migration, prompt optimization as the novelty claim.

## Codex's claims, checked

| Claim | Result |
|---|---|
| 103 offline tests pass | Verified |
| A/C are blocked from reports | Verified |
| All four arms get training/validation curves | **False in code:** only first and last losses. Fixed. |
| Ready for a GPU VM | **No:** sequential, no resume, no data pipeline, no study figures. Built. |
| Recipes and budgets | Recommended by Codex; the user never explicitly approved them. |

## What Claude changed

- **Design fidelity:** curves for all arms; reports with class names, mistaken text, and tabular feature
  contrasts (they were bare IDs); identical prompts across arms; an impossible tool removed from
  B/D; the baseline always visible; citation slips logged instead of costing a trial.
- **API:** retries, outages that stop rather than consume trials, exact provider cost, reasoning effort.
- **Execution:** `baselines.py`, `study.py`, `autopilot.py`; crash recovery; run identity independent of
  which GPU was used.
- **Data:** `prepare.py`, with pinned sources and recorded splits.
- **Results:** `figures.py` and six figure types with CSV twins.

## Verification state

- The last run on the Mac, before the user stopped local runs: **126 tests passed**, including the
  whole pipeline with real subprocess training on tiny fixtures.
- **Code changed after that run.** An independent review and a read-only multi-agent check (38 claims,
  27 confirmed, no test predicted to fail) led to fixes and new tests that have been checked only by
  static tools. `vm_setup.sh` runs them first. If a test fails there, fix it before starting.
- Left as is, by choice: `kill -9` of the autopilot alone can make a rerun collide with live jobs;
  Ctrl-C and tmux are safe. Machine faults have no per-trial cap, so a recipe that repeatedly kills its
  worker can leave a run unfinished. Spambase duplicates can straddle splits, which is documented.
- Five real `gpt-6-luna` calls ($0.0022): all parsed, and all four recipes were valid.
- **Not yet run:** any CUDA path, any real dataset, real BERT-Tiny loading. `vm_setup.sh` and the first
  autopilot stage are where these get proved.

## Next

1. The user supplies the VM details. The defaults are already approved.
2. Sync, run `vm_setup.sh`, then start `autopilot` under tmux.
3. Self-improvement, the proposal's original central question, is a later stage. It is untested by
   A–D, and the existing `improve` command is a reflective loop, not GEPA.
