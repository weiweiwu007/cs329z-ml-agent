"""Matched A-D suites and plots. Planning is read-only; execution is explicit."""

from __future__ import annotations

import csv
from pathlib import Path

from .evaluation import _MATCH_FIELDS, compare_runs, progress_rows
from .schemas import Experiment, RunSummary, WorkflowVariant, digest
from .storage import read_events, read_json, write_json

VARIANTS = {name: WorkflowVariant(name=name).settings() for name in ("A", "B", "C", "D")}


def unresolved(value, prefix="") -> list[str]:
    if value is None:
        return [prefix]
    if isinstance(value, dict):
        return [
            p for key, item in value.items() for p in unresolved(item, f"{prefix}.{key}".strip("."))
        ]
    if isinstance(value, list):
        return [
            p for index, item in enumerate(value) for p in unresolved(item, f"{prefix}[{index}]")
        ]
    return []


def plan_suite(path: Path) -> dict:
    """Readable draft status without importing optional models or constructing backends."""
    document = read_json(path)
    missing = unresolved(document)
    tasks = []
    for case in document.get("cases", []):
        task_path = (path.resolve().parent / case["task"]).resolve()
        task = read_json(task_path)
        tasks.append(
            {
                "path": str(task_path),
                "task_id": task["task_id"],
                "adapter": task["adapter"],
                "status": task["protocol_status"],
                "unresolved": unresolved(task),
                "limits": task.get("limits"),
                "seeds": case.get("seeds"),
            }
        )
    return {
        "mode": "plan_only",
        "training_launched": False,
        "api_calls": 0,
        "suite_status": document.get("protocol_status"),
        "unresolved": missing,
        "variants": VARIANTS,
        "cases": tasks,
        "note": "Null entries mark decisions to finalize; optional unused recipe fields may remain null after schema validation.",
    }


def behavior(records: list[Experiment], metric: str) -> dict:
    counts = {
        key: 0
        for key in (
            "baseline",
            "improvement",
            "regression",
            "tie",
            "failed",
            "timeout",
            "rejected",
            "repeated_recipe",
        )
    }
    best, seen = None, set()
    for record in records:
        if record.recipe is not None:
            key = digest(record.recipe)
            counts["repeated_recipe"] += int(key in seen)
            seen.add(key)
        if record.result.status != "completed":
            counts[record.result.status] += 1
            continue
        score = record.result.metrics[metric]
        if best is None:
            counts["baseline"] += 1
        elif score > best:
            counts["improvement"] += 1
        elif score < best:
            counts["regression"] += 1
        else:
            counts["tie"] += 1
        best = score if best is None else max(best, score)
    return counts


def compare_design_runs(summaries: dict[str, RunSummary], task) -> dict:
    if set(summaries) != set(VARIANTS):
        raise ValueError("A-D comparison needs exactly four variants.")
    first = summaries["A"]
    for label, summary in summaries.items():
        if summary.variant != label:
            raise ValueError("Summary variant differs from its assigned comparison cell.")
        mismatches = [
            key
            for key in (*_MATCH_FIELDS, "policy_hash")
            if key != "variant" and getattr(first, key) != getattr(summary, key)
        ]
        if mismatches:
            raise ValueError("Unmatched A-D runs: " + ", ".join(mismatches))
    contrasts = {}
    for before, after, question in (
        ("A", "B", "report_with_single_agent"),
        ("C", "D", "report_with_reviewer"),
        ("A", "C", "reviewer_without_detailed_reports"),
        ("B", "D", "reviewer_with_detailed_reports"),
    ):
        # Use the established accounting with the intentionally manipulated variant removed.
        a = summaries[before].model_copy(update={"variant": "A"})
        b = summaries[after].model_copy(update={"variant": "A"})
        result = compare_runs(
            a,
            b,
            compute_budget=task.limits.total_compute_seconds,
            trial_budget=task.limits.max_trials,
        )
        result["eligible_for_promotion"] = False
        contrasts[f"{after}_vs_{before}"] = {"question": question, **result}
    return {
        "task_id": task.task_id,
        "seed": task.seed,
        "metric": task.metric,
        "contrasts": contrasts,
        "runs": {k: v.model_dump(mode="json") for k, v in summaries.items()},
    }


def compare_designs(suite_path: Path, policy_path: Path, backend_name: str, output: Path) -> dict:
    from .cli import backend, load_policy, suite_cases
    from .workflow import Workflow, preflight

    document = read_json(suite_path)
    if document.get("protocol_status") != "ready":
        raise ValueError("A-D suite is draft; finalize settings before execution.")
    tasks = suite_cases(suite_path)
    orders = document.get("orders")
    if (
        not isinstance(orders, list)
        or len(orders) != len(tasks)
        or any(
            not isinstance(order, list) or len(order) != 4 or set(order) != set(VARIANTS)
            for order in orders
        )
    ):
        raise ValueError("Declare one explicit A-D execution order per task/seed case.")
    for task in tasks:
        problems = preflight(task, backend_name)
        if problems:
            raise ValueError("\n".join(problems))
    policy = load_policy(policy_path)
    metadata = backend(backend_name).metadata  # Missing keys fail before any preparation.
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "suite.json",
        {
            "tasks": [t.model_dump(mode="json") for t in tasks],
            "orders": orders,
            "variants": VARIANTS,
            "policy_hash": digest(policy),
            "backend": metadata,
            "maximum_execution_seconds": sum(4 * t.limits.total_compute_seconds for t in tasks),
            "maximum_api_calls": sum(4 * t.limits.max_api_calls for t in tasks),
            "interpretation": "Matched workflow comparison; no policy promotion or assumed winner.",
        },
    )
    cases = []
    for index, (task, order) in enumerate(zip(tasks, orders)):
        summaries = {}
        directories = {}
        for label in order:
            directory = output / f"case-{index:03d}-{label}"
            directories[label] = directory
            summaries[label] = Workflow(
                task, policy, backend(backend_name), directory, variant=label
            ).run()
        case = compare_design_runs(summaries, task)
        case["behavior"] = {}
        for label, directory in directories.items():
            records = [
                Experiment.model_validate(r)
                for r in read_json(directory / "state.json")["experiments"]
            ]
            case["behavior"][label] = behavior(records, task.metric)
        cases.append(case)
        write_json(
            output / "comparison.json", {"cases": cases, "completed": len(cases) == len(tasks)}
        )
        plot_case(output, index, task, summaries, directories, case["behavior"])
    return {"cases": cases, "completed": True}


def plot_case(output, index, task, summaries, directories, behaviors):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    csv_rows = []
    for label in VARIANTS:
        directory = directories[label]
        records = [
            Experiment.model_validate(r) for r in read_json(directory / "state.json")["experiments"]
        ]
        events = read_events(directory / "events.jsonl")
        costs = [event["payload"] for event in events if event["event"] == "diagnostic_cost"]
        rows = progress_rows(records, task.metric)
        for row in rows:
            row["cumulative_compute_seconds"] += sum(
                c["seconds"] for c in costs if c["after_attempt"] < row["attempt"]
            )
            csv_rows.append({"variant": label, **row})
        axes[0].step(
            [r["cumulative_compute_seconds"] for r in rows],
            [r["best_score"] for r in rows],
            where="post",
            label=label,
        )
    axes[0].axhline(task.target, linestyle="--", color="black", label="target")
    axes[0].set(xlabel="Cumulative execution + diagnostic seconds", ylabel=task.metric)
    axes[0].legend()
    values = [
        summaries[k].compute_to_target
        if summaries[k].reached_target
        and summaries[k].compute_to_target <= task.limits.total_compute_seconds
        else task.limits.total_compute_seconds
        for k in VARIANTS
    ]
    bars = axes[1].bar(list(VARIANTS), values)
    for bar, key in zip(bars, VARIANTS):
        if (
            not summaries[key].reached_target
            or summaries[key].compute_to_target > task.limits.total_compute_seconds
        ):
            bar.set_hatch("//")
    axes[1].set(ylabel="Seconds to target (hatched = not reached)")
    bottom = np.zeros(4)
    for category in (
        "baseline",
        "improvement",
        "regression",
        "tie",
        "failed",
        "timeout",
        "rejected",
    ):
        values = [behaviors[key][category] for key in VARIANTS]
        axes[2].bar(list(VARIANTS), values, bottom=bottom, label=category)
        bottom += values
    axes[2].set(ylabel="Experiment outcomes versus prior best")
    axes[2].legend(fontsize=7)
    fig.suptitle(
        f"{task.task_id} / seed {task.seed} — {'software fixture' if task.purpose == 'demo' else 'measured validation results'}"
    )
    fig.tight_layout()
    fig.savefig(output / f"case-{index:03d}.svg")
    plt.close(fig)
    if csv_rows:
        with (output / f"case-{index:03d}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)


import numpy as np
