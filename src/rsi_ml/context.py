"""Focused working context reconstructed from durable experiment evidence."""

from __future__ import annotations

import json
from copy import deepcopy
from importlib.resources import files

from .schemas import Policy, TaskSpec, WorkflowVariant
from .tasks.registry import recipe_rules


def knowledge_for(adapter: str) -> list[dict]:
    return [
        card
        for card in json.loads(files("rsi_ml").joinpath("knowledge.json").read_text())
        if adapter in card["families"]
    ]


def public_task(task: TaskSpec) -> dict:
    result = task.model_dump(mode="json")
    result.pop("random_search", None)  # The reference arm's ranges are not a menu for agents.
    for key in (
        "train_csv",
        "validation_csv",
        "test_csv",
        "train_file",
        "validation_file",
        "test_file",
        "test_sha256",
        "test_rows",
        "split_manifest",
        "split_manifest_sha256",
    ):
        result["dataset"].pop(key, None)
    if result.get("text_model"):
        result["text_model"].pop("local_path", None)
    return result


def attempt_table(task: TaskSpec, records: list[dict]) -> list[dict]:
    """One short row per attempt, so an agent with many attempts forgets none of them.

    Only ordinary results: nothing here comes from a detailed error report.
    """
    starting = task.initial_recipe.model_dump(mode="json")
    rows = []
    for record in records:
        result, recipe = record["result"], record.get("recipe")
        rows.append(
            {
                "trial_id": record["trial_id"],
                "status": result["status"],
                "score": result.get("metrics", {}).get(task.metric),
                "seconds": round(result.get("compute_seconds", 0), 1),
                "parameters": result.get("parameter_count"),
                "changed_from_starting_recipe": (
                    {k: v for k, v in recipe.items() if v != starting.get(k)} if recipe else None
                ),
            }
        )
    return rows


def validate_prior_evidence(evidence: list[dict], variant: WorkflowVariant) -> None:
    """Reject prior evidence that could reintroduce excluded reports through prose."""
    if variant.detailed_error_reports:
        return

    def has_report(value):
        if isinstance(value, dict):
            return any(
                (key in {"diagnostic_report", "diagnostic_reports"} and bool(item))
                or has_report(item)
                for key, item in value.items()
            )
        return isinstance(value, list) and any(has_report(item) for item in value)

    for prior in evidence:
        summary = prior.get("summary", {})
        if (
            summary.get("variant") in {"B", "D"}
            or summary.get("diagnostic_seconds", 0) > 0
            or summary.get("diagnostic_requests", 0) > 0
            or has_report(prior)
        ):
            raise ValueError(
                f"Variant {variant.name} cannot use prior evidence containing detailed error "
                "reports or decisions made with them. Supply report-free prior evidence."
            )


def make_context(
    task: TaskSpec, policy: Policy, state: dict, role: str, *, variant: WorkflowVariant
) -> dict:
    validate_prior_evidence(state.get("prior_evidence", []), variant)
    records = deepcopy(state.get("experiments", []))
    # Compact scalar loss summaries remain ordinary logs for every variant.
    # Cached or embedded reports must never bypass A/C's access restriction.
    reports = state.get("diagnostic_reports", {}) if variant.detailed_error_reports else {}
    for record in records:
        record.pop("diagnostic_report", None)
        # Hundreds of per-epoch file hashes are bookkeeping, not evidence an agent can use.
        record["result"].pop("artifact_hashes", None)
        if record["trial_id"] in reports:
            record["diagnostic_report"] = reports[record["trial_id"]]
    best = sorted(
        [r for r in records if r["result"]["status"] == "completed"],
        key=lambda r: r["result"]["metrics"].get(task.metric, -1),
        reverse=True,
    )[:2]
    # The baseline stays visible: every later change is judged against it.
    selected = {r["trial_id"]: r for r in [*records[:1], *best, *records[-4:]]}
    # Agents are blind to the study design: no arm label, no other role's instructions.
    # The only per-arm differences are report data and (C/D) recorded reviews.
    context = {
        "task": public_task(task),
        "role": role,
        "records": [selected[trial_id] for trial_id in sorted(selected)],
        "history_count": len(records),
        "history_omitted": len(records) - len(selected),
        "remaining": {
            "trials": task.limits.max_trials - len(records),
            "compute_seconds": max(
                0,
                task.limits.total_compute_seconds
                - sum(r["result"]["compute_seconds"] for r in records),
            ),
        },
        "knowledge": knowledge_for(task.adapter),
        "prior_evidence": state.get("prior_evidence", []),
        "prior_evidence_omitted": 0,
        "tools": {},
    }
    # Advertise the optional request tool only when a request can succeed. With a zero
    # allowance, B/D still get automatic reports, and an impossible tool cannot lure
    # them into a rejected plan that A/C could never make.
    requests_remaining = task.limits.max_diagnostic_requests - state.get("diagnostic_requests", 0)
    if variant.detailed_error_reports and requests_remaining > 0:
        context["tools"] = {
            "diagnostics": "Set diagnostic_trial_id to a completed visible trial ID to request its error report. No training is launched; a new decision follows the tool result.",
            "requests_remaining": requests_remaining,
        }
    if task.recipe_policy == "constraints":
        context["recipe_rules"] = recipe_rules(task)
        context["all_attempts"] = attempt_table(task, records)
    if role == "planner" and state.get("recipe_feedback"):
        context["recipe_feedback"] = state["recipe_feedback"]
    if role in ("developer", "reviewer"):
        context["current_plan"] = state.get("plan")
    if role == "reviewer" and records:
        context["current_result"] = records[-1]["result"]
    context["remaining"]["compute_seconds"] = max(
        0, context["remaining"]["compute_seconds"] - state.get("diagnostic_seconds", 0)
    )
    # Reduce old episodic records, never the task's immutable constraints.
    # Reserve space for role instructions and serialized schema overhead.
    context_limit = (
        task.limits.max_context_chars - 2500 - len(getattr(policy, f"{role}_instructions", ""))
    )
    while len(json.dumps(context)) > context_limit and context["records"]:
        context["records"].pop(0)
        context["history_omitted"] += 1
    context["prior_evidence"] = list(context["prior_evidence"])
    while len(json.dumps(context)) > context_limit and context["prior_evidence"]:
        context["prior_evidence"].pop(0)
        context["prior_evidence_omitted"] += 1
    if len(json.dumps(context)) > context_limit:
        raise ValueError(
            "Task, policy, and current result exceed context budget; increase the declared limit."
        )
    return context
