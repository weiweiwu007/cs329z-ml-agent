"""Independent accounting and a conservative, explicit workflow-promotion rule.

The language model never grades its own proposals. These functions use only
completed execution results. A missed target is censored at the declared budget,
so an agent cannot win by stopping early or producing only rejected experiments.
"""

from __future__ import annotations

import math
from typing import Any

from .schemas import Experiment, Policy, RunSummary, TaskSpec, Usage, digest


def progress_rows(experiments: list[Experiment], metric: str) -> list[dict[str, Any]]:
    """One row per attempt, including rejected, failed, and timed-out trials."""
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    best: float | None = None
    seen: set[str] = set()
    for attempt, experiment in enumerate(experiments, 1):
        if experiment.trial_id in seen:
            raise ValueError(f"Duplicate trial ID: {experiment.trial_id}")
        seen.add(experiment.trial_id)
        result = experiment.result
        if not math.isfinite(result.compute_seconds) or result.compute_seconds < 0:
            raise ValueError("Execution compute must be a finite nonnegative duration.")
        cumulative += result.compute_seconds
        score = result.metrics.get(metric) if result.status == "completed" else None
        if result.status == "completed" and score is None:
            raise ValueError(f"Completed trial {experiment.trial_id} lacks metric {metric}.")
        if score is not None:
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid classification score in {experiment.trial_id}.")
            best = score if best is None else max(best, score)
        rows.append(
            {
                "attempt": attempt,
                "trial_id": experiment.trial_id,
                "status": result.status,
                "score": score,
                "best_score": best,
                "compute_seconds": result.compute_seconds,
                "cumulative_compute_seconds": cumulative,
            }
        )
    return rows


def summarize_run(
    run_id: str,
    task: TaskSpec,
    policy: Policy,
    backend: str,
    experiments: list[Experiment],
    usage: Usage,
    prepared: dict,
    stop_reason: str,
) -> RunSummary:
    rows = progress_rows(experiments, task.metric)
    crossing = next(
        (r for r in rows if r["best_score"] is not None and r["best_score"] >= task.target), None
    )
    return RunSummary(
        run_id=run_id,
        task_id=task.task_id,
        purpose=task.purpose,
        backend=backend,
        policy_hash=digest(policy),
        task_hash=digest(task),
        starting_model_hash=prepared["starting_model_hash"],
        dataset_hash=prepared["dataset_hash"],
        seed=task.seed,
        metric=task.metric,
        target=task.target,
        device=task.device,
        reached_target=crossing is not None,
        trials_to_target=crossing["attempt"] if crossing else None,
        compute_to_target=crossing["cumulative_compute_seconds"] if crossing else None,
        best_score=rows[-1]["best_score"] if rows else None,
        total_trials=len(rows),
        total_compute_seconds=rows[-1]["cumulative_compute_seconds"] if rows else 0,
        stop_reason=stop_reason,
        usage=usage,
        trial_ids=[r["trial_id"] for r in rows],
    )


_MATCH_FIELDS = (
    "task_id",
    "task_hash",
    "dataset_hash",
    "starting_model_hash",
    "seed",
    "metric",
    "target",
    "device",
    "backend",
    "purpose",
    "backend_fingerprint",
    "execution_fingerprint",
    "memory_fingerprint",
    "variant",
)


def _validate_summary(summary: RunSummary) -> None:
    if summary.reached_target:
        if (
            summary.trials_to_target is None
            or summary.compute_to_target is None
            or summary.best_score is None
            or summary.best_score < summary.target
        ):
            raise ValueError("Target-attainment summary is internally inconsistent.")
        if not 1 <= summary.trials_to_target <= summary.total_trials:
            raise ValueError("Trials to target exceeds recorded attempts.")
        if not 0 <= summary.compute_to_target <= summary.total_compute_seconds:
            raise ValueError("Compute to target exceeds recorded compute.")
    elif summary.trials_to_target is not None or summary.compute_to_target is not None:
        raise ValueError("A missed target must have null time and attempts to target.")
    if summary.total_compute_seconds < 0 or summary.total_trials < 0:
        raise ValueError("Run cost must be nonnegative.")


def compare_runs(
    baseline: RunSummary,
    candidate: RunSummary,
    *,
    compute_budget: float | None = None,
    trial_budget: int | None = None,
) -> dict[str, Any]:
    """Describe one matched comparison; reject changes in its task or resources.

    Without the original task's two declared budgets, this function can describe
    runs but cannot supply evidence for automatic promotion.
    """
    for summary in (baseline, candidate):
        _validate_summary(summary)
    mismatches = [key for key in _MATCH_FIELDS if getattr(baseline, key) != getattr(candidate, key)]
    if mismatches:
        raise ValueError("Cannot compare unmatched runs: " + ", ".join(mismatches))
    if baseline.run_id == candidate.run_id:
        raise ValueError("A run cannot serve as both sides of a comparison.")
    has_budgets = compute_budget is not None and trial_budget is not None
    if (compute_budget is None) != (trial_budget is None):
        raise ValueError("Provide both compute_budget and trial_budget, or neither.")
    if has_budgets:
        if not math.isfinite(compute_budget) or compute_budget <= 0:
            raise ValueError("Compute budget must be finite and positive.")
        if isinstance(trial_budget, bool) or not isinstance(trial_budget, int) or trial_budget < 1:
            raise ValueError("Trial budget must be a positive integer.")

    def side(summary: RunSummary) -> dict[str, Any]:
        # An over-budget target crossing does not count as reaching the target
        # inside the comparison's allowed resources.
        attained = summary.reached_target
        if has_budgets and attained:
            attained = (
                summary.compute_to_target <= compute_budget
                and summary.trials_to_target <= trial_budget
            )
        capped_compute = None
        capped_trials = None
        if has_budgets:
            capped_compute = summary.compute_to_target if attained else compute_budget
            capped_trials = summary.trials_to_target if attained else trial_budget
        return {
            "run_id": summary.run_id,
            "policy_hash": summary.policy_hash,
            "reached_target": attained,
            "best_score": summary.best_score,
            "trials_to_target": summary.trials_to_target if attained else None,
            "compute_to_target": summary.compute_to_target if attained else None,
            "capped_compute": capped_compute,
            "capped_trials": capped_trials,
            "actual_total_compute": summary.total_compute_seconds,
            "actual_total_trials": summary.total_trials,
            "api_usage": summary.usage.model_dump(mode="json"),
        }

    return {
        "identity": {key: getattr(baseline, key) for key in _MATCH_FIELDS},
        "compute_budget": compute_budget,
        "trial_budget": trial_budget,
        "eligible_for_promotion": (
            has_budgets
            and baseline.purpose == "development"
            and not baseline.backend.lower().startswith("mock")
            and bool(baseline.backend_fingerprint)
            and bool(baseline.execution_fingerprint)
            and bool(baseline.memory_fingerprint)
            and baseline.policy_hash != candidate.policy_hash
        ),
        "baseline": side(baseline),
        "candidate": side(candidate),
        "interpretation": "Exploratory matched comparison; no statistical significance claim.",
        "discovery_cost_included": False,
    }


def promotion_decision(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the documented v1 gate to paired development-case results.

    Require no drop in target attainment, fewer capped attempts, and less capped
    compute. Misses consume the full declared comparison budget. The caller must
    verify receipt integrity and candidate lineage before using this decision.
    """
    rule = (
        "Development comparisons only; at least one candidate target attained; "
        "no decrease in target attainment; strictly fewer summed capped attempts "
        "AND strictly less summed capped compute."
    )
    result: dict[str, Any] = {
        "accepted": False,
        "rule": rule,
        "pairs": len(pairs),
        "interpretation": "Exploratory selection rule, not proof of generalization.",
        "discovery_cost_included": False,
        "discovery_cost_note": "Report workflow generation and selection costs separately.",
    }
    if not pairs:
        return {**result, "reason": "No matched development comparisons supplied."}
    if not all(p.get("eligible_for_promotion") for p in pairs):
        return {
            **result,
            "reason": "Missing budgets/fingerprints, mock backend, non-development data, or unchanged policy.",
        }
    identities = [p["identity"] for p in pairs]
    cases = [(i["task_hash"], i["seed"]) for i in identities]
    if len(cases) != len(set(cases)):
        return {**result, "reason": "Duplicate task/seed pairs are not independent cases."}
    if any(i["purpose"] != "development" for i in identities):
        return {**result, "reason": "Evaluation and demo cases cannot select a policy."}
    if (
        len({p["baseline"]["policy_hash"] for p in pairs}) != 1
        or len({p["candidate"]["policy_hash"] for p in pairs}) != 1
    ):
        return {**result, "reason": "All pairs must compare the same two fixed policies."}
    totals: dict[str, dict[str, Any]] = {}
    for side in ("baseline", "candidate"):
        costs = [p[side]["capped_compute"] for p in pairs]
        trials = [p[side]["capped_trials"] for p in pairs]
        if any(v is None or not math.isfinite(v) or v < 0 for v in costs + trials):
            return {**result, "reason": "Invalid capped accounting."}
        totals[side] = {
            "targets_attained": sum(bool(p[side]["reached_target"]) for p in pairs),
            "capped_compute_seconds": sum(costs),
            "capped_attempts": sum(trials),
            "mean_capped_compute_seconds": sum(costs) / len(pairs),
        }
    baseline, candidate = totals["baseline"], totals["candidate"]
    accepted = (
        candidate["targets_attained"] > 0
        and candidate["targets_attained"] >= baseline["targets_attained"]
        and candidate["capped_attempts"] < baseline["capped_attempts"]
        and candidate["capped_compute_seconds"] < baseline["capped_compute_seconds"]
    )
    return {
        **result,
        "accepted": accepted,
        "totals": totals,
        "reason": "Candidate meets the declared selection rule."
        if accepted
        else "Candidate does not improve both resource measures without losing target attainment.",
    }
