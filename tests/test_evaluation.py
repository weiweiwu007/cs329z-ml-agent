from __future__ import annotations

import csv
from copy import deepcopy

import pytest

from rsi_ml.evaluation import compare_runs, progress_rows, promotion_decision, summarize_run
from rsi_ml.report import write_report
from rsi_ml.schemas import Experiment, Policy, RunSummary, TaskSpec, TrialResult, Usage


def task() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "task_id": "accounting-test",
            "adapter": "mlp_classification",
            "purpose": "demo",
            "protocol_status": "demo",
            "metric": "accuracy",
            "target": 0.8,
            "seed": 4,
            "device": "cpu",
            "dataset": {
                "kind": "synthetic_fixture",
                "identity": "unit-test",
                "revision": "1",
                "input_dim": 3,
                "num_classes": 2,
                "standardize": True,
                "seed": 10,
                "train_rows": 8,
                "validation_rows": 4,
                "test_rows": 4,
            },
            "initial_recipe": {
                "hidden_sizes": [4],
                "activation": "relu",
                "normalization": "none",
                "dropout": 0,
                "optimizer": "adamw",
                "learning_rate": 0.01,
                "weight_decay": 0,
                "batch_size": 4,
                "epochs": 2,
                "schedule": "constant",
                "warmup_fraction": 0,
                "decay_fraction": 0.2,
            },
            "limits": {
                "max_trials": 5,
                "max_parameters": 1000,
                "max_epochs": 5,
                "per_trial_seconds": 10,
                "total_compute_seconds": 40,
                "max_api_calls": 10,
                "max_context_chars": 8000,
                "max_output_tokens": 1000,
            },
        }
    )


def policy() -> Policy:
    return Policy(
        policy_id="baseline",
        version=0,
        status="baseline",
        planner_instructions="Plan measurable experiments.",
        developer_instructions="Implement the proposed configuration.",
        reviewer_instructions="Review actual measurements only.",
    )


def experiment(number: int, status: str, cost: float, score: float | None = None) -> Experiment:
    return Experiment(
        trial_id=f"trial-{number}",
        index=number,
        result=TrialResult(
            status=status,
            compute_seconds=cost,
            metrics={} if score is None else {"accuracy": score},
        ),
    )


def summary(experiments=None, **updates) -> RunSummary:
    if experiments is None:
        experiments = [experiment(0, "completed", 5, 0.6), experiment(1, "completed", 5, 0.85)]
    value = summarize_run(
        "baseline-run",
        task(),
        policy(),
        "dspy",
        experiments,
        Usage(),
        {"dataset_hash": "dataset", "starting_model_hash": "weights"},
        "target_reached",
    )
    return value.model_copy(
        update={
            "backend_fingerprint": "fake-backend",
            "execution_fingerprint": "fake-execution",
            "memory_fingerprint": "same-history",
            **updates,
        }
    )


def candidate(base: RunSummary, **updates) -> RunSummary:
    return base.model_copy(
        update={"run_id": "candidate-run", "policy_hash": "new-policy", **updates}
    )


def pair(base: RunSummary, cand: RunSummary):
    return compare_runs(base, cand, compute_budget=40, trial_budget=5)


def test_first_crossing_counts_rejected_and_failed_attempts():
    experiments = [
        experiment(0, "completed", 2, 0.5),
        experiment(1, "rejected", 0),
        experiment(2, "timeout", 7),
        experiment(3, "completed", 3, 0.8),
        experiment(4, "completed", 4, 0.9),
    ]
    result = summary(experiments)
    assert result.trials_to_target == 4
    assert result.compute_to_target == 12
    assert result.total_trials == 5
    assert result.total_compute_seconds == 16
    assert result.best_score == 0.9
    assert [r["best_score"] for r in progress_rows(experiments, "accuracy")] == [
        0.5,
        0.5,
        0.5,
        0.8,
        0.9,
    ]


def test_failed_trial_cannot_supply_a_score():
    result = summary([experiment(0, "failed", 1, 0.99)])
    assert result.best_score is None
    assert not result.reached_target
    assert result.compute_to_target is None


def test_completed_missing_score_and_duplicate_attempt_rejected():
    with pytest.raises(ValueError, match="lacks metric"):
        progress_rows([experiment(0, "completed", 2)], "accuracy")
    with pytest.raises(ValueError, match="Duplicate"):
        progress_rows([experiment(0, "completed", 2, 0.5)] * 2, "accuracy")


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 12),
        ("starting_model_hash", "other"),
        ("dataset_hash", "other"),
        ("device", "cuda"),
        ("task_hash", "other"),
        ("backend", "mock"),
        ("backend_fingerprint", "different-model"),
        ("execution_fingerprint", "different-hardware"),
        ("memory_fingerprint", "different-history"),
    ],
)
def test_unmatched_comparisons_rejected(field, value):
    base = summary()
    with pytest.raises(ValueError, match=field):
        compare_runs(base, candidate(base, **{field: value}))


def test_stopping_early_without_target_has_full_censored_cost():
    base = summary(purpose="development")
    failed = candidate(
        base,
        reached_target=False,
        best_score=0.7,
        trials_to_target=None,
        compute_to_target=None,
        total_compute_seconds=1,
        total_trials=1,
    )
    comparison = pair(base, failed)
    assert comparison["candidate"]["capped_compute"] == 40
    assert comparison["candidate"]["capped_trials"] == 5
    assert not promotion_decision([comparison])["accepted"]


def test_candidate_requires_both_fewer_trials_and_less_compute():
    base = summary(purpose="development", trials_to_target=3, total_trials=3)
    successful = candidate(
        base, trials_to_target=2, total_trials=2, compute_to_target=8, total_compute_seconds=8
    )
    assert promotion_decision([pair(base, successful)])["accepted"]
    tied_attempts = successful.model_copy(update={"trials_to_target": 3, "total_trials": 3})
    assert not promotion_decision([pair(base, tied_attempts)])["accepted"]


@pytest.mark.parametrize(
    "purpose,backend", [("demo", "dspy"), ("evaluation", "dspy"), ("development", "mock")]
)
def test_demo_evaluation_and_mock_cannot_promote(purpose, backend):
    base = summary(purpose=purpose, backend=backend)
    assert not promotion_decision([pair(base, candidate(base))])["accepted"]


def test_missing_budgets_cannot_promote_and_partial_budgets_invalid():
    base = summary(purpose="development")
    assert not promotion_decision([compare_runs(base, candidate(base))])["accepted"]
    with pytest.raises(ValueError, match="both"):
        compare_runs(base, candidate(base), compute_budget=40)


@pytest.mark.parametrize(
    "field", ["backend_fingerprint", "execution_fingerprint", "memory_fingerprint"]
)
def test_missing_fingerprints_cannot_promote(field):
    base = summary(purpose="development", **{field: ""})
    assert not promotion_decision([pair(base, candidate(base))])["accepted"]


def test_target_after_budget_is_a_miss():
    base = summary(purpose="development")
    late = candidate(base, compute_to_target=41, total_compute_seconds=41)
    comparison = pair(base, late)
    assert not comparison["candidate"]["reached_target"]
    assert comparison["candidate"]["capped_compute"] == 40


def test_duplicate_pairs_and_mixed_policies_cannot_promote():
    base = summary(purpose="development", trials_to_target=3, total_trials=3)
    cand = candidate(
        base, trials_to_target=2, total_trials=2, compute_to_target=8, total_compute_seconds=8
    )
    comparison = pair(base, cand)
    assert not promotion_decision([comparison, comparison])["accepted"]
    different = deepcopy(comparison)
    different["identity"]["seed"] = 5
    different["candidate"]["policy_hash"] = "third-policy"
    assert not promotion_decision([comparison, different])["accepted"]


def test_report_has_all_attempts_and_escapes_untrusted_text(tmp_path):
    experiments = [experiment(0, "completed", 2, 0.5), experiment(1, "failed", 1)]
    result = summary(experiments, run_id="<script>alert(1)</script>", backend="mock")
    write_report(tmp_path, result, experiments)
    html = (tmp_path / "report.html").read_text()
    assert "SOFTWARE DEMO" in html
    assert "scripted mock" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<svg" in html
    with (tmp_path / "curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[1]["best_score"] == "0.5"
    assert rows[1]["cumulative_compute_seconds"] == "3.0"
