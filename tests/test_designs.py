"""Control comparisons using synthetic receipts, never benchmark results."""

import json
from pathlib import Path

import pytest
from test_workflow import fake_tools

from rsi_ml.agents import MockBackend
from rsi_ml.cli import load_policy, load_task, main
from rsi_ml.comparison import compare_design_runs, plan_suite
from rsi_ml.context import make_context
from rsi_ml.schemas import WorkflowVariant
from rsi_ml.storage import read_json
from rsi_ml.workflow import Workflow

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "variant,reviewer,automatic",
    [("A", False, False), ("B", False, True), ("C", True, False), ("D", True, True)],
)
def test_four_variants_enforce_report_access_and_reviewer(
    tmp_path, monkeypatch, variant, reviewer, automatic
):
    task, policy = (
        load_task(ROOT / "configs/mlp-demo.json"),
        load_policy(ROOT / "configs/policy.json"),
    )
    fake_tools(monkeypatch, [0.5, 1.0])
    reports = []

    def report(path):
        reports.append(path)
        return {"measured_error_pattern": "fixture", "trial_id": path.name}

    monkeypatch.setattr("rsi_ml.workflow.diagnostic_report", report)

    class RecordingBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.roles = []
            self.contexts = []

        def call(self, role, context, policy, limits):
            self.roles.append(role)
            self.contexts.append(context)
            return super().call(role, context, policy, limits)

    backend = RecordingBackend()
    summary = Workflow(task, policy, backend, tmp_path / variant, variant=variant).run()
    assert ("reviewer" in backend.roles) == reviewer
    assert "developer" not in backend.roles
    assert bool(reports) == automatic
    assert summary.variant == variant
    assert summary.total_compute_seconds >= 2
    # Automatic reports do not depend on an optional request allowance.
    assert task.limits.max_diagnostic_requests == 0
    assert summary.diagnostic_requests == 0
    assert len(reports) == (2 if automatic else 0)
    assert (
        any(
            "diagnostic_report" in record
            for context in backend.contexts
            for record in context["records"]
        )
        == automatic
    )
    # With a zero allowance no arm is offered the optional request tool.
    assert all("diagnostics" not in c["tools"] for c in backend.contexts)
    # Agents are blind to the study design.
    assert all("workflow" not in c and "policy" not in c for c in backend.contexts)
    for context in backend.contexts:
        # Context ranks best/recent trials; identity must not depend on their order.
        baseline = next(r for r in context["records"] if r["trial_id"] == "trial-0000")
        assert baseline["result"]["metrics"]["accuracy"] == 0.5
    for role, context in zip(backend.roles, backend.contexts):
        if role == "reviewer":
            current = next(
                r for r in context["records"] if r["result"] == context["current_result"]
            )
            assert ("diagnostic_report" in current) == automatic
            if automatic:
                assert current["diagnostic_report"]["trial_id"] == current["trial_id"]
    assert read_json(tmp_path / variant / "manifest.json")["variant_settings"] == {
        "detailed_error_reports": automatic,
        "automatic_diagnostics": automatic,
        "separate_reviewer": reviewer,
    }


@pytest.mark.parametrize("variant,role", [("B", "planner"), ("D", "reviewer")])
def test_report_enabled_roles_can_reuse_report_without_an_extra_trial(
    tmp_path, monkeypatch, variant, role
):
    task, policy = (
        load_task(ROOT / "configs/mlp-demo.json"),
        load_policy(ROOT / "configs/policy.json"),
    )
    task.limits.max_diagnostic_requests = 1
    calls = fake_tools(monkeypatch, [0.5, 1.0])
    monkeypatch.setattr("rsi_ml.workflow.diagnostic_report", lambda path: {"report": "fixture"})

    class RequestingBackend(MockBackend):
        requested = False

        def call(self, called_role, context, policy, limits):
            result = super().call(called_role, context, policy, limits)
            if called_role == role and not self.requested:
                assert "diagnostic_report" in context["records"][0]
                result.diagnostic_trial_id = "trial-0000"
                self.requested = True
            return result

    summary = Workflow(task, policy, RequestingBackend(), tmp_path / variant, variant=variant).run()
    assert len(calls) == 2
    assert summary.diagnostic_requests == 1
    assert summary.diagnostic_seconds > 0


@pytest.mark.parametrize("variant,role", [("A", "planner"), ("C", "planner"), ("C", "reviewer")])
def test_forbidden_request_cannot_bypass_report_access(tmp_path, monkeypatch, variant, role):
    task = load_task(ROOT / "configs/mlp-demo.json")
    policy = load_policy(ROOT / "configs/policy.json")
    task.limits.max_trials = 2
    task.limits.max_diagnostic_requests = 10  # A positive allowance never grants access.
    calls = fake_tools(monkeypatch, [0.5, 1.0])

    def forbidden(*args, **kwargs):
        pytest.fail("A/C must never execute the diagnostic tool")

    monkeypatch.setattr("rsi_ml.workflow.diagnostic_report", forbidden)

    class RequestingBackend(MockBackend):
        def call(self, called_role, context, policy, limits):
            assert "diagnostics" not in context["tools"]
            result = super().call(called_role, context, policy, limits)
            if called_role == role:
                result.diagnostic_trial_id = "trial-0000"
            return result

    directory = tmp_path / variant
    summary = Workflow(task, policy, RequestingBackend(), directory, variant=variant).run()
    assert len(calls) == (1 if role == "planner" else 2)
    assert summary.diagnostic_requests == summary.diagnostic_seconds == 0
    state = read_json(directory / "state.json")
    assert state["diagnostic_reports"] == {}
    assert not list(directory.rglob("diagnostic_report.json"))
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert any(
        event["event"] == "diagnostic_access_denied" and event["payload"]["caller"] == role
        for event in events
    )


@pytest.mark.parametrize("variant", ["A", "C"])
def test_stale_diagnostic_node_is_rejected_before_reading_outputs(tmp_path, monkeypatch, variant):
    task = load_task(ROOT / "configs/mlp-demo.json")
    policy = load_policy(ROOT / "configs/policy.json")

    def forbidden(*args, **kwargs):
        pytest.fail("Forbidden graph node must not read raw validation outputs")

    monkeypatch.setattr("rsi_ml.workflow.diagnostic_report", forbidden)
    workflow = Workflow(task, policy, MockBackend(), tmp_path, variant=variant)
    with pytest.raises(ValueError, match="unavailable"):
        workflow._diagnostics({"diagnostic_trial_id": "trial-0000", "experiments": []})


@pytest.mark.parametrize("variant", ["A", "C"])
@pytest.mark.parametrize("role", ["planner", "reviewer"])
def test_cached_or_embedded_reports_never_enter_restricted_context(variant, role):
    task = load_task(ROOT / "configs/mlp-demo.json")
    policy = load_policy(ROOT / "configs/policy.json")
    record = {
        "trial_id": "trial-0000",
        "result": {
            "status": "completed",
            "metrics": {"accuracy": 0.5},
            "diagnostics": {"last_train_loss": 0.4, "last_validation_loss": 0.8},
            "compute_seconds": 1,
        },
        "diagnostic_report": {"confident_errors": ["embedded-error-must-not-leak"]},
    }
    state = {
        "experiments": [record],
        "diagnostic_reports": {"trial-0000": {"confident_errors": ["cached-error-must-not-leak"]}},
    }
    context = make_context(task, policy, state, role, variant=WorkflowVariant(name=variant))
    serialized = json.dumps(context)
    assert "must-not-leak" not in serialized
    assert "diagnostic_report" not in serialized
    assert context["records"][0]["result"] == record["result"]
    if role == "reviewer":
        assert context["current_result"] == record["result"]
    assert "diagnostic_report" in record  # The durable source is not mutated.


@pytest.mark.parametrize("variant", ["A", "C"])
@pytest.mark.parametrize(
    "prior",
    [
        {"summary": {"run_id": "prior", "variant": "B"}, "records": []},
        {"summary": {"run_id": "prior", "variant": "D"}, "records": []},
        {"summary": {"run_id": "legacy-C", "variant": "C", "diagnostic_requests": 1}},
        {"summary": {"run_id": "prior"}, "diagnostic_reports": {"trial-0000": {"errors": [1]}}},
    ],
)
def test_report_informed_prior_memory_is_rejected_before_preparation(
    tmp_path, monkeypatch, variant, prior
):
    task = load_task(ROOT / "configs/mlp-demo.json")
    policy = load_policy(ROOT / "configs/policy.json")

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid evidence access must fail before preparation/training")

    monkeypatch.setattr("rsi_ml.workflow.prepare_task", forbidden)
    monkeypatch.setattr("rsi_ml.workflow.run_trial", forbidden)
    directory = tmp_path / "not-created"
    with pytest.raises(ValueError, match="cannot use prior evidence"):
        Workflow(
            task, policy, MockBackend(), directory, shared_memory=[prior], variant=variant
        ).run()
    assert not directory.exists()


def test_variant_change_cannot_resume_run(tmp_path, monkeypatch):
    task, policy = (
        load_task(ROOT / "configs/mlp-demo.json"),
        load_policy(ROOT / "configs/policy.json"),
    )
    fake_tools(monkeypatch, [1.0])
    Workflow(task, policy, MockBackend(), tmp_path / "run", variant="A").run()
    with pytest.raises(ValueError, match="unchanged"):
        Workflow(task, policy, MockBackend(), tmp_path / "run", variant="B").run(resume=True)


def test_draft_planning_and_cli_never_call_training_or_provider(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("No execution or backend construction during planning")

    monkeypatch.setattr("rsi_ml.workflow.run_trial", forbidden)
    monkeypatch.setattr("rsi_ml.cli.backend", forbidden)
    plan = plan_suite(ROOT / "configs/abcd-suite.template.json")
    assert len(plan["cases"]) == 4
    assert not plan["training_launched"]
    assert all(case["status"] == "draft" for case in plan["cases"])
    assert all(all(v is None for v in case["limits"].values()) for case in plan["cases"])
    assert plan["variants"]["A"]["detailed_error_reports"] is False
    assert plan["variants"]["C"]["detailed_error_reports"] is False
    assert plan["variants"]["B"]["automatic_diagnostics"] is True
    assert plan["variants"]["D"]["automatic_diagnostics"] is True
    assert main(["compare-designs", "--suite", str(ROOT / "configs/abcd-suite.template.json")]) == 0
    assert (
        main(
            [
                "run",
                "--task",
                "does-not-exist",
                "--policy",
                "does-not-exist",
                "--out",
                "not-created",
                "--backend",
                "dspy",
                "--variant",
                "A",
            ]
        )
        == 1
    )
    for name in ("spambase", "cifar100", "sst2", "mrpc"):
        assert (
            main(
                [
                    "preflight",
                    "--task",
                    str(ROOT / f"configs/{name}.template.json"),
                    "--backend",
                    "dspy",
                ]
            )
            == 1
        )


def test_comparison_and_plots_on_stubbed_runs(tmp_path, monkeypatch):
    from rsi_ml.comparison import compare_designs

    task_path = ROOT / "configs/mlp-demo.json"
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "protocol_status": "ready",
                "cases": [{"task": str(task_path), "seeds": [3]}],
                "orders": [["A", "B", "C", "D"]],
            }
        )
    )
    fake_tools(monkeypatch, [0.5, 1.0] * 4)
    monkeypatch.setattr("rsi_ml.workflow.diagnostic_report", lambda path: {"report": "fixture"})
    result = compare_designs(suite, ROOT / "configs/policy.json", "mock", tmp_path / "compared")
    assert result["completed"]
    assert set(result["cases"][0]["contrasts"]) == {"B_vs_A", "D_vs_C", "C_vs_A", "D_vs_B"}
    assert (tmp_path / "compared/case-000.svg").exists()
    summaries = {
        key: __import__("rsi_ml.schemas", fromlist=["RunSummary"]).RunSummary.model_validate(value)
        for key, value in result["cases"][0]["runs"].items()
    }
    summaries["B"].starting_model_hash = "different"
    with pytest.raises(ValueError, match="Unmatched"):
        compare_design_runs(summaries, load_task(task_path))


def test_text_paths_and_test_labels_excluded_from_context(tmp_path):
    from test_adapters import text_task

    task = text_task(tmp_path)
    policy = load_policy(ROOT / "configs/policy.json")
    context = make_context(
        task,
        policy,
        {"experiments": [], "prior_evidence": []},
        "planner",
        variant=WorkflowVariant(name="C"),
    )
    serialized = json.dumps(context)
    assert task.dataset.test_file not in serialized
    assert task.dataset.test_sha256 not in serialized
    assert task.text_model.local_path not in serialized
