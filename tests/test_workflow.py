"""Check orchestration invariants with synthetic tool receipts, not benchmark claims."""

from pathlib import Path

import pytest

from rsi_ml.agents import MockBackend
from rsi_ml.cli import load_policy, load_task
from rsi_ml.context import make_context
from rsi_ml.improvement import propose_policy
from rsi_ml.schemas import TrialResult, WorkflowVariant, digest
from rsi_ml.storage import Store, read_json, write_json
from rsi_ml.workflow import Workflow, preflight

ROOT = Path(__file__).parents[1]


@pytest.fixture
def setup():
    task = load_task(ROOT / "configs/mlp-demo.json")
    policy = load_policy(ROOT / "configs/policy.json")
    return task, policy


def fake_tools(monkeypatch, scores):
    calls = []
    scores = iter(scores)

    def prepare(task, directory):
        return {"starting_model_hash": "fixed-initial-weights", "dataset_hash": "fixed-data"}

    def execute(task, recipe, prepared, out, timeout):
        calls.append(recipe)
        score = next(scores)
        return TrialResult(
            status="failed" if score is None else "completed",
            compute_seconds=1,
            metrics={} if score is None else {"accuracy": score, "macro_f1": score},
            error="Intentional synthetic failure" if score is None else None,
        )

    monkeypatch.setattr("rsi_ml.workflow.prepare_task", prepare)
    monkeypatch.setattr("rsi_ml.workflow.run_trial", execute)
    return calls


def test_count_initial_failed_attempts_and_stop_at_target(tmp_path, monkeypatch, setup):
    task, policy = setup
    calls = fake_tools(monkeypatch, [0.5, None, 1.0])
    run = tmp_path / "inner-loop"
    summary = Workflow(task, policy, MockBackend(), run).run()
    assert calls[0] == task.initial_recipe
    assert summary.reached_target
    assert summary.trials_to_target == 3
    assert summary.compute_to_target == 3
    assert summary.total_trials == 3
    assert read_json(run / "state.json")["next_node"] == "done"
    assert len(read_json(run / "state.json")["experiments"]) == 3
    assert (run / "report.html").is_file()


def test_completed_resume_does_not_execute_again(tmp_path, monkeypatch, setup):
    task, policy = setup
    calls = fake_tools(monkeypatch, [1.0])
    run = tmp_path / "resume"
    original = Workflow(task, policy, MockBackend(), run).run()
    resumed = Workflow(task, policy, MockBackend(), run).run(resume=True)
    assert original == resumed
    assert len(calls) == 1
    changed = policy.model_copy(
        update={"planner_instructions": "Different instructions that must be rejected."}
    )
    with pytest.raises(ValueError, match="unchanged"):
        Workflow(task, changed, MockBackend(), run).run(resume=True)


def test_no_silent_rerun_of_inflight_operation(tmp_path):
    write_json(tmp_path / "state.json", {"next_node": "execute", "completed_operation": "old"})
    write_json(tmp_path / "inflight.json", {"operation": "unknown", "node": "execute"})
    with pytest.raises(RuntimeError, match="No automatic rerun"):
        Store(tmp_path).resumable_state()


def test_trusted_builder_rejects_out_of_space_recipe(tmp_path, monkeypatch, setup):
    task, policy = setup
    calls = fake_tools(monkeypatch, [0.5])

    class InvalidPlanner(MockBackend):
        def call(self, role, context, policy, limits):
            result = super().call(role, context, policy, limits)
            assert role != "developer", "A-D use a trusted builder, not another LLM developer."
            if role == "planner":
                result.recipe.dropout = 0.75
            return result

    task.limits.max_trials = 2
    summary = Workflow(task, policy, InvalidPlanner(), tmp_path / "rejected").run()
    assert len(calls) == 1
    assert summary.total_trials == 2
    assert summary.total_compute_seconds == 1
    state = read_json(tmp_path / "rejected/state.json")
    assert state["experiments"][1]["result"]["status"] == "rejected"


def test_prior_memory_is_hashed_and_no_test_path_enters_context(tmp_path, monkeypatch, setup):
    task, policy = setup
    task.dataset.test_csv = "/private/held-out-labels.csv"
    fake_tools(monkeypatch, [1.0])
    memory = [{"summary": {"run_id": "prior"}, "records": []}]
    run = tmp_path / "memory"
    summary = Workflow(task, policy, MockBackend(), run, shared_memory=memory).run()
    assert summary.memory_fingerprint == digest(memory)
    context = make_context(
        task, policy, read_json(run / "state.json"), "planner", variant=WorkflowVariant(name="C")
    )
    assert "test_csv" not in context["task"]["dataset"]
    with pytest.raises(ValueError, match="unchanged"):
        Workflow(task, policy, MockBackend(), run, shared_memory=[]).run(resume=True)


def test_policy_revision_is_separate_and_cannot_read_final_eval(tmp_path, monkeypatch, setup):
    task, policy = setup
    fake_tools(monkeypatch, [0.5, 0.6, 0.7, 0.8])
    run = tmp_path / "development-demo"
    Workflow(task, policy, MockBackend(), run).run()
    original_hash = digest(policy)
    candidate = propose_policy(policy, [run], MockBackend(), tmp_path / "proposal")
    assert digest(policy) == original_hash
    assert candidate.status == "candidate"
    assert candidate.parent_hash == original_hash
    assert candidate.developer_instructions == policy.developer_instructions
    assert (tmp_path / "proposal/shared_evidence.json").is_file()
    summary = read_json(run / "summary.json")
    summary["purpose"] = "evaluation"
    write_json(run / "summary.json", summary)
    with pytest.raises(ValueError, match="development runs"):
        propose_policy(policy, [run], MockBackend(), tmp_path / "forbidden")


def test_draft_protocol_never_executes(tmp_path, monkeypatch, setup):
    task, policy = setup
    task.protocol_status = "draft"
    calls = fake_tools(monkeypatch, [])
    assert preflight(task, "mock")
    with pytest.raises(ValueError, match="draft"):
        Workflow(task, policy, MockBackend(), tmp_path / "not-started").run()
    assert calls == []
    assert not (tmp_path / "not-started").exists()


def test_invalid_citation_is_logged_and_removed(tmp_path, monkeypatch, setup):
    task, policy = setup
    task.limits.max_trials = 2
    fake_tools(monkeypatch, [0.5, 0.6])

    class Citing(MockBackend):
        def call(self, role, context, policy, limits):
            result = super().call(role, context, policy, limits)
            if role == "planner":
                result.evidence_ids = ["trial-0000", "made-up-paper"]
            return result

    run = tmp_path / "citing"
    summary = Workflow(task, policy, Citing(), run).run()
    assert summary.total_trials == 2  # The second attempt still trained.
    state = read_json(run / "state.json")
    assert state["experiments"][1]["result"]["status"] == "completed"
    assert state["experiments"][1]["plan"]["evidence_ids"] == ["trial-0000"]
    events = (run / "events.jsonl").read_text()
    assert "invalid_evidence_citation" in events and "made-up-paper" in events


def test_stop_on_target_false_keeps_spending_the_budget(tmp_path, monkeypatch, setup):
    task, policy = setup
    task.stop_on_target = False
    task.limits.max_trials = 3
    calls = fake_tools(monkeypatch, [1.0, 0.5, 0.5])
    summary = Workflow(task, policy, MockBackend(), tmp_path / "continues").run()
    assert len(calls) == 3
    assert summary.stop_reason == "trial_budget_exhausted"
    # Time to target stays derivable from the complete trajectory.
    assert summary.reached_target and summary.trials_to_target == 1
