"""Machine failures are not agent decisions, and behavior metrics mean what they say.

Runs on the VM with the rest of the suite. Nothing here needs a GPU.
"""

import json
import subprocess
import time
from pathlib import Path

import pytest
import torch
from test_training import tiny_task

from rsi_ml.agents import MockBackend
from rsi_ml.cli import load_policy, load_task
from rsi_ml.context import make_context
from rsi_ml.diagnostics import _validation_text
from rsi_ml.figures import Run, decision_rows
from rsi_ml.schemas import Experiment, Plan, Recipe, TaskSpec, TrialResult, WorkflowVariant
from rsi_ml.storage import Store, read_events, write_json
from rsi_ml.tasks.base import INFRASTRUCTURE_EXIT_CODE, InfrastructureFault
from rsi_ml.tasks.mlp import prepare_task, run_trial
from rsi_ml.worker import is_infrastructure_fault
from rsi_ml.workflow import Workflow

ROOT = Path(__file__).parents[1]


def _demo():
    return load_task(ROOT / "configs/mlp-demo.json"), load_policy(ROOT / "configs/policy.json")


def _fake_tools(monkeypatch, scores):
    calls = []
    remaining = iter(scores)

    def execute(task, recipe, prepared, out, timeout):
        calls.append(recipe)
        score = next(remaining)
        return TrialResult(
            status="completed", compute_seconds=1, metrics={"accuracy": score, "macro_f1": score}
        )

    monkeypatch.setattr(
        "rsi_ml.workflow.prepare_task",
        lambda task, directory: {"starting_model_hash": "w", "dataset_hash": "d"},
    )
    monkeypatch.setattr("rsi_ml.workflow.run_trial", execute)
    return calls


class _AlwaysStops(MockBackend):
    def call(self, role, context, policy, limits):
        result = super().call(role, context, policy, limits)
        if role == "planner":
            result.stop = True
        return result


def test_agent_stop_is_ignored_when_the_budget_must_be_spent(tmp_path, monkeypatch):
    task, policy = _demo()
    task.stop_on_target = False
    task.limits.max_trials = 3
    calls = _fake_tools(monkeypatch, [0.5, 0.6, 0.7])
    run = tmp_path / "keeps-going"
    summary = Workflow(task, policy, _AlwaysStops(), run, variant="A").run()
    assert len(calls) == 3
    assert summary.stop_reason == "trial_budget_exhausted"
    assert "agent_stop_ignored" in (run / "events.jsonl").read_text()


def test_agent_stop_still_ends_the_run_when_stopping_is_allowed(tmp_path, monkeypatch):
    task, policy = _demo()
    assert task.stop_on_target
    calls = _fake_tools(monkeypatch, [0.5, 0.6])
    summary = Workflow(task, policy, _AlwaysStops(), tmp_path / "stops", variant="A").run()
    assert len(calls) == 1
    assert summary.stop_reason == "planner_stopped"


class _DeadWorker:
    pid = 0

    def __init__(self, code):
        self.code = code

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code


class _SlowWorker(_DeadWorker):
    """Would outlast any limit."""

    def __init__(self):
        super().__init__(0)

    def wait(self, timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("worker", timeout)
        return -9


@pytest.mark.parametrize(
    "budget,expected",
    [(5, "Compute budget ran out"), (10_000, "per-trial time limit")],
)
def test_a_stopped_trial_says_which_limit_stopped_it(tmp_path, monkeypatch, budget, expected):
    task = tiny_task()
    prepare_task(task, tmp_path / "prepared")
    monkeypatch.setattr("rsi_ml.tasks.mlp.subprocess.Popen", lambda *a, **k: _SlowWorker())
    monkeypatch.setattr("rsi_ml.tasks.mlp.os.killpg", lambda *a: None)
    result = run_trial(task, task.initial_recipe, tmp_path / "prepared", tmp_path / "t", budget)
    assert result.status == "timeout" and expected in result.error


def test_process_startup_is_recorded_but_not_charged(tmp_path):
    task = tiny_task()
    prepare_task(task, tmp_path / "prepared")
    started = time.monotonic()
    result = run_trial(task, task.initial_recipe, tmp_path / "prepared", tmp_path / "t", 60)
    wall = time.monotonic() - started
    assert result.status == "completed", (tmp_path / "t" / "stderr.log").read_text()
    assert result.startup_seconds > 0 and result.compute_seconds > 0
    assert result.compute_seconds + result.startup_seconds <= wall
    assert (tmp_path / "t" / "worker_ready").exists()


def test_a_run_ends_when_too_little_budget_is_left_to_be_useful(tmp_path, monkeypatch):
    task, policy = _demo()
    task.limits.total_compute_seconds = 2.05  # Two 1 s attempts leave only 5% of a baseline.
    task.limits.max_trials = 4
    calls = _fake_tools(monkeypatch, [0.5, 0.6, 0.7, 0.8])
    summary = Workflow(task, policy, MockBackend(), tmp_path / "run", variant="A").run()
    assert len(calls) == 2 and summary.stop_reason == "compute_budget_exhausted"


def test_a_schema_break_tells_the_agent_which_rule_without_echoing_the_response():
    from pydantic import ValidationError

    from rsi_ml.agents import _schema_problem

    task, _ = _demo()
    values = task.initial_recipe.model_dump()
    with pytest.raises(ValidationError) as caught:
        Recipe.model_validate({**values, "dropout": 5.0})
    problem = _schema_problem(caught.value)
    assert "dropout" in problem and "5.0" not in problem
    assert _schema_problem(RuntimeError("Authorization: Bearer secret")) == ""


def test_random_search_draws_conventional_rates_and_runs_that_can_finish():
    from rsi_ml.baseline import MAX_STEPS_VS_BASELINE, RandomSearchBackend

    protocol = json.loads((ROOT / "configs/study/mlp-demo.json").read_text())
    spec = protocol["tasks"]["spambase"]
    initial = {**_demo()[0].initial_recipe.model_dump(mode="json"), **spec["initial_recipe"]}
    backend = RandomSearchBackend()
    for count in range(40):
        recipe = backend._plan(
            {
                "task": {"seed": 1, "initial_recipe": initial},
                "records": [],
                "history_count": count,
                "random_search": spec["random_search"],
            }
        ).recipe
        steps = (recipe.epochs / initial["epochs"]) * (initial["batch_size"] / recipe.batch_size)
        assert steps <= MAX_STEPS_VS_BASELINE
        if recipe.optimizer == "adamw":
            assert recipe.learning_rate <= 0.01
            assert recipe.momentum == 0


def test_random_search_honors_a_declared_longer_step_cap():
    from rsi_ml.baseline import RandomSearchBackend

    protocol = json.loads((ROOT / "configs/study/mlp-covertype-600s.json").read_text())
    spec = protocol["tasks"]["covertype"]
    initial = {**_demo()[0].initial_recipe.model_dump(mode="json"), **spec["initial_recipe"]}
    backend = RandomSearchBackend()
    longest = 0.0
    for count in range(80):
        recipe = backend._plan(
            {
                "task": {"seed": 1, "initial_recipe": initial},
                "records": [],
                "history_count": count,
                "random_search": spec["random_search"],
            }
        ).recipe
        steps = (recipe.epochs / initial["epochs"]) * (initial["batch_size"] / recipe.batch_size)
        assert steps <= spec["random_search"]["max_steps_vs_baseline"]
        longest = max(longest, steps)
    assert longest > 2.5


@pytest.mark.parametrize("code", [INFRASTRUCTURE_EXIT_CODE, -9])
def test_a_machine_failure_stops_the_run_and_is_not_an_attempt(tmp_path, monkeypatch, code):
    task = tiny_task()
    prepare_task(task, tmp_path / "prepared")
    monkeypatch.setattr(
        "rsi_ml.tasks.mlp.subprocess.Popen", lambda *args, **kwargs: _DeadWorker(code)
    )
    with pytest.raises(InfrastructureFault):
        run_trial(task, task.initial_recipe, tmp_path / "prepared", tmp_path / "trial", 60)
    assert not (tmp_path / "trial" / "result.json").exists()


def test_prepared_data_faults_and_spawn_faults_are_not_attempts(tmp_path, monkeypatch):
    task = tiny_task()
    prepare_task(task, tmp_path / "prepared")

    def cannot_start(*args, **kwargs):
        raise OSError(11, "Resource temporarily unavailable")

    monkeypatch.setattr("rsi_ml.tasks.mlp.subprocess.Popen", cannot_start)
    with pytest.raises(InfrastructureFault, match="could not be started"):
        run_trial(task, task.initial_recipe, tmp_path / "prepared", tmp_path / "spawn", 60)
    (tmp_path / "prepared" / "manifest.json").chmod(0o644)
    (tmp_path / "prepared" / "manifest.json").write_text('{"task_hash": "changed"}')
    with pytest.raises(InfrastructureFault, match="failed verification"):
        run_trial(task, task.initial_recipe, tmp_path / "prepared", tmp_path / "changed", 60)


def test_a_lost_gpu_is_refused_before_a_run_records_its_identity(tmp_path, monkeypatch):
    task, policy = _demo()
    task.device = "cuda"
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    directory = tmp_path / "not-created"
    with pytest.raises(InfrastructureFault, match="unavailable"):
        Workflow(task, policy, MockBackend(), directory, variant="A").run()
    assert not directory.exists()


def test_a_disk_fault_during_a_report_stops_rather_than_dropping_the_report(tmp_path, monkeypatch):
    task, policy = _demo()
    task.limits.max_trials = 2
    _fake_tools(monkeypatch, [0.5, 0.6])

    def broken_disk(path):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr("rsi_ml.workflow.diagnostic_report", broken_disk)
    run = tmp_path / "stopped"
    with pytest.raises(OSError):
        Workflow(task, policy, MockBackend(), run, variant="B").run()
    assert (run / "inflight.json").exists()


def test_a_recipe_failure_is_still_an_attempt_and_says_why(tmp_path, monkeypatch):
    task = tiny_task()
    prepare_task(task, tmp_path / "prepared")

    def popen(*args, stderr, **kwargs):
        stderr.write(b"Traceback...\nValueError: Nonfinite training loss.\n")
        stderr.flush()
        return _DeadWorker(1)

    monkeypatch.setattr("rsi_ml.tasks.mlp.subprocess.Popen", popen)
    result = run_trial(task, task.initial_recipe, tmp_path / "prepared", tmp_path / "trial", 60)
    assert result.status == "failed"
    assert "Nonfinite training loss" in result.error


@pytest.mark.parametrize(
    "error,machine",
    [
        (InfrastructureFault("no GPU"), True),
        (MemoryError(), True),
        (OSError(28, "No space left on device"), True),
        (RuntimeError("CUDA error: device lost"), True),
        (RuntimeError("CUDA error: CUBLAS_STATUS_INTERNAL_ERROR"), True),
        (RuntimeError("[enforce fail] PytorchStreamWriter failed writing file"), True),
        # Depends on the model and batch size chosen, on hardware every arm shares.
        (torch.cuda.OutOfMemoryError("CUDA out of memory."), False),
        (RuntimeError("CUDA error: out of memory"), False),
        (RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED"), False),
        (ValueError("Nonfinite training loss."), False),
        (RuntimeError("mat1 and mat2 shapes cannot be multiplied"), False),
    ],
)
def test_worker_separates_machine_failures_from_recipe_failures(error, machine):
    assert is_infrastructure_fault(error) is machine


def test_a_line_cut_off_by_a_crash_cannot_swallow_the_next_event(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "events.jsonl").write_text('{"event": "first", "payload": {}}\n{"event": "cut')
    write_json(root / "state.json", {"experiments": [], "next_node": "planner"})
    write_json(root / "inflight.json", {"operation": "abcd1234", "node": "planner"})
    store = Store(root)
    store.resumable_state(recover=True)
    store.event("after", {})
    assert [e["event"] for e in read_events(root / "events.jsonl")] == [
        "first",
        "interrupted_operation_recovered",
        "after",
    ]


def test_unicode_line_separators_do_not_split_text_rows(tmp_path):
    record = {"row_id": "1", "sentence": "a\u2028b\u0085c"}
    (tmp_path / "validation.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert _validation_text(tmp_path, ["sentence"])["1"]["sentence"] == record["sentence"]


def _constraints_task():
    task, policy = _demo()
    return task.model_copy(update={"recipe_policy": "constraints", "search_space": {}}), policy


class _OversizedFirst(MockBackend):
    """Breaks the parameter ceiling a set number of times, then behaves."""

    def __init__(self, oversized_tries):
        super().__init__()
        self.tries = oversized_tries
        self.feedback = []

    def call(self, role, context, policy, limits):
        result = super().call(role, context, policy, limits)
        if role == "planner":
            self.feedback.append(context.get("recipe_feedback"))
            if self.tries:
                self.tries -= 1
                result.recipe = result.recipe.model_copy(update={"hidden_sizes": [4096]})
        return result


def test_a_recipe_over_a_limit_goes_back_to_the_agent_without_costing_an_attempt(
    tmp_path, monkeypatch
):
    task, policy = _constraints_task()
    task.limits.max_trials = 2
    calls = _fake_tools(monkeypatch, [0.5, 0.6])
    backend = _OversizedFirst(2)
    run = tmp_path / "repaired"
    summary = Workflow(task, policy, backend, run, variant="A").run()
    assert summary.total_trials == 2 and len(calls) == 2
    assert "parameters" in backend.feedback[1]["problem"]
    assert backend.feedback[2]["repairs_left"] == 0
    assert (run / "events.jsonl").read_text().count("recipe_repair_requested") == 2


def test_repairs_are_bounded_then_the_attempt_is_lost(tmp_path, monkeypatch):
    task, policy = _constraints_task()
    task.limits.max_trials = 2
    calls = _fake_tools(monkeypatch, [0.5, 0.6])
    summary = Workflow(task, policy, _OversizedFirst(3), tmp_path / "lost", variant="A").run()
    assert summary.total_trials == 2 and len(calls) == 1
    assert summary.stop_reason == "trial_budget_exhausted"


def test_a_menu_task_keeps_its_old_behavior(tmp_path, monkeypatch):
    task, policy = _demo()
    task.limits.max_trials = 2
    calls = _fake_tools(monkeypatch, [0.5, 0.6])
    Workflow(task, policy, _OversizedFirst(1), tmp_path / "menu", variant="A").run()
    assert len(calls) == 1  # Rejected immediately, with no repair round.


def test_recipe_shapes_are_free_but_must_be_consistent():
    task, _ = _demo()
    values = task.initial_recipe.model_dump()
    Recipe.model_validate({**values, "cnn_channels": [8] * 5, "cnn_blocks": [2] * 5})
    Recipe.model_validate({**values, "hidden_sizes": [8] * 16})
    with pytest.raises(ValueError, match="same length"):
        Recipe.model_validate({**values, "cnn_channels": [8, 16], "cnn_blocks": [1, 1, 1]})


def test_a_constraints_task_cannot_also_carry_a_menu():
    task, _ = _demo()
    with pytest.raises(ValueError, match="cannot also declare"):
        TaskSpec.model_validate({**task.model_dump(), "recipe_policy": "constraints"})


def test_figures_report_the_size_of_the_selected_model(tmp_path):
    from rsi_ml.figures import _best_model_parameters

    run = _behavior_run(tmp_path, None, [(0.5, {}), (0.7, {"dropout": 0.1}), (0.6, {})], "o")
    for size, record in zip((100, 900, 50), run.records):
        record.result.parameter_count = size
    assert _best_model_parameters(run) == 900


def test_every_attempt_stays_visible_even_when_only_a_few_are_shown_in_full():
    task, policy = _constraints_task()
    recipe = task.initial_recipe.model_dump(mode="json")
    records = [
        {
            "trial_id": f"trial-{i:04d}",
            "recipe": {**recipe, "dropout": 0.1} if i else recipe,
            "result": {
                "status": "completed",
                "metrics": {"accuracy": 0.5 + i / 100},
                "compute_seconds": 1.26,
                "parameter_count": 100,
            },
        }
        for i in range(12)
    ]
    state = {
        "experiments": records,
        "diagnostic_reports": {"trial-0000": {"confident_errors": ["report-only-text"]}},
    }
    context = make_context(task, policy, state, "planner", variant=WorkflowVariant(name="A"))
    assert len(context["records"]) <= 7
    table = context["all_attempts"]
    assert [row["trial_id"] for row in table] == [r["trial_id"] for r in records]
    assert table[0]["changed_from_starting_recipe"] == {}
    assert table[3]["changed_from_starting_recipe"] == {"dropout": 0.1}
    assert table[3]["score"] == pytest.approx(0.53) and table[3]["seconds"] == 1.3
    assert "report-only-text" not in json.dumps(context)  # Still no report in A.


def test_general_knowledge_cards_reach_every_task():
    from rsi_ml.context import knowledge_for

    expected = {"controlled_experiment", "wsd", "bohb"}
    for adapter in ("mlp_classification", "cifar100_classification", "text_classification"):
        assert expected <= {card["id"] for card in knowledge_for(adapter)}


def test_file_hashes_stay_out_of_agent_context():
    task, policy = _demo()
    result = {
        "status": "completed",
        "metrics": {"accuracy": 0.5},
        "artifact_hashes": {"checkpoint.pt": "hash-agents-must-not-receive"},
        "compute_seconds": 1,
    }
    state = {"experiments": [{"trial_id": "trial-0000", "result": result}]}
    context = make_context(task, policy, state, "reviewer", variant=WorkflowVariant(name="D"))
    assert "hash-agents-must-not-receive" not in json.dumps(context)
    assert "artifact_hashes" in state["experiments"][0]["result"]  # The record itself is intact.


def _behavior_run(tmp_path, class_names, updates, text):
    """Trial 0 is the baseline; each later (score, recipe changes) is one decision."""
    task, _ = _demo()
    if class_names:
        dataset = task.dataset.model_copy(
            update={"num_classes": len(class_names), "class_names": class_names}
        )
        task = task.model_copy(update={"dataset": dataset})
    records = []
    for index, (score, changes) in enumerate(updates):
        recipe = task.initial_recipe.model_copy(update=changes)
        plan = None
        if index:
            plan = Plan(
                observation=text,
                hypothesis="h",
                predicted_effect="p",
                experiment="e",
                recipe=recipe,
            )
        records.append(
            Experiment(
                trial_id=f"trial-{index:04d}",
                index=index,
                recipe=recipe,
                plan=plan,
                result=TrialResult(
                    status="completed",
                    compute_seconds=1,
                    metrics={"accuracy": score, "macro_f1": score},
                ),
            )
        )
    return Run("t", 1, "A", tmp_path, task, None, records, [])


def test_built_on_best_describes_the_new_recipe(tmp_path):
    good = {"dropout": 0.1}
    worse = {"dropout": 0.1, "learning_rate": 0.005}
    further = {"dropout": 0.1, "learning_rate": 0.005, "weight_decay": 0.001}
    back_to_best = {"dropout": 0.1, "weight_decay": 0.001}
    rows = decision_rows(
        _behavior_run(
            tmp_path,
            None,
            [(0.5, {}), (0.6, good), (0.4, worse), (0.3, further), (0.5, back_to_best)],
            "o",
        )
    )
    assert [r["built_on_best"] for r in rows] == [True, True, False, True]


@pytest.mark.parametrize(
    "names,text,expected",
    [
        (["train", "can", "maple_tree", "oak_tree"], "We can train longer.", False),
        (["train", "can", "maple_tree", "oak_tree"], "maple tree errors dominate.", True),
        (["negative", "positive"], "A positive effect on negative class.", None),
    ],
)
def test_class_mentions_ignore_ordinary_words(tmp_path, names, text, expected):
    rows = decision_rows(_behavior_run(tmp_path, names, [(0.5, {}), (0.6, {"dropout": 0.1})], text))
    assert rows[0]["plan_mentions_class_name"] is expected
