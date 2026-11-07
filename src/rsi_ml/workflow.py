"""Frozen-policy inner loop, coordinated by an explicit LangGraph state graph."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .agents import AgentBackend, BackendError, BudgetExceeded
from .context import knowledge_for, make_context, validate_prior_evidence
from .diagnostics import diagnostic_report
from .evaluation import summarize_run
from .report import write_report
from .schemas import (
    Development,
    Experiment,
    Plan,
    Policy,
    Review,
    RunSummary,
    TaskSpec,
    TrialResult,
    Usage,
    WorkflowVariant,
    digest,
)
from .storage import (
    RunLock,
    Store,
    comparable_environment,
    environment,
    read_events,
    read_json,
    write_json,
)
from .tasks.base import require_device
from .tasks.registry import prepare_task, run_trial, validate_recipe

# Checking a recipe against the limits is free in real work, so a recipe that breaks one goes
# back to the agent before it can cost an attempt. Bounded, and the same in every arm.
MAX_RECIPE_REPAIRS = 2
MIN_USEFUL_BUDGET_VS_BASELINE = 0.1


class State(TypedDict, total=False):
    recipe_feedback: dict | None
    decision_repairs: int
    next_node: str
    experiments: list[dict]
    plan: dict | None
    recipe: dict | None
    stop_reason: str
    usage: dict
    completed_operation: str
    prior_evidence: list[dict]
    diagnostic_reports: dict
    diagnostic_seconds: float
    diagnostic_requests: int
    diagnostic_trial_id: str
    diagnostic_return: str


def failure_reason(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(f"{entry['loc']}: {entry['type']}" for entry in error.errors())[:1000]
    # BackendError messages are sanitized at the provider boundary; remaining
    # ValueErrors here come from trusted contract/limit checks.
    return str(error)[:1000]


def preflight(task: TaskSpec, backend_name: str) -> list[str]:
    problems = []
    if task.protocol_status == "draft":
        problems.append(
            "Research protocol is draft: choose dataset splits, starting model, target, and budgets before marking it ready."
        )
    if task.purpose != "demo" and task.protocol_status != "ready":
        problems.append("Research tasks need an explicitly ready protocol.")
    if task.purpose != "demo" and task.artifact_policy is None:
        problems.append(
            "Choose artifact_policy explicitly before a research run: per-epoch checkpoints and raw logits."
        )
    if task.adapter != "mlp_classification":
        for split in ("train", "validation", "test"):
            for key in (f"{split}_file", f"{split}_sha256"):
                if not getattr(task.dataset, key):
                    problems.append(f"Missing dataset.{key}.")
        paths = [getattr(task.dataset, f"{s}_file") for s in ("train", "validation", "test")]
        if len({str(Path(p).resolve()) for p in paths if p}) != 3:
            problems.append("Train, validation, and test require separate paths.")
        hashes = [getattr(task.dataset, f"{s}_sha256") for s in ("train", "validation", "test")]
        if len(set(hashes)) != 3:
            problems.append("Train, validation, and test require distinct hashes.")
        if task.recipe_policy == "menu" and not task.search_space:
            problems.append("Declare the search space before execution.")
        if not task.dataset.split_manifest or not task.dataset.split_manifest_sha256:
            problems.append("Declare a split manifest and its hash for source provenance.")
    if task.adapter == "cifar100_classification":
        if task.dataset.num_classes != 100:
            problems.append("CIFAR-100 requires 100 classes.")
        for name in ("image_mean", "image_std"):
            values = getattr(task.dataset, name)
            if not values or len(values) != 3 or not all(math.isfinite(v) for v in values):
                problems.append(f"Declare three finite dataset.{name} values.")
        if task.dataset.image_std and min(task.dataset.image_std) <= 0:
            problems.append("Image standard deviations must be positive.")
    if task.adapter == "text_classification" and task.text_model is None:
        problems.append("Declare a local, pinned pretrained text_model.")
    if backend_name == "mock" and task.purpose != "demo":
        problems.append(
            "Mock decisions are restricted to software demos; use DSPy or the random baseline for real tasks."
        )
    if task.dataset.kind == "local_csv":
        if task.recipe_policy == "menu" and not task.search_space:
            problems.append(
                "Declare search_space so agents and conventional search have identical allowed changes."
            )
        for key in (
            "train_csv",
            "validation_csv",
            "train_sha256",
            "validation_sha256",
            "test_csv",
            "test_sha256",
        ):
            if not getattr(task.dataset, key):
                problems.append(f"Missing dataset.{key}.")
        hashes = [getattr(task.dataset, f"{s}_sha256") for s in ("train", "validation", "test")]
        if len(set(hashes)) != 3:
            problems.append("Train, validation, and test file hashes must all differ.")
        paths = [getattr(task.dataset, f"{s}_csv") for s in ("train", "validation", "test")]
        if len({str(Path(p).resolve()) for p in paths if p}) != 3:
            problems.append("Train, validation, and test must have separate paths.")
        if not task.dataset.revision.strip():
            problems.append("Dataset revision must be explicit.")
    try:
        validate_recipe(task, task.initial_recipe)
    except (ValueError, OSError, ImportError) as exc:
        problems.append(str(exc))
    if backend_name == "dspy" and task.limits.max_api_calls < 3:
        problems.append("DSPy mode needs a nonzero declared API call budget (at least 3).")
    return problems


class Workflow:
    def __init__(
        self,
        task: TaskSpec,
        policy: Policy,
        backend: AgentBackend,
        directory: Path,
        shared_memory: list[dict] | None = None,
        variant: str = "C",
        baseline_dir: Path | None = None,
    ):
        self.task, self.policy, self.backend = task, policy, backend
        self.root = directory.resolve()
        self.store = Store(self.root)
        self.shared_memory = shared_memory or []
        self.variant = WorkflowVariant(name=variant)
        self.baseline_dir = Path(baseline_dir).resolve() if baseline_dir else None
        if hasattr(backend, "bind_validator"):
            backend.bind_validator(lambda recipe: validate_recipe(task, recipe))

    def _stop_reason(self, state: State) -> str | None:
        records = state["experiments"]
        if self.task.stop_on_target and any(
            r["result"]["status"] == "completed"
            and r["result"]["metrics"].get(self.task.metric, -1) >= self.task.target
            for r in records
        ):
            return "target_reached"
        if len(records) >= self.task.limits.max_trials:
            return "trial_budget_exhausted"
        used = sum(r["result"]["compute_seconds"] for r in records)
        used += state.get("diagnostic_seconds", 0)
        # Too little left for a useful experiment: otherwise every run would end by being
        # forced into one last attempt that is certain to be cut off.
        useful = (
            MIN_USEFUL_BUDGET_VS_BASELINE * records[0]["result"]["compute_seconds"]
            if records
            else 0
        )
        if self.task.limits.total_compute_seconds - used <= useful:
            return "compute_budget_exhausted"
        # One proposing agent call produces the executable recipe. The trusted
        # builder uses no API; C/D charge reviewer calls to the same API budget.
        if (
            self.backend.name == "dspy"
            and self.backend.usage.calls + 1 > self.task.limits.max_api_calls
        ):
            return "api_budget_exhausted"
        return None

    def _call(self, role: str, state: State) -> Any:
        context = make_context(self.task, self.policy, state, role, variant=self.variant)
        if self.backend.name == "random":
            context["random_search"] = self.task.random_search
        self.store.event("agent_input", {"role": role, "context": context})
        try:
            result = self.backend.call(role, context, self.policy, self.task.limits)
        finally:
            state["usage"] = self.backend.usage.model_dump(mode="json")
        self.store.event(
            "agent_output",
            {"role": role, "output": result.model_dump(mode="json"), "usage": state["usage"]},
        )
        return result

    def _trial_id(self, state: State) -> str:
        return f"trial-{len(state['experiments']):04d}"

    def _record(self, state: State, result: TrialResult, recipe=None) -> None:
        trial_id = self._trial_id(state)
        record = Experiment(
            trial_id=trial_id,
            index=len(state["experiments"]),
            recipe=recipe,
            plan=state.get("plan"),
            result=result,
        )
        state["experiments"].append(record.model_dump(mode="json"))
        write_json(self.root / "trials" / trial_id / "experiment.json", record)
        self.store.event("experiment_recorded", record.model_dump(mode="json"))

    def _guard(self, state: State) -> State:
        reason = self._stop_reason(state)
        state["stop_reason"] = reason or ""
        state["plan"], state["recipe"] = None, None
        state["next_node"] = "finish" if reason else "planner"
        return state

    def _planner(self, state: State) -> State:
        state["plan"] = None  # Never let an earlier reply's plan stand in for this one.
        try:
            plan = self._call("planner", state)
            if not isinstance(plan, Plan):
                raise BackendError("Planner returned the wrong contract.")
            valid_ids = {r["trial_id"] for r in state["experiments"]}
            valid_ids.update(c["id"] for c in knowledge_for(self.task.adapter))
            for prior in state.get("prior_evidence", []):
                run_id = prior["summary"]["run_id"]
                valid_ids.add(run_id)
                valid_ids.update(f"{run_id}:{r['trial_id']}" for r in prior.get("records", []))
            invalid = sorted(set(plan.evidence_ids) - valid_ids)
            if invalid:
                # Measured behavior, not an execution failure: see agents._check_evidence.
                self.store.event("invalid_evidence_citation", {"cited": invalid})
                plan.evidence_ids = [e for e in plan.evidence_ids if e in valid_ids]
            state["plan"] = plan.model_dump(mode="json")
            if plan.diagnostic_trial_id:
                self._request_diagnostics(state, plan.diagnostic_trial_id, "planner")
            elif plan.stop and self.task.stop_on_target:
                state["stop_reason"], state["next_node"] = "planner_stopped", "finish"
            else:
                if plan.stop:
                    # Honoring this would make best-score contrasts partly measure stopping,
                    # which random search never does. The wish is kept as behavior data.
                    self.store.event("agent_stop_ignored", {"attempts": len(state["experiments"])})
                if self.task.recipe_policy == "constraints":
                    validate_recipe(self.task, plan.recipe)
                state["recipe_feedback"], state["decision_repairs"] = None, 0
                state["next_node"] = "developer"
        except (BackendError, ValueError) as exc:
            if self._return_for_repair(state, exc):
                return state
            self._record(
                state,
                TrialResult(
                    status="rejected",
                    compute_seconds=0,
                    error=f"Planning failed: {failure_reason(exc)}",
                ),
            )
            state["next_node"] = "guard"
        return state

    def _return_for_repair(self, state: State, error: Exception) -> bool:
        """Send an invalid recipe back to the agent, at no cost in attempts, a few times."""
        used = state.get("decision_repairs", 0)
        if (
            self.task.recipe_policy != "constraints"
            or isinstance(error, BudgetExceeded)
            or used >= MAX_RECIPE_REPAIRS
        ):
            state["recipe_feedback"], state["decision_repairs"] = None, 0
            return False
        state["decision_repairs"] = used + 1
        plan = state.get("plan")
        state["plan"] = None
        state["recipe_feedback"] = {
            "problem": failure_reason(error),
            "rejected_recipe": plan["recipe"] if plan else None,
            "note": "Your last recipe was not run and did not use an attempt. Fix this and resend.",
            "repairs_left": MAX_RECIPE_REPAIRS - used - 1,
        }
        self.store.event("recipe_repair_requested", state["recipe_feedback"])
        state["next_node"] = "planner"
        return True

    def _developer(self, state: State) -> State:
        try:
            development = Development(
                recipe=Plan.model_validate(state["plan"]).recipe,
                implementation_notes="Trusted declarative builder; no extra LLM role.",
            )
            # The developer implements the planner's declared experiment; changing
            # its knobs silently would invalidate the recorded hypothesis.
            if development.recipe != Plan.model_validate(state["plan"]).recipe:
                raise ValueError("Developer changed the planned recipe; replan explicitly.")
            validate_recipe(self.task, development.recipe)
            state["recipe"] = development.recipe.model_dump(mode="json")
            state["next_node"] = "execute"
        except (BackendError, ValueError) as exc:
            self._record(
                state,
                TrialResult(
                    status="rejected",
                    compute_seconds=0,
                    error=f"Development rejected: {failure_reason(exc)}",
                ),
            )
            state["next_node"] = "reviewer" if self.variant.separate_reviewer else "guard"
        return state

    def _execute(self, state: State) -> State:
        from .schemas import Recipe

        recipe = Recipe.model_validate(state["recipe"])
        used = sum(r["result"]["compute_seconds"] for r in state["experiments"]) + state.get(
            "diagnostic_seconds", 0
        )
        timeout = min(
            self.task.limits.per_trial_seconds, self.task.limits.total_compute_seconds - used
        )
        if timeout <= 0:
            state["stop_reason"], state["next_node"] = "compute_budget_exhausted", "finish"
            return state
        if self.baseline_dir and not state["experiments"]:
            # The first attempt is the identical recipe and seed in every arm, so it is
            # trained once. Its full measured cost is still charged to this run.
            from .baselines import adopt_baseline

            result = adopt_baseline(
                self.task,
                self.baseline_dir,
                self.root / "trials" / self._trial_id(state),
                read_json(self.root / "manifest.json")["prepared"],
            )
            self.store.event(
                "baseline_adopted",
                {"source": str(self.baseline_dir), "compute_seconds": result.compute_seconds},
            )
        else:
            result = run_trial(
                self.task,
                recipe,
                self.root / "prepared",
                self.root / "trials" / self._trial_id(state),
                timeout,
            )
        self._record(state, result, recipe)
        next_node = "reviewer" if self.variant.separate_reviewer else "guard"
        state["next_node"] = next_node
        if self.variant.automatic_diagnostics and result.status == "completed":
            state["diagnostic_trial_id"] = state["experiments"][-1]["trial_id"]
            state["diagnostic_return"] = next_node
            state["next_node"] = "diagnostics"
        return state

    def _request_diagnostics(self, state: State, trial_id: str, destination: str):
        self._require_diagnostic_access(trial_id, destination)
        if state.get("diagnostic_requests", 0) >= self.task.limits.max_diagnostic_requests:
            raise ValueError("Diagnostic request budget exhausted.")
        if not any(
            r["trial_id"] == trial_id and r["result"]["status"] == "completed"
            for r in state["experiments"]
        ):
            raise ValueError("Diagnostics require a completed current-run trial.")
        state["diagnostic_requests"] = state.get("diagnostic_requests", 0) + 1
        state["diagnostic_trial_id"], state["diagnostic_return"] = trial_id, destination
        state["next_node"] = "diagnostics"

    def _require_diagnostic_access(self, trial_id: str, caller: str) -> None:
        if not self.variant.detailed_error_reports:
            self.store.event(
                "diagnostic_access_denied",
                {"variant": self.variant.name, "trial_id": trial_id, "caller": caller},
            )
            raise ValueError(
                f"Detailed error reports are unavailable in variant {self.variant.name}."
            )

    def _diagnostics(self, state: State) -> State:
        trial_id = state["diagnostic_trial_id"]
        # Check the execution boundary too, including restored graph states.
        self._require_diagnostic_access(trial_id, "diagnostics_node")
        started = time.monotonic()
        used = sum(r["result"]["compute_seconds"] for r in state["experiments"]) + state.get(
            "diagnostic_seconds", 0
        )
        if used >= self.task.limits.total_compute_seconds:
            state["stop_reason"], state["next_node"] = "compute_budget_exhausted", "finish"
            return state
        try:
            reports = state.setdefault("diagnostic_reports", {})
            if trial_id not in reports:
                reports[trial_id] = diagnostic_report(self.root / "trials" / trial_id)
            self.store.event(
                "diagnostic_report", {"trial_id": trial_id, "report": reports[trial_id]}
            )
        except (ValueError, KeyError, FileNotFoundError) as exc:
            # Other I/O errors propagate: a machine fault must not silently remove a report
            # from a B/D decision. The run stops at a resumable point instead.
            self.store.event("diagnostics_failed", {"trial_id": trial_id, "error": str(exc)})
        finally:
            elapsed = time.monotonic() - started
            state["diagnostic_seconds"] = state.get("diagnostic_seconds", 0) + elapsed
            self.store.event(
                "diagnostic_cost",
                {
                    "trial_id": trial_id,
                    "seconds": elapsed,
                    "after_attempt": len(state["experiments"]),
                },
            )
        state["next_node"] = state["diagnostic_return"]
        if used + elapsed >= self.task.limits.total_compute_seconds:
            state["stop_reason"], state["next_node"] = "compute_budget_exhausted", "finish"
        return state

    def _reviewer(self, state: State) -> State:
        try:
            review = self._call("reviewer", state)
            if not isinstance(review, Review):
                raise BackendError("Reviewer returned the wrong contract.")
            if review.diagnostic_trial_id:
                self._request_diagnostics(state, review.diagnostic_trial_id, "reviewer")
                return state
            state["experiments"][-1]["review"] = review.model_dump(mode="json")
            record = state["experiments"][-1]
            write_json(self.root / "trials" / record["trial_id"] / "experiment.json", record)
        except (BackendError, ValueError) as exc:
            self.store.event(
                "review_failed",
                {
                    "error_type": type(exc).__name__,
                    "error": failure_reason(exc),
                    "note": "Code-generated metrics are retained unchanged.",
                },
            )
        state["next_node"] = "guard"
        return state

    def _finish(self, state: State) -> State:
        state["next_node"] = "done"
        return state

    def graph(self):
        builder = StateGraph(State)
        names = ["guard", "planner", "developer", "execute", "reviewer", "diagnostics", "finish"]
        for name in names:
            function = getattr(self, f"_{name}")

            def node(state, fn=function, stage=name):
                state = deepcopy(state)
                token = self.store.begin(stage)
                result = fn(state)
                self.store.commit(stage, token, result)
                return result

            builder.add_node(name, node)
        routes = {n: n for n in names} | {"done": END}
        builder.add_conditional_edges(START, lambda s: s["next_node"], routes)
        for name in names:
            builder.add_conditional_edges(name, lambda s: s["next_node"], routes)
        return builder.compile()

    def run(self, *, resume: bool = False, recover: bool = False) -> RunSummary:
        problems = preflight(self.task, self.backend.name)
        if problems:
            raise ValueError("\n".join(problems))
        validate_prior_evidence(self.shared_memory, self.variant)
        require_device(self.task)
        identity = {
            "variant": self.variant.name,
            "variant_settings": self.variant.settings(),
            "task_hash": digest(self.task),
            "policy_hash": digest(self.policy),
            "backend_fingerprint": digest(self.backend.metadata),
            "execution_fingerprint": digest(comparable_environment()),
            "memory_fingerprint": digest(self.shared_memory),
        }
        if not resume:
            self.root.mkdir(parents=True, exist_ok=False)
        with RunLock(self.root, recover=recover):
            if resume:
                manifest = read_json(self.root / "manifest.json")
                if any(manifest.get(k) != v for k, v in identity.items()):
                    raise ValueError(
                        "Resume requires unchanged task, policy, backend, and source/environment."
                    )
                state = self.store.resumable_state(recover=recover)
                self.backend.restore_usage(Usage.model_validate(state["usage"]))
                prepared = prepare_task(self.task, self.root / "prepared")
            else:
                write_json(self.root / "task.json", self.task)
                write_json(self.root / "policy.json", self.policy)
                write_json(self.root / "shared_memory.json", self.shared_memory)
                preparation_start = time.monotonic()
                prepared = prepare_task(self.task, self.root / "prepared")
                prepared["preparation_seconds"] = time.monotonic() - preparation_start
                write_json(
                    self.root / "manifest.json",
                    {
                        **identity,
                        "run_id": self.root.name,
                        "backend": self.backend.metadata,
                        "environment": environment(),
                        "baseline_dir": str(self.baseline_dir) if self.baseline_dir else None,
                        "prepared": prepared,
                        "mock_warning": self.backend.name == "mock",
                    },
                )
                # Everyone starts with the exact same supplied baseline recipe.
                # It is a counted training experiment, not a free warm start.
                state = State(
                    next_node="execute",
                    experiments=[],
                    recipe=self.task.initial_recipe.model_dump(mode="json"),
                    plan=None,
                    stop_reason="",
                    usage=self.backend.usage.model_dump(mode="json"),
                    prior_evidence=self.shared_memory,
                    diagnostic_reports={},
                    diagnostic_seconds=0,
                    diagnostic_requests=0,
                )
                write_json(self.root / "state.json", state)
            if state["next_node"] != "done":
                state = self.graph().invoke(
                    state,
                    {
                        # Up to eight nodes per decision, plus recipe repairs.
                        "recursion_limit": self.task.limits.max_trials * 14
                        + self.task.limits.max_diagnostic_requests * 4
                        + 20
                    },
                )
            records = [Experiment.model_validate(r) for r in state["experiments"]]
            summary = summarize_run(
                self.root.name,
                self.task,
                self.policy,
                self.backend.name,
                records,
                self.backend.usage,
                prepared,
                state["stop_reason"],
            )
            summary.backend_fingerprint = identity["backend_fingerprint"]
            summary.execution_fingerprint = identity["execution_fingerprint"]
            summary.memory_fingerprint = identity["memory_fingerprint"]
            summary.variant = self.variant.name
            summary.diagnostic_seconds = state.get("diagnostic_seconds", 0)
            summary.diagnostic_requests = state.get("diagnostic_requests", 0)
            summary.preparation_seconds = read_json(self.root / "manifest.json")["prepared"].get(
                "preparation_seconds", 0
            )
            summary.total_compute_seconds += summary.diagnostic_seconds
            if summary.reached_target:
                # Costs charged before the target trial; the report afterwards is total overhead.
                summary.compute_to_target += sum(
                    e["payload"]["seconds"]
                    for e in read_events(self.root / "events.jsonl")
                    if e["event"] == "diagnostic_cost"
                    and e["payload"]["after_attempt"] < summary.trials_to_target
                )
            write_json(self.root / "summary.json", summary)
            write_report(self.root, summary, records)
            return summary
