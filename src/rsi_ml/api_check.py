"""Check the API key and, only on request, make one real planner call."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from .agents import DSPyBackend, Usage
from .study import load_api_key


def _get(url: str, key: str | None = None) -> dict:
    headers = {"User-Agent": "rsi-ml-api-check"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def api_check(
    env_file: Path,
    *,
    model: str | None,
    reasoning_effort: str | None,
    api_base: str,
    live: bool,
) -> dict:
    key = load_api_key(env_file)["RSI_API_KEY"]
    data = _get(f"{api_base}/key", key).get("data", {})
    # The provider's `label` contains a key prefix, so only numeric fields are kept.
    report = {
        "credit_limit_usd": data.get("limit"),
        "credit_remaining_usd": data.get("limit_remaining"),
        "credit_used_usd": data.get("usage"),
        "live_call": False,
    }
    if model:
        name = model.removeprefix("openrouter/")
        match = next((m for m in _get(f"{api_base}/models")["data"] if m["id"] == name), None)
        report["model_found"] = match is not None
        if match:
            report["price_usd_per_million_tokens"] = {
                "input": float(match["pricing"]["prompt"]) * 1e6,
                "output": float(match["pricing"]["completion"]) * 1e6,
            }
    if live:
        if not model:
            raise ValueError("--live needs --model.")
        report.update(_live_call(model, key, reasoning_effort, api_base))
        after = _get(f"{api_base}/key", key).get("data", {})
        report["credit_used_after_usd"] = after.get("usage")
    return report


def _live_call(model: str, key: str, effort: str | None, api_base: str) -> dict:
    from importlib.resources import files

    from .cli import load_policy, load_task
    from .context import make_context
    from .schemas import WorkflowVariant

    root = Path(str(files("rsi_ml"))).parents[1]
    task = load_task(root / "configs" / "mlp-demo.json")
    policy = load_policy(root / "configs" / "policy.json")
    context = make_context(
        task,
        policy,
        {"experiments": [], "prior_evidence": []},
        "planner",
        variant=WorkflowVariant(name="A"),
    )
    backend = DSPyBackend(
        model=model, api_key=key, api_base=api_base, reasoning_effort=effort, timeout=300
    )
    plan = backend.call("planner", context, policy, task.limits)
    return {
        "live_call": True,
        "parsed_plan": True,
        "recipe_valid_for_task": _valid(task, plan.recipe),
        "usage": Usage.model_validate(backend.usage).model_dump(mode="json"),
    }


def study_api_key(env_file: Path | None) -> str:
    if env_file:
        return load_api_key(env_file)["RSI_API_KEY"]
    if not os.environ.get("RSI_API_KEY"):
        raise ValueError("The agent stage needs --env-file or RSI_API_KEY in the environment.")
    return os.environ["RSI_API_KEY"]


def _largest_context(task, policy) -> dict:
    """A D-arm planner prompt at a late decision: a long history, each attempt with a report."""
    from .context import make_context
    from .schemas import WorkflowVariant

    recipe = task.initial_recipe.model_dump(mode="json")
    names = task.dataset.class_names or [str(i) for i in range(task.dataset.num_classes)]
    text = {column: "placeholder " * 25 for column in task.dataset.text_columns or []}
    report = {
        "scores": {"accuracy": 0.5},
        "error_count": 50,
        "validation_examples": 100,
        "lowest_recall_classes": [
            {"class_name": names[i % len(names)], "support": 50, "recall": 0.5, "precision": 0.5}
            for i in range(10)
        ],
        "largest_confusions": [
            {"true": names[0], "predicted": names[-1], "count": 5, "share_of_true_class": 0.1}
            for _ in range(10)
        ],
        "confident_errors": [
            {"row_id": str(i), "true": names[0], "predicted": names[-1], "text": text}
            for i in range(10)
        ],
        "interpretation": "Placeholder values for a connectivity check.",
    }
    curve = [
        {"epoch": float(i), "train_loss": 1.0, "validation_loss": 1.0, "accuracy": 0.5}
        for i in range(1, 11)
    ]
    records = [
        {
            "trial_id": f"trial-{i:04d}",
            "index": i,
            "recipe": recipe,
            "plan": {"observation": "placeholder", "hypothesis": "placeholder", "recipe": recipe},
            "review": {"interpretation": "placeholder", "next_step": "placeholder"},
            "result": {
                "status": "completed",
                "metrics": {"accuracy": 0.5, task.metric: 0.5},
                "compute_seconds": 1.0,
                "curve_summary": curve,
            },
        }
        for i in range(task.limits.max_trials - 1)
    ]
    state = {
        "experiments": records,
        "prior_evidence": [],
        "diagnostic_reports": {record["trial_id"]: report for record in records},
    }
    return make_context(task, policy, state, "planner", variant=WorkflowVariant(name="D"))


def api_preflight(study_dir: Path, api_key: str) -> dict:
    """Before any training: one real call with the study's model and largest prompt shape.

    Proves the key, credit, and model work, and that a response of this size fits the
    output limit rather than being cut off. Costs cents and saves a wasted run.
    """
    from .cli import load_policy, load_task
    from .storage import read_json

    study_dir = Path(study_dir)
    study = read_json(study_dir / "study.json")
    stage = study["agent_stage"]
    order = study["task_order"]
    tasks = {name: f"seed-{study['tasks'][name]['seeds'][0]}.json" for name in order}
    # A text task has the longest report, since it carries the mistaken sentences.
    chosen = next((n for n in order if n in {"mrpc", "sst2"}), order[0])
    task = load_task(study_dir / "tasks" / "baselines" / chosen / tasks[chosen])
    policy = load_policy(study_dir / "policy.json")
    context = _largest_context(task, policy)
    backend = DSPyBackend(
        model=stage["model"],
        api_key=api_key,
        api_base=stage.get("api_base"),
        reasoning_effort=stage.get("reasoning_effort"),
        timeout=stage["api_timeout_seconds"],
    )
    plan = backend.call("planner", context, policy, task.limits)
    return {
        "model": stage["model"],
        "task": chosen,
        "context_characters": len(json.dumps(context)),
        "recipe_valid_for_task": _valid(task, plan.recipe),
        "usage": backend.usage.model_dump(mode="json"),
    }


def _valid(task, recipe) -> bool:
    from .tasks.registry import validate_recipe

    try:
        validate_recipe(task, recipe)
    except ValueError:
        return False
    return True
