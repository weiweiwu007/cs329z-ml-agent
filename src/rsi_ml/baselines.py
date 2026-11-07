"""Train each task/seed's starting recipe once; every arm adopts that one result.

Trial 0 is the same recipe, seed, data, and initial weights in A, B, C, D, and the
random-search arm, so training it per arm would only repeat the most expensive stage.
Adoption charges each run the full measured seconds, so accounting does not change,
and all arms share one measured starting point. Stage 1 therefore also measures
throughput before any API call is made.
"""

from __future__ import annotations

import os
from pathlib import Path

from .schemas import TaskSpec, TrialResult, digest
from .storage import environment, read_json, training_fingerprint, write_json
from .tasks.base import require_device
from .tasks.mlp import file_hash
from .tasks.registry import prepare_task, run_trial

_PATH_FIELDS = tuple(
    f"{split}_{kind}" for split in ("train", "validation", "test") for kind in ("file", "csv")
) + ("split_manifest",)


def training_identity(task: TaskSpec) -> str:
    """Everything that determines trial 0's computation, nothing that does not.

    Targets, budgets, and search spaces only affect later agent decisions. Paths are
    excluded because split SHA-256 values already identify the bytes.
    """
    data = task.model_dump(mode="json")
    dataset = {k: v for k, v in data["dataset"].items() if k not in _PATH_FIELDS}
    model = dict(data["text_model"]) if data.get("text_model") else None
    if model:
        model.pop("local_path", None)
    return digest(
        {
            "adapter": data["adapter"],
            "dataset": dataset,
            "initial_recipe": data["initial_recipe"],
            "seed": data["seed"],
            "device": data["device"],
            "text_model": model,
            "artifact_policy": data["artifact_policy"],
        }
    )


def run_baseline(task: TaskSpec, output: Path) -> dict:
    """Stage 1: one training run with no agent, API, or test-data access."""
    require_device(task)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared = prepare_task(task, output / "prepared")
    result = run_trial(
        task,
        task.initial_recipe,
        output / "prepared",
        output / "trial-0000",
        task.limits.per_trial_seconds,
    )
    receipt = {
        "kind": "shared_baseline",
        "task_id": task.task_id,
        "seed": task.seed,
        "metric": task.metric,
        "training_identity": training_identity(task),
        "training_fingerprint": training_fingerprint(),
        "starting_model_hash": prepared["starting_model_hash"],
        "dataset_hash": prepared["dataset_hash"],
        "result": result.model_dump(mode="json"),
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": environment(),
    }
    write_json(output / "baseline.json", receipt)
    return receipt


def adopt_baseline(task: TaskSpec, source: Path, destination: Path, prepared: dict) -> TrialResult:
    """Link the verified baseline into a run as its first trial."""
    source = Path(source)
    receipt = read_json(source / "baseline.json")
    checks = {
        "training identity": (receipt["training_identity"], training_identity(task)),
        "training code/environment": (receipt["training_fingerprint"], training_fingerprint()),
        "starting weights": (receipt["starting_model_hash"], prepared["starting_model_hash"]),
        "dataset": (receipt["dataset_hash"], prepared["dataset_hash"]),
    }
    mismatched = [name for name, (saved, current) in checks.items() if saved != current]
    if mismatched:
        raise ValueError("Shared baseline does not match this run: " + ", ".join(mismatched))
    result = TrialResult.model_validate(receipt["result"])
    if result.status != "completed":
        raise ValueError(f"Shared baseline did not complete: {result.status} ({result.error}).")
    trial = source / "trial-0000"
    for name, expected in result.artifact_hashes.items():
        if file_hash(trial / name) != expected:
            raise ValueError(f"Shared baseline artifact changed: {name}.")
    destination.mkdir(parents=True, exist_ok=False)
    # This run's own task record replaces the stage-1 copy (targets and budgets differ).
    own = {"task.json", "result.json", "experiment.json"}
    for path in sorted(trial.rglob("*")):
        relative = path.relative_to(trial)
        if path.is_file() and str(relative) not in own:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.hardlink_to(path)
    write_json(destination / "task.json", task)
    adopted = result.model_copy(
        update={
            "artifact_hashes": {
                **result.artifact_hashes,
                "task.json": file_hash(destination / "task.json"),
            }
        }
    )
    write_json(destination / "result.json", adopted)
    return adopted
