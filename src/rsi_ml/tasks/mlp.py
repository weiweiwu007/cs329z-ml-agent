"""Trusted MLP builder and bounded worker launcher.

This adapter executes declarative recipes, never LLM-produced Python. CSV splits
must be supplied explicitly; no split, label map, or dataset is silently chosen.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from rsi_ml.schemas import Recipe, TaskSpec, TrialResult, digest

from .base import INFRASTRUCTURE_EXIT_CODE, InfrastructureFault

STARTUP_LIMIT_SECONDS = 300
BUDGET_CUT_ERROR = "Compute budget ran out during this attempt."


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def parameter_count(task: TaskSpec, recipe: Recipe) -> int:
    widths = [task.dataset.input_dim, *recipe.hidden_sizes, task.dataset.num_classes]
    count = sum(a * b + b for a, b in pairwise(widths))
    if recipe.normalization == "layer_norm":
        count += 2 * sum(recipe.hidden_sizes)
    return count


def validate_recipe(task: TaskSpec, recipe: Recipe) -> int:
    space = getattr(task, "search_space", {})
    if space:
        original = task.initial_recipe.model_dump()
        for field, value in recipe.model_dump().items():
            if field in space:
                if value not in space[field]:
                    raise ValueError(f"Recipe field {field} is outside the declared search space.")
            elif value != original[field]:
                raise ValueError(f"Recipe field {field} is fixed by the declared search space.")
    count = parameter_count(task, recipe)
    if count > task.limits.max_parameters:
        raise ValueError(f"Recipe needs {count} parameters; limit is {task.limits.max_parameters}.")
    if recipe.epochs > task.limits.max_epochs:
        raise ValueError(f"Recipe needs {recipe.epochs} epochs; limit is {task.limits.max_epochs}.")
    return count


def architecture_key(recipe: Recipe) -> tuple:
    # Activation/dropout do not change the learnable tensor shapes, so retain
    # identical initial weights when testing those interventions as well.
    return (tuple(recipe.hidden_sizes), recipe.normalization)


def build_model(input_dim: int, num_classes: int, recipe: Recipe):
    from torch import nn

    activation = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[recipe.activation]
    layers = []
    previous = input_dim
    for width in recipe.hidden_sizes:
        layers.append(nn.Linear(previous, width))
        if recipe.normalization == "layer_norm":
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        # Keep Dropout nodes even at p=0 so state-dict keys remain stable.
        layers.append(nn.Dropout(recipe.dropout))
        previous = width
    layers.append(nn.Linear(previous, num_classes))
    return nn.Sequential(*layers)


def state_hash(state: dict) -> str:
    """Hash tensor content, not serialization container metadata."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def _array_hash(arrays: dict[str, np.ndarray], metadata: dict) -> str:
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for name, array in sorted(arrays.items()):
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def _load_csv(path: str | None, expected_sha: str | None, rows: int, task: TaskSpec):
    if not path or not expected_sha:
        raise ValueError(
            "Local CSV requires explicit paths and SHA-256 values for train/validation."
        )
    source = Path(path).resolve()
    if file_hash(source) != expected_sha:
        raise ValueError(f"SHA-256 mismatch for {source.name}.")
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if len(set(names)) != len(names) or task.dataset.label_column not in names:
            raise ValueError("CSV needs unique headers including the declared label column.")
        features = [n for n in names if n != task.dataset.label_column]
        if len(features) != task.dataset.input_dim:
            raise ValueError("CSV feature count differs from declared input_dim.")
        data = list(reader)
    if len(data) != rows:
        raise ValueError(f"CSV row count is {len(data)}; expected {rows}.")
    x = np.asarray([[float(row[n]) for n in features] for row in data], dtype=np.float32)
    raw_y = np.asarray([float(row[task.dataset.label_column]) for row in data], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(raw_y).all():
        raise ValueError(
            "CSV has nonfinite or missing numeric values; declare preprocessing first."
        )
    if not np.equal(raw_y, np.floor(raw_y)).all():
        raise ValueError("Labels must be explicit integer class IDs; no implicit label encoding.")
    y = raw_y.astype(np.int64)
    if (y < 0).any() or (y >= task.dataset.num_classes).any():
        raise ValueError("Labels must be in [0, num_classes).")
    return x, y, features


def prepare_task(task: TaskSpec, directory: Path) -> dict[str, Any]:
    """Stage only train/validation data and preserve immutable initial weights.

    Test labels are never opened. Local split files are read, checked and copied
    as tensors so later source changes cannot silently alter an active run.
    """
    import torch

    if task.purpose != "demo" and task.protocol_status != "ready":
        raise ValueError("Research task protocol must be explicitly marked ready before execution.")
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["task_hash"] != digest(task):
            raise ValueError("Prepared task belongs to a different protocol or seed.")
        for name, expected in manifest["artifact_hashes"].items():
            if file_hash(directory / name) != expected:
                raise ValueError(f"Prepared artifact changed: {name}.")
        return manifest
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Refusing to replace an incomplete/nonempty prepared task directory.")
    validate_recipe(task, task.initial_recipe)
    from .registry import verify_split_manifest

    verify_split_manifest(task)
    ds = task.dataset
    if ds.kind == "synthetic_fixture":
        if task.purpose != "demo" or task.device != "cpu":
            raise ValueError("Synthetic fixture is restricted to a CPU software demo.")
        if ds.train_rows + ds.validation_rows > 4096 or ds.input_dim > 64 or ds.num_classes > 16:
            raise ValueError("Synthetic fixture exceeds the bounded software-demo size.")
        # The demo is a seeded noisy nonlinear classifier. It is deliberately
        # not a benchmark and does not simulate LLM decisions or fake scores.
        rng = np.random.default_rng(ds.seed)
        teacher = rng.normal(size=(ds.input_dim, ds.num_classes)).astype(np.float32)
        raw_x = rng.normal(size=(ds.train_rows + ds.validation_rows, ds.input_dim)).astype(
            np.float32
        )
        scores = np.tanh(raw_x) @ teacher
        scores += rng.normal(scale=0.1, size=scores.shape)
        labels = scores.argmax(axis=1).astype(np.int64)
        train_x, val_x = raw_x[: ds.train_rows], raw_x[ds.train_rows :]
        train_y, val_y = labels[: ds.train_rows], labels[ds.train_rows :]
        features = [f"x{i}" for i in range(ds.input_dim)]
    else:
        train_x, train_y, features = _load_csv(ds.train_csv, ds.train_sha256, ds.train_rows, task)
        val_x, val_y, val_features = _load_csv(
            ds.validation_csv, ds.validation_sha256, ds.validation_rows, task
        )
        if features != val_features:
            raise ValueError("Train/validation CSV feature order must be identical.")
        if Path(ds.train_csv).resolve() == Path(ds.validation_csv).resolve():
            raise ValueError("Train and validation must be separate split files.")
    directory.mkdir(parents=True, exist_ok=True)
    mean = train_x.astype(np.float64).mean(axis=0) if ds.standardize else np.zeros(ds.input_dim)
    scale = train_x.astype(np.float64).std(axis=0) if ds.standardize else np.ones(ds.input_dim)
    scale = np.where(scale == 0, 1.0, scale)
    # Raw numeric features are retained alongside derived normalized tensors.
    np.savez(
        directory / "train.npz",
        raw_x=train_x,
        x=((train_x - mean) / scale).astype(np.float32),
        y=train_y,
        row_id=np.arange(len(train_y)),
    )
    np.savez(
        directory / "validation.npz",
        raw_x=val_x,
        x=((val_x - mean) / scale).astype(np.float32),
        y=val_y,
        row_id=np.arange(len(val_y)),
    )
    write_json(
        directory / "normalization.json",
        {
            "standardize": ds.standardize,
            "fitted_on": "train_only",
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "statistics_dtype": "float64",
            "model_input_dtype": "float32",
            "features": features,
        },
    )
    torch.manual_seed(task.seed)
    model = build_model(ds.input_dim, ds.num_classes, task.initial_recipe)
    torch.save(model.state_dict(), directory / "initial_weights.pt")
    write_json(directory / "task.json", task.model_dump(mode="json"))
    if ds.split_manifest:
        import shutil

        shutil.copyfile(ds.split_manifest, directory / "split_manifest.json")
    dataset_metadata = {
        "identity": ds.identity,
        "revision": ds.revision,
        "seed": ds.seed,
        "standardize": ds.standardize,
        "features": features,
        "source_train_sha256": ds.train_sha256,
        "source_validation_sha256": ds.validation_sha256,
    }
    artifact_hashes = {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}
    manifest = {
        "task_hash": digest(task),
        "dataset_hash": _array_hash(
            {"train_x": train_x, "train_y": train_y, "validation_x": val_x, "validation_y": val_y},
            dataset_metadata,
        ),
        "starting_model_hash": state_hash(model.state_dict()),
        "parameter_count": parameter_count(task, task.initial_recipe),
        "dataset": dataset_metadata,
        "test_access": "not read; excluded from agent context and staged files",
        "artifact_hashes": artifact_hashes,
        "initialization": {
            "seed": task.seed,
            "method": "PyTorch Linear/LayerNorm default initialization; saved exact tensors",
        },
    }
    write_json(manifest_path, manifest)
    for path in directory.iterdir():
        if path.is_file():
            path.chmod(0o444)
    return manifest


def learning_rate_factor(step: int, total_steps: int, recipe: Recipe) -> float:
    """Step-based warmup; cosine or piecewise linear WSD cooldown.

    Warmup starts above zero. Final scheduled step is zero for a cooldown with
    at least two steps. This exact convention is preserved in each recipe.
    """
    warmup = int(total_steps * recipe.warmup_fraction)
    if warmup and step < warmup:
        return (step + 1) / warmup
    if recipe.schedule == "constant":
        return 1.0
    if recipe.schedule == "cosine":
        position = (step - warmup) / max(1, total_steps - warmup - 1)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, position)))
    decay_steps = max(1, int(total_steps * recipe.decay_fraction))
    start = total_steps - decay_steps
    if step < start:
        return 1.0
    return max(0.0, 1 - (step - start) / max(1, decay_steps - 1))


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if (
        logits.ndim != 2
        or logits.shape[1] < 2
        or len(logits) == 0
        or labels.ndim != 1
        or len(logits) != len(labels)
        or not np.isfinite(logits).all()
        or not np.issubdtype(labels.dtype, np.integer)
        or (labels < 0).any()
        or (labels >= logits.shape[1]).any()
    ):
        raise ValueError("Invalid/nonfinite validation logits or labels.")
    prediction = logits.argmax(axis=1)
    f1s = []
    for cls in range(logits.shape[1]):
        tp = np.sum((prediction == cls) & (labels == cls))
        fp = np.sum((prediction == cls) & (labels != cls))
        fn = np.sum((prediction != cls) & (labels == cls))
        denominator = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denominator) if denominator else 0.0)
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    loss = -shifted[np.arange(len(labels)), labels] + logsumexp
    result = {
        "accuracy": float(np.mean(prediction == labels)),
        "macro_f1": float(np.mean(f1s)),
        "validation_loss": float(loss.mean()),
    }
    if logits.shape[1] == 2:
        result["f1"] = f1s[1]  # GLUE MRPC: positive-class F1, not macro-F1.
    if logits.shape[1] >= 5:
        top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
        result["top5_accuracy"] = float(np.mean((top5 == labels[:, None]).any(axis=1)))
    return result


def generated_model_source(task: TaskSpec, recipe: Recipe) -> str:
    """Standalone model source assembled only from validated numeric/enumerated fields."""
    if task.adapter != "mlp_classification":
        return (
            "# Trusted adapter; source hashes are recorded in the run manifest.\n"
            "# Run from this trial directory. Local files only; no downloads.\n"
            "from pathlib import Path\nfrom rsi_ml.schemas import TaskSpec, Recipe\n"
            "from rsi_ml.tasks.registry import build_model\n"
            "task = TaskSpec.model_validate_json(Path('task.json').read_text())\n"
            "recipe = Recipe.model_validate_json(Path('recipe.json').read_text())\n"
            "def make_model():\n    return build_model(task, recipe)\n"
        )
    layers = []
    previous = task.dataset.input_dim
    activations = {"relu": "ReLU", "tanh": "Tanh", "gelu": "GELU"}
    for width in recipe.hidden_sizes:
        layers.append(f"        nn.Linear({previous}, {width}),")
        if recipe.normalization == "layer_norm":
            layers.append(f"        nn.LayerNorm({width}),")
        layers.append(f"        nn.{activations[recipe.activation]}(),")
        layers.append(f"        nn.Dropout({recipe.dropout!r}),")
        previous = width
    layers.append(f"        nn.Linear({previous}, {task.dataset.num_classes}),")
    return (
        "# Generated by the trusted recipe builder; no LLM code is executed.\nimport torch.nn as nn\n\ndef build_model():\n    return nn.Sequential(\n"
        + "\n".join(layers)
        + "\n    )\n"
    )


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): file_hash(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "result.json"
    }


CURVE_POINTS = 10
CURVE_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "accuracy",
    "macro_f1",
    "f1",
    "top5_accuracy",
    "last_learning_rate",
)


def curve_summary(curve: list[dict]) -> list[dict[str, float]]:
    """Evenly spaced epochs, always including the first and last, for agent context."""
    if len(curve) <= CURVE_POINTS:
        selected = curve
    else:
        positions = np.linspace(0, len(curve) - 1, CURVE_POINTS).round().astype(int)
        selected = [curve[i] for i in sorted(set(positions.tolist()))]
    return [
        {key: round(float(row[key]), 6) for key in CURVE_FIELDS if key in row} for row in selected
    ]


def _partial_curve(trial_dir: Path) -> list[dict[str, float]]:
    """What a stopped trial finished, so the agent can judge how far it got."""
    try:
        return curve_summary(json.loads((trial_dir / "learning_curve.json").read_text()))
    except (OSError, ValueError):
        return []


def _await_ready(process, ready: Path) -> None:
    """Wait for the worker to finish importing and initializing, which is not charged."""
    deadline = time.monotonic() + STARTUP_LIMIT_SECONDS
    while not ready.exists() and process.poll() is None:
        if time.monotonic() > deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise InfrastructureFault(f"Worker did not start in {STARTUP_LIMIT_SECONDS} s.")
        time.sleep(0.02)


def _worker_failure(trial_dir: Path, code: int) -> str:
    """The exception line from the worker's log, so an attempt's failure has a reason."""
    lines = (trial_dir / "stderr.log").read_text(errors="replace").strip().split("\n")
    return f"Training worker exited with code {code}: {lines[-1][:300]}"


def run_trial(
    task: TaskSpec, recipe: Recipe, prepared_dir: Path, trial_dir: Path, timeout: float
) -> TrialResult:
    """Launch a trusted worker with a real wall-time limit and no inherited API keys."""
    from .registry import prepare_task, validate_recipe

    started = time.monotonic()
    prepared_dir, trial_dir = Path(prepared_dir).resolve(), Path(trial_dir).resolve()
    trial_dir.mkdir(parents=True, exist_ok=False)
    (trial_dir / "stdout.log").touch()
    (trial_dir / "stderr.log").touch()
    write_json(trial_dir / "recipe.json", recipe.model_dump(mode="json"))
    write_json(trial_dir / "task.json", task.model_dump(mode="json"))
    (trial_dir / "model.py").write_text(generated_model_source(task, recipe))
    try:
        count = validate_recipe(task, recipe)
        if timeout <= 0:
            raise ValueError("No execution time remains.")
    except (ValueError, OSError) as exc:
        result = TrialResult(
            status="rejected",
            compute_seconds=time.monotonic() - started,
            error=str(exc),
            artifact_hashes=_artifact_hashes(trial_dir),
        )
        write_json(trial_dir / "result.json", result.model_dump(mode="json"))
        return result
    try:
        prepare_task(task, prepared_dir)  # Verify immutable preparation before each run.
    except (ValueError, OSError) as exc:
        raise InfrastructureFault(f"Prepared data failed verification: {exc}") from exc
    command = [
        sys.executable,
        "-m",
        "rsi_ml.worker",
        "--task",
        str(trial_dir / "task.json"),
        "--recipe",
        str(trial_dir / "recipe.json"),
        "--prepared",
        str(prepared_dir),
        "--out",
        str(trial_dir),
    ]
    # This is subprocess isolation, not an OS security sandbox. The worker
    # contains trusted code and has no network/API operations.
    # A credential-free allowlist; CUDA libraries on GPU VMs may need the last group.
    env = {
        key: os.environ[key]
        for key in (
            "PATH",
            "SYSTEMROOT",
            "TMPDIR",
            "TEMP",
            "CUDA_VISIBLE_DEVICES",
            "PYTHONPATH",
            "HOME",
            "LD_LIBRARY_PATH",
            "CUDA_DEVICE_ORDER",
            "NVIDIA_VISIBLE_DEVICES",
        )
        if key in os.environ
    }
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "RSI_SUPERVISOR_PID": str(os.getpid()),
        }
    )
    status, error = "completed", None
    limit = min(timeout, task.limits.per_trial_seconds)
    with (
        (trial_dir / "stdout.log").open("wb") as stdout,
        (trial_dir / "stderr.log").open("wb") as stderr,
    ):
        try:
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, env=env, start_new_session=True
            )
        except OSError as exc:
            raise InfrastructureFault("Training worker could not be started.") from exc
        _await_ready(process, trial_dir / "worker_ready")
        charge_started = time.monotonic()
        try:
            code = process.wait(timeout=limit)
            # A negative code is a signal, such as the kernel's out-of-memory killer. The
            # timeout path below sends its own kill and never reaches this check.
            if code == INFRASTRUCTURE_EXIT_CODE or code < 0:
                raise InfrastructureFault(
                    f"Worker stopped by the machine (exit {code}); see {trial_dir}/stderr.log."
                )
            if code:
                status, error = "failed", _worker_failure(trial_dir, code)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            status = "timeout"
            error = (
                BUDGET_CUT_ERROR
                if timeout < task.limits.per_trial_seconds
                else "Worker exceeded the per-trial time limit."
            )
    elapsed = time.monotonic() - charge_started
    details = {}
    if status == "completed":
        try:
            details = json.loads((trial_dir / "worker_result.json").read_text())
            if not details["metrics"] or not all(
                math.isfinite(v) for v in details["metrics"].values()
            ):
                raise ValueError("Worker returned missing/nonfinite metrics.")
        except (OSError, ValueError, KeyError, TypeError):
            status, error = "failed", "Worker result was missing or invalid."
            details = {}
    result = TrialResult(
        status=status,
        metrics=details.get("metrics", {}),
        compute_seconds=elapsed,
        training_seconds=details.get("training_seconds", 0),
        startup_seconds=charge_started - started,
        diagnostics=details.get("diagnostics", {}),
        parameter_count=count,
        error=error,
        artifact_hashes=_artifact_hashes(trial_dir),
        curve_summary=details.get("curve_summary") or _partial_curve(trial_dir),
    )
    write_json(trial_dir / "result.json", result.model_dump(mode="json"))
    return result


def score_test(task: TaskSpec, prepared_dir: Path, trial_dir: Path, output_path: Path) -> dict:
    """Supervisor-only final grading. Never called by the agent graph.

    The selected completed model is evaluated once per output path, using only
    the normalization fitted on training data. No model choice is made here.
    """
    import torch

    if task.purpose == "demo" or task.dataset.kind != "local_csv":
        raise ValueError("Demo fixtures have no final test set; supply a real declared CSV task.")
    if task.dataset.test_sha256 and task.dataset.test_sha256 in {
        task.dataset.train_sha256,
        task.dataset.validation_sha256,
    }:
        raise ValueError("Test CSV must not be a byte-identical copy of train or validation.")
    prepared_dir, trial_dir, output_path = Path(prepared_dir), Path(trial_dir), Path(output_path)
    outputs_path = output_path.with_suffix(".outputs.npz")
    if output_path.exists() or outputs_path.exists():
        raise ValueError("Refusing to overwrite a final-test evaluation.")
    prepare_task(task, prepared_dir)
    saved_task = TaskSpec.model_validate_json((trial_dir / "task.json").read_text())
    if digest(saved_task) != digest(task):
        raise ValueError("Selected trial belongs to a different task protocol.")
    trial = TrialResult.model_validate_json((trial_dir / "result.json").read_text())
    if trial.status != "completed":
        raise ValueError("Final grading requires a completed selected trial.")
    if file_hash(trial_dir / "checkpoint.pt") != trial.artifact_hashes.get("checkpoint.pt"):
        raise ValueError("Selected checkpoint differs from the recorded artifact hash.")
    recipe = Recipe.model_validate_json((trial_dir / "recipe.json").read_text())
    x, labels, features = _load_csv(
        task.dataset.test_csv, task.dataset.test_sha256, task.dataset.test_rows, task
    )
    stats = json.loads((prepared_dir / "normalization.json").read_text())
    if features != stats["features"]:
        raise ValueError("Test feature order differs from training feature order.")
    if Path(task.dataset.test_csv).resolve() in {
        Path(task.dataset.train_csv).resolve(),
        Path(task.dataset.validation_csv).resolve(),
    }:
        raise ValueError("Test CSV must be separate from train and validation.")
    preprocessing = trial_dir / "preprocessing.json"
    if preprocessing.exists():
        stats = json.loads(preprocessing.read_text())
        if stats.get("method") == "signed_log_standard":
            x = np.sign(x) * np.log1p(np.abs(x))
    normalized = ((x - np.asarray(stats["mean"])) / np.asarray(stats["scale"])).astype(np.float32)
    if task.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but unavailable for final grading.")
    torch.set_num_threads(1)
    model = build_model(task.dataset.input_dim, task.dataset.num_classes, recipe)
    checkpoint = torch.load(trial_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(task.device).eval()
    if task.device == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    with torch.no_grad():
        # Bounded batches avoid allocating the entire held-out set on GPU.
        logits = np.concatenate(
            [
                model(torch.from_numpy(normalized[i : i + recipe.batch_size]).to(task.device))
                .cpu()
                .numpy()
                for i in range(0, len(normalized), recipe.batch_size)
            ]
        )
    metrics = classification_metrics(logits, labels)
    if task.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outputs_path, logits=logits, labels=labels, row_id=np.arange(len(labels)))
    result = {
        "kind": "final_test_evaluation",
        "task_id": task.task_id,
        "task_hash": digest(task),
        "trial_directory": str(trial_dir.resolve()),
        "model_hash": state_hash(checkpoint["model"]),
        "checkpoint_sha256": trial.artifact_hashes["checkpoint.pt"],
        "test_sha256": task.dataset.test_sha256,
        "metrics": metrics,
        "evaluation_seconds": elapsed,
        "device": task.device,
        "outputs": str(outputs_path.resolve()),
        "outputs_sha256": file_hash(outputs_path),
        "normalization_sha256": file_hash(prepared_dir / "normalization.json"),
        "preprocessing_sha256": file_hash(preprocessing) if preprocessing.exists() else None,
        "selection": "externally selected model; test score not supplied to agents",
    }
    write_json(output_path, result)
    return result
