"""Batched trusted training shared by all three task families; no API/network I/O."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from rsi_ml.schemas import Recipe, TaskSpec, TrialResult, digest

from .base import InfrastructureFault
from .mlp import (
    classification_metrics,
    curve_summary,
    file_hash,
    learning_rate_factor,
    state_hash,
    write_json,
)
from .registry import build_model, load_split, prepare_task, validate_recipe


def _features(task, recipe, prepared, output):
    with np.load(prepared / "train.npz", allow_pickle=False) as train:
        raw_train, train_y = train["raw_x"].copy(), train["y"].copy()
    with np.load(prepared / "validation.npz", allow_pickle=False) as val:
        raw_val, val_y, ids = val["raw_x"].copy(), val["y"].copy(), val["row_id"].copy()
    method = recipe.feature_preprocessing or ("standard" if task.dataset.standardize else "none")
    if method == "signed_log_standard":
        raw_train = np.sign(raw_train) * np.log1p(np.abs(raw_train))
        raw_val = np.sign(raw_val) * np.log1p(np.abs(raw_val))
    mean = (
        raw_train.mean(axis=0, dtype=np.float64)
        if method != "none"
        else np.zeros(raw_train.shape[1])
    )
    std = (
        raw_train.std(axis=0, dtype=np.float64) if method != "none" else np.ones(raw_train.shape[1])
    )
    std = np.where(std == 0, 1.0, std)
    write_json(
        output / "preprocessing.json",
        {"method": method, "mean": mean.tolist(), "scale": std.tolist(), "fitted_on": "train_only"},
    )
    return (
        torch.from_numpy(((raw_train - mean) / std).astype(np.float32)),
        train_y,
        torch.from_numpy(((raw_val - mean) / std).astype(np.float32)),
        val_y,
        ids,
    )


def inputs(task, recipe, prepared, output):
    if task.adapter == "mlp_classification":
        return _features(task, recipe, prepared, output)
    extension = "npz" if task.adapter == "cifar100_classification" else "jsonl"
    train, train_y, _ = load_split(task, "train", source=prepared / f"train.{extension}")
    val, val_y, ids = load_split(task, "validation", source=prepared / f"validation.{extension}")
    if task.adapter == "cifar100_classification":
        return torch.from_numpy(train), train_y, torch.from_numpy(val), val_y, ids
    from .text import tokenize

    train_x, val_x = tokenize(task, recipe, train), tokenize(task, recipe, val)
    for name, data in (("train", train_x), ("validation", val_x)):
        np.savez(
            output / f"{name}_tokens.npz", **{key: value.numpy() for key, value in data.items()}
        )
    return train_x, train_y, val_x, val_y, ids


def batch_input(task, recipe, data, indices, device, *, training):
    if isinstance(data, dict):
        return {key: value[indices].to(device) for key, value in data.items()}
    x = data[indices]
    if task.adapter == "cifar100_classification":
        from .cnn import transform

        x = transform(x, recipe, task.dataset, training=training)
    return x.to(device)


def predict(model, task, recipe, data, size):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, size, recipe.batch_size):
            x = batch_input(
                task,
                recipe,
                data,
                slice(start, start + recipe.batch_size),
                task.device,
                training=False,
            )
            logits.append(model(x).cpu().numpy())
    return np.concatenate(logits)


def execute(task: TaskSpec, recipe: Recipe, prepared: Path, output: Path) -> dict:
    from rsi_ml.worker import _retain_epoch

    if task.purpose != "demo" and task.protocol_status != "ready":
        raise ValueError("Research protocol is not ready.")
    started_total = time.monotonic()
    # From the launcher, not getppid(): it can already have died during the slow imports.
    supervisor = int(os.environ.get("RSI_SUPERVISOR_PID", os.getppid()))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(task.seed)
    random.seed(task.seed)
    np.random.seed(task.seed % 2**32)
    if task.device == "cuda" and not torch.cuda.is_available():
        raise InfrastructureFault("CUDA requested but unavailable; no fallback.")
    if task.device == "cuda":
        torch.zeros(1, device="cuda")  # A fixed cost of starting, not of this recipe.
    # Everything before this point is process startup. The compute charge starts here.
    (output / "worker_ready").touch()
    count = validate_recipe(task, recipe)
    train_x, labels, val_x, val_y, ids = inputs(task, recipe, prepared, output)
    model = build_model(task, recipe, pretrained=task.adapter == "text_classification")
    original = torch.load(prepared / "initial_weights.pt", map_location="cpu", weights_only=True)
    candidate = model.state_dict()
    compatible = candidate.keys() == original.keys() and all(
        candidate[name].shape == original[name].shape for name in candidate
    )
    if compatible:
        model.load_state_dict(original)
    initial_hash = state_hash(model.state_dict())
    torch.save(model.state_dict(), output / "initial_state.pt")
    write_json(
        output / "initialization.json",
        {
            "seed": task.seed,
            "initial_state_hash": initial_hash,
            "starting_model_hash": json.loads((prepared / "manifest.json").read_text())[
                "starting_model_hash"
            ],
            "source": "saved_initial_weights"
            if compatible
            else (
                "pinned_pretrained_encoder_new_seeded_head"
                if task.text_model
                else "same_seed_new_architecture"
            ),
        },
    )
    model.to(task.device)
    train_y = torch.from_numpy(labels)
    parameters = [p for p in model.parameters() if p.requires_grad]
    kwargs = {"lr": recipe.learning_rate, "weight_decay": recipe.weight_decay}
    optimizer = (
        torch.optim.AdamW(parameters, **kwargs)
        if recipe.optimizer == "adamw"
        else torch.optim.SGD(parameters, momentum=recipe.momentum, **kwargs)
    )
    counts = np.bincount(labels, minlength=task.dataset.num_classes)
    inverse = np.divide(
        len(labels),
        task.dataset.num_classes * counts,
        out=np.zeros(len(counts), dtype=float),
        where=counts != 0,
    )
    weights = (
        torch.tensor(inverse, dtype=torch.float32, device=task.device)
        if recipe.class_weighting == "balanced"
        else None
    )
    sample_weights = torch.tensor(inverse[labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(task.seed)
    total_steps = recipe.epochs * math.ceil(len(labels) / recipe.batch_size)
    curve, step = [], 0
    if task.device == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    for epoch in range(recipe.epochs):
        model.train()
        order = (
            torch.randperm(len(labels), generator=generator)
            if recipe.sampling == "shuffle"
            else torch.multinomial(
                sample_weights, len(labels), replacement=True, generator=generator
            )
        )
        total_loss, total_correct, first_lr, last_lr = 0.0, 0, None, None
        for offset in range(0, len(labels), recipe.batch_size):
            indices = order[offset : offset + recipe.batch_size]
            x = batch_input(task, recipe, train_x, indices, task.device, training=True)
            y = train_y[indices].to(task.device)
            lr = recipe.learning_rate * learning_rate_factor(step, total_steps, recipe)
            for group in optimizer.param_groups:
                group["lr"] = lr
            first_lr = lr if first_lr is None else first_lr
            last_lr = lr
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits, y, weight=weights, label_smoothing=recipe.label_smoothing
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("Nonfinite gradient.")
            optimizer.step()
            total_loss += loss.item() * len(indices)
            total_correct += (logits.argmax(1) == y).sum().item()
            step += 1
        logits = predict(model, task, recipe, val_x, len(val_y))
        metrics = classification_metrics(logits, val_y)
        # Workers run in their own session, so one can outlive a killed supervisor. Stop before
        # writing: recovery may already have started this trial again at the same path.
        if os.getppid() != supervisor:
            raise InfrastructureFault("Supervisor exited; abandoning this trial.")
        if task.device == "cuda":
            torch.cuda.synchronize()
        curve.append(
            {
                "epoch": epoch + 1,
                "step": step,
                "train_loss": total_loss / len(labels),
                # Running accuracy in train mode, the ordinary log line; not re-evaluated.
                "train_accuracy": total_correct / len(labels),
                **metrics,
                "first_learning_rate": first_lr,
                "last_learning_rate": last_lr,
                "training_seconds": time.monotonic() - started,
            }
        )
        write_json(output / "learning_curve.json", curve)
        with (output / "validation_outputs.tmp").open("wb") as stream:
            np.savez(stream, logits=logits, labels=val_y, row_id=ids, epoch=epoch + 1)
        (output / "validation_outputs.tmp").replace(output / "validation_outputs.npz")
        snapshot = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "step": step,
            "torch_rng": torch.get_rng_state(),
            "shuffle_rng": generator.get_state(),
            "python_rng": random.getstate(),
            "numpy_rng": {
                "generator": np.random.get_state()[0],
                "state": torch.from_numpy(np.random.get_state()[1].astype(np.int64)),
                "position": np.random.get_state()[2],
                "has_gauss": np.random.get_state()[3],
                "cached_gaussian": np.random.get_state()[4],
            },
            "cuda_rng": torch.cuda.get_rng_state_all() if task.device == "cuda" else [],
            "recipe": recipe.model_dump(),
            "seed": task.seed,
        }
        torch.save(snapshot, output / "checkpoint.tmp")
        (output / "checkpoint.tmp").replace(output / "checkpoint.pt")
        if task.artifact_policy and task.artifact_policy.retain_epoch_checkpoints:
            _retain_epoch(output / "checkpoint.pt", output / "epochs" / f"epoch-{epoch + 1:04d}.pt")
        if task.artifact_policy and task.artifact_policy.retain_epoch_logits:
            _retain_epoch(
                output / "validation_outputs.npz",
                output / "epochs" / f"epoch-{epoch + 1:04d}.outputs.npz",
            )
    diagnostics = {
        "first_train_loss": curve[0]["train_loss"],
        "last_train_loss": curve[-1]["train_loss"],
        "first_validation_loss": curve[0]["validation_loss"],
        "last_validation_loss": curve[-1]["validation_loss"],
        "final_learning_rate": curve[-1]["last_learning_rate"],
        "epochs_completed": float(len(curve)),
    }
    result = {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "parameter_count": count,
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "training_seconds": time.monotonic() - started,
        "worker_seconds": time.monotonic() - started_total,
        "curve_summary": curve_summary(curve),
        "optimizer_steps": step,
        "examples_processed": recipe.epochs * len(labels),
        "gpu_name": torch.cuda.get_device_name(0) if task.device == "cuda" else None,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated()
        if task.device == "cuda"
        else None,
        "device": task.device,
        "artifact_policy": task.artifact_policy.model_dump() if task.artifact_policy else {},
        "evaluation": "final epoch; unweighted unsmoothed validation loss; positive-class F1 for binary tasks",
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "worker_result.json", result)
    return result


def score_multimodal_test(task, prepared_dir, trial_dir, output_path):
    """Read protected test data only on an explicit supervisor command."""
    prepared_dir, trial_dir, output_path = map(Path, (prepared_dir, trial_dir, output_path))
    outputs_path = output_path.with_suffix(".outputs.npz")
    if output_path.exists() or outputs_path.exists():
        raise ValueError("Refusing to overwrite final-test evaluation.")
    prepare_task(task, prepared_dir)
    if digest(TaskSpec.model_validate_json((trial_dir / "task.json").read_text())) != digest(task):
        raise ValueError("Selected trial belongs to a different task.")
    result = TrialResult.model_validate_json((trial_dir / "result.json").read_text())
    if result.status != "completed" or file_hash(
        trial_dir / "checkpoint.pt"
    ) != result.artifact_hashes.get("checkpoint.pt"):
        raise ValueError("Final grading needs a completed, intact checkpoint.")
    recipe = Recipe.model_validate_json((trial_dir / "recipe.json").read_text())
    rows, labels, ids = load_split(task, "test")
    extension = "npz" if task.adapter == "cifar100_classification" else "jsonl"
    for split in ("train", "validation"):
        _, _, used_ids = load_split(task, split, source=prepared_dir / f"{split}.{extension}")
        if set(map(str, ids)) & set(map(str, used_ids)):
            raise ValueError("Test source IDs overlap a development split.")
    if task.adapter == "text_classification":
        from .text import tokenize

        data = tokenize(task, recipe, rows)
    else:
        data = torch.from_numpy(rows)
    checkpoint = torch.load(trial_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = build_model(task, recipe)
    model.load_state_dict(checkpoint["model"])
    model.to(task.device)
    start = time.monotonic()
    logits = predict(model, task, recipe, data, len(labels))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outputs_path, logits=logits, labels=labels, row_id=ids)
    receipt = {
        "kind": "final_test_evaluation",
        "task_hash": digest(task),
        "metrics": classification_metrics(logits, labels),
        "evaluation_seconds": time.monotonic() - start,
        "checkpoint_sha256": result.artifact_hashes["checkpoint.pt"],
        "test_sha256": task.dataset.test_sha256,
        "outputs": str(outputs_path.resolve()),
        "outputs_sha256": file_hash(outputs_path),
        "selection": "externally selected; never supplied to agents",
    }
    write_json(output_path, receipt)
    return receipt
