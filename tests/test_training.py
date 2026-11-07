"""Tests use tiny CPU fixtures; these are software checks, not project results."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from rsi_ml.schemas import DatasetSpec, Limits, Recipe, TaskSpec
from rsi_ml.tasks.mlp import (
    classification_metrics,
    file_hash,
    learning_rate_factor,
    prepare_task,
    run_trial,
    state_hash,
    validate_recipe,
)


def tiny_task() -> TaskSpec:
    return TaskSpec(
        task_id="training-test",
        adapter="mlp_classification",
        purpose="demo",
        protocol_status="demo",
        metric="accuracy",
        target=0.8,
        seed=17,
        device="cpu",
        dataset=DatasetSpec(
            kind="synthetic_fixture",
            identity="unit-test-fixture",
            revision="1",
            input_dim=4,
            num_classes=2,
            standardize=True,
            seed=25,
            train_rows=32,
            validation_rows=16,
            test_rows=8,
        ),
        initial_recipe=Recipe(
            hidden_sizes=[8],
            activation="relu",
            normalization="none",
            dropout=0.0,
            optimizer="adamw",
            learning_rate=0.01,
            weight_decay=0.0,
            batch_size=8,
            epochs=2,
            schedule="constant",
            warmup_fraction=0.0,
            decay_fraction=0.2,
        ),
        limits=Limits(
            max_trials=3,
            max_parameters=10000,
            max_epochs=5,
            per_trial_seconds=20,
            total_compute_seconds=60,
            max_api_calls=20,
            max_context_chars=10000,
            max_output_tokens=1000,
        ),
    )


def test_preparation_is_reproducible_and_keeps_raw_data(tmp_path):
    task = tiny_task()
    a, b = tmp_path / "a", tmp_path / "b"
    ma, mb = prepare_task(task, a), prepare_task(task, b)
    assert ma["starting_model_hash"] == mb["starting_model_hash"]
    assert ma["dataset_hash"] == mb["dataset_hash"]
    assert not any("test" in p.name for p in a.iterdir())
    stats = json.loads((a / "normalization.json").read_text())
    with np.load(a / "train.npz") as train:
        np.testing.assert_allclose(train["raw_x"].mean(axis=0, dtype=np.float64), stats["mean"])
        np.testing.assert_allclose(train["x"].mean(axis=0), 0, atol=1e-6)
    assert prepare_task(task, a) == ma


def test_deterministic_grader_checks_outputs():
    logits = np.array([[4.0, -2.0], [3.0, 1.0], [-1.0, 3.0], [-2.0, 4.0]], dtype=np.float32)
    labels = np.array([0, 1, 1, 1])
    metrics = classification_metrics(logits, labels)
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    with pytest.raises(ValueError, match="nonfinite"):
        classification_metrics(logits * np.nan, labels)


def test_worker_trains_and_preserves_exact_start(tmp_path):
    task = tiny_task()
    prepared, trial = tmp_path / "prepared", tmp_path / "trial"
    manifest = prepare_task(task, prepared)
    result = run_trial(task, task.initial_recipe, prepared, trial, timeout=20)
    assert result.status == "completed", (trial / "stderr.log").read_text()
    assert result.training_seconds > 0
    assert result.compute_seconds >= result.training_seconds
    with np.load(trial / "validation_outputs.npz") as outputs:
        assert classification_metrics(outputs["logits"], outputs["labels"]) == result.metrics
    initial = torch.load(trial / "initial_state.pt", weights_only=True)
    assert state_hash(initial) == manifest["starting_model_hash"]
    curve = json.loads((trial / "learning_curve.json").read_text())
    assert len(curve) == 2
    assert result.diagnostics == {
        "first_train_loss": curve[0]["train_loss"],
        "last_train_loss": curve[-1]["train_loss"],
        "first_validation_loss": curve[0]["validation_loss"],
        "last_validation_loss": curve[-1]["validation_loss"],
        "final_learning_rate": curve[-1]["last_learning_rate"],
        "epochs_completed": 2.0,
    }
    checkpoint = torch.load(trial / "checkpoint.pt", weights_only=False, map_location="cpu")
    assert checkpoint["epoch"] == 2
    assert {"model", "optimizer", "torch_rng", "shuffle_rng"} <= set(checkpoint)
    for name, expected in result.artifact_hashes.items():
        assert file_hash(trial / name) == expected


def test_timeout_is_counted_and_preserves_request(tmp_path):
    task = tiny_task()
    prepared, trial = tmp_path / "prepared", tmp_path / "timeout"
    prepare_task(task, prepared)
    result = run_trial(task, task.initial_recipe, prepared, trial, timeout=0.001)
    assert result.status == "timeout"
    assert result.compute_seconds > 0
    assert result.metrics == {}
    assert (trial / "recipe.json").exists()
    assert (trial / "stderr.log").exists()


def test_limits_reject_before_execution(tmp_path):
    task = tiny_task()
    oversized = task.initial_recipe.model_copy(update={"hidden_sizes": [8192]})
    with pytest.raises(ValueError, match="parameters"):
        validate_recipe(task, oversized)
    result = run_trial(task, oversized, tmp_path / "unused", tmp_path / "rejected", 1)
    assert result.status == "rejected"
    assert not (tmp_path / "unused").exists()


def test_csv_splits_hashes_and_train_only_normalization(tmp_path):
    train, val = tmp_path / "train.csv", tmp_path / "validation.csv"
    train.write_text("x,label\n0,0\n2,1\n")
    val.write_text("x,label\n100,0\n200,1\n")
    task = tiny_task().model_copy(update={"purpose": "development", "protocol_status": "ready"})
    task.dataset = DatasetSpec(
        kind="local_csv",
        identity="csv-unit-test",
        revision="fixture-v1",
        input_dim=1,
        num_classes=2,
        standardize=True,
        seed=0,
        train_rows=2,
        validation_rows=2,
        test_rows=2,
        train_csv=str(train),
        validation_csv=str(val),
        test_csv=str(tmp_path / "unavailable-secret-test.csv"),
        train_sha256=file_hash(train),
        validation_sha256=file_hash(val),
    )
    directory = tmp_path / "prepared"
    prepare_task(task, directory)
    stats = json.loads((directory / "normalization.json").read_text())
    assert stats["mean"] == [1.0]
    with np.load(directory / "validation.npz") as data:
        np.testing.assert_equal(data["x"].ravel(), [99.0, 199.0])
    # The unavailable test file proves it was not read even for a content hash.
    train.write_text("x,label\n0,1\n2,0\n")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_task(task, tmp_path / "different")


def test_wsd_schedule_has_stable_phase_and_cooldown():
    recipe = tiny_task().initial_recipe.model_copy(
        update={"schedule": "wsd", "warmup_fraction": 0.2, "decay_fraction": 0.3}
    )
    rates = [learning_rate_factor(i, 10, recipe) for i in range(10)]
    assert rates[:2] == [0.5, 1.0]
    assert rates[2:8] == [1.0] * 6
    assert rates[-2:] == [0.5, 0.0]


def test_final_test_is_separate_and_checkpoint_is_verified(tmp_path):
    from rsi_ml.tasks.mlp import score_test

    paths = {}
    contents = {
        "train": "x,label\n0,0\n1,1\n2,1\n3,0\n",
        "validation": "x,label\n4,1\n5,0\n",
        "test": "x,label\n6,0\n7,1\n",
    }
    for name, content in contents.items():
        paths[name] = tmp_path / f"{name}.csv"
        paths[name].write_text(content)
    task = tiny_task().model_copy(update={"purpose": "development", "protocol_status": "ready"})
    task.dataset = DatasetSpec(
        kind="local_csv",
        identity="final-test-fixture",
        revision="v1",
        input_dim=1,
        num_classes=2,
        standardize=True,
        seed=2,
        train_rows=4,
        validation_rows=2,
        test_rows=2,
        train_csv=str(paths["train"]),
        validation_csv=str(paths["validation"]),
        test_csv=str(paths["test"]),
        train_sha256=file_hash(paths["train"]),
        validation_sha256=file_hash(paths["validation"]),
        test_sha256=file_hash(paths["test"]),
    )
    prepared, trial = tmp_path / "prepared", tmp_path / "trial"
    prepare_task(task, prepared)
    result = run_trial(task, task.initial_recipe, prepared, trial, 20)
    assert result.status == "completed", (trial / "stderr.log").read_text()
    receipt = score_test(task, prepared, trial, tmp_path / "final.json")
    assert receipt["kind"] == "final_test_evaluation"
    with np.load(receipt["outputs"]) as data:
        assert classification_metrics(data["logits"], data["labels"]) == receipt["metrics"]
    with pytest.raises(ValueError, match="overwrite"):
        score_test(task, prepared, trial, tmp_path / "final.json")
    with (trial / "checkpoint.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checkpoint"):
        score_test(task, prepared, trial, tmp_path / "another.json")


def test_declared_search_space_is_enforced():
    task = tiny_task()
    task.search_space = {"learning_rate": [0.01, 0.02], "hidden_sizes": [[8], [16]]}
    validate_recipe(task, task.initial_recipe.model_copy(update={"learning_rate": 0.02}))
    with pytest.raises(ValueError, match="outside"):
        validate_recipe(task, task.initial_recipe.model_copy(update={"learning_rate": 0.03}))
    with pytest.raises(ValueError, match="fixed"):
        validate_recipe(task, task.initial_recipe.model_copy(update={"dropout": 0.1}))


def test_final_test_rejects_renamed_copy_before_reading_labels(tmp_path):
    from rsi_ml.tasks.mlp import score_test

    task = tiny_task().model_copy(update={"purpose": "development", "protocol_status": "ready"})
    task.dataset = task.dataset.model_copy(
        update={
            "kind": "local_csv",
            "train_csv": "train.csv",
            "validation_csv": "val.csv",
            "test_csv": "renamed-val.csv",
            "train_sha256": "a" * 64,
            "validation_sha256": "b" * 64,
            "test_sha256": "b" * 64,
        }
    )
    with pytest.raises(ValueError, match="byte-identical"):
        score_test(
            task, tmp_path / "absent-prepared", tmp_path / "absent-trial", tmp_path / "final.json"
        )


def test_requested_epoch_artifacts_are_retained(tmp_path):
    from rsi_ml.schemas import ArtifactPolicy

    task = tiny_task()
    task.artifact_policy = ArtifactPolicy(retain_epoch_checkpoints=True, retain_epoch_logits=True)
    prepared, trial = tmp_path / "prepared", tmp_path / "trial"
    prepare_task(task, prepared)
    result = run_trial(task, task.initial_recipe, prepared, trial, 20)
    assert result.status == "completed", (trial / "stderr.log").read_text()
    for epoch in (1, 2):
        checkpoint = torch.load(
            trial / "epochs" / f"epoch-{epoch:04d}.pt", map_location="cpu", weights_only=True
        )
        assert checkpoint["epoch"] == epoch
        with np.load(trial / "epochs" / f"epoch-{epoch:04d}.outputs.npz") as outputs:
            assert int(outputs["epoch"]) == epoch
            assert outputs["logits"].shape == (
                task.dataset.validation_rows,
                task.dataset.num_classes,
            )
        assert f"epochs/epoch-{epoch:04d}.pt" in result.artifact_hashes
        assert f"epochs/epoch-{epoch:04d}.outputs.npz" in result.artifact_hashes
    assert file_hash(trial / "epochs" / "epoch-0002.pt") == file_hash(trial / "checkpoint.pt")
    assert (
        json.loads((trial / "worker_result.json").read_text())["artifact_policy"]
        == task.artifact_policy.model_dump()
    )
