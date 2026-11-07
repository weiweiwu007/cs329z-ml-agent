"""Tiny local software fixtures only: no project data, hub downloads, API or GPU."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_training import tiny_task

from rsi_ml.schemas import DatasetSpec, TextModelSpec
from rsi_ml.tasks.cnn import CifarResNet, transform
from rsi_ml.tasks.mlp import classification_metrics, file_hash
from rsi_ml.tasks.registry import prepare_task, run_trial, score_test


def cnn_task(tmp_path):
    ds = {
        "kind": "cifar100_npz",
        "identity": "unit-test-images",
        "revision": "fixture",
        "input_dim": 3072,
        "num_classes": 100,
        "standardize": False,
        "seed": 0,
        "train_rows": 4,
        "validation_rows": 2,
        "test_rows": 2,
        "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3,
    }
    for split, size in (("train", 4), ("validation", 2), ("test", 2)):
        path = tmp_path / f"{split}.npz"
        rng = np.random.default_rng(size + len(split))
        np.savez(
            path,
            images=rng.integers(0, 256, (size, 3, 32, 32), dtype=np.uint8),
            labels=np.arange(size) % 2,
            row_id=np.array([f"{split}-{i}" for i in range(size)]),
        )
        ds[f"{split}_file"], ds[f"{split}_sha256"] = str(path), file_hash(path)
    task = tiny_task().model_copy(
        update={
            "adapter": "cifar100_classification",
            "purpose": "development",
            "protocol_status": "ready",
            "dataset": DatasetSpec(**ds),
        }
    )
    task.initial_recipe = task.initial_recipe.model_copy(
        update={
            "hidden_sizes": [],
            "cnn_channels": [4, 8, 8, 8],
            "cnn_blocks": [1, 1, 1, 1],
            "epochs": 1,
            "batch_size": 2,
            "augmentation": "crop_flip",
        }
    )
    task.limits.max_parameters = 100000
    return task


def test_cifar_resnet18_shape_and_augmentation():
    recipe = tiny_task().initial_recipe.model_copy(
        update={"hidden_sizes": [], "cnn_channels": [64, 128, 256, 512], "cnn_blocks": [2, 2, 2, 2]}
    )
    with torch.device("meta"):
        model = CifarResNet(100, recipe)
        assert model(torch.empty(2, 3, 32, 32)).shape == (2, 100)
    assert model.stem[0].kernel_size == (3, 3)
    assert model.stem[0].stride == (1, 1)
    assert len(model.stages) == 8


def test_cifar_worker_and_final_test_use_raw_local_data(tmp_path):
    task = cnn_task(tmp_path)
    prepared, trial = tmp_path / "prepared", tmp_path / "trial"
    test_path = Path(task.dataset.test_file)
    backup = test_path.read_bytes()
    test_path.unlink()  # Preparation and training must not inspect held-out data.
    manifest = prepare_task(task, prepared)
    result = run_trial(task, task.initial_recipe, prepared, trial, timeout=20)
    assert result.status == "completed", (trial / "stderr.log").read_text()
    assert result.parameter_count == manifest["parameter_count"]
    assert result.metrics["top5_accuracy"] >= result.metrics["accuracy"]
    assert (
        json.loads((trial / "initialization.json").read_text())["source"] == "saved_initial_weights"
    )
    test_path.write_bytes(backup)
    receipt = score_test(task, prepared, trial, tmp_path / "final.json")
    assert "accuracy" in receipt["metrics"]
    with pytest.raises(ValueError, match="overwrite"):
        score_test(task, prepared, trial, tmp_path / "final.json")


def test_cifar_augmentation_only_changes_training(tmp_path):
    task = cnn_task(tmp_path)
    images = torch.arange(2 * 3 * 32 * 32).reshape(2, 3, 32, 32).to(torch.uint8)
    evaluation = transform(images, task.initial_recipe, task.dataset, training=False)
    expected = (images.float() / 255 - 0.5) / 0.5
    torch.testing.assert_close(evaluation, expected)
    assert (
        transform(images, task.initial_recipe, task.dataset, training=True).shape == expected.shape
    )


def test_binary_f1_is_not_macro_f1():
    logits = np.array([[3, 0], [3, 0], [3, 0], [0, 3]], dtype=np.float32)
    metrics = classification_metrics(logits, np.array([0, 0, 1, 1]))
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] != metrics["f1"]


def text_task(tmp_path):
    transformers = pytest.importorskip("transformers")
    checkpoint = tmp_path / "fixture-bert"
    checkpoint.mkdir()
    (checkpoint / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nmovie\n")
    tokenizer = transformers.BertTokenizerFast(
        vocab_file=str(checkpoint / "vocab.txt"), do_lower_case=True
    )
    tokenizer.save_pretrained(checkpoint)
    config = transformers.BertConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
    )
    transformers.BertModel(config, add_pooling_layer=False).save_pretrained(checkpoint)
    spec = TextModelSpec(
        model_id="software-fixture-not-a-pretrained-benchmark",
        revision="fixture",
        local_path=str(checkpoint),
        file_sha256={p.name: file_hash(p) for p in checkpoint.iterdir()},
    )
    ds = {
        "kind": "text_jsonl",
        "identity": "unit-test-text",
        "revision": "fixture",
        "input_dim": 1,
        "num_classes": 2,
        "standardize": False,
        "seed": 0,
        "train_rows": 4,
        "validation_rows": 2,
        "test_rows": 2,
        "text_columns": ["sentence1", "sentence2"],
    }
    for split, size in (("train", 4), ("validation", 2), ("test", 2)):
        path = tmp_path / f"{split}.jsonl"
        rows = [
            {
                "row_id": f"{split}-{i}",
                "sentence1": "good movie",
                "sentence2": "bad movie",
                "label": i % 2,
            }
            for i in range(size)
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        ds[f"{split}_file"], ds[f"{split}_sha256"] = str(path), file_hash(path)
    task = tiny_task().model_copy(
        update={
            "adapter": "text_classification",
            "purpose": "development",
            "protocol_status": "ready",
            "dataset": DatasetSpec(**ds),
            "text_model": spec,
            "metric": "f1",
        }
    )
    task.initial_recipe = task.initial_recipe.model_copy(
        update={"hidden_sizes": [], "max_length": 8, "epochs": 1, "batch_size": 2}
    )
    return task


def test_local_bert_worker_freezing_pooling_and_token_retention(tmp_path, monkeypatch):
    from rsi_ml.tasks.text import BertClassifier

    task = text_task(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    prepared, trial = tmp_path / "prepared", tmp_path / "trial"
    prepare_task(task, prepared)
    result = run_trial(task, task.initial_recipe, prepared, trial, timeout=30)
    assert result.status == "completed", (trial / "stderr.log").read_text()
    assert (trial / "validation_tokens.npz").exists()
    assert "f1" in result.metrics
    modified = task.initial_recipe.model_copy(
        update={"pooling": "mean", "freeze_encoder_layers": 1, "freeze_embeddings": True}
    )
    model = BertClassifier(task, modified, pretrained=True)
    assert not any(p.requires_grad for p in model.encoder.encoder.layer[0].parameters())
    assert any(p.requires_grad for p in model.encoder.encoder.layer[1].parameters())
    receipt = score_test(task, prepared, trial, tmp_path / "final.json")
    assert "f1" in receipt["metrics"]
    (Path(task.text_model.local_path) / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        prepare_task(task, prepared)


def test_train_validation_overlap_rejected(tmp_path):
    task = cnn_task(tmp_path)
    path = Path(task.dataset.validation_file)
    with np.load(path) as d:
        images, labels = d["images"].copy(), d["labels"].copy()
    np.savez(path, images=images, labels=labels, row_id=np.array(["train-0", "validation-1"]))
    task.dataset.validation_sha256 = file_hash(path)
    with pytest.raises(ValueError, match="overlap"):
        prepare_task(task, tmp_path / "prepared")


def test_adapter_rejects_a_silently_ignored_setting(tmp_path):
    from rsi_ml.tasks.registry import validate_recipe

    task = cnn_task(tmp_path)
    task.initial_recipe.normalization = "layer_norm"
    with pytest.raises(ValueError, match="ignored"):
        validate_recipe(task, task.initial_recipe)


def test_sst2_tokenization_handles_single_sentence(tmp_path):
    from rsi_ml.tasks.text import tokenize

    task = text_task(tmp_path)
    task.dataset.text_columns = ["sentence"]
    encoded = tokenize(task, task.initial_recipe, [{"sentence": "good movie"}])
    assert encoded["input_ids"].shape == (1, task.initial_recipe.max_length)
    assert encoded["attention_mask"].sum() == 4  # CLS, two words, SEP.
