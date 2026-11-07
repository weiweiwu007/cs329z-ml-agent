"""Local import and numerical report checks; no real datasets or experiments."""

import json

import numpy as np
import pytest

from rsi_ml.data import import_data
from rsi_ml.diagnostics import diagnostic_report
from rsi_ml.storage import read_json, write_json
from rsi_ml.tasks.mlp import file_hash


@pytest.mark.parametrize("kind", ["cifar100_binary", "numeric_csv", "text_tsv"])
def test_explicit_import_preserves_rows_and_provenance(tmp_path, kind):
    source = tmp_path / "source"
    plan = {
        "protocol_status": "ready",
        "identity": "software-fixture",
        "revision": "v1",
        "kind": kind,
        "selection_rule": "explicit fixture indices",
        "selection_seed": 0,
        "splits": {
            "train": {"source": "source", "indices": [0, 1]},
            "validation": {"source": "source", "indices": [2, 3]},
            "test": {"source": "source", "indices": [4, 5]},
        },
    }
    if kind == "cifar100_binary":
        original = np.arange(6 * 3074).reshape(6, 3074).astype(np.uint8)
        original[:, 0] = 0
        original[:, 1] = np.arange(6)
        source.write_bytes(original.tobytes())
    elif kind == "numeric_csv":
        source.write_text("".join(f"{i},{i % 2}\n" for i in range(6)))
        plan.update(
            delimiter=",", fieldnames=["x", "label"], feature_columns=["x"], label_column="label"
        )
    else:
        source.write_text("sentence\tlabel\n" + "".join(f"text {i}\t{i % 2}\n" for i in range(6)))
        plan.update(delimiter="\t", text_columns={"sentence": "sentence"}, label_column="label")
    plan["sources"] = {"source": {"path": str(source), "sha256": file_hash(source)}}
    path = tmp_path / "plan.json"
    write_json(path, plan)
    manifest = import_data(path, tmp_path / "data")
    for split in ("train", "validation", "test"):
        entry = manifest["splits"][split]
        assert entry["rows"] == 2
        assert entry["source_indices"] == plan["splits"][split]["indices"]
    if kind == "cifar100_binary":
        with np.load(tmp_path / "data/validation.npz") as data:
            np.testing.assert_array_equal(data["images"].reshape(2, -1), original[2:4, 2:])
            assert data["images"].dtype == np.uint8
    elif kind == "text_tsv":
        assert (
            json.loads((tmp_path / "data/test.jsonl").read_text().splitlines()[0])["sentence"]
            == "text 4"
        )
    plan["splits"]["test"]["indices"] = [0, 4]
    write_json(path, plan)
    with pytest.raises(ValueError, match="overlap"):
        import_data(path, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_import_draft_does_not_read_sources_or_create_output(tmp_path):
    path = tmp_path / "draft.json"
    write_json(path, {"protocol_status": "draft"})
    with pytest.raises(ValueError, match="draft"):
        import_data(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_split_manifest_checks_identity_without_reading_test_data(tmp_path):
    from test_training import tiny_task

    from rsi_ml.tasks.registry import verify_split_manifest

    manifest = {
        "identity": "unit-test-fixture",
        "revision": "1",
        "sources": {"s": {"sha256": "source"}},
        "splits": {
            name: {"source": "s", "source_indices": [2 * i, 2 * i + 1], "rows": 2, "sha256": name}
            for i, name in enumerate(("train", "validation", "test"))
        },
    }
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    task = tiny_task()
    for split in ("train", "validation", "test"):
        setattr(task.dataset, f"{split}_rows", 2)
        setattr(task.dataset, f"{split}_sha256", split)
    task.dataset.split_manifest = str(path)
    task.dataset.split_manifest_sha256 = file_hash(path)
    verify_split_manifest(task)
    manifest["splits"]["test"]["source_indices"] = [0, 5]
    write_json(path, manifest)
    task.dataset.split_manifest_sha256 = file_hash(path)
    with pytest.raises(ValueError, match="overlap"):
        verify_split_manifest(task)


def test_diagnostic_report_is_derived_from_raw_outputs_and_checks_integrity(tmp_path):
    output = tmp_path / "validation_outputs.npz"
    np.savez(
        output,
        logits=np.array([[3.0, 0.0], [0.0, 3.0], [4.0, 0.0], [0.0, 4.0]]),
        labels=np.array([0, 0, 1, 1]),
        row_id=np.array(["a", "b", "c", "d"]),
    )
    write_json(
        tmp_path / "result.json",
        {"status": "completed", "artifact_hashes": {"validation_outputs.npz": file_hash(output)}},
    )
    write_json(tmp_path / "learning_curve.json", [{"epoch": 1, "train_loss": 1.0}])
    report = diagnostic_report(tmp_path)
    assert report["scores"]["accuracy"] == 0.5
    assert report["confident_errors"][0]["row_id"] == "c"
    assert read_json(tmp_path / "diagnostic_report.json")["confusion_matrix"] == [[1, 1], [1, 1]]
    with output.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="intact"):
        diagnostic_report(tmp_path)
