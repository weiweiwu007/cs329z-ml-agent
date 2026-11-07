"""Measured error analysis; no LLM judging and no extra training experiments.

The report is the study's independent variable, so it must be something an ML
engineer could act on: named classes, the actual mistaken text, and which tabular
features distinguish errors. Everything is validation-only and derived from raw
logits. Test data are never opened. B/D agents see `compact`; `full` is on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .storage import read_json, write_json
from .tasks.mlp import classification_metrics, file_hash

MAX_CLASSES = 10
MAX_CONFUSIONS = 10
MAX_ERRORS_IN_CONTEXT = 10
MAX_ERRORS_ON_DISK = 20
MAX_TEXT_CHARS = 300
MAX_FEATURES = 8


def _task_info(trial: Path) -> dict:
    path = trial / "task.json"
    return read_json(path) if path.exists() else {}


def _short_id(row_id: str) -> str:
    # Source IDs are "<source sha256>:<row>"; the hash is noise in agent context.
    return str(row_id).rsplit(":", 1)[-1]


def _validation_text(prepared: Path, columns: list[str]) -> dict[str, dict[str, str]]:
    path = prepared / "validation.jsonl"
    if not columns or not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            record = json.loads(line)
            rows[str(record["row_id"])] = {c: str(record[c])[:MAX_TEXT_CHARS] for c in columns}
    return rows


def _feature_contrast(prepared: Path, wrong: np.ndarray) -> list[dict]:
    """Which raw features differ most between misclassified and correct validation rows."""
    data_path, names_path = prepared / "validation.npz", prepared / "normalization.json"
    if not (data_path.exists() and names_path.exists()) or wrong.all() or not wrong.any():
        return []
    with np.load(data_path, allow_pickle=False) as data:
        if "raw_x" not in data or len(data["raw_x"]) != len(wrong):
            return []
        features = data["raw_x"].astype(np.float64)
    names = read_json(names_path)["features"]
    scale = features.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    in_errors, in_correct = features[wrong].mean(axis=0), features[~wrong].mean(axis=0)
    difference = (in_errors - in_correct) / scale
    order = sorted(range(len(names)), key=lambda i: (-abs(float(difference[i])), i))
    return [
        {
            "feature": names[i],
            "mean_in_errors": float(in_errors[i]),
            "mean_in_correct": float(in_correct[i]),
            "standardized_difference": float(difference[i]),
        }
        for i in order[:MAX_FEATURES]
    ]


def diagnostic_report(trial: Path) -> dict:
    outputs = trial / "validation_outputs.npz"
    result = read_json(trial / "result.json")
    expected = result.get("artifact_hashes", {}).get("validation_outputs.npz")
    if result["status"] != "completed" or not expected or file_hash(outputs) != expected:
        raise ValueError("Diagnostics require intact outputs from a completed validation run.")
    with np.load(outputs, allow_pickle=False) as data:
        logits, labels, ids = data["logits"], data["labels"], data["row_id"]
    dataset = _task_info(trial).get("dataset", {})
    names = dataset.get("class_names")
    prepared = trial.parent.parent / "prepared"

    def label(class_id: int) -> str:
        return names[class_id] if names else str(class_id)

    scores = classification_metrics(logits, labels)
    predicted = logits.argmax(1)
    wrong = predicted != labels
    confusion = np.zeros((logits.shape[1], logits.shape[1]), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    probabilities = np.exp(logits - logits.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    confidence = probabilities[np.arange(len(labels)), predicted]
    classes = []
    for c in range(len(confusion)):
        support, predicted_count, correct = (
            int(confusion[c].sum()),
            int(confusion[:, c].sum()),
            int(confusion[c, c]),
        )
        classes.append(
            {
                "class_id": c,
                "class_name": label(c),
                "support": support,
                "predicted": predicted_count,
                "recall": correct / support if support else None,
                "precision": correct / predicted_count if predicted_count else None,
            }
        )
    pairs = [
        {
            "true": label(i),
            "predicted": label(j),
            "count": int(confusion[i, j]),
            "share_of_true_class": float(confusion[i, j] / confusion[i].sum()),
        }
        for i in range(len(confusion))
        for j in range(len(confusion))
        if i != j and confusion[i, j]
    ]
    pairs.sort(key=lambda row: (-row["count"], row["true"], row["predicted"]))
    mistakes = sorted(np.flatnonzero(wrong), key=lambda i: (-float(confidence[i]), int(i)))
    text = _validation_text(prepared, dataset.get("text_columns") or [])
    error_examples = []
    for i in mistakes[:MAX_ERRORS_ON_DISK]:
        example = {
            "row_id": _short_id(ids[i]),
            "true": label(int(labels[i])),
            "predicted": label(int(predicted[i])),
            "predicted_probability": float(confidence[i]),
        }
        if str(ids[i]) in text:
            example["text"] = text[str(ids[i])]
        error_examples.append(example)
    calibration = {
        "mean_confidence_when_correct": float(confidence[~wrong].mean())
        if (~wrong).any()
        else None,
        "mean_confidence_when_wrong": float(confidence[wrong].mean()) if wrong.any() else None,
    }
    curve = read_json(trial / "learning_curve.json")
    compact = {
        "scores": scores,
        "error_count": int(wrong.sum()),
        "validation_examples": len(labels),
        "calibration": calibration,
        "lowest_recall_classes": sorted(
            [r for r in classes if r["support"]], key=lambda r: (r["recall"], r["class_id"])
        )[:MAX_CLASSES],
        "largest_confusions": pairs[:MAX_CONFUSIONS],
        "confident_errors": error_examples[:MAX_ERRORS_IN_CONTEXT],
        "curve_endpoints": [curve[0], curve[-1]] if curve else [],
        "interpretation": "Measured validation evidence; causes remain hypotheses. Training loss may use weighting or smoothing.",
        "validation_outputs_sha256": expected,
    }
    contrast = _feature_contrast(prepared, wrong)
    if contrast:
        compact["features_distinguishing_errors"] = contrast
    full = {
        **compact,
        "confident_errors": error_examples,
        "per_class": classes,
        "confusion_matrix": confusion.tolist(),
        "curve": curve,
        "selection": f"up to {MAX_ERRORS_ON_DISK} incorrect examples sorted by predicted-class probability",
    }
    write_json(trial / "diagnostic_report.json", full)
    return compact
