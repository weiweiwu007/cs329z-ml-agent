"""Explicit local import: never download datasets or invent data splits."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .storage import read_json, write_json
from .tasks.mlp import file_hash


def import_data(plan_path: Path, output: Path) -> dict:
    """Convert official local files using explicit, disjoint source row indices.

    CIFAR binary records contain a coarse label, fine label, then 3072 RGB bytes.
    CSV/TSV supports Spambase/OpenML and SST-2/MRPC respectively. Unlabeled GLUE
    test rows cannot be graded: explicitly select a held-out labeled split.
    """
    plan = read_json(plan_path)
    required = ("identity", "revision", "selection_rule", "selection_seed")
    if plan.get("protocol_status") != "ready" or any(plan.get(k) is None for k in required):
        raise ValueError(
            "Data import plan is draft or has unresolved provenance/selection settings."
        )
    if set(plan.get("splits", {})) != {"train", "validation", "test"}:
        raise ValueError("Declare all three split selections.")
    root = plan_path.resolve().parent
    sources, source_info = {}, {}
    kind = plan["kind"]
    for name, entry in plan["sources"].items():
        path = (root / entry["path"]).resolve()
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")
        if kind == "cifar100_binary":
            raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            if len(raw) % 3074:
                raise ValueError("CIFAR-100 binary source length must be a multiple of 3074.")
            sources[name] = raw.reshape(-1, 3074)
        elif kind in {"numeric_csv", "text_tsv"}:
            with path.open(newline="", encoding="utf-8") as stream:
                sources[name] = list(
                    csv.DictReader(
                        stream, delimiter=plan["delimiter"], fieldnames=plan.get("fieldnames")
                    )
                )
        else:
            raise ValueError("Unsupported local source format.")
        source_info[name] = {
            "path": str(path),
            "sha256": entry["sha256"],
            "rows": len(sources[name]),
        }
    selected, seen = {}, set()
    for split, selection in plan["splits"].items():
        source_name = selection["source"]
        source = sources[source_name]
        indices = selection["indices"]
        if indices == "all":
            indices = list(range(len(source)))
        elif isinstance(indices, str):
            indices = read_json(root / indices)
        if not isinstance(indices, list) or len(indices) < 2 or len(set(indices)) != len(indices):
            raise ValueError("Each split needs unique row indices or 'all'.")
        if any(type(i) is not int or not 0 <= i < len(source) for i in indices):
            raise ValueError("Invalid row index.")
        ids = [f"{source_info[source_name]['sha256']}:{i}" for i in indices]
        if set(ids) & seen:
            raise ValueError("Selected rows overlap across splits.")
        seen.update(ids)
        selected[split] = (source_name, indices, ids)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, (source_name, indices, ids) in selected.items():
        source = sources[source_name]
        if kind == "cifar100_binary":
            rows = source[indices]
            if (rows[:, 1] >= 100).any():
                raise ValueError("Invalid CIFAR fine label.")
            path = output / f"{split}.npz"
            np.savez(
                path,
                images=rows[:, 2:].reshape(-1, 3, 32, 32),
                labels=rows[:, 1].astype(np.int64),
                row_id=np.asarray(ids),
            )
        elif kind == "numeric_csv":
            path = output / f"{split}.csv"
            features, label = plan["feature_columns"], plan["label_column"]
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([*features, "label"])
                for i in indices:
                    writer.writerow([*[source[i][c] for c in features], source[i][label]])
        else:
            path = output / f"{split}.jsonl"
            records = []
            for i, row_id in zip(indices, ids):
                record = {
                    target: source[i][column] for target, column in plan["text_columns"].items()
                }
                record.update(row_id=row_id, label=int(source[i][plan["label_column"]]))
                if record["label"] not in (0, 1):
                    raise ValueError("Text classification source needs public binary labels.")
                records.append(record)
            path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
        files[split] = {
            "path": str(path.resolve()),
            "sha256": file_hash(path),
            "rows": len(indices),
            "source": source_name,
            "source_indices": indices,
        }
    manifest = {
        "identity": plan["identity"],
        "revision": plan["revision"],
        "kind": kind,
        "selection_rule": plan["selection_rule"],
        "selection_seed": plan["selection_seed"],
        "sources": source_info,
        "splits": files,
        "normalization": "none; raw inputs retained for later preprocessing",
    }
    write_json(output / "manifest.json", manifest)
    return manifest
