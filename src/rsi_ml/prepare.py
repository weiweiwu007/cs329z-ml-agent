"""Fetch pinned sources and produce exact, recorded splits for the A-D study.

Downloads happen only with --execute. Every remote file is addressed by an immutable
revision where the host offers one, and verified by SHA-256 where a hash is published.
Splits are deterministic from one recorded seed; test rows are never used to fit anything.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import import_data
from .storage import now, read_json, write_json
from .tasks.mlp import file_hash

SPLIT_SEED = 20261104
GLUE_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
SETFIT_SST2_REVISION = "00ea8ccb7a54b4e3780a3e51aa3f80361ff849c0"
BERT_TINY_ID = "google/bert_uncased_L-2_H-128_A-2"
BERT_TINY_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
BINARY_NAMES = {
    "spambase": ["not_spam", "spam"],
    "sst2": ["negative", "positive"],
    "mrpc": ["not_paraphrase", "paraphrase"],
}
# Column order and class numbering from the archive's covtype.info.
COVERTYPE_FEATURES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    *(f"Wilderness_Area{i}" for i in range(1, 5)),
    *(f"Soil_Type{i}" for i in range(1, 41)),
]
COVERTYPE_CLASSES = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    size: int | None = None
    sha256: str | None = None


def _hf(kind: str, repo: str, revision: str, path: str) -> str:
    prefix = "datasets/" if kind == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{path}"


def _glue(config: str, split: str, sha256: str, size: int) -> Source:
    path = f"{config}/{split}-00000-of-00001.parquet"
    return Source(
        f"glue-{config}-{split}.parquet",
        _hf("dataset", "nyu-mll/glue", GLUE_REVISION, path),
        size,
        sha256,
    )


SOURCES = (
    # Hosts publish no SHA-256: the size and the structural checks below verify these,
    # and the hash computed at download is recorded in sources.lock.json.
    Source(
        "cifar-100-binary.tar.gz",
        "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
        168513733,
    ),
    Source("spambase.zip", "https://archive.ics.uci.edu/static/public/94/spambase.zip"),
    # The host publishes no hash. This one is from the copy checked on 2026-11-04 against
    # the row and class counts in its documentation.
    Source(
        "covertype.zip",
        "https://archive.ics.uci.edu/static/public/31/covertype.zip",
        11251206,
        "89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d",
    ),
    _glue(
        "sst2",
        "train",
        "66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229",
        3110468,
    ),
    _glue(
        "sst2",
        "validation",
        "a1371f3b3a7b0bcefa8388799a9359dc3ce76c349cc0079507a7991364fd2a9b",
        72819,
    ),
    _glue(
        "sst2", "test", "e9d23cf0067211d2baf018328b507f5153fb6704d75117295a8bda47c7adccb1", 147793
    ),
    _glue(
        "mrpc", "train", "61fd41301e0e244b0420c4350a170c8e7cf64740335fc875a4af2d79af0df0af", 649281
    ),
    _glue(
        "mrpc",
        "validation",
        "33c007dbf5bfa8463d87a13e6226df8c0fcf2596c2cd39d0f3bb79754e00f50f",
        75678,
    ),
    _glue(
        "mrpc", "test", "a623ed1cbdf445b11f8e249acbf649d7d3a5ee58c918554c40cbd8307e488693", 308441
    ),
    # GLUE withholds SST-2 test labels. These are the same public sentences with labels,
    # matched by exact text below; a run stops if any sentence fails to match.
    Source(
        "setfit-sst2-test.jsonl",
        _hf("dataset", "SetFit/sst2", SETFIT_SST2_REVISION, "test.jsonl"),
        281037,
    ),
    Source(
        "bert-tiny-config.json", _hf("model", BERT_TINY_ID, BERT_TINY_REVISION, "config.json"), 382
    ),
    Source(
        "bert-tiny-vocab.txt", _hf("model", BERT_TINY_ID, BERT_TINY_REVISION, "vocab.txt"), 231508
    ),
    Source(
        "bert-tiny-model.safetensors",
        _hf("model", BERT_TINY_ID, BERT_TINY_REVISION, "model.safetensors"),
        17739144,
        "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414",
    ),
)


def source_plan() -> dict:
    return {
        "mode": "plan_only",
        "downloaded": False,
        "known_bytes": sum(s.size or 0 for s in SOURCES),
        "sources": [
            {"name": s.name, "url": s.url, "bytes": s.size, "sha256_pinned": bool(s.sha256)}
            for s in SOURCES
        ],
    }


def _download(source: Source, directory: Path) -> Path:
    destination = directory / source.name
    if destination.exists():
        return destination
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(source.url, headers={"User-Agent": "rsi-ml-data-prep"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(block)
            stream.write(block)
    if source.sha256 and digest.hexdigest() != source.sha256:
        partial.unlink()
        raise ValueError(f"SHA-256 mismatch for {source.name}; refusing to use it.")
    if source.size and partial.stat().st_size != source.size:
        partial.unlink()
        raise ValueError(f"Size mismatch for {source.name}; refusing to use it.")
    if source.name.endswith(".zip") and not zipfile.is_zipfile(partial):
        partial.unlink()  # This host publishes no hash or size to catch a truncated download.
        raise ValueError(f"{source.name} is not a complete archive; rerun to fetch it again.")
    partial.replace(destination)
    return destination


_SOURCE_TASKS = {
    "cifar-100": {"cifar100"},
    "spambase": {"spambase"},
    "covertype": {"covertype"},
    "glue-sst2": {"sst2"},
    "setfit": {"sst2"},
    "glue-mrpc": {"mrpc"},
    "bert-tiny": {"sst2", "mrpc"},
}


def _needed(source: Source, tasks: set[str] | None) -> bool:
    if tasks is None:
        return True
    return any(
        source.name.startswith(prefix) and owners & tasks
        for prefix, owners in _SOURCE_TASKS.items()
    )


def fetch_sources(root: Path, tasks: set[str] | None = None) -> dict:
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    # Fetching one task must not erase the record of what earlier tasks downloaded.
    lock_path = root / "sources.lock.json"
    entries = read_json(lock_path)["sources"] if lock_path.exists() else {}
    for source in SOURCES:
        if not _needed(source, tasks):
            continue
        path = _download(source, downloads)
        entries[source.name] = {
            "url": source.url,
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }
    lock = {"fetched": now(), "sources": entries}
    write_json(lock_path, lock)
    return lock


def stratified_indices(labels, fractions: dict[str, float], seed: int) -> dict[str, list[int]]:
    """Deterministic per-class split; the last named split takes any rounding remainder."""
    labels = np.asarray(labels)
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Split fractions must sum to 1.")
    rng = np.random.default_rng(seed)
    names, selected = list(fractions), {name: [] for name in fractions}
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        rng.shuffle(members)
        start = 0
        for position, name in enumerate(names):
            size = (
                len(members) - start
                if position == len(names) - 1
                else round(len(members) * fractions[name])
            )
            selected[name].extend(members[start : start + size].tolist())
            start += size
    return {name: sorted(values) for name, values in selected.items()}


def _write_indices(root: Path, task: str, splits: dict[str, list[int]]) -> dict[str, str]:
    directory = root / "plans" / f"{task}-indices"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, values in splits.items():
        write_json(directory / f"{split}.json", values)
        paths[split] = f"{task}-indices/{split}.json"
    return paths


def _finish(root: Path, task: str, plan: dict, **extra) -> dict:
    plan_path = root / "plans" / f"{task}.json"
    write_json(plan_path, {"protocol_status": "ready", "identity": task, **plan})
    output = root / "selected" / task
    manifest = import_data(plan_path, output)
    manifest.update(extra)
    write_json(output / "manifest.json", manifest)
    return manifest


def build_cifar100(root: Path, seed: int) -> dict:
    target = root / "sources" / "cifar100"
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(root / "downloads" / "cifar-100-binary.tar.gz") as archive:
        for member in ("train.bin", "test.bin", "fine_label_names.txt"):
            stream = archive.extractfile(f"cifar-100-binary/{member}")
            (target / member).write_bytes(stream.read())
    names = [
        n.strip() for n in (target / "fine_label_names.txt").read_text().splitlines() if n.strip()
    ]
    if len(names) != 100:
        raise ValueError(f"Expected 100 CIFAR-100 fine labels, found {len(names)}.")
    raw = np.fromfile(target / "train.bin", dtype=np.uint8).reshape(-1, 3074)
    splits = stratified_indices(raw[:, 1], {"train": 0.9, "validation": 0.1}, seed)
    _write_indices(root, "cifar100", splits)
    plan = {
        "revision": "cs.toronto.edu/~kriz/cifar-100-binary.tar.gz sha256:"
        + file_hash(root / "downloads" / "cifar-100-binary.tar.gz"),
        "kind": "cifar100_binary",
        "selection_rule": "stratified 90/10 of the official train file; official test file is test",
        "selection_seed": seed,
        "sources": {
            "official_train": {
                "path": "../sources/cifar100/train.bin",
                "sha256": file_hash(target / "train.bin"),
            },
            "official_test": {
                "path": "../sources/cifar100/test.bin",
                "sha256": file_hash(target / "test.bin"),
            },
        },
        "splits": {
            "train": {"source": "official_train", "indices": "cifar100-indices/train.json"},
            "validation": {
                "source": "official_train",
                "indices": "cifar100-indices/validation.json",
            },
            "test": {"source": "official_test", "indices": "all"},
        },
    }
    manifest_path = root / "selected" / "cifar100"
    manifest = _finish(root, "cifar100", plan, class_names=names)
    # Normalization is fitted on the selected training images only.
    with np.load(manifest_path / "train.npz", allow_pickle=False) as data:
        images = data["images"].astype(np.float64) / 255.0
    manifest["image_mean"] = images.mean(axis=(0, 2, 3)).tolist()
    manifest["image_std"] = images.std(axis=(0, 2, 3)).tolist()
    manifest["normalization_fitted_on"] = "selected train split only"
    write_json(manifest_path / "manifest.json", manifest)
    return manifest


def _spambase_columns(names_text: str) -> list[str]:
    columns = []
    for line in names_text.splitlines():
        name, separator, kind = line.partition(":")
        if separator and kind.strip() == "continuous." and not name.startswith("|"):
            columns.append(name.strip())
    if len(columns) != 57:
        raise ValueError(f"Expected 57 Spambase feature names, found {len(columns)}.")
    return columns


def build_spambase(root: Path, seed: int) -> dict:
    target = root / "sources" / "spambase"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "downloads" / "spambase.zip") as archive:
        (target / "spambase.data").write_bytes(archive.read("spambase.data"))
        (target / "spambase.names").write_bytes(archive.read("spambase.names"))
    columns = _spambase_columns((target / "spambase.names").read_text(errors="replace"))
    with (target / "spambase.data").open(newline="") as stream:
        labels = [int(float(row[-1])) for row in csv.reader(stream) if row]
    splits = stratified_indices(labels, {"train": 0.6, "validation": 0.2, "test": 0.2}, seed)
    _write_indices(root, "spambase", splits)
    plan = {
        "revision": "UCI ML Repository dataset 94, spambase.zip sha256:"
        + file_hash(root / "downloads" / "spambase.zip"),
        "kind": "numeric_csv",
        "selection_rule": "stratified 60/20/20 of all rows; duplicate rows are not removed",
        "selection_seed": seed,
        "delimiter": ",",
        "fieldnames": [*columns, "label"],
        "feature_columns": columns,
        "label_column": "label",
        "sources": {
            "labeled_source": {
                "path": "../sources/spambase/spambase.data",
                "sha256": file_hash(target / "spambase.data"),
            }
        },
        "splits": {
            split: {"source": "labeled_source", "indices": f"spambase-indices/{split}.json"}
            for split in splits
        },
    }
    return _finish(root, "spambase", plan, class_names=BINARY_NAMES["spambase"])


def build_covertype(root: Path, seed: int) -> dict:
    target = root / "sources" / "covertype"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "downloads" / "covertype.zip") as archive:
        compressed = archive.read("covtype.data.gz")
        (target / "covtype.info").write_bytes(archive.read("covtype.info"))
    rows = list(csv.reader(io.StringIO(gzip.decompress(compressed).decode("ascii"))))
    if any(len(row) != len(COVERTYPE_FEATURES) + 1 for row in rows):
        raise ValueError("Covertype rows must hold 54 features and a label.")
    # The source numbers its classes 1 to 7; the trainer needs explicit IDs from 0.
    labels = [int(row[-1]) - 1 for row in rows]
    if not set(labels) <= set(range(len(COVERTYPE_CLASSES))):
        raise ValueError("Covertype labels must be between 1 and 7.")
    with (target / "covtype.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows([*row[:-1], label] for row, label in zip(rows, labels))
    splits = stratified_indices(labels, {"train": 0.6, "validation": 0.2, "test": 0.2}, seed)
    _write_indices(root, "covertype", splits)
    plan = {
        "revision": "UCI ML Repository dataset 31, covertype.zip sha256:"
        + file_hash(root / "downloads" / "covertype.zip"),
        "kind": "numeric_csv",
        "selection_rule": "stratified 60/20/20 of all rows; labels renumbered from 1-7 to 0-6",
        "selection_seed": seed,
        "delimiter": ",",
        "fieldnames": [*COVERTYPE_FEATURES, "label"],
        "feature_columns": COVERTYPE_FEATURES,
        "label_column": "label",
        "sources": {
            "labeled_source": {
                "path": "../sources/covertype/covtype.csv",
                "sha256": file_hash(target / "covtype.csv"),
            }
        },
        "splits": {
            split: {"source": "labeled_source", "indices": f"covertype-indices/{split}.json"}
            for split in splits
        },
    }
    return _finish(root, "covertype", plan, class_names=COVERTYPE_CLASSES)


def _parquet_rows(path: Path) -> list[dict]:
    try:
        from pyarrow import parquet
    except ImportError:
        raise ValueError("Install the project's [data] extra to read GLUE parquet files.") from None
    return parquet.read_table(path).to_pylist()


def _write_tsv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[c] for c in columns])


def _text_plan(root: Path, task: str, columns: list[str], revision: str, rule: str) -> dict:
    target = root / "sources" / task
    return {
        "revision": revision,
        "kind": "text_tsv",
        "selection_rule": rule,
        "selection_seed": 0,
        "delimiter": "\t",
        "label_column": "label",
        "text_columns": {column: column for column in columns},
        "sources": {
            split: {
                "path": f"../sources/{task}/{split}.tsv",
                "sha256": file_hash(target / f"{split}.tsv"),
            }
            for split in ("train", "validation", "test")
        },
        "splits": {
            split: {"source": split, "indices": "all"} for split in ("train", "validation", "test")
        },
    }


def build_sst2(root: Path) -> dict:
    downloads, target = root / "downloads", root / "sources" / "sst2"
    labels = {}
    for line in (downloads / "setfit-sst2-test.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            record = json.loads(line)
            labels[record["text"].strip()] = record["label"]
    test = []
    unmatched = 0
    for row in _parquet_rows(downloads / "glue-sst2-test.parquet"):
        label = labels.get(row["sentence"].strip())
        unmatched += label is None
        test.append({"sentence": row["sentence"], "label": label})
    if unmatched:
        raise ValueError(
            f"{unmatched} GLUE SST-2 test sentences have no labeled match; the labeled-test "
            "decision needs revisiting before this dataset can be used."
        )
    columns = ["sentence", "label"]
    for split in ("train", "validation"):
        _write_tsv(
            target / f"{split}.tsv",
            columns,
            _parquet_rows(downloads / f"glue-sst2-{split}.parquet"),
        )
    _write_tsv(target / "test.tsv", columns, test)
    plan = _text_plan(
        root,
        "sst2",
        ["sentence"],
        f"nyu-mll/glue@{GLUE_REVISION}; test labels from SetFit/sst2@{SETFIT_SST2_REVISION}",
        "GLUE train and validation as published; GLUE test sentences joined to public labels "
        "by exact text",
    )
    return _finish(root, "sst2", plan, class_names=BINARY_NAMES["sst2"])


def build_mrpc(root: Path) -> dict:
    downloads, target = root / "downloads", root / "sources" / "mrpc"
    columns = ["sentence1", "sentence2", "label"]
    for split in ("train", "validation", "test"):
        _write_tsv(
            target / f"{split}.tsv",
            columns,
            _parquet_rows(downloads / f"glue-mrpc-{split}.parquet"),
        )
    plan = _text_plan(
        root,
        "mrpc",
        ["sentence1", "sentence2"],
        f"nyu-mll/glue@{GLUE_REVISION}",
        "GLUE train, validation, and labeled test as published",
    )
    return _finish(root, "mrpc", plan, class_names=BINARY_NAMES["mrpc"])


def build_bert_tiny(root: Path) -> dict:
    downloads, target = root / "downloads", root / "models" / "bert-tiny"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "vocab.txt", "model.safetensors"):
        shutil.copyfile(downloads / f"bert-tiny-{name}", target / name)
    pin = {
        "model_id": BERT_TINY_ID,
        "revision": BERT_TINY_REVISION,
        "local_path": str(target.resolve()),
        "file_sha256": {p.name: file_hash(p) for p in sorted(target.iterdir())},
    }
    # Outside the checkpoint: its directory must contain exactly the pinned files.
    write_json(root / "models" / "bert-tiny.pin.json", pin)
    return pin


def prepare_all(
    root: Path, *, seed: int = SPLIT_SEED, fetch: bool = True, tasks: set[str] | None = None
) -> dict:
    root = Path(root).resolve()
    if fetch:
        fetch_sources(root, tasks)
    (root / "plans").mkdir(parents=True, exist_ok=True)
    builders = {
        "spambase": lambda: build_spambase(root, seed),
        "covertype": lambda: build_covertype(root, seed),
        "mrpc": lambda: build_mrpc(root),
        "sst2": lambda: build_sst2(root),
        "cifar100": lambda: build_cifar100(root, seed),
    }
    manifests = {}
    # Restartable: finished datasets are kept and never overwritten, so a rerun cannot
    # silently change a split. Output with no manifest is an interrupted conversion.
    for task, build in builders.items():
        if tasks is not None and task not in tasks:
            continue
        output = root / "selected" / task
        if (output / "manifest.json").exists():
            manifests[task] = read_json(output / "manifest.json")
            continue
        if output.exists():
            shutil.rmtree(output)
        manifests[task] = build()
    expected_rows = {
        "cifar100": {"train": 45000, "validation": 5000, "test": 10000},
        "sst2": {"train": 67349, "validation": 872, "test": 1821},
        "mrpc": {"train": 3668, "validation": 408, "test": 1725},
    }
    rows = {
        task: {split: entry["rows"] for split, entry in manifest["splits"].items()}
        for task, manifest in manifests.items()
    }
    if {"mrpc", "sst2"} & set(manifests):
        build_bert_tiny(root)
    receipt = {
        "prepared": now(),
        "split_seed": seed,
        "rows": rows,
        "row_counts_match_published": {
            **{task: rows[task] == expected_rows[task] for task in rows if task in expected_rows},
            **({"spambase": sum(rows["spambase"].values()) == 4601} if "spambase" in rows else {}),
            **(
                {"covertype": sum(rows["covertype"].values()) == 581012}
                if "covertype" in rows
                else {}
            ),
        },
    }
    mismatched = [k for k, ok in receipt["row_counts_match_published"].items() if not ok]
    if fetch and mismatched:
        raise ValueError(f"Row counts differ from the published datasets: {mismatched}: {rows}")
    write_json(root / "prepared.json", receipt)
    return receipt


def load_selected(root: Path, task: str) -> dict:
    return read_json(Path(root) / "selected" / task / "manifest.json")
