"""One trusted interface for MLP, CIFAR-100, and BERT classification."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from rsi_ml.schemas import Recipe, TaskSpec, digest

from . import mlp


def build_model(task: TaskSpec, recipe: Recipe, *, pretrained: bool = False):
    if task.adapter == "mlp_classification":
        return mlp.build_model(task.dataset.input_dim, task.dataset.num_classes, recipe)
    if task.adapter == "cifar100_classification":
        from .cnn import CifarResNet

        return CifarResNet(task.dataset.num_classes, recipe)
    from .text import BertClassifier

    return BertClassifier(task, recipe, pretrained=pretrained)


def parameter_count(task: TaskSpec, recipe: Recipe) -> int:
    if task.adapter == "mlp_classification":
        return mlp.parameter_count(task, recipe)
    import torch

    with torch.device("meta"):
        return sum(p.numel() for p in build_model(task, recipe).parameters())


# Fields a task family does not use. These stay at their inactive value; everything else is
# the agent's decision under a constraints task.
INACTIVE_FIELDS = {
    "mlp_classification": {
        "cnn_channels",
        "cnn_blocks",
        "augmentation",
        "pooling",
        "freeze_encoder_layers",
        "freeze_embeddings",
        "max_length",
    },
    "cifar100_classification": {
        "feature_preprocessing",
        "pooling",
        "freeze_encoder_layers",
        "freeze_embeddings",
        "max_length",
        "normalization",
    },
    "text_classification": {
        "feature_preprocessing",
        "cnn_channels",
        "cnn_blocks",
        "augmentation",
        "normalization",
    },
}
_FAMILY_NOTES = {
    "mlp_classification": (
        "hidden_sizes sets depth and width, one entry per hidden layer; an empty list is a "
        "linear model, which has no activation, dropout, or normalization."
    ),
    "cifar100_classification": (
        "cnn_channels and cnn_blocks set the residual stages and must have equal length; each "
        "stage after the first halves the image. hidden_sizes adds classifier layers. "
        "Residual blocks always use ReLU, and activation affects only classifier layers."
    ),
    "text_classification": (
        "The pretrained encoder is fixed. hidden_sizes sets the classifier head, "
        "freeze_encoder_layers cannot exceed the encoder's layers, and max_length cannot "
        "exceed its position embeddings. activation affects only classifier layers."
    ),
}


def recipe_rules(task: TaskSpec) -> dict:
    """What a constraints task tells every agent, identically across arms."""
    return {
        "you_decide": (
            "Every recipe field not listed in keep_at_initial_value, using any value the recipe "
            "schema allows. There is no menu: model size and shape, training length, "
            "optimization, regularization, and input handling are all your decisions."
        ),
        "constraints": (
            "task.limits: max_parameters, max_epochs, per_trial_seconds, and the remaining "
            "attempts and compute. A trial stops at per_trial_seconds or when remaining compute "
            "runs out, whichever comes first, and still counts as an attempt. The run ends when "
            "compute runs out."
        ),
        "keep_at_initial_value": sorted(INACTIVE_FIELDS[task.adapter]),
        "notes": [
            _FAMILY_NOTES[task.adapter],
            "momentum applies only to sgd; keep it 0 with adamw.",
            "warmup_fraction applies to every schedule; decay_fraction applies only to wsd.",
            (
                "Schema limits: at most 16 hidden layers, each width 1 to 8192; CNN 1 to 6 "
                "stages, widths up to 2048, at most 64 blocks per stage; with schedule wsd, "
                "warmup_fraction plus decay_fraction cannot exceed 1."
            ),
            (
                "context.all_attempts lists every attempt so far; context.records shows the most "
                "relevant ones in full."
            ),
        ],
    }


def validate_recipe(task: TaskSpec, recipe: Recipe) -> int:
    original = task.initial_recipe.model_dump()
    if task.recipe_policy == "menu":
        # An empty research space means no permitted changes, never unlimited edits.
        for name, value in recipe.model_dump().items():
            if name in task.search_space:
                if value not in task.search_space[name]:
                    raise ValueError(f"Recipe field {name} is outside the declared search space.")
            elif value != original[name] and (task.search_space or task.purpose != "demo"):
                raise ValueError(f"Recipe field {name} is fixed by the declared search space.")
    if (
        task.recipe_policy == "menu"
        and task.adapter == "mlp_classification"
        and not recipe.hidden_sizes
    ):
        raise ValueError("MLP needs hidden layers.")
    if task.recipe_policy == "constraints" and not recipe.hidden_sizes:
        # These would train identically and cost a full attempt while seeming to test something.
        unused = ["activation"]
        if task.adapter == "mlp_classification":
            unused += ["dropout", "normalization"]
        for name in unused:
            if getattr(recipe, name) != original[name]:
                raise ValueError(
                    f"{name} has no effect without hidden_sizes; keep it or add layers."
                )
    irrelevant = INACTIVE_FIELDS
    for name in irrelevant[task.adapter]:
        if getattr(recipe, name) != original[name]:
            raise ValueError(f"{name} is not an editable field for {task.adapter}.")
    defaults = {
        "cnn_channels": None,
        "cnn_blocks": None,
        "augmentation": "none",
        "pooling": "cls",
        "freeze_encoder_layers": 0,
        "freeze_embeddings": False,
        "max_length": None,
        "feature_preprocessing": None,
        "normalization": "none",
    }
    for name in irrelevant[task.adapter]:
        if getattr(recipe, name) != defaults[name]:
            raise ValueError(
                f"{name} would be ignored by {task.adapter}; keep it at its inactive value."
            )
    if recipe.optimizer != "sgd" and recipe.momentum != 0:
        raise ValueError("Momentum is an SGD-only setting; use zero for AdamW.")
    count = parameter_count(task, recipe)
    if count > task.limits.max_parameters:
        raise ValueError(f"Recipe needs {count} parameters; limit is {task.limits.max_parameters}.")
    if recipe.epochs > task.limits.max_epochs:
        raise ValueError("Recipe exceeds the declared epoch limit.")
    return count


def split_path(task: TaskSpec, split: str) -> Path:
    value = getattr(task.dataset, f"{split}_file")
    if not value:
        raise ValueError(f"Missing {split}_file.")
    return Path(value).resolve()


def verify_split_manifest(task: TaskSpec) -> None:
    """Validate split provenance without opening held-out data files."""
    if not task.dataset.split_manifest:
        return
    path = Path(task.dataset.split_manifest)
    if mlp.file_hash(path) != task.dataset.split_manifest_sha256:
        raise ValueError("Split manifest hash mismatch.")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("identity") != task.dataset.identity
        or manifest.get("revision") != task.dataset.revision
    ):
        raise ValueError("Split manifest identity/revision differs from the declared dataset.")
    seen = set()
    for split in ("train", "validation", "test"):
        entry = manifest["splits"][split]
        if entry["sha256"] != getattr(task.dataset, f"{split}_sha256") or entry["rows"] != getattr(
            task.dataset, f"{split}_rows"
        ):
            raise ValueError("Split manifest differs from declared split hashes/row counts.")
        source = manifest["sources"][entry["source"]]
        ids = [(source["sha256"], i) for i in entry["source_indices"]]
        if len(ids) != entry["rows"] or len(set(ids)) != len(ids) or set(ids) & seen:
            raise ValueError("Source indices overlap or do not match split size.")
        seen.update(ids)


def load_split(task: TaskSpec, split: str, *, source: Path | None = None):
    path = source or split_path(task, split)
    expected = getattr(task.dataset, f"{split}_sha256")
    if not expected or mlp.file_hash(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {split} split.")
    size = getattr(task.dataset, f"{split}_rows")
    if task.adapter == "cifar100_classification":
        with np.load(path, allow_pickle=False) as data:
            x, labels, ids = data["images"].copy(), data["labels"].copy(), data["row_id"].copy()
        if x.shape != (size, 3, 32, 32) or x.dtype != np.uint8:
            raise ValueError("CIFAR images must be uint8 [N,3,32,32], with declared N.")
        rows = x
    else:
        # Only a newline ends a row: text may contain U+0085 or U+2028, which splitlines() breaks on.
        text = path.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.split("\n") if line.strip()]
        if len(rows) != size or not task.dataset.text_columns:
            raise ValueError("Text split row count or text_columns is invalid.")
        if any(not isinstance(r.get(c), str) for r in rows for c in task.dataset.text_columns):
            raise ValueError("Text columns must contain strings.")
        labels = np.asarray([r[task.dataset.label_column] for r in rows])
        ids = np.asarray([r["row_id"] for r in rows])
    if (
        labels.shape != (size,)
        or not np.issubdtype(labels.dtype, np.integer)
        or (labels < 0).any()
        or (labels >= task.dataset.num_classes).any()
    ):
        raise ValueError("Split needs integer labels in the declared class range.")
    if ids.shape != (size,) or len(set(map(str, ids))) != size:
        raise ValueError("Each split needs unique source row IDs.")
    return rows, labels.astype(np.int64), ids


def prepare_task(task: TaskSpec, directory: Path) -> dict:
    if task.adapter == "mlp_classification":
        return mlp.prepare_task(task, directory)
    import torch

    if task.protocol_status != "ready":
        raise ValueError("Research protocol must be explicitly ready before preparation.")
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["task_hash"] != digest(task):
            raise ValueError("Prepared task protocol changed.")
        for name, expected in manifest["artifact_hashes"].items():
            if mlp.file_hash(directory / name) != expected:
                raise ValueError(f"Prepared artifact changed: {name}")
        if task.adapter == "text_classification":
            from .text import verify_checkpoint

            verify_checkpoint(task)
        return manifest
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Refusing to replace incomplete preparation.")
    verify_split_manifest(task)
    validate_recipe(task, task.initial_recipe)
    train = load_split(task, "train")
    val = load_split(task, "validation")
    if set(map(str, train[2])) & set(map(str, val[2])):
        raise ValueError("Train and validation source IDs overlap.")
    if task.adapter == "text_classification":
        from .text import verify_checkpoint

        verify_checkpoint(task)
    directory.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        shutil.copyfile(
            split_path(task, split),
            directory
            / f"{split}.{'npz' if task.adapter == 'cifar100_classification' else 'jsonl'}",
        )
    if task.dataset.split_manifest:
        if mlp.file_hash(Path(task.dataset.split_manifest)) != task.dataset.split_manifest_sha256:
            raise ValueError("Split manifest hash mismatch.")
        shutil.copyfile(task.dataset.split_manifest, directory / "split_manifest.json")
    torch.manual_seed(task.seed)
    model = build_model(task, task.initial_recipe, pretrained=task.adapter == "text_classification")
    torch.save(model.state_dict(), directory / "initial_weights.pt")
    mlp.write_json(directory / "task.json", task.model_dump(mode="json"))
    manifest = {
        "task_hash": digest(task),
        "dataset_hash": digest(task.dataset),
        "starting_model_hash": mlp.state_hash(model.state_dict()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "test_access": "not opened; supervisor-only scoring",
        "initialization": "pinned pretrained BERT plus seeded head"
        if task.text_model
        else "seeded random weights",
        "artifact_hashes": {p.name: mlp.file_hash(p) for p in directory.iterdir() if p.is_file()},
    }
    mlp.write_json(manifest_path, manifest)
    for path in directory.iterdir():
        if path.is_file():
            path.chmod(0o444)
    return manifest


def run_trial(task, recipe, prepared_dir, trial_dir, timeout):
    # Shared subprocess launcher dispatches the correct trusted adapter.
    return mlp.run_trial(task, recipe, prepared_dir, trial_dir, timeout)


def score_test(task, prepared_dir, trial_dir, output_path):
    if task.adapter == "mlp_classification":
        return mlp.score_test(task, prepared_dir, trial_dir, output_path)
    from .training import score_multimodal_test

    return score_multimodal_test(task, prepared_dir, trial_dir, output_path)
