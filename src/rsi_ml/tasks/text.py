"""Offline-only BERT encoder, configurable head, pooling, and freezing."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from rsi_ml.schemas import Recipe, TaskSpec

from .cnn import head
from .mlp import file_hash


def verify_checkpoint(task: TaskSpec) -> Path:
    spec = task.text_model
    if spec is None or not spec.revision.strip() or not spec.file_sha256:
        raise ValueError("Text task needs an explicit local checkpoint revision and file hashes.")
    root = Path(spec.local_path).resolve()
    actual = {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}
    if set(actual) != set(spec.file_sha256):
        raise ValueError("Checkpoint directory must contain exactly the pinned files.")
    for name, path in actual.items():
        if not path.resolve().is_relative_to(root) or file_hash(path) != spec.file_sha256[name]:
            raise ValueError(f"Checkpoint file failed integrity check: {name}")
    if "config.json" not in actual or not any(name.endswith(".safetensors") for name in actual):
        raise ValueError("Use a local BERT config and safetensors checkpoint.")
    return root


def configuration(task: TaskSpec, recipe: Recipe):
    try:
        from transformers import BertConfig
    except ImportError:
        raise ValueError("Text tasks require the optional [text] dependencies.") from None
    if task.text_model is None:
        raise ValueError("Missing text_model.")
    data = json.loads((Path(task.text_model.local_path) / "config.json").read_text())
    if data.get("model_type") != "bert":
        raise ValueError("Only the declared BERT model family is supported.")
    config = BertConfig.from_dict(data)
    if recipe.max_length is None or recipe.max_length > config.max_position_embeddings:
        raise ValueError("Explicit max_length must fit the encoder's position embeddings.")
    if recipe.freeze_encoder_layers > config.num_hidden_layers:
        raise ValueError("Cannot freeze more encoder layers than exist.")
    config.hidden_dropout_prob = recipe.dropout
    config.attention_probs_dropout_prob = recipe.dropout
    config._attn_implementation = "eager"
    return config


class BertClassifier(nn.Module):
    def __init__(self, task: TaskSpec, recipe: Recipe, *, pretrained: bool):
        super().__init__()
        from transformers import BertModel

        config = configuration(task, recipe)
        if pretrained:
            self.encoder, loading = BertModel.from_pretrained(
                task.text_model.local_path,
                config=config,
                local_files_only=True,
                use_safetensors=True,
                add_pooling_layer=False,
                output_loading_info=True,
            )
            if loading.get("missing_keys") or loading.get("mismatched_keys"):
                raise ValueError("Pretrained encoder has missing or mismatched weights.")
        else:
            self.encoder = BertModel(config, add_pooling_layer=False)
        self.pooling = recipe.pooling
        self.classifier = head(config.hidden_size, task.dataset.num_classes, recipe)
        if recipe.freeze_embeddings:
            self.encoder.embeddings.requires_grad_(False)
        for layer in self.encoder.encoder.layer[: recipe.freeze_encoder_layers]:
            layer.requires_grad_(False)

    def forward(self, inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(pooled)


def tokenize(task: TaskSpec, recipe: Recipe, rows: list[dict]) -> dict[str, torch.Tensor]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        task.text_model.local_path, local_files_only=True, trust_remote_code=False
    )
    columns = task.dataset.text_columns
    if columns is None or len(columns) not in (1, 2):
        raise ValueError("Declare one SST-2 or two MRPC text columns.")
    arguments = [[row[column] for row in rows] for column in columns]
    return dict(
        tokenizer(
            *arguments,
            padding="max_length",
            truncation="longest_first",
            max_length=recipe.max_length,
            return_tensors="pt",
        )
    )
