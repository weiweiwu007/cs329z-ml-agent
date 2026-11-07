"""Strict contracts shared by agents, execution tools, and independent graders."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


def digest(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class Recipe(StrictModel):
    hidden_sizes: list[int] = Field(max_length=16)
    activation: Literal["relu", "tanh", "gelu"]
    normalization: Literal["none", "layer_norm"]
    dropout: float = Field(ge=0, lt=1)
    optimizer: Literal["adamw", "sgd"]
    learning_rate: float = Field(gt=0, le=1)
    weight_decay: float = Field(ge=0, le=1)
    batch_size: int = Field(ge=1, le=65536)
    epochs: int = Field(ge=1, le=100000)
    schedule: Literal["constant", "cosine", "wsd"]
    warmup_fraction: float = Field(ge=0, lt=1)
    decay_fraction: float = Field(gt=0, le=1)
    momentum: float = Field(default=0, ge=0, lt=1)
    label_smoothing: float = Field(default=0, ge=0, lt=1)
    class_weighting: Literal["none", "balanced"] = "none"
    sampling: Literal["shuffle", "balanced"] = "shuffle"
    feature_preprocessing: Literal["standard", "none", "signed_log_standard"] | None = None
    cnn_channels: list[int] | None = None
    cnn_blocks: list[int] | None = None
    augmentation: Literal["none", "crop_flip", "crop_flip_color"] = "none"
    pooling: Literal["cls", "mean"] = "cls"
    freeze_encoder_layers: int = Field(default=0, ge=0)
    freeze_embeddings: bool = False
    max_length: int | None = Field(default=None, ge=4, le=512)

    @model_validator(mode="after")
    def valid_shape(self):
        if any(w < 1 or w > 8192 for w in self.hidden_sizes):
            raise ValueError("Each hidden width must be in [1, 8192].")
        if self.schedule == "wsd" and self.warmup_fraction + self.decay_fraction > 1:
            raise ValueError("WSD warmup + decay fractions must not exceed 1.")
        # Six stages is the most a 32x32 image can halve through. Beyond that, the parameter
        # ceiling and time limits are what bound a model, not this schema.
        if self.cnn_channels is not None and (
            not 1 <= len(self.cnn_channels) <= 6
            or any(w < 1 or w > 2048 for w in self.cnn_channels)
        ):
            raise ValueError("CNN needs one to six stage widths, each in [1, 2048].")
        if self.cnn_blocks is not None and (
            not 1 <= len(self.cnn_blocks) <= 6 or any(n < 1 or n > 64 for n in self.cnn_blocks)
        ):
            raise ValueError("CNN needs one to six block counts, each in [1, 64].")
        if (
            self.cnn_channels is not None
            and self.cnn_blocks is not None
            and len(self.cnn_channels) != len(self.cnn_blocks)
        ):
            raise ValueError("cnn_channels and cnn_blocks must have the same length.")
        return self


class DatasetSpec(StrictModel):
    kind: Literal["synthetic_fixture", "local_csv", "cifar100_npz", "text_jsonl"]
    identity: str
    revision: str
    input_dim: int = Field(ge=1)
    num_classes: int = Field(ge=2)
    standardize: bool
    seed: int = Field(ge=0)
    train_rows: int = Field(ge=2)
    validation_rows: int = Field(ge=2)
    test_rows: int = Field(ge=2)
    train_csv: str | None = None
    validation_csv: str | None = None
    test_csv: str | None = None
    label_column: str = "label"
    train_sha256: str | None = None
    validation_sha256: str | None = None
    test_sha256: str | None = None
    train_file: str | None = None
    validation_file: str | None = None
    test_file: str | None = None
    text_columns: list[str] | None = None
    image_mean: list[float] | None = None
    image_std: list[float] | None = None
    split_manifest: str | None = None
    split_manifest_sha256: str | None = None
    # Human-readable label names; error reports are not interpretable without them.
    class_names: list[str] | None = None

    @model_validator(mode="after")
    def class_names_match(self):
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError("class_names must list exactly num_classes names.")
        return self


class TextModelSpec(StrictModel):
    model_id: str
    revision: str
    local_path: str
    file_sha256: dict[str, str]


class Limits(StrictModel):
    max_trials: int = Field(ge=1, le=10000)
    max_parameters: int = Field(ge=1)
    max_epochs: int = Field(ge=1)
    per_trial_seconds: float = Field(gt=0)
    total_compute_seconds: float = Field(gt=0)
    max_api_calls: int = Field(ge=0)
    max_context_chars: int = Field(ge=4096)
    max_output_tokens: int = Field(ge=256)
    max_diagnostic_requests: int = Field(default=0, ge=0)


class ArtifactPolicy(StrictModel):
    retain_epoch_checkpoints: bool
    retain_epoch_logits: bool


class TaskSpec(StrictModel):
    task_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")
    adapter: Literal["mlp_classification", "cifar100_classification", "text_classification"]
    purpose: Literal["demo", "development", "evaluation"]
    protocol_status: Literal["demo", "draft", "ready"]
    dataset: DatasetSpec
    initial_recipe: Recipe
    metric: Literal["accuracy", "macro_f1", "f1"]
    target: float = Field(gt=0, le=1)
    seed: int = Field(ge=0)
    device: Literal["cpu", "cuda"]
    limits: Limits
    # "constraints": agents choose any value the recipe schema allows, within `limits`.
    # "menu": only values listed in search_space, for software demos and tests.
    recipe_policy: Literal["menu", "constraints"] = "menu"
    search_space: dict[str, list[Any]] = Field(default_factory=dict)
    # Sampling ranges for the random-search reference arm only. Never shown to agents.
    random_search: dict[str, Any] = Field(default_factory=dict)
    artifact_policy: ArtifactPolicy | None = None
    text_model: TextModelSpec | None = None
    # False keeps spending the budget after the target, so best-score-versus-compute
    # curves stay complete and time-to-target is still derivable from the trajectory.
    stop_on_target: bool = True

    @model_validator(mode="after")
    def consistent_mode(self):
        if self.dataset.kind == "synthetic_fixture" and self.purpose != "demo":
            raise ValueError("Synthetic fixtures are software demos, not research tasks.")
        if self.protocol_status == "demo" and self.purpose != "demo":
            raise ValueError("Only demo tasks may use demo protocol status.")
        allowed = {
            "mlp_classification": {"synthetic_fixture", "local_csv"},
            "cifar100_classification": {"cifar100_npz"},
            "text_classification": {"text_jsonl"},
        }
        if self.dataset.kind not in allowed[self.adapter]:
            raise ValueError("Dataset kind does not match the task adapter.")
        if self.adapter == "mlp_classification" and not self.initial_recipe.hidden_sizes:
            raise ValueError("MLP needs at least one hidden layer.")
        if self.metric == "f1" and self.dataset.num_classes != 2:
            raise ValueError("f1 is binary positive-class (label 1) F1; use macro_f1 otherwise.")
        if self.recipe_policy == "constraints" and self.search_space:
            raise ValueError("A constraints task cannot also declare a search_space menu.")
        for key, values in self.search_space.items():
            if key not in Recipe.model_fields or not values:
                raise ValueError(
                    "Search space needs valid recipe field names and nonempty choices."
                )
            if getattr(self.initial_recipe, key) not in values:
                raise ValueError(f"Search space must include initial {key}.")
            for value in values:
                Recipe.model_validate({**self.initial_recipe.model_dump(), key: value})
        return self


class Policy(StrictModel):
    policy_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")
    version: int = Field(ge=0)
    parent_hash: str | None = None
    status: Literal["baseline", "candidate", "accepted"]
    planner_instructions: str = Field(min_length=10, max_length=8000)
    developer_instructions: str = Field(min_length=10, max_length=4000)
    reviewer_instructions: str = Field(min_length=10, max_length=4000)
    evidence_run_ids: list[str] = Field(default_factory=list)
    rationale: str = "Initial workflow."


class Plan(StrictModel):
    observation: str
    hypothesis: str
    predicted_effect: str
    experiment: str
    recipe: Recipe
    evidence_ids: list[str] = Field(default_factory=list)
    stop: bool = False
    diagnostic_trial_id: str | None = None


class Development(StrictModel):
    recipe: Recipe
    implementation_notes: str


class Review(StrictModel):
    interpretation: str
    prediction_supported: Literal["yes", "no", "uncertain"]
    next_step: str
    diagnostic_trial_id: str | None = None


class PolicyChange(StrictModel):
    planner_instructions: str = Field(min_length=10, max_length=8000)
    rationale: str = Field(min_length=10, max_length=4000)
    evidence_run_ids: list[str] = Field(min_length=1)


class TrialResult(StrictModel):
    status: Literal["completed", "failed", "timeout", "rejected"]
    metrics: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, float] = Field(default_factory=dict)
    compute_seconds: float = Field(ge=0)
    training_seconds: float = Field(default=0, ge=0)
    # Process startup, recorded but not charged to compute_seconds.
    startup_seconds: float = Field(default=0, ge=0)
    parameter_count: int = Field(default=0, ge=0)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    # Ordinary training/validation curves, shown to every A-D arm. Evenly spaced
    # epochs ending at the final one; the complete curve stays in learning_curve.json.
    curve_summary: list[dict[str, float]] = Field(default_factory=list)


class Experiment(StrictModel):
    trial_id: str
    index: int
    recipe: Recipe | None = None
    plan: Plan | None = None
    result: TrialResult
    review: Review | None = None


class Usage(StrictModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    tokens_known: bool = True
    # Provider-side failures retried automatically; they do not consume `calls`.
    transient_retries: int = 0


class RunSummary(StrictModel):
    run_id: str
    task_id: str
    purpose: str
    backend: str
    policy_hash: str
    task_hash: str
    starting_model_hash: str
    dataset_hash: str
    seed: int
    metric: str
    target: float
    device: str
    reached_target: bool
    trials_to_target: int | None
    compute_to_target: float | None
    best_score: float | None
    total_trials: int
    total_compute_seconds: float
    stop_reason: str
    usage: Usage
    trial_ids: list[str]
    backend_fingerprint: str = ""
    execution_fingerprint: str = ""
    memory_fingerprint: str = ""
    variant: Literal["A", "B", "C", "D"] = "C"
    diagnostic_seconds: float = 0
    diagnostic_requests: int = 0
    preparation_seconds: float = 0


class WorkflowVariant(StrictModel):
    name: Literal["A", "B", "C", "D"]

    @property
    def detailed_error_reports(self) -> bool:
        return self.name in {"B", "D"}

    @property
    def automatic_diagnostics(self) -> bool:
        return self.detailed_error_reports

    @property
    def separate_reviewer(self) -> bool:
        return self.name in {"C", "D"}

    def settings(self) -> dict[str, bool]:
        return {
            "detailed_error_reports": self.detailed_error_reports,
            "automatic_diagnostics": self.automatic_diagnostics,
            "separate_reviewer": self.separate_reviewer,
        }
