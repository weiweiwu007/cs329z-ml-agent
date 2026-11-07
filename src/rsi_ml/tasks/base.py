"""Adapter contract: trusted task tools own data, training, and grading."""

from pathlib import Path
from typing import Any, Protocol

from rsi_ml.schemas import Recipe, TaskSpec, TrialResult

INFRASTRUCTURE_EXIT_CODE = 75  # EX_TEMPFAIL


class InfrastructureFault(RuntimeError):
    """The machine failed, not the experiment: a lost GPU, a full disk, a killed worker.

    Deliberately never recorded as an attempt. Counting it would charge an arm for a
    hardware fault and mark the run finished. The run stops at a resumable boundary.
    """


def require_device(task: TaskSpec) -> None:
    """Call before creating a run: its manifest would otherwise record a GPU-less identity."""
    if task.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise InfrastructureFault("CUDA requested but unavailable; no fallback.")


class TaskAdapter(Protocol):
    def prepare_task(self, task: TaskSpec, directory: Path) -> dict[str, Any]: ...

    def run_trial(
        self, task: TaskSpec, recipe: Recipe, prepared_dir: Path, trial_dir: Path, timeout: float
    ) -> TrialResult: ...
