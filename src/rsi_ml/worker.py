"""Trusted worker entry point; only local data/checkpoints, no agent or network calls."""

from __future__ import annotations

import argparse
import shutil
import traceback
from pathlib import Path

from rsi_ml.schemas import Recipe, TaskSpec
from rsi_ml.tasks.base import INFRASTRUCTURE_EXIT_CODE, InfrastructureFault
from rsi_ml.tasks.training import execute

_MACHINE_FAULTS = (
    "cuda error",
    "no cuda gpus",
    "nvidia driver",
    "device not available",
    "cudnn_status",
    "cublas_status",
    # A full disk during torch.save surfaces as a RuntimeError, not an OSError.
    "pytorchstreamwriter",
    "file write failed",
    "unexpected pos",
    "no space left",
)
_RECIPE_MEMORY_FAULTS = ("out of memory", "alloc_failed")


def is_infrastructure_fault(error: BaseException) -> bool:
    """Separate machine failures from failures an experiment's recipe caused.

    GPU out-of-memory is deliberately excluded: it depends on the model and batch size
    chosen, on hardware every arm shares, so it counts as that attempt's failure.
    """
    if isinstance(error, (InfrastructureFault, MemoryError, OSError)):
        return True
    if not isinstance(error, RuntimeError) or type(error).__name__ == "OutOfMemoryError":
        return False
    text = str(error).lower()
    if any(fault in text for fault in _RECIPE_MEMORY_FAULTS):
        return False
    return any(fault in text for fault in _MACHINE_FAULTS)


def _retain_epoch(source: Path, destination: Path) -> None:
    """Hard-link an immutable epoch artifact, with a portable copy fallback."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Refusing to overwrite a retained epoch artifact.")
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copyfile(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    for arg in ("task", "recipe", "prepared", "out"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    args = parser.parse_args()
    try:
        execute(
            TaskSpec.model_validate_json(args.task.read_text()),
            Recipe.model_validate_json(args.recipe.read_text()),
            args.prepared,
            args.out,
        )
    except Exception as error:
        if not is_infrastructure_fault(error):
            raise
        traceback.print_exc()
        raise SystemExit(INFRASTRUCTURE_EXIT_CODE) from None


if __name__ == "__main__":
    main()
