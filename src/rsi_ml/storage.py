"""Local, append-only traces and atomic checkpoints. Credentials never belong here."""

from __future__ import annotations

import json
import os
import platform
import ssl
import subprocess
import sys
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .schemas import digest


def now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_ca_bundle() -> None:
    """Use certifi's roots only when the system trust store is empty.

    python.org builds of Python on macOS ship with none, which breaks TLS verification
    for downloads and API calls. A configured or populated system store always wins.
    """
    if (
        os.environ.get("SSL_CERT_FILE")
        or ssl.create_default_context().cert_store_stats()["x509_ca"]
    ):
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def environment() -> dict:
    packages = {}
    for package in (
        "rsi-ml",
        "torch",
        "numpy",
        "langgraph",
        "pydantic",
        "dspy",
        "transformers",
        "tokenizers",
        "safetensors",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    root = Path(__file__).parent
    sources = {
        str(p.relative_to(root)): __import__("hashlib").sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    knowledge = root / "knowledge.json"
    if knowledge.exists():
        sources["knowledge.json"] = __import__("hashlib").sha256(knowledge.read_bytes()).hexdigest()
    cpu = platform.processor()
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        cpu = result.stdout.strip() or cpu
    hardware = {"cpu": cpu, "logical_cpus": os.cpu_count(), "cuda_devices": []}
    if packages["torch"]:
        import torch

        hardware["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(index)
                hardware["cuda_devices"].append(
                    {
                        "name": device.name,
                        "memory": device.total_memory,
                        "capability": [device.major, device.minor],
                        "uuid": str(getattr(device, "uuid", "unknown")),
                    }
                )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": hardware,
        "packages": packages,
        "source_hashes": sources,
    }


def comparable_environment(env: dict | None = None) -> dict:
    """Identity for matching runs: which GPU model, not which physical device.

    Parallel jobs are pinned to different GPUs of the same model; their results are
    still matched. Physical UUIDs remain in each run's full `environment` record.
    """
    env = deepcopy(env if env is not None else environment())
    devices = env["hardware"].pop("cuda_devices", [])
    env["hardware"]["cuda_models"] = sorted({f"{d['name']}|{d['memory']}" for d in devices})
    return env


def training_fingerprint(env: dict | None = None) -> str:
    """What can change the training computation; unaffected by agent-code edits."""
    env = comparable_environment(env)
    return digest(
        {
            "sources": {
                name: value
                for name, value in env["source_hashes"].items()
                if name.startswith("tasks/") or name == "worker.py"
            },
            "packages": {
                name: env["packages"].get(name)
                for name in ("torch", "numpy", "transformers", "tokenizers", "safetensors")
            },
            "python": env["python"].split()[0],
            "hardware": env["hardware"],
        }
    )


def read_events(path: Path) -> list[dict]:
    """Parse an event log, skipping lines a crash cut off. state.json is the record of truth."""
    events = []
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def event(self, kind: str, payload: Any) -> None:
        record = {"timestamp": now(), "event": kind, "payload": payload}
        with (self.root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def begin(self, node: str) -> str:
        token = uuid.uuid4().hex
        write_json(
            self.root / "inflight.json",
            {"operation": token, "node": node, "pid": os.getpid(), "timestamp": now()},
        )
        self.event("node_started", {"node": node, "operation": token})
        return token

    def commit(self, node: str, token: str, state: dict) -> None:
        state["completed_operation"] = token
        write_json(self.root / "state.json", state)
        self.event("node_completed", {"node": node, "operation": token})
        (self.root / "inflight.json").unlink(missing_ok=True)

    def resumable_state(self, *, recover: bool = False) -> dict:
        state = read_json(self.root / "state.json")
        pending = self.root / "inflight.json"
        if pending.exists():
            operation = read_json(pending)
            if state.get("completed_operation") == operation["operation"]:
                pending.unlink()
            elif recover:
                self._recover(state, operation)
                pending.unlink()
            else:
                raise RuntimeError(
                    f"Interrupted {operation['node']} operation detected. No automatic rerun: "
                    "inspect inflight.json and the process/artifacts first, then resume with "
                    "--recover-interrupted. Completed trials are preserved."
                )
        return state

    def _recover(self, state: dict, operation: dict) -> None:
        """Rerun an interrupted node from the last committed state, explicitly and on record.

        A partial trial is kept aside, never deleted. Its elapsed time and any API call
        lost with the interrupted node are not charged to the agent's budget; the event
        records that they happened.
        """
        # A line cut off by the crash must not swallow the next event.
        log = self.root / "events.jsonl"
        if log.exists() and log.read_bytes()[-1:] not in (b"", b"\n"):
            with log.open("ab") as handle:
                handle.write(b"\n")
        moved = None
        if operation["node"] == "execute":
            trial = self.root / "trials" / f"trial-{len(state['experiments']):04d}"
            if trial.exists():
                moved = self.root / "interrupted" / f"{trial.name}-{operation['operation'][:8]}"
                moved.parent.mkdir(parents=True, exist_ok=True)
                trial.replace(moved)
        self.event(
            "interrupted_operation_recovered",
            {
                "node": operation["node"],
                "operation": operation["operation"],
                "started": operation.get("timestamp"),
                "partial_trial_moved_to": str(moved) if moved else None,
                "not_charged": "elapsed time and API usage of the interrupted operation",
            },
        )


class RunLock:
    """Fail closed on simultaneous writers; a stale lock needs explicit human inspection."""

    def __init__(self, root: Path, *, recover: bool = False):
        self.path = root / ".run.lock"
        self.recover = recover

    def __enter__(self):
        if self.recover and self.path.exists():
            holder = json.loads(self.path.read_text()).get("pid")
            try:
                os.kill(holder, 0)
            except ProcessLookupError:
                self.path.unlink()  # The holder is gone: the lock is stale.
            except PermissionError:
                pass  # A live process owned by someone else: leave the lock in place.
            else:
                raise RuntimeError(f"Run is still held by live process {holder}.")
        try:
            with self.path.open("x") as handle:
                json.dump({"pid": os.getpid(), "timestamp": now()}, handle)
        except FileExistsError as exc:
            raise RuntimeError(
                "Run is locked. Inspect .run.lock before attempting recovery."
            ) from exc
        return self

    def __exit__(self, *args):
        self.path.unlink(missing_ok=True)


def execution_identity() -> str:
    return digest(comparable_environment())
