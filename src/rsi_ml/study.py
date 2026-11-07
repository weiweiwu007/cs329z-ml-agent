"""Build and run the staged A-D study across GPUs.

Stage 1 (`baselines`) trains each task/seed's starting recipe once, with no API, which
also measures throughput. Stage 2 (`agents`) runs A, B, C, D, and random search, each
adopting that baseline. One job per GPU keeps wall-clock cost comparable. Everything is
resumable, and every stage needs an explicit --execute.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .agents import PROVIDER_UNAVAILABLE_EXIT_CODE
from .schemas import TaskSpec, digest
from .storage import comparable_environment, now, read_json, write_json
from .tasks.mlp import file_hash
from .tasks.registry import parameter_count

STAGES = ("baselines", "agents")
LATIN_SQUARE = (
    ("A", "B", "C", "D"),
    ("B", "C", "D", "A"),
    ("C", "D", "A", "B"),
    ("D", "A", "B", "C"),
)
# A slot whose jobs keep dying within seconds is broken (a lost GPU, for example). It stops
# taking work, so it cannot drain the shared queue while healthy slots are busy.
MAX_FAST_FAILURES = 2
FAST_FAILURE_SECONDS = 120
UNREACHABLE_TARGET = 1.0  # Stage 1 never stops on a score; real targets exist only in stage 2.


def _protocol(path: Path) -> dict:
    protocol = read_json(path)
    if protocol.get("status") not in {"proposed", "approved"}:
        raise ValueError("Protocol status must be 'proposed' or 'approved'.")
    return protocol


def _dataset(template: dict, selected: Path) -> dict:
    manifest_path = (selected / "manifest.json").resolve()
    manifest = read_json(manifest_path)
    field = "csv" if template["kind"] == "local_csv" else "file"
    dataset = {
        **template,
        "revision": manifest["revision"],
        "seed": manifest["selection_seed"],
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": file_hash(manifest_path),
        "class_names": manifest["class_names"],
    }
    for split in ("train", "validation", "test"):
        entry = manifest["splits"][split]
        dataset[f"{split}_{field}"] = entry["path"]
        dataset[f"{split}_sha256"] = entry["sha256"]
        dataset[f"{split}_rows"] = entry["rows"]
    if template["kind"] == "cifar100_npz":
        dataset["image_mean"], dataset["image_std"] = manifest["image_mean"], manifest["image_std"]
    return dataset


def _task(
    protocol: dict,
    base: Path,
    task_id: str,
    data_root: Path,
    seed: int,
    target: float,
    per_trial_seconds: float,
    total_compute_seconds: float,
) -> TaskSpec:
    spec = protocol["tasks"][task_id]
    template = read_json(base / spec["template"])
    recipe = {**template["initial_recipe"], **spec["initial_recipe"]}
    limits = {
        **protocol["common_limits"],
        "max_parameters": spec["max_parameters"],
        "max_epochs": spec.get("max_epochs", recipe["epochs"]),
        "per_trial_seconds": per_trial_seconds,
        "total_compute_seconds": total_compute_seconds,
    }
    data = {
        **template,
        "protocol_status": "ready",
        "device": spec["device"],
        "seed": seed,
        "stop_on_target": protocol["stop_on_target"],
        "artifact_policy": protocol["artifact_policy"],
        "initial_recipe": recipe,
        "recipe_policy": spec.get("recipe_policy", "menu"),
        "search_space": spec.get("search_space", {}),
        "random_search": spec.get("random_search", {}),
        "limits": limits,
        "target": target,
        "dataset": _dataset(template["dataset"], data_root / "selected" / task_id),
    }
    if "text_model" in template:
        data["text_model"] = read_json(data_root / "models" / "bert-tiny.pin.json")
    return TaskSpec.model_validate(data)


def unresolved_agent_settings(protocol: dict) -> list[str]:
    stage = protocol["agent_stage"]
    missing = [] if stage.get("model") else ["agent_stage.model"]
    for group in ("targets", "per_trial_seconds", "total_compute_seconds"):
        missing += [f"agent_stage.{group}.{k}" for k, v in stage[group].items() if v is None]
    return missing


def build_study(protocol_path: Path, data_root: Path, output: Path) -> dict:
    """Write runnable task files. Refuses an unapproved protocol."""
    from .workflow import preflight

    protocol_path, data_root = Path(protocol_path).resolve(), Path(data_root).resolve()
    output = Path(output).resolve()
    protocol = _protocol(protocol_path)
    if protocol["status"] != "approved":
        raise ValueError(
            "Protocol is 'proposed'. Review docs/DECISION_RECORD.md and set status to 'approved'."
        )
    base = protocol_path.parent
    missing = unresolved_agent_settings(protocol)
    stages = ["baselines"] + ([] if missing else ["agents"])
    tasks = {}
    written = []
    for task_id, spec in protocol["tasks"].items():
        orders = {}
        for index, seed in enumerate(spec["seeds"]):
            orders[str(seed)] = list(LATIN_SQUARE[index % 4])
            ceiling = protocol["baseline_stage"]["per_trial_seconds"][task_id]
            first = _task(
                protocol, base, task_id, data_root, seed, UNREACHABLE_TARGET, ceiling, ceiling
            )
            written.append((output / "tasks" / "baselines" / task_id / f"seed-{seed}.json", first))
            if "agents" in stages:
                stage = protocol["agent_stage"]
                task = _task(
                    protocol,
                    base,
                    task_id,
                    data_root,
                    seed,
                    stage["targets"][task_id],
                    stage["per_trial_seconds"][task_id],
                    stage["total_compute_seconds"][task_id],
                )
                problems = preflight(task, "dspy")
                if problems:
                    raise ValueError(f"{task_id} seed {seed}: " + "; ".join(problems))
                written.append((output / "tasks" / "agents" / task_id / f"seed-{seed}.json", task))
        tasks[task_id] = {
            "metric": first.metric,
            "device": spec["device"],
            "seeds": spec["seeds"],
            "orders": orders,
            "initial_parameters": parameter_count(first, first.initial_recipe),
            "class_names_loaded": bool(first.dataset.class_names),
        }
    for path, task in written:
        write_json(path, task)
    policy_source = (base / protocol["policy"]).resolve()
    write_json(output / "policy.json", read_json(policy_source))
    write_json(output / "protocol.json", protocol)
    study = {
        "study_id": protocol["study_id"],
        "built": now(),
        "protocol_sha256": file_hash(protocol_path),
        "policy_sha256": file_hash(policy_source),
        "stages": stages,
        "unresolved_for_agent_stage": missing,
        "arms": protocol["arms"],
        "agent_stage": {
            key: protocol["agent_stage"][key]
            for key in ("model", "reasoning_effort", "api_base", "api_timeout_seconds")
        },
        "max_trials": protocol["common_limits"]["max_trials"],
        # Stored explicitly because JSON records sort their keys: this order is the queue.
        "task_order": protocol.get("task_order") or list(protocol["tasks"]),
        "tasks": tasks,
    }
    write_json(output / "study.json", study)
    return study


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    task_id: str
    seed: int
    arm: str
    device: str
    task_file: Path
    out: Path


def _load(study_dir: Path) -> dict:
    return read_json(Path(study_dir) / "study.json")


def run_dir(study_dir: Path, task_id: str, seed: int, arm: str) -> Path:
    return Path(study_dir).resolve() / "runs" / task_id / f"seed-{seed}" / arm


def stage_jobs(
    study_dir: Path,
    stage: str,
    *,
    tasks: set[str] | None = None,
    seeds: set[int] | None = None,
    arms: set[str] | None = None,
) -> list[Job]:
    """Queue order is protocol order: cheap tasks first, so problems surface early."""
    study_dir = Path(study_dir).resolve()
    study = _load(study_dir)
    if stage not in study["stages"]:
        raise ValueError(
            f"Stage {stage} is not built. Unresolved: {study['unresolved_for_agent_stage']}"
        )
    jobs = []
    for task_id in study["task_order"]:
        entry = study["tasks"][task_id]
        if tasks and task_id not in tasks:
            continue
        for seed in entry["seeds"]:
            if seeds and seed not in seeds:
                continue
            names = ["baseline"]
            if stage == "agents":
                # The ordering balances A-D runs; it does not authorize extra arms.
                # A single-design mission must only queue the declared design.
                names = [a for a in entry["orders"][str(seed)] if a in study["arms"]]
                if "random" in study["arms"]:
                    names.append("random")
            for arm in names:
                if arms and stage == "agents" and arm not in arms:
                    continue
                jobs.append(
                    Job(
                        f"{task_id}.seed-{seed}.{arm}",
                        stage,
                        task_id,
                        seed,
                        arm,
                        entry["device"],
                        study_dir / "tasks" / stage / task_id / f"seed-{seed}.json",
                        run_dir(study_dir, task_id, seed, arm),
                    )
                )
    return jobs


def job_action(job: Job) -> str:
    """`done`, `start`, `resume`, or `restart` (partial work with nothing resumable)."""
    if job.stage == "baselines":
        receipt = job.out / "baseline.json"
        if receipt.exists() and read_json(receipt)["result"]["status"] == "completed":
            return "done"
        return "restart" if job.out.exists() else "start"
    if (job.out / "summary.json").exists():
        return "done"
    if (job.out / "state.json").exists() and (job.out / "manifest.json").exists():
        return "resume"
    return "restart" if job.out.exists() else "start"


def job_command(job: Job, action: str, study_dir: Path, *, recover: bool) -> list[str]:
    python = [sys.executable, "-m", "rsi_ml"]
    if job.stage == "baselines":
        return [
            *python,
            "baseline-job",
            "--task",
            str(job.task_file),
            "--out",
            str(job.out),
            "--execute",
        ]
    if action == "resume":
        return [
            *python,
            "resume",
            str(job.out),
            "--execute",
            *(["--recover-interrupted"] if recover else []),
        ]
    baseline = run_dir(study_dir, job.task_id, job.seed, "baseline")
    if not (baseline / "baseline.json").exists():
        raise ValueError(f"{job.job_id} needs its stage-1 baseline first: {baseline}")
    # The random arm is variant A's workflow (no reviewer, no reports) with random decisions.
    backend, variant = ("random", "A") if job.arm == "random" else ("dspy", job.arm)
    return [
        *python,
        "run",
        "--task",
        str(job.task_file),
        "--policy",
        str(Path(study_dir).resolve() / "policy.json"),
        "--backend",
        backend,
        "--variant",
        variant,
        "--baseline-dir",
        str(baseline),
        "--out",
        str(job.out),
        "--execute",
    ]


def load_api_key(path: Path) -> dict[str, str]:
    """Read only the API key from an env file; its value is never logged or returned elsewhere."""
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    key = values.get("RSI_API_KEY") or values.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(f"No RSI_API_KEY or OPENROUTER_API_KEY in {path}.")
    return {"RSI_API_KEY": key}


def agent_environment(study: dict, env_file: Path | None) -> dict[str, str]:
    stage = study["agent_stage"]
    env = {"RSI_MODEL": stage["model"], "RSI_API_TIMEOUT": str(stage["api_timeout_seconds"])}
    if stage.get("api_base"):
        env["RSI_API_BASE"] = stage["api_base"]
    if stage.get("reasoning_effort"):
        env["RSI_REASONING_EFFORT"] = stage["reasoning_effort"]
    if env_file:
        env.update(load_api_key(env_file))
    elif not os.environ.get("RSI_API_KEY"):
        raise ValueError("Agent stage needs --env-file or RSI_API_KEY in the environment.")
    return env


class _StudyLock:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        if self.path.exists():
            holder = read_json(self.path)["pid"]
            try:
                os.kill(holder, 0)
            except ProcessLookupError:
                self.path.unlink()
            else:
                raise RuntimeError(f"Another runner holds this stage: pid {holder}.")
        write_json(self.path, {"pid": os.getpid(), "started": now()})
        return self

    def __exit__(self, *args):
        self.path.unlink(missing_ok=True)


def run_stage(
    study_dir: Path,
    stage: str,
    *,
    gpus: list[str],
    cpu_slots: int = 2,
    jobs_per_gpu: int = 1,
    env_file: Path | None = None,
    recover: bool = False,
    tasks: set[str] | None = None,
    seeds: set[int] | None = None,
    arms: set[str] | None = None,
    guard: Callable[[], str | None] | None = None,
) -> dict:
    """Run a stage to completion; a failed job does not stop the others.

    `guard` is checked before each job starts. Once it names a reason, running jobs
    finish and nothing new starts; held jobs simply run on the next invocation.
    """
    study_dir = Path(study_dir).resolve()
    study = _load(study_dir)
    jobs = stage_jobs(study_dir, stage, tasks=tasks, seeds=seeds, arms=arms)
    if any(job.device == "cuda" for job in jobs) and not gpus:
        raise ValueError("This stage has CUDA jobs; pass --gpus.")
    extra = {}
    if stage == "agents" and any(job.arm != "random" for job in jobs):
        extra = agent_environment(study, env_file)
    gpu_queue: queue.Queue = queue.Queue()
    cpu_queue: queue.Queue = queue.Queue()
    status = {}
    lock, children = threading.Lock(), set()
    status_path = study_dir / "status" / f"{stage}.json"
    logs = study_dir / "logs" / stage
    logs.mkdir(parents=True, exist_ok=True)

    def record(job_id, **fields):
        with lock:
            status.setdefault(job_id, {}).update(fields)
            write_json(status_path, {"updated": now(), "stage": stage, "jobs": status})

    # Jobs that ran and failed last time go last, so they cannot starve untouched work.
    earlier = read_json(status_path).get("jobs", {}) if status_path.exists() else {}
    for job in sorted(
        jobs, key=lambda j: earlier.get(j.job_id, {}).get("exit_code") not in (None, 0)
    ):
        action = job_action(job)
        if action == "done":
            record(job.job_id, state="done_before")
        else:
            (gpu_queue if job.device == "cuda" else cpu_queue).put((job, action))
            record(job.job_id, state="queued", action=action)

    def worker(pending: queue.Queue, device: str):
        fast_failures = 0
        while fast_failures < MAX_FAST_FAILURES:
            try:
                job, action = pending.get_nowait()
            except queue.Empty:
                return
            reason = guard() if guard else None
            if reason:
                record(job.job_id, state="held", reason=reason)
                continue
            if action == "restart":
                # Partial work with no resumable state is kept aside, never deleted.
                aside = job.out.with_name(f"{job.out.name}.aborted-{int(time.time())}")
                job.out.replace(aside)
                record(job.job_id, set_aside=str(aside))
            job.out.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            record(job.job_id, state="running", device=device, started=now())
            # PCI order, matching nvidia-smi's indices, whichever GPU CUDA ranks fastest.
            environment = {
                **os.environ,
                **extra,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": device,
            }
            try:
                command = job_command(job, action, study_dir, recover=recover)
                with (logs / f"{job.job_id}.log").open("ab") as log:
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, env=environment
                    )
                    with lock:
                        children.add(process)
                    code = process.wait()
                    with lock:
                        children.discard(process)
            except (OSError, ValueError) as exc:
                code = -1
                record(job.job_id, error=str(exc)[:500])
            seconds = time.monotonic() - started
            if code == PROVIDER_UNAVAILABLE_EXIT_CODE:
                pass  # The provider failed, not this slot.
            elif code and seconds < FAST_FAILURE_SECONDS:
                fast_failures += 1
            else:
                fast_failures = 0
            record(
                job.job_id,
                state="completed" if code == 0 else "failed",
                exit_code=code,
                finished=now(),
                seconds=seconds,
            )

    def stop(signum, frame):
        with lock:
            for process in children:
                process.terminate()
        raise SystemExit(128 + signum)

    threads = [
        threading.Thread(target=worker, args=(gpu_queue, gpu), daemon=True)
        for gpu in gpus
        for _ in range(jobs_per_gpu)
    ] + [
        threading.Thread(target=worker, args=(cpu_queue, ""), daemon=True) for _ in range(cpu_slots)
    ]
    with _StudyLock(study_dir / "status" / f".{stage}.lock"):
        previous = signal.signal(signal.SIGTERM, stop)
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            signal.signal(signal.SIGTERM, previous)
    for pending in (gpu_queue, cpu_queue):
        while not pending.empty():
            job, _ = pending.get_nowait()
            record(job.job_id, state="failed", error="No healthy worker slot remained.")
    failed = sorted(k for k, v in status.items() if v["state"] == "failed")
    return {
        "stage": stage,
        "jobs": len(jobs),
        "completed": sum(v["state"] == "completed" for v in status.values()),
        "already_done": sum(v["state"] == "done_before" for v in status.values()),
        "failed": failed,
        "held": sorted(k for k, v in status.items() if v["state"] == "held"),
        "execution_identity": digest(comparable_environment()),
        "gpus": gpus,
        "jobs_per_gpu": jobs_per_gpu,
    }


def run_baseline_job(task_file: Path, output: Path) -> dict:
    from .baselines import run_baseline
    from .cli import load_task

    receipt = run_baseline(load_task(task_file), output)
    result = receipt["result"]
    return {k: receipt[k] for k in ("task_id", "seed", "metric")} | {
        "status": result["status"],
        "metrics": result["metrics"],
        "compute_seconds": result["compute_seconds"],
        "error": result["error"],
    }


def study_status(study_dir: Path) -> dict:
    """Read-only progress, costs, and a projection from measured baseline times."""
    study_dir = Path(study_dir).resolve()
    study = _load(study_dir)
    rows, remaining_seconds = [], {}
    for task_id in study["task_order"]:
        entry = study["tasks"][task_id]
        durations = []
        for seed in entry["seeds"]:
            receipt_path = run_dir(study_dir, task_id, seed, "baseline") / "baseline.json"
            row = {"task": task_id, "seed": seed, "arm": "baseline", "state": "pending"}
            if receipt_path.exists():
                result = read_json(receipt_path)["result"]
                row.update(
                    state=result["status"],
                    trials=1,
                    best=result["metrics"].get(entry["metric"]),
                    compute_seconds=result["compute_seconds"],
                )
                durations.append(result["compute_seconds"])
            elif run_dir(study_dir, task_id, seed, "baseline").exists():
                row["state"] = "running_or_interrupted"
            rows.append(row)
        mean = sum(durations) / len(durations) if durations else None
        left = 0
        for seed in entry["seeds"]:
            for arm in [*entry["orders"][str(seed)], *[a for a in study["arms"] if a == "random"]]:
                directory = run_dir(study_dir, task_id, seed, arm)
                row = {"task": task_id, "seed": seed, "arm": arm, "state": "pending"}
                trials = 0
                if (directory / "summary.json").exists():
                    summary = read_json(directory / "summary.json")
                    trials = study["max_trials"]
                    row.update(
                        state=summary["stop_reason"],
                        trials=summary["total_trials"],
                        best=summary["best_score"],
                        compute_seconds=summary["total_compute_seconds"],
                        api_calls=summary["usage"]["calls"],
                        api_retries=summary["usage"].get("transient_retries", 0),
                        api_cost_usd=summary["usage"]["cost_usd"],
                    )
                elif (directory / "state.json").exists():
                    state = read_json(directory / "state.json")
                    trials = len(state["experiments"])
                    row.update(
                        state="running_or_interrupted",
                        trials=trials,
                        interrupted=(directory / "inflight.json").exists(),
                        api_calls=state["usage"]["calls"],
                    )
                left += max(0, study["max_trials"] - max(trials, 1))
                rows.append(row)
        remaining_seconds[task_id] = {
            "baseline_seconds_mean": mean,
            "agent_trials_remaining_at_most": left,
            "gpu_or_cpu_hours_remaining_if_like_baseline": mean * left / 3600 if mean else None,
        }
    return {
        "study_id": study["study_id"],
        "stages_built": study["stages"],
        "unresolved_for_agent_stage": study["unresolved_for_agent_stage"],
        "projection": remaining_seconds,
        "runs": rows,
    }


def best_trial(state: dict, metric: str) -> str | None:
    """Selection uses validation scores only; ties go to the earliest trial."""
    best, chosen = None, None
    for record in state["experiments"]:
        result = record["result"]
        if result["status"] == "completed":
            score = result["metrics"][metric]
            if best is None or score > best:
                best, chosen = score, record["trial_id"]
    return chosen


def score_study_test_sets(study_dir: Path) -> dict:
    """Supervisor-only: score each finished run's validation-selected model once on test."""
    from .cli import load_task
    from .tasks.registry import score_test

    study_dir = Path(study_dir).resolve()
    study = _load(study_dir)
    scored, skipped = [], []
    for task_id in study["task_order"]:
        entry = study["tasks"][task_id]
        for seed in entry["seeds"]:
            for arm in [*entry["orders"][str(seed)], *[a for a in study["arms"] if a == "random"]]:
                directory = run_dir(study_dir, task_id, seed, arm)
                output = directory / "final_test.json"
                if not (directory / "summary.json").exists():
                    skipped.append(f"{task_id}.seed-{seed}.{arm}: unfinished")
                    continue
                if output.exists():
                    receipt = read_json(output)
                else:
                    task = load_task(directory / "task.json")
                    trial = best_trial(read_json(directory / "state.json"), task.metric)
                    if trial is None:
                        skipped.append(f"{task_id}.seed-{seed}.{arm}: no completed trial")
                        continue
                    # Outputs with no receipt are an interrupted scoring; keep them aside.
                    partial = output.with_suffix(".outputs.npz")
                    if partial.exists():
                        partial.replace(
                            partial.with_name(f"final_test.partial-{int(time.time())}.npz")
                        )
                    try:
                        receipt = {
                            **score_test(
                                task, directory / "prepared", directory / "trials" / trial, output
                            ),
                            "selected_trial": trial,
                        }
                    except (ValueError, OSError, RuntimeError) as exc:
                        reason = f"{type(exc).__name__}: {exc}"[:300]
                        skipped.append(f"{task_id}.seed-{seed}.{arm}: {reason}")
                        continue
                    write_json(output, receipt)
                metrics = dict(receipt["metrics"])
                # classification_metrics names its loss for validation; this one is test loss.
                metrics["test_loss"] = metrics.pop("validation_loss")
                scored.append(
                    {
                        "task": task_id,
                        "seed": seed,
                        "arm": arm,
                        "trial": receipt.get("selected_trial"),
                        **metrics,
                    }
                )
    write_json(study_dir / "final_test.json", {"scored": scored, "skipped": skipped})
    return {"scored": len(scored), "skipped": skipped}
