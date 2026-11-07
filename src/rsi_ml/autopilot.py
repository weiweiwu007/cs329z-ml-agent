"""The whole study, one command, no human in the loop.

data -> baselines -> targets and budgets from measured baselines -> all agent runs ->
figures -> final test scores. Failed jobs recover and retry on their own. Rerunning the
same command continues from where it stopped and never repeats finished work.

Guardrails are the corner cases where continuing could cause harm or cost money for
nothing: low disk, the API spending cap, or a baseline that cannot train. Running jobs
finish, nothing new starts, and the report says why. Agents cannot affect any of this:
they return schema-validated recipe values and have no shell, file, or network tools.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from .agents import BackendError, ProviderUnavailable
from .api_check import api_preflight, study_api_key
from .storage import now, read_events, read_json, write_json
from .study import build_study, job_action, run_dir, run_stage, stage_jobs

# If a provider ever omits the charged amount, the cap must still count that call. Generous
# enough to stop runaway spending, and low enough that a normal study never trips it.
UNKNOWN_COST_PER_CALL_USD = 0.05
MAX_WAIT_SECONDS = 3600


def parse_gpus(value: str) -> list[str]:
    if value == "none":
        return []
    if value == "auto":
        return detect_gpus()
    return [g for g in value.split(",") if g]


def detect_gpus() -> list[str]:
    """Every GPU of the most common model, so wall-clock cost stays comparable."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = [[part.strip() for part in line.split(",", 1)] for line in output.splitlines() if line]
    if not rows:
        return []
    model = Counter(row[1] for row in rows).most_common(1)[0][0]
    return [row[0] for row in rows if row[1] == model]


def derive_agent_stage(protocol: dict, study_dir: Path) -> dict:
    """Fill unset targets and budgets by the declared rules, before any agent runs.

    Target: remove a fixed share of the baseline's remaining error, which is comparable
    across tasks that start at very different scores. Values already set are kept.
    """
    rules = protocol["autopilot"]
    stage = json.loads(json.dumps(protocol["agent_stage"]))
    measured = {}
    for task_id, spec in protocol["tasks"].items():
        receipts = [
            read_json(run_dir(study_dir, task_id, seed, "baseline") / "baseline.json")
            for seed in spec["seeds"]
        ]
        metric = receipts[0]["metric"]
        scores = [r["result"]["metrics"][metric] for r in receipts]
        seconds = [r["result"]["compute_seconds"] for r in receipts]
        median = statistics.median(scores)
        target = min(0.999, median + rules["target_error_reduction"] * (1 - median))
        per_trial = math.ceil(max(seconds) * rules["per_trial_seconds_vs_baseline"])
        total = math.ceil(statistics.mean(seconds) * rules["total_compute_vs_baseline_runs"])
        for group, value in (
            ("targets", round(target, 4)),
            ("per_trial_seconds", per_trial),
            ("total_compute_seconds", total),
        ):
            stage[group] = {k: v for k, v in stage[group].items() if k in protocol["tasks"]}
            if stage[group].get(task_id) is None:
                stage[group][task_id] = value
        measured[task_id] = {"median_score": median, "scores": scores, "seconds": seconds}
    return {"agent_stage": stage, "measured": measured}


def api_spend(study_dir: Path) -> float:
    """Provider-reported spend so far, including runs still in progress."""
    total = 0.0
    for path in Path(study_dir).glob("runs/*/seed-*/*/state.json"):
        usage = read_json(path).get("usage", {})
        cost = usage.get("cost_usd")
        total += cost if cost is not None else usage.get("calls", 0) * UNKNOWN_COST_PER_CALL_USD
    return total


def _guard(study_dir: Path, rules: dict, *, spending: bool) -> Callable[[], str | None]:
    minimum = rules["min_free_disk_gb"] * 2**30

    def check() -> str | None:
        free = shutil.disk_usage(study_dir).free
        if free < minimum:
            return f"Free disk {free / 2**30:.1f} GB is below {rules['min_free_disk_gb']} GB."
        if spending and api_spend(study_dir) >= rules["max_api_usd"]:
            return f"API spending reached the ${rules['max_api_usd']} cap."
        return None

    return check


def _run_until_settled(study_dir, stage, rules, sleep, log, *, gpus, **options) -> dict:
    """Retry failed jobs until a pass finishes everything or progress stops.

    Only passes that finish nothing count toward the limit, so separate faults on a long
    run do not add up. The GPU order rotates, so a dead first GPU cannot take the same job
    on every pass. Held jobs wait for the guardrail instead of being retried.
    """
    attempt = stalled = 0
    while True:
        shift = attempt % len(gpus) if gpus else 0
        attempt += 1
        result = run_stage(
            study_dir, stage, recover=True, gpus=gpus[shift:] + gpus[:shift], **options
        )
        log(
            f"{stage}_pass",
            attempt=attempt,
            **{k: result[k] for k in ("completed", "failed", "held")},
        )
        if not result["failed"] or result["held"]:
            return result
        stalled = 0 if result["completed"] else stalled + 1
        if stalled >= rules["max_passes"]:
            return result
        sleep(min(rules["seconds_between_passes"] * 2 ** max(stalled - 1, 0), MAX_WAIT_SECONDS))


def _freeze_key(protocol: dict) -> dict:
    """The settings that define a study. Guardrail limits may change, and seeds may grow."""
    key = {
        k: v for k, v in protocol.items() if k not in {"autopilot", "status", "approval", "note"}
    }
    key["tasks"] = {
        name: {k: v for k, v in spec.items() if k != "seeds"} for name, spec in key["tasks"].items()
    }
    return key


def _freeze_inputs(protocol: dict, base: Path, out: Path) -> dict:
    """Copy the policy and templates in, so later edits to them cannot change this study."""
    frozen = out / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    resolved = json.loads(json.dumps(protocol))
    # Recorded explicitly because JSON records sort their keys: this order is the queue.
    resolved["task_order"] = list(protocol["tasks"])
    shutil.copyfile(base / protocol["policy"], frozen / "policy.json")
    resolved["policy"] = str(frozen / "policy.json")
    for name, spec in resolved["tasks"].items():
        target = frozen / f"{name}.template.json"
        shutil.copyfile(base / spec["template"], target)
        spec["template"] = str(target)
    return resolved


def _sync_seeds(path: Path, protocol: dict) -> None:
    """Added seeds join the study with the targets it already fixed; none may be removed."""
    current = read_json(path)
    for name, spec in protocol["tasks"].items():
        seeds = current["tasks"][name]["seeds"]
        if set(seeds) - set(spec["seeds"]):
            raise ValueError(f"Seeds cannot be removed from a study ({name}).")
        seeds.extend(seed for seed in spec["seeds"] if seed not in seeds)
    write_json(path, current)


def _write_figures(out: Path, log) -> None:
    from .figures import write_figures

    try:
        log("figures", count=len(write_figures(out)["figures"]))
    except Exception as exc:  # noqa: BLE001 - a plot must not hide the study's status.
        log("figures_failed", error=f"{type(exc).__name__}: {exc}"[:500])


def _score_test_sets(out: Path, gpus: list[str], log) -> None:
    """In a child process, because this one has already initialized CUDA on all GPUs."""
    command = [sys.executable, "-m", "rsi_ml", "study-score-test", "--study", str(out), "--execute"]
    final = out / "final_test.json"
    for device in gpus or [""]:
        environment = {
            **os.environ,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": device,
        }
        code = subprocess.run(command, env=environment, capture_output=True, check=False).returncode
        if code == 0 and not read_json(final)["skipped"]:
            break  # Otherwise try the next GPU; finished scores are kept and never repeated.
    outcome = read_json(final) if final.exists() else {"scored": [], "skipped": []}
    log("final_test", exit_code=code, scored=len(outcome["scored"]), skipped=outcome["skipped"])


def autopilot(
    protocol_path: Path,
    data_root: Path,
    out: Path,
    *,
    gpus: list[str],
    cpu_slots: int = 2,
    env_file: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    protocol_path, data_root, out = (Path(p).resolve() for p in (protocol_path, data_root, out))
    protocol = read_json(protocol_path)
    if protocol.get("status") != "approved":
        raise ValueError("Protocol is not approved; nothing was started.")
    needs_api = any(arm != "random" for arm in protocol["arms"])
    if needs_api:
        # Before any work: finding this after hours of baselines would waste the whole night.
        api_key = study_api_key(env_file)
    rules = protocol["autopilot"]
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "autopilot.json"
    previous = read_json(report_path) if report_path.exists() else {}
    report = {
        "started": previous.get("started", now()),
        "gpus": gpus,
        "log": previous.get("log", []),
    }

    def log(step, **details):
        report["log"].append({"at": now(), "step": step, **details})
        report["updated"] = now()
        write_json(report_path, report)

    def finish(status, **details):
        report.update(status=status, spend_usd=api_spend(out), **details)
        log("finished", status=status)
        return report

    log("started")
    # A study's settings are fixed by its first run. A rerun continues exactly that study.
    source_path, initial_path = out / "protocol.source.json", out / "protocol.initial.json"
    resolved_path = out / "protocol.resolved.json"
    if initial_path.exists():
        if _freeze_key(read_json(source_path)) != _freeze_key(protocol):
            raise ValueError(
                f"{out} was built from different settings. Restore them, or use a new --out."
            )
    else:
        write_json(source_path, protocol)
        write_json(initial_path, _freeze_inputs(protocol, protocol_path.parent, out))
    current_path = resolved_path if resolved_path.exists() else initial_path
    _sync_seeds(current_path, protocol)
    if not all((data_root / "selected" / t / "manifest.json").exists() for t in protocol["tasks"]):
        from .prepare import prepare_all

        log("data", **prepare_all(data_root, tasks=set(protocol["tasks"])))
    build_study(current_path, data_root, out)

    preflight_path = out / "api_preflight.json"
    if needs_api and not preflight_path.exists():
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                write_json(preflight_path, api_preflight(out, api_key))
                log("api_preflight", **read_json(preflight_path))
                break
            except (ProviderUnavailable, BackendError) as exc:
                log("api_preflight_failed", attempt=attempt, error=str(exc)[:500])
                if attempt == attempts:
                    return finish("stopped_api_preflight", reason=str(exc)[:500])
                sleep(rules["seconds_between_passes"])

    # Finished baselines are skipped at no cost; only new seeds or failures train here.
    guard = _guard(out, rules, spending=False)
    result = _run_until_settled(
        out, "baselines", rules, sleep, log, gpus=gpus, cpu_slots=cpu_slots, guard=guard
    )
    if result["held"]:
        return finish("stopped_at_guardrail", reason=guard(), held=result["held"])
    broken = [
        job.job_id
        for job in stage_jobs(out, "baselines")
        if not (job.out / "baseline.json").exists()
        or read_json(job.out / "baseline.json")["result"]["status"] != "completed"
    ]
    if broken:
        return finish("stopped_baseline_failed", failed=broken)
    if not resolved_path.exists():
        resolved = read_json(initial_path)
        derived = derive_agent_stage(resolved, out)
        resolved["agent_stage"] = derived["agent_stage"]
        write_json(resolved_path, resolved)
        write_json(out / "agent_stage.derived.json", derived)
        log("agent_stage_derived", measured=derived["measured"])
        build_study(resolved_path, data_root, out)

    guard = _guard(out, rules, spending=True)
    result = _run_until_settled(
        out,
        "agents",
        rules,
        sleep,
        log,
        gpus=gpus,
        cpu_slots=cpu_slots,
        env_file=env_file,
        guard=guard,
    )
    # Figures are always written, so a stopped study still shows what it has.
    _write_figures(out, log)
    if result["held"]:
        return finish("stopped_at_guardrail", reason=guard(), held=result["held"])
    unfinished = sorted(
        {
            *result["failed"],
            *(j.job_id for j in stage_jobs(out, "agents") if job_action(j) != "done"),
        }
    )
    if unfinished:
        reasons = {}
        for job_id in unfinished:
            task_id, seed, arm = job_id.split(".")
            events = run_dir(out, task_id, int(seed.removeprefix("seed-")), arm) / "events.jsonl"
            reasons[job_id] = read_events(events)[-1]["event"] if events.exists() else "not started"
        return finish("incomplete_after_retries", failed=unfinished, last_event=reasons)
    if rules["score_test_at_end"]:
        _score_test_sets(out, gpus, log)
        _write_figures(out, log)
    return finish("complete", figures=str(out / "figures"))
