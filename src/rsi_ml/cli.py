"""Local entry points. API and research execution are always explicit choices."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .agents import (
    PROVIDER_UNAVAILABLE_EXIT_CODE,
    BackendError,
    DSPyBackend,
    MockBackend,
    ProviderUnavailable,
)
from .baseline import RandomSearchBackend
from .evaluation import compare_runs, promotion_decision
from .improvement import propose_policy
from .schemas import Policy, TaskSpec, digest
from .storage import ensure_ca_bundle, read_json, write_json
from .workflow import Workflow, preflight


def backend(name: str):
    return {"mock": MockBackend, "dspy": DSPyBackend, "random": RandomSearchBackend}[name]()


def load_task(path: Path) -> TaskSpec:
    data = read_json(path)
    # Dataset files are relative to their task config, never an implicit cwd.
    for field in (
        "train_csv",
        "validation_csv",
        "test_csv",
        "train_file",
        "validation_file",
        "test_file",
        "split_manifest",
    ):
        value = data.get("dataset", {}).get(field)
        if value and not Path(value).is_absolute():
            data["dataset"][field] = str((path.resolve().parent / value).resolve())
    model = data.get("text_model")
    if model and model.get("local_path") and not Path(model["local_path"]).is_absolute():
        model["local_path"] = str((path.resolve().parent / model["local_path"]).resolve())
    return TaskSpec.model_validate(data)


def load_policy(path: Path) -> Policy:
    return Policy.model_validate(read_json(path))


def fresh_output(label: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"{label}-{stamp}-{uuid.uuid4().hex[:6]}"


def suite_cases(path: Path) -> list[TaskSpec]:
    document = read_json(path)
    cases, identities = [], set()
    if not document.get("cases"):
        raise ValueError("Suite must declare at least one task and explicit seeds.")
    for item in document["cases"]:
        if not item.get("seeds"):
            raise ValueError("Each suite case must declare its paired seeds.")
        task = load_task((path.resolve().parent / item["task"]).resolve())
        for seed in item["seeds"]:
            case = TaskSpec.model_validate({**task.model_dump(), "seed": seed})
            key = digest(case)
            if key in identities:
                raise ValueError("Suite contains duplicate task/seed pairs.")
            identities.add(key)
            cases.append(case)
    return cases


def compare_suite(
    suite: Path, baseline_path: Path, candidate_path: Path, backend_name: str, output: Path
) -> dict:
    original, candidate = load_policy(baseline_path), load_policy(candidate_path)
    if candidate.parent_hash != digest(original) or candidate.status != "candidate":
        raise ValueError(
            "Candidate must be an unpromoted direct revision of the supplied baseline."
        )
    for field in ("developer_instructions", "reviewer_instructions"):
        if getattr(original, field) != getattr(candidate, field):
            raise ValueError("V1 comparisons change planning instructions only.")
    cases = suite_cases(suite)
    memory_path = candidate_path.parent / "shared_evidence.json"
    if not memory_path.exists():
        raise ValueError(
            "Candidate requires shared_evidence.json from the improve command for an equal-history comparison."
        )
    shared_memory = read_json(memory_path)
    for task in cases:
        problems = preflight(task, backend_name)
        if problems:
            raise ValueError("\n".join(problems))
    # Instantiate once before writing output so missing API configuration is clear.
    descriptor = backend(backend_name).metadata
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "suite.json",
        {
            "cases": [t.model_dump(mode="json") for t in cases],
            "backend": descriptor,
            "baseline_hash": digest(original),
            "candidate_hash": digest(candidate),
            "memory_fingerprint": digest(shared_memory),
            "maximum_allocated_compute_seconds": sum(
                2 * t.limits.total_compute_seconds for t in cases
            ),
            "maximum_api_calls": sum(2 * t.limits.max_api_calls for t in cases),
            "criterion": "No fewer targets reached; strictly fewer capped attempts and lower capped compute.",
            "interpretation": "Exploratory finite-suite comparison; not a statistical significance claim.",
        },
    )
    pairs, all_summaries = [], []
    for index, task in enumerate(cases):
        # Counterbalance order to reduce systematic warm-cache/runtime advantage.
        order = [("original", original), ("candidate", candidate)]
        if index % 2:
            order.reverse()
        summaries = {}
        for label, policy in order:
            directory = output / f"case-{index:03d}-{label}"
            summaries[label] = Workflow(
                task, policy, backend(backend_name), directory, shared_memory=shared_memory
            ).run()
            all_summaries.append(summaries[label])
        pairs.append(
            compare_runs(
                summaries["original"],
                summaries["candidate"],
                compute_budget=task.limits.total_compute_seconds,
                trial_budget=task.limits.max_trials,
            )
        )
        write_json(output / "pairs.json", pairs)
    decision = promotion_decision(pairs)
    result = {
        "decision": decision,
        "pairs": pairs,
        "validation_compute_seconds": sum(s.total_compute_seconds for s in all_summaries),
        "validation_api_calls": sum(s.usage.calls for s in all_summaries),
        "discovery_receipt": None,
        "transfer_claim": "None. Inspect discovery cases and evaluation settings before claiming transfer.",
    }
    discovery_path = candidate_path.parent / "discovery.json"
    if discovery_path.exists():
        result["discovery_receipt"] = read_json(discovery_path)
    # Candidate status and acceptance are separate artifacts. No existing policy is overwritten.
    if decision.get("accept", decision.get("accepted", False)):
        accepted = candidate.model_copy(update={"status": "accepted"})
        write_json(output / "accepted_policy.json", accepted)
        result["accepted_from_candidate_hash"] = digest(candidate)
        result["accepted_policy_hash"] = digest(accepted)
    write_json(output / "comparison.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    ensure_ca_bundle()
    parser = argparse.ArgumentParser(description="Measured multi-agent ML development, v1.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Run a bounded CPU fixture with scripted agents; no API."
    )
    demo.add_argument("--out", type=Path)
    run = commands.add_parser("run", help="Run one task under a frozen policy.")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--policy", type=Path, required=True)
    run.add_argument("--backend", choices=["mock", "dspy", "random"], required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--variant", choices=["A", "B", "C", "D"], required=True)
    run.add_argument("--baseline-dir", type=Path, help="Adopt this stage-1 baseline as trial 0.")
    run.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize execution of a finalized protocol.",
    )
    resume = commands.add_parser(
        "resume", help="Resume an unchanged run at a completed node boundary."
    )
    resume.add_argument("run", type=Path)
    resume.add_argument("--execute", action="store_true")
    resume.add_argument(
        "--recover-interrupted",
        action="store_true",
        help="After a crash: rerun the interrupted node, keeping any partial trial aside.",
    )
    check = commands.add_parser(
        "preflight", help="Validate a protocol without training or API calls."
    )
    check.add_argument("--task", type=Path, required=True)
    check.add_argument("--backend", choices=["mock", "dspy", "random"], required=True)
    improve = commands.add_parser(
        "improve", help="Propose a candidate planner policy from development evidence."
    )
    improve.add_argument("--policy", type=Path, required=True)
    improve.add_argument("--runs", type=Path, nargs="+", required=True)
    improve.add_argument("--backend", choices=["mock", "dspy"], required=True)
    improve.add_argument("--out", type=Path, required=True)
    improve.add_argument("--execute", action="store_true")
    compare = commands.add_parser(
        "compare", help="Evaluate original/candidate workflows on matched cases."
    )
    compare.add_argument("--suite", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--backend", choices=["mock", "dspy"], required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--execute", action="store_true")
    designs = commands.add_parser(
        "compare-designs", help="Plan an A-D suite; runs only with --execute."
    )
    designs.add_argument("--suite", type=Path, required=True)
    designs.add_argument("--policy", type=Path)
    designs.add_argument("--backend", choices=["mock", "dspy"])
    designs.add_argument("--out", type=Path)
    designs.add_argument("--execute", action="store_true")
    data_import = commands.add_parser(
        "import-data", help="Convert local sources using an explicit split plan."
    )
    data_import.add_argument("--plan", type=Path, required=True)
    data_import.add_argument("--out", type=Path, required=True)
    data_import.add_argument("--execute", action="store_true")
    score = commands.add_parser(
        "score-test", help="Explicit supervisor-only final evaluation; never an agent tool."
    )
    score.add_argument("run", type=Path)
    score.add_argument("--trial", required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--execute", action="store_true")
    sources = commands.add_parser(
        "prepare-sources", help="Show, or with --execute download and split, the pinned data."
    )
    sources.add_argument("--root", type=Path, default=Path("data"))
    sources.add_argument("--execute", action="store_true")
    build = commands.add_parser(
        "study-build", help="Write runnable task files from an approved protocol."
    )
    build.add_argument("--protocol", type=Path, required=True)
    build.add_argument("--data", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--execute", action="store_true")
    stage = commands.add_parser(
        "study-run", help="List a stage's jobs, or with --execute run them across GPUs."
    )
    stage.add_argument("--study", type=Path, required=True)
    stage.add_argument("--stage", choices=["baselines", "agents"], required=True)
    stage.add_argument("--gpus", default="", help="Comma-separated GPU indices, e.g. 0,1.")
    stage.add_argument("--cpu-slots", type=int, default=2)
    stage.add_argument("--jobs-per-gpu", type=int, default=1)
    stage.add_argument("--env-file", type=Path, help="File containing the API key.")
    stage.add_argument("--tasks", help="Comma-separated subset of tasks.")
    stage.add_argument("--seeds", help="Comma-separated subset of seeds.")
    stage.add_argument("--arms", help="Comma-separated subset of arms (agents stage).")
    stage.add_argument("--recover", action="store_true", help="Recover interrupted runs.")
    stage.add_argument("--execute", action="store_true")
    auto = commands.add_parser(
        "autopilot", help="Run the whole study unattended, from data to final figures."
    )
    auto.add_argument("--protocol", type=Path, required=True)
    auto.add_argument("--data", type=Path, default=Path("data"))
    auto.add_argument("--out", type=Path, required=True)
    auto.add_argument("--gpus", default="auto", help="'auto', 'none', or indices such as 0,1.")
    auto.add_argument("--cpu-slots", type=int, default=2)
    auto.add_argument("--env-file", type=Path, help="File containing the API key.")
    auto.add_argument("--execute", action="store_true")
    job = commands.add_parser("baseline-job", help="One stage-1 training run; used by study-run.")
    job.add_argument("--task", type=Path, required=True)
    job.add_argument("--out", type=Path, required=True)
    job.add_argument("--execute", action="store_true")
    status = commands.add_parser("study-status", help="Read-only progress and cost.")
    status.add_argument("--study", type=Path, required=True)
    figures = commands.add_parser("study-figures", help="Write result figures and tables.")
    figures.add_argument("--study", type=Path, required=True)
    figures.add_argument("--out", type=Path)
    test_scores = commands.add_parser(
        "study-score-test", help="Supervisor-only final test scoring of selected models."
    )
    test_scores.add_argument("--study", type=Path, required=True)
    test_scores.add_argument("--execute", action="store_true")
    check = commands.add_parser(
        "api-check", help="Show remaining credit; with --live --execute make one real call."
    )
    check.add_argument("--env-file", type=Path, required=True)
    check.add_argument("--model")
    check.add_argument("--reasoning-effort")
    check.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    check.add_argument("--live", action="store_true")
    check.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if (
            args.command
            in {
                "run",
                "resume",
                "improve",
                "compare",
                "score-test",
                "import-data",
                "baseline-job",
                "study-score-test",
                "autopilot",
            }
            and not args.execute
        ):
            raise ValueError(
                "Execution is disabled. Finalize the protocol and explicitly pass --execute."
            )
        if args.command == "demo":
            root = Path(__file__).resolve().parents[2]
            output = args.out or fresh_output("demo")
            result = Workflow(
                load_task(root / "configs/mlp-demo.json"),
                load_policy(root / "configs/policy.json"),
                MockBackend(),
                output,
            ).run()
            print(json.dumps(result.model_dump(mode="json"), indent=2))
            print(f"Report: {(output / 'report.html').resolve()}")
        elif args.command == "run":
            result = Workflow(
                load_task(args.task),
                load_policy(args.policy),
                backend(args.backend),
                args.out,
                variant=args.variant,
                baseline_dir=args.baseline_dir,
            ).run()
            print(json.dumps(result.model_dump(mode="json"), indent=2))
        elif args.command == "resume":
            manifest = read_json(args.run / "manifest.json")
            result = Workflow(
                load_task(args.run / "task.json"),
                load_policy(args.run / "policy.json"),
                backend(manifest["backend"]["backend"]),
                args.run,
                shared_memory=read_json(args.run / "shared_memory.json"),
                variant=manifest["variant"],
                baseline_dir=manifest.get("baseline_dir"),
            ).run(resume=True, recover=args.recover_interrupted)
            print(json.dumps(result.model_dump(mode="json"), indent=2))
        elif args.command == "preflight":
            from pydantic import ValidationError

            from .comparison import unresolved

            try:
                load_task(args.task)
            except ValidationError as exc:
                print(
                    json.dumps(
                        {
                            "ready": False,
                            "training_launched": False,
                            "api_calls": 0,
                            "unresolved": unresolved(read_json(args.task)),
                            "problems": [
                                {"field": ".".join(map(str, e["loc"])), "type": e["type"]}
                                for e in exc.errors()
                            ],
                        },
                        indent=2,
                    )
                )
                return 1
            task = load_task(args.task)
            problems = preflight(task, args.backend)
            from .tasks.registry import parameter_count

            try:
                initial_parameters = parameter_count(task, task.initial_recipe)
            except (ValueError, OSError, ImportError):
                initial_parameters = None
            print(
                json.dumps(
                    {
                        "ready": not problems,
                        "problems": problems,
                        "task_id": task.task_id,
                        "purpose": task.purpose,
                        "limits": task.limits.model_dump(),
                        "initial_parameters": initial_parameters,
                        "note": "Schema checks only; source data hashes are verified when staging a run.",
                    },
                    indent=2,
                )
            )
            return 1 if problems else 0
        elif args.command == "improve":
            result = propose_policy(
                load_policy(args.policy), args.runs, backend(args.backend), args.out
            )
            print(json.dumps(result.model_dump(mode="json"), indent=2))
        elif args.command == "compare":
            result = compare_suite(
                args.suite, args.baseline, args.candidate, args.backend, args.out
            )
            print(json.dumps(result["decision"], indent=2))
            print(f"Comparison: {(args.out / 'comparison.json').resolve()}")
        elif args.command == "score-test":
            from .tasks.registry import score_test

            if Path(args.trial).name != args.trial or not args.trial.startswith("trial-"):
                raise ValueError("Trial must be a local trial ID.")
            result = score_test(
                load_task(args.run / "task.json"),
                args.run / "prepared",
                args.run / "trials" / args.trial,
                args.out,
            )
            print(json.dumps(result, indent=2))
        elif args.command == "compare-designs":
            from .comparison import compare_designs, plan_suite

            if not args.execute:
                result = plan_suite(args.suite)
            else:
                if not all((args.policy, args.backend, args.out)):
                    raise ValueError("Execution requires --policy, --backend, and --out.")
                result = compare_designs(args.suite, args.policy, args.backend, args.out)
            print(json.dumps(result, indent=2))
        elif args.command == "prepare-sources":
            from .prepare import prepare_all, source_plan

            result = prepare_all(args.root) if args.execute else source_plan()
            print(json.dumps(result, indent=2))
        elif args.command == "study-build":
            from .study import build_study, unresolved_agent_settings

            if not args.execute:
                protocol = read_json(args.protocol)
                print(
                    json.dumps(
                        {
                            "mode": "plan_only",
                            "status": protocol["status"],
                            "tasks": {
                                k: {"seeds": v["seeds"], "device": v["device"]}
                                for k, v in protocol["tasks"].items()
                            },
                            "arms": protocol["arms"],
                            "unresolved_for_agent_stage": unresolved_agent_settings(protocol),
                        },
                        indent=2,
                    )
                )
            else:
                print(json.dumps(build_study(args.protocol, args.data, args.out), indent=2))
        elif args.command == "autopilot":
            from .autopilot import autopilot, parse_gpus

            result = autopilot(
                args.protocol,
                args.data,
                args.out,
                gpus=parse_gpus(args.gpus),
                cpu_slots=args.cpu_slots,
                env_file=args.env_file,
            )
            print(json.dumps(result, indent=2))
            if result["status"] != "complete":
                return 1
        elif args.command == "study-run":
            from .study import job_action, run_stage, stage_jobs

            def subset(value, kind=str):
                return {kind(v) for v in value.split(",")} if value else None

            selection = {
                "tasks": subset(args.tasks),
                "seeds": subset(args.seeds, int),
                "arms": subset(args.arms),
            }
            gpus = [g for g in args.gpus.split(",") if g != ""]
            if not args.execute:
                jobs = stage_jobs(args.study, args.stage, **selection)
                listing = {
                    "mode": "plan_only",
                    "stage": args.stage,
                    "jobs": [
                        {"job": j.job_id, "device": j.device, "action": job_action(j)} for j in jobs
                    ],
                }
                print(json.dumps(listing, indent=2))
            else:
                result = run_stage(
                    args.study,
                    args.stage,
                    gpus=gpus,
                    cpu_slots=args.cpu_slots,
                    jobs_per_gpu=args.jobs_per_gpu,
                    env_file=args.env_file,
                    recover=args.recover,
                    **selection,
                )
                print(json.dumps(result, indent=2))
                if result["failed"]:
                    return 1
        elif args.command == "baseline-job":
            from .study import run_baseline_job

            result = run_baseline_job(args.task, args.out)
            print(json.dumps(result, indent=2))
            if result["status"] != "completed":
                return 1
        elif args.command == "study-status":
            from .study import study_status

            print(json.dumps(study_status(args.study), indent=2))
        elif args.command == "study-figures":
            from .figures import write_figures

            print(json.dumps(write_figures(args.study, args.out), indent=2))
        elif args.command == "study-score-test":
            from .study import score_study_test_sets

            print(json.dumps(score_study_test_sets(args.study), indent=2))
        elif args.command == "api-check":
            from .api_check import api_check

            print(
                json.dumps(
                    api_check(
                        args.env_file,
                        model=args.model,
                        reasoning_effort=args.reasoning_effort,
                        api_base=args.api_base,
                        live=args.live and args.execute,
                    ),
                    indent=2,
                )
            )
        elif args.command == "import-data":
            from .data import import_data

            print(json.dumps(import_data(args.plan, args.out), indent=2))
    except ProviderUnavailable as exc:
        # A distinct code: the scheduler must not read a provider outage as a broken GPU slot.
        print(f"Error: {exc}", file=sys.stderr)
        return PROVIDER_UNAVAILABLE_EXIT_CODE
    except (BackendError, ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
