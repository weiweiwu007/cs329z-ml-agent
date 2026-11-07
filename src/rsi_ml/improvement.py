"""Outer loop: propose planner instructions, then independently compare them."""

from __future__ import annotations

from pathlib import Path

from .agents import AgentBackend
from .context import knowledge_for, public_task
from .schemas import Experiment, Policy, PolicyChange, RunSummary, TaskSpec, digest
from .storage import Store, read_json, write_json


def propose_policy(parent: Policy, runs: list[Path], backend: AgentBackend, output: Path) -> Policy:
    if not runs:
        raise ValueError("Provide development runs containing measured experiment records.")
    evidence, source_receipts, seen = [], [], set()
    reference_task = None
    for path in runs:
        summary = RunSummary.model_validate(read_json(path / "summary.json"))
        task = TaskSpec.model_validate(read_json(path / "task.json"))
        if summary.purpose not in ({"demo"} if backend.name == "mock" else {"development"}):
            raise ValueError(
                "Workflow revision may use only development runs (demo runs in mock mode)."
            )
        if summary.policy_hash != digest(parent) or summary.task_hash != digest(task):
            raise ValueError(
                "Development evidence must come from this parent policy and intact task protocol."
            )
        if summary.backend_fingerprint != digest(backend.metadata):
            raise ValueError(
                "Workflow improvement must use the same underlying model/backend settings."
            )
        if summary.run_id in seen:
            raise ValueError("Development run IDs must be unique.")
        seen.add(summary.run_id)
        reference_task = reference_task or task
        state = read_json(path / "state.json")
        records = [
            Experiment.model_validate(r).model_dump(mode="json") for r in state["experiments"]
        ]
        evidence.append(
            {
                "summary": summary.model_dump(mode="json"),
                "task": public_task(task),
                "records": records[-5:],
            }
        )
        source_receipts.append(
            {
                "run_id": summary.run_id,
                "summary_hash": digest(summary),
                "task_hash": summary.task_hash,
                "dataset_hash": summary.dataset_hash,
                "task_id": task.task_id,
                "seed": task.seed,
            }
        )
    assert reference_task is not None
    output.mkdir(parents=True, exist_ok=False)
    store = Store(output)
    context = {
        "policy": parent.model_dump(mode="json"),
        "development_runs": evidence,
        "knowledge": knowledge_for(reference_task.adapter),
    }
    write_json(output / "shared_evidence.json", evidence)
    store.event("agent_input", {"role": "improver", "context": context})
    try:
        change = backend.call("improver", context, parent, reference_task.limits)
        if not isinstance(change, PolicyChange) or set(change.evidence_run_ids) - seen:
            raise ValueError("Improver must cite available development evidence.")
        candidate = parent.model_copy(
            update={
                "policy_id": f"policy-v{parent.version + 1}-{digest(change)[:8]}",
                "version": parent.version + 1,
                "parent_hash": digest(parent),
                "status": "candidate",
                "planner_instructions": change.planner_instructions,
                "rationale": change.rationale,
                "evidence_run_ids": change.evidence_run_ids,
            }
        )
        candidate = Policy.model_validate(candidate.model_dump())
        write_json(output / "policy.json", candidate)
        store.event("candidate_proposed", candidate.model_dump(mode="json"))
    finally:
        write_json(
            output / "discovery.json",
            {
                "backend": backend.metadata,
                "usage": backend.usage.model_dump(mode="json"),
                "source_runs": source_receipts,
                "parent_hash": digest(parent),
                "source_compute_seconds": sum(
                    e["summary"]["total_compute_seconds"] for e in evidence
                ),
                "note": "Proposal cost and source-run costs are separate from later policy-validation cost.",
            },
        )
    return candidate
