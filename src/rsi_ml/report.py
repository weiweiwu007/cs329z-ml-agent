"""Portable local reports using measured results only; no external assets."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .evaluation import progress_rows
from .schemas import Experiment, RunSummary
from .storage import read_events


def _number(value: float | None, precision: int = 4) -> str:
    return "—" if value is None else f"{value:.{precision}f}"


def _markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _plot(rows: list[dict], x_key: str, x_label: str, target: float, metric: str) -> str:
    # Standard plotting backend; standalone SVGs are also exported below.
    import io
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rsi-ml-matplotlib"))
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    figure = Figure(figsize=(6.2, 3.1), layout="constrained")
    FigureCanvasSVG(figure)
    axis = figure.subplots()
    completed = [row for row in rows if row["best_score"] is not None]
    if completed:
        axis.step(
            [r[x_key] for r in completed],
            [r["best_score"] for r in completed],
            where="post",
            marker="o",
            markersize=3,
            color="#2563eb",
            label="Best score",
        )
    else:
        axis.text(0.5, 0.5, "No completed scores", ha="center", transform=axis.transAxes)
    axis.axhline(target, linestyle="--", color="#c2410c", linewidth=1.2, label="Target")
    axis.set(
        xlabel=x_label,
        ylabel=metric,
        ylim=(0, 1.03),
        xlim=(0, max([float(row[x_key]) for row in rows] + [1.0]) * 1.03),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", fontsize=8)
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", metadata={"Date": None})
    svg = buffer.getvalue()
    return svg[svg.index("<svg") :]


def write_report(directory: Path, summary: RunSummary, experiments: list[Experiment]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = progress_rows(experiments, summary.metric)
    events_path = directory / "events.jsonl"
    if events_path.exists():
        costs = [
            event["payload"]
            for event in read_events(events_path)
            if event["event"] == "diagnostic_cost"
        ]
        for row in rows:
            row["cumulative_compute_seconds"] += sum(
                cost["seconds"] for cost in costs if cost["after_attempt"] < row["attempt"]
            )
    fieldnames = [
        "attempt",
        "trial_id",
        "status",
        "score",
        "best_score",
        "compute_seconds",
        "cumulative_compute_seconds",
    ]
    with (directory / "curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    label = (
        "SOFTWARE DEMO — not benchmark evidence"
        if summary.purpose == "demo"
        else (
            "Development result — used for workflow selection"
            if summary.purpose == "development"
            else "Evaluation result — never used for workflow selection"
        )
    )
    if summary.backend.lower().startswith("mock"):
        label += "; scripted mock decisions, measured training results"
    attainment = "Reached" if summary.reached_target else "Not reached within this run"
    facts = [
        ("Task", summary.task_id),
        ("Mode", label),
        ("Backend", summary.backend),
        ("Workflow variant", summary.variant),
        ("Diagnostic seconds (included in total)", _number(summary.diagnostic_seconds)),
        ("Initial preparation seconds (separate)", _number(summary.preparation_seconds)),
        ("Validation metric / target", f"{summary.metric} / {summary.target:g}"),
        ("Target", attainment),
        (
            "Attempts to target",
            str(summary.trials_to_target) if summary.reached_target else "Not attained",
        ),
        ("Compute to target (seconds)", _number(summary.compute_to_target)),
        ("Best validation score", _number(summary.best_score)),
        ("All attempted experiments", str(summary.total_trials)),
        ("Total execution seconds", _number(summary.total_compute_seconds)),
        ("Execution device", summary.device),
        ("Agent calls", str(summary.usage.calls)),
        (
            "API cost (USD)",
            _number(summary.usage.cost_usd)
            if summary.usage.cost_usd is not None
            else "Unknown / not supplied",
        ),
        (
            "Input / output tokens",
            f"{summary.usage.input_tokens} / {summary.usage.output_tokens}"
            if summary.usage.tokens_known
            else "Unknown / incomplete provider usage",
        ),
        ("Stop reason", summary.stop_reason),
    ]
    md = [
        f"# ML development run: {_markdown(summary.run_id)}",
        "",
        label,
        "",
        "| Measurement | Value |",
        "|---|---|",
    ]
    md.extend(f"| {_markdown(k)} | {_markdown(v)} |" for k, v in facts)
    md.extend(
        [
            "",
            (
                "Execution time includes worker startup, training, validation, artifact writing, and diagnostics. "
                "It is elapsed allocation time, not measured GPU kernel utilization. "
                "All failures and rejected attempts remain in the trial count. "
                "These are validation scores; no final test score is implied."
            ),
            "",
            "## Attempts",
            "",
            "| Attempt | Status | Score | Best so far | Cumulative seconds |",
            "|---|---|---:|---:|---:|",
        ]
    )
    md.extend(
        f"| {row['attempt']} | {row['status']} | {_number(row['score'])} | "
        f"{_number(row['best_score'])} | {_number(row['cumulative_compute_seconds'])} |"
        for row in rows
    )
    md.extend(
        [
            "",
            "## Comparison identity",
            "",
            "```json",
            json.dumps(
                {
                    "task_hash": summary.task_hash,
                    "policy_hash": summary.policy_hash,
                    "dataset_hash": summary.dataset_hash,
                    "starting_model_hash": summary.starting_model_hash,
                    "backend_fingerprint": summary.backend_fingerprint,
                    "execution_fingerprint": summary.execution_fingerprint,
                    "memory_fingerprint": summary.memory_fingerprint,
                    "seed": summary.seed,
                },
                indent=2,
            ),
            "```",
            "",
            (
                "A single run cannot establish self-improvement. Compare fixed policies under the same "
                "task, starting weights, data, seed, tools, model, and budgets. Report workflow discovery "
                "and selection costs separately."
            ),
            "",
        ]
    )
    (directory / "report.md").write_text("\n".join(md))

    fact_html = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in facts
    )
    rows_html = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(v))}</td>"
            for v in (
                row["attempt"],
                row["status"],
                _number(row["score"]),
                _number(row["best_score"]),
                _number(row["cumulative_compute_seconds"]),
            )
        )
        + "</tr>"
        for row in rows
    )
    html_document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>ML development run</title><style>
body{font-family:system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 20px;color:#172033;background:#f8fafc}
h1{font-size:26px}.notice{padding:14px;background:#fff7ed;border-left:4px solid #c2410c}
table{border-collapse:collapse;width:100%;background:white;margin:18px 0;font-size:14px}
th,td{padding:9px 12px;border-bottom:1px solid #e2e8f0;text-align:left;overflow-wrap:anywhere}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
svg{width:100%;background:white;border:1px solid #e2e8f0;border-radius:8px}
p{line-height:1.55}.muted{color:#475569;font-size:14px}
</style></head><body>"""
    html_document += f"<h1>ML development run: {html.escape(summary.run_id)}</h1>"
    html_document += f'<p class="notice">{html.escape(label)}</p><table>{fact_html}</table>'
    attempt_plot = _plot(rows, "attempt", "Attempt", summary.target, summary.metric)
    compute_plot = _plot(
        rows,
        "cumulative_compute_seconds",
        "Cumulative execution seconds",
        summary.target,
        summary.metric,
    )
    (directory / "score_by_attempt.svg").write_text(attempt_plot)
    (directory / "score_by_compute.svg").write_text(compute_plot)
    html_document += '<div class="charts">' + attempt_plot + compute_plot + "</div>"
    html_document += (
        '<p class="muted">Blue: best validation score so far. Dashed orange: declared target. '
    )
    html_document += "Execution time includes worker startup, training, validation, artifact writing, and diagnostics; "
    html_document += (
        "it is not measured GPU kernel utilization. Failed and rejected attempts are included.</p>"
    )
    html_document += "<h2>Attempts</h2><table><tr><th>Attempt</th><th>Status</th><th>Score</th>"
    html_document += f"<th>Best so far</th><th>Cumulative seconds</th></tr>{rows_html}</table>"
    html_document += '<p class="muted">These are validation results. No final test performance or '
    html_document += (
        "self-improvement claim follows from a single run. Workflow discovery and selection "
    )
    html_document += "costs must be reported separately.</p></body></html>"
    (directory / "report.html").write_text(html_document)
