"""Study-level results: the agreed A-D figures, behavior analysis, and their data tables.

Every number comes from recorded run artifacts. Each figure has a CSV twin. Works on a
partly finished study and says what is missing. Post-hoc analyses of stored validation
outputs are never shown to agents, so computing them for A/C does not break their
report restriction.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .comparison import compare_design_runs
from .evaluation import progress_rows
from .schemas import Experiment, RunSummary, TaskSpec, digest
from .storage import read_events, read_json, write_json
from .study import run_dir
from .tasks.mlp import BUDGET_CUT_ERROR

ARMS = ("A", "B", "C", "D", "random")
ARM_LABELS = {
    "A": "A: agent alone",
    "B": "B: + error reports",
    "C": "C: + reviewer",
    "D": "D: reports + reviewer",
    "random": "Random search",
}
# The arms are a 2x2, so the encoding is too. Color is the error report (blue none, orange
# present); line and marker are the reviewer (solid circle none, dashed diamond present).
# Blue/orange pass the palette checks under all pairs, which no four-hue set here does.
# Random search is a neutral reference, not another identity hue.
REVIEWER_LINE = (0, (3.5, 1.8))
ARM_STYLE = {
    "A": {"color": "#2a78d6", "marker": "o", "line": "-"},
    "B": {"color": "#eb6834", "marker": "o", "line": "-"},
    "C": {"color": "#2a78d6", "marker": "D", "line": REVIEWER_LINE},
    "D": {"color": "#eb6834", "marker": "D", "line": REVIEWER_LINE},
    "random": {"color": "#898781", "marker": "X", "line": "-"},
}
OUTCOME_STYLE = {
    "improvement": ("Improved best", "#0ca30c"),
    "tie": ("Tied best", "#c3c2b7"),
    "regression": ("Below best", "#ec835a"),
    "budget_cut": ("Cut off by budget", "#898781"),
    "failure": ("Failed or rejected", "#d03b3b"),
}
KNOB_GROUPS = {
    "Optimization": (
        "learning_rate",
        "schedule",
        "batch_size",
        "momentum",
        "optimizer",
        "epochs",
        "warmup_fraction",
        "decay_fraction",
    ),
    "Regularization": ("dropout", "weight_decay", "label_smoothing"),
    "Architecture": (
        "hidden_sizes",
        "cnn_channels",
        "cnn_blocks",
        "activation",
        "normalization",
        "pooling",
        "freeze_encoder_layers",
        "freeze_embeddings",
    ),
    "Input handling": ("augmentation", "feature_preprocessing", "max_length"),
    "Class balance": ("class_weighting", "sampling"),
}
FIELD_SHORT = {
    "learning_rate": "lr",
    "weight_decay": "wd",
    "batch_size": "batch",
    "hidden_sizes": "hidden",
    "cnn_channels": "width",
    "cnn_blocks": "blocks",
    "label_smoothing": "smooth",
    "feature_preprocessing": "features",
    "freeze_encoder_layers": "freeze",
    "class_weighting": "class wt",
    "normalization": "norm",
    "max_length": "len",
    "augmentation": "augment",
    "decay_fraction": "decay",
    "warmup_fraction": "warmup",
}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
CONTRASTS = (
    ("B_vs_A", "B", "A", "Error reports, no reviewer"),
    ("D_vs_C", "D", "C", "Error reports, with reviewer"),
    ("C_vs_A", "C", "A", "Reviewer, no error reports"),
    ("D_vs_B", "D", "B", "Reviewer, with error reports"),
)
METRIC_LABELS = {"accuracy": "accuracy", "f1": "positive-class F1", "macro_f1": "macro-F1"}
WEAK_CLASSES = 10
# CIFAR-100 labels that are also ordinary words in an ML plan ("train loss", "can reduce").
GENERIC_CLASS_WORDS = {"train", "can", "plain", "forest"}


@dataclass
class Run:
    task_id: str
    seed: int
    arm: str
    directory: Path
    task: TaskSpec
    summary: RunSummary | None
    records: list[Experiment]
    events: list[dict]


def load_runs(study_dir: Path) -> tuple[dict, list[Run], list[str]]:
    study = read_json(Path(study_dir) / "study.json")
    runs, missing = [], []
    for task_id in study["task_order"]:
        entry = study["tasks"][task_id]
        for seed in entry["seeds"]:
            for arm in study["arms"]:
                directory = run_dir(study_dir, task_id, seed, arm)
                if not (directory / "state.json").exists():
                    missing.append(f"{task_id}.seed-{seed}.{arm}")
                    continue
                state = read_json(directory / "state.json")
                summary = None
                if (directory / "summary.json").exists():
                    summary = RunSummary.model_validate(read_json(directory / "summary.json"))
                else:
                    missing.append(f"{task_id}.seed-{seed}.{arm} (unfinished)")
                runs.append(
                    Run(
                        task_id,
                        seed,
                        arm,
                        directory,
                        TaskSpec.model_validate(read_json(directory / "task.json")),
                        summary,
                        [Experiment.model_validate(r) for r in state["experiments"]],
                        read_events(directory / "events.jsonl"),
                    )
                )
    return study, runs, missing


def trajectory(run: Run) -> list[dict]:
    """Best score against execution plus diagnostic seconds charged before each attempt."""
    costs = [e["payload"] for e in run.events if e["event"] == "diagnostic_cost"]
    rows = progress_rows(run.records, run.task.metric)
    for row in rows:
        row["cumulative_compute_seconds"] += sum(
            c["seconds"] for c in costs if c["after_attempt"] < row["attempt"]
        )
    return rows


def changed_fields(recipe: dict | None, reference: dict | None) -> list[str]:
    if not recipe or not reference:
        return []
    return sorted(k for k in recipe if recipe[k] != reference.get(k))


def _recalls(trial: Path) -> tuple[np.ndarray, float] | None:
    path = trial / "validation_outputs.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        logits, labels = data["logits"], data["labels"]
    predicted = logits.argmax(1)
    classes = logits.shape[1]
    recalls = np.full(classes, np.nan)
    for c in range(classes):
        members = labels == c
        if members.any():
            recalls[c] = (predicted[members] == c).mean()
    return recalls, float((predicted == labels).mean())


def weak_class_change(reference: Path, current: Path) -> dict | None:
    """Did this change help the classes that were weakest in the model it set out to beat?

    Weak classes tend to rise by regression to the mean, so read this as a comparison
    between arms and against overall change, never as an absolute effect.
    """
    before, after = _recalls(reference), _recalls(current)
    if before is None or after is None:
        return None
    recalls, accuracy = before
    count = min(WEAK_CLASSES, max(1, len(recalls) // 2))
    weakest = np.argsort(np.nan_to_num(recalls, nan=2.0), kind="stable")[:count]
    return {
        "weak_class_recall_change": float(np.nanmean(after[0][weakest] - recalls[weakest])),
        "accuracy_change": after[1] - accuracy,
    }


def decision_rows(run: Run) -> list[dict]:
    """One row per attempt after the baseline: what changed, what happened, what was said."""
    # Not meaningful on binary tasks, where "positive" and "negative" are ordinary prose.
    names = (
        [n for n in run.task.dataset.class_names or [] if n.lower() not in GENERIC_CLASS_WORDS]
        if run.task.dataset.num_classes > 2
        else []
    )
    pattern = (
        re.compile(
            r"\b(" + "|".join(re.escape(n.replace("_", " ")) for n in names) + r")\b",
            re.IGNORECASE,
        )
        if names
        else None
    )
    cited = [e["payload"]["cited"] for e in run.events if e["event"] == "invalid_evidence_citation"]
    rows, best, best_recipe, best_trial, previous, seen = [], None, None, None, None, set()
    for index, record in enumerate(run.records):
        recipe = record.recipe.model_dump(mode="json") if record.recipe else None
        result = record.result
        score = result.metrics.get(run.task.metric) if result.status == "completed" else None
        if index > 0:
            outcome = "budget_cut" if result.error == BUDGET_CUT_ERROR else "failure"
            if score is not None:
                outcome = "improvement" if score > best else "regression" if score < best else "tie"
            changed = changed_fields(recipe, best_recipe)
            text = (
                " ".join(
                    getattr(record.plan, name)
                    for name in ("observation", "hypothesis", "predicted_effect", "experiment")
                )
                if record.plan
                else ""
            )
            row = {
                "task": run.task_id,
                "seed": run.seed,
                "arm": run.arm,
                "decision": index,
                "trial": record.trial_id,
                "status": result.status,
                "outcome": outcome,
                "score": score,
                "gain_over_best": score - best if score is not None and best is not None else None,
                "compute_seconds": result.compute_seconds,
                "fields_changed_from_best": ";".join(changed),
                "number_changed_from_best": len(changed),
                "fields_changed_from_previous": ";".join(changed_fields(recipe, previous)),
                "exact_repeat": bool(recipe) and digest(recipe) in seen,
                # This recipe is at least as close to the best so far as to the last attempt.
                "built_on_best": bool(recipe)
                and len(changed) <= len(changed_fields(recipe, previous)),
                "plan_mentions_class_name": bool(pattern.search(text.replace("_", " ")))
                if pattern
                else None,
                "error": result.error,
                "observation": record.plan.observation if record.plan else "",
                "hypothesis": record.plan.hypothesis if record.plan else "",
                "predicted_effect": record.plan.predicted_effect if record.plan else "",
                "experiment": record.plan.experiment if record.plan else "",
                "review_prediction_supported": record.review.prediction_supported
                if record.review
                else None,
                "review_next_step": record.review.next_step if record.review else "",
            }
            for group, fields in KNOB_GROUPS.items():
                row[f"changed:{group}"] = any(f in changed for f in fields)
            change = (
                weak_class_change(
                    run.directory / "trials" / best_trial,
                    run.directory / "trials" / record.trial_id,
                )
                if score is not None and best_trial
                else None
            )
            row.update(change or {"weak_class_recall_change": None, "accuracy_change": None})
            rows.append(row)
        if recipe:
            seen.add(digest(recipe))
            previous = recipe
        if score is not None and (best is None or score > best):
            best, best_recipe, best_trial = score, recipe, record.trial_id
    if rows:
        rows[0]["invalid_citations_in_run"] = sum(len(c) for c in cited)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.75,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.75,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
        }
    )


def _axes(plt, count: int, *, rows: int = 1, height: float = 3.3, sharex=False):
    """Small multiples, one panel per dataset; different metrics are never pooled."""
    columns = min(count, 4)
    figure, axes = plt.subplots(
        rows * max(1, int(np.ceil(count / columns))),
        columns,
        # Wide enough for the legend and note, even with a single panel.
        figsize=(max(3.45 * columns + 0.3, 8.5), height * rows + 0.75),
        squeeze=False,
        sharex=sharex,
    )
    return figure, axes


def _arm_legend(figure, arms, *, y: float = 1.0) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            color=ARM_STYLE[a]["color"],
            marker=ARM_STYLE[a]["marker"],
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
            linewidth=1.5,
            linestyle=ARM_STYLE[a]["line"],
            label=ARM_LABELS[a],
        )
        for a in arms
    ]
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.008, y),
        ncol=max(1, len(handles)),
        handlelength=2.6,
        columnspacing=1.6,
        labelcolor=INK_2,
    )


def _note(figure, text: str) -> None:
    figure.text(0.008, 0.006, text, ha="left", va="bottom", color=INK_2, fontsize=8)


def _time_unit(seconds: float) -> tuple[float, str]:
    if seconds >= 2 * 3600:
        return 3600.0, "hours"
    if seconds >= 120:
        return 60.0, "minutes"
    return 1.0, "seconds"


def _save(figure, directory: Path, name: str) -> list[str]:
    outputs = []
    for extension in ("svg", "png"):
        path = directory / f"{name}.{extension}"
        figure.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
        outputs.append(str(path))
    return outputs


def _panel_title(run: Run) -> str:
    return f"{run.task_id}: {METRIC_LABELS.get(run.task.metric, run.task.metric)}"


def _mean_step(traces: list[tuple[list[float], list[float]]]):
    """Mean best-so-far over seeds; a finished run holds its final best.

    Starts once every seed has a score, so seeds joining cannot make a best-so-far fall.
    """
    start = max(xs[0] for xs, _ in traces)
    times = sorted({t for xs, _ in traces for t in xs if t >= start})
    means = []
    for moment in times:
        values = []
        for xs, ys in traces:
            position = np.searchsorted(xs, moment, side="right") - 1
            if position >= 0:
                values.append(ys[position])
        means.append(float(np.mean(values)))
    return times, means


def figure_quality(
    plt, runs: list[Run], output: Path, *, x_axis: str
) -> tuple[list[str], list[dict]]:
    """Main result: best validation score so far, one line per arm, one panel per dataset."""
    tasks = list(dict.fromkeys(r.task_id for r in runs))
    figure, axes = _axes(plt, len(tasks))
    table = []
    arms_seen = []
    for axis, task_id in zip(axes.flat, tasks):
        members = [r for r in runs if r.task_id == task_id]
        spans = [row["cumulative_compute_seconds"] for r in members for row in trajectory(r)]
        scale, unit = _time_unit(max(spans, default=1)) if x_axis == "compute" else (1.0, "")
        low, high = [], []
        for arm in ARMS:
            traces = []
            for run in [r for r in members if r.arm == arm]:
                rows = [row for row in trajectory(run) if row["best_score"] is not None]
                if not rows:
                    continue
                xs = [
                    row["cumulative_compute_seconds"] / scale
                    if x_axis == "compute"
                    else float(row["attempt"])
                    for row in rows
                ]
                ys = [row["best_score"] for row in rows]
                traces.append((xs, ys))
                for row in rows:
                    table.append(
                        {
                            "task": task_id,
                            "seed": run.seed,
                            "arm": arm,
                            "attempt": row["attempt"],
                            "cumulative_compute_seconds": row["cumulative_compute_seconds"],
                            "score": row["score"],
                            "best_score": row["best_score"],
                        }
                    )
            if not traces:
                continue
            if arm not in arms_seen:
                arms_seen.append(arm)
            style = ARM_STYLE[arm]
            for xs, ys in traces:
                axis.step(xs, ys, where="post", color=style["color"], linewidth=0.8, alpha=0.28)
            times, means = _mean_step(traces)
            axis.step(
                times,
                means,
                where="post",
                color=style["color"],
                linewidth=1.5,
                linestyle=style["line"],
                solid_capstyle="round",
            )
            # Mean steps fall at every seed's timing on the compute axis, so only the end
            # is marked there; on the experiment axis each point is one experiment.
            marked = (times, means) if x_axis == "experiments" else (times[-1:], means[-1:])
            axis.plot(
                *marked,
                linestyle="none",
                marker=style["marker"],
                markersize=5.5,
                color=style["color"],
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
            )
            low.append(min(min(ys) for _, ys in traces))
            high.append(max(max(ys) for _, ys in traces))
        target = members[0].task.target
        if low:
            margin = max(0.002, (max(high) - min(low)) * 0.12)
            bottom, top = min(low) - margin, max(high) + margin
            # A target near the data is drawn; one far outside would flatten the scores
            # that matter, so its distance is stated in the panel instead.
            if target < 1.0 and bottom - 3 * margin <= target <= top + 3 * margin:
                axis.axhline(target, color=INK_2, linewidth=0.75)
                axis.annotate(
                    "target",
                    (1, target),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    color=INK_2,
                    fontsize=8,
                    annotation_clip=False,
                )
                bottom, top = min(bottom, target - margin), max(top, target + margin)
            elif target < 1.0:
                axis.annotate(
                    f"target {target:.3f} is off this scale",
                    (1, 0.02),
                    xycoords="axes fraction",
                    ha="right",
                    color=INK_2,
                    fontsize=8,
                )
            axis.set_ylim(bottom, top)
        axis.set_title(_panel_title(members[0]))
        axis.set_ylabel("Best validation score so far")
        axis.set_xlabel(
            f"Cumulative training + analysis {unit}" if x_axis == "compute" else "Experiment number"
        )
        if x_axis == "experiments":
            from matplotlib.ticker import MaxNLocator

            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    for axis in axes.flat[len(tasks) :]:
        axis.set_visible(False)
    figure.tight_layout(rect=(0, 0.05, 1, 0.92))
    _arm_legend(figure, arms_seen)
    if x_axis == "compute":
        _note(figure, "A flat end means that run has used its budget.")
    name = "fig1_quality_vs_compute" if x_axis == "compute" else "fig2_quality_vs_experiments"
    outputs = _save(figure, output, name)
    plt.close(figure)
    return outputs, table


def figure_cost_to_target(plt, runs: list[Run], output: Path) -> tuple[list[str], list[dict]]:
    """Experiments and compute to reach the fixed target; misses stay visible as hollow marks."""
    finished = [r for r in runs if r.summary and r.task.target < 1.0]
    tasks = list(dict.fromkeys(r.task_id for r in finished))
    table: list[dict] = []
    if not tasks:
        return [], table
    figure, axes = _axes(plt, len(tasks), rows=2, height=2.6)
    arms_seen = []
    for column, task_id in enumerate(tasks):
        members = [r for r in finished if r.task_id == task_id]
        limits = members[0].task.limits
        scale, unit = _time_unit(limits.total_compute_seconds)
        panels = (
            (axes[0][column], "trials_to_target", None, 1.0, "Experiments to target"),
            (
                axes[1][column],
                "compute_to_target",
                limits.total_compute_seconds,
                scale,
                f"{unit.capitalize()} to target",
            ),
        )
        for axis, field, cap, divide, label in panels:
            arms_here = [a for a in ARMS if any(r.arm == a for r in members)]
            highest = 1.0
            for position, arm in enumerate(arms_here):
                style = ARM_STYLE[arm]
                group = [r for r in members if r.arm == arm]
                values = []
                for offset, run in enumerate(group):
                    reached = getattr(run.summary, field)
                    # A miss is drawn at what that run actually spent: most runs can never
                    # reach the attempt cap, so it would distort the scale and the mean.
                    missed = cap if cap is not None else run.summary.total_trials
                    value = (reached if reached is not None else missed) / divide
                    values.append(value)
                    highest = max(highest, value)
                    jitter = (offset - (len(group) - 1) / 2) * 0.09
                    axis.scatter(
                        position + jitter,
                        value,
                        s=34,
                        marker=style["marker"],
                        facecolor=style["color"] if reached is not None else SURFACE,
                        edgecolor=style["color"] if reached is None else SURFACE,
                        linewidth=1.1 if reached is None else 0.6,
                        zorder=3,
                    )
                    if field == "trials_to_target":
                        table.append(
                            {
                                "task": task_id,
                                "seed": run.seed,
                                "arm": arm,
                                "target": run.task.target,
                                "reached": run.summary.reached_target,
                                "experiments_to_target": run.summary.trials_to_target,
                                "compute_seconds_to_target": run.summary.compute_to_target,
                            }
                        )
                # On the experiments row a miss sits at what that run spent, so averaging it in
                # would flatter an arm that ran out early. Its line covers only runs that hit.
                line = (
                    values
                    if cap is not None
                    else [v for v, r in zip(values, group) if getattr(r.summary, field) is not None]
                )
                if line:
                    axis.hlines(
                        np.mean(line),
                        position - 0.28,
                        position + 0.28,
                        color=INK_2,
                        linewidth=1.5,
                        zorder=2,
                    )
                reached_count = sum(getattr(r.summary, field) is not None for r in group)
                axis.annotate(
                    f"{reached_count}/{len(group)}",
                    (position, 1),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=INK_2,
                )
                if arm not in arms_seen:
                    arms_seen.append(arm)
            if cap is not None:
                axis.axhline(cap / divide, color=AXIS, linewidth=0.75)
            axis.set_xticks(range(len(arms_here)))
            axis.set_xticklabels(["Random" if a == "random" else a for a in arms_here])
            axis.set_xlim(-0.6, len(arms_here) - 0.4)
            axis.set_ylim(0, (cap / divide if cap is not None else highest) * 1.08)
            axis.set_ylabel(label)
            axis.grid(axis="x", visible=False)
        axes[0][column].set_title(_panel_title(members[0]), pad=14)
    figure.tight_layout(rect=(0, 0.04, 1, 0.9))
    _arm_legend(figure, arms_seen)
    _note(
        figure,
        "Line: mean (experiments row: runs reaching the target only). Hollow: missed, at what that run spent. n/n: runs reaching target.",
    )
    outputs = _save(figure, output, "fig3_cost_to_target")
    plt.close(figure)
    return outputs, table


def figure_outcomes(plt, decisions: list[dict], output: Path) -> tuple[list[str], list[dict]]:
    """What each arm's decisions produced, relative to its own best so far."""
    tasks = list(dict.fromkeys(d["task"] for d in decisions))
    table: list[dict] = []
    if not tasks:
        return [], table
    from matplotlib.patches import Patch

    figure, axes = _axes(plt, len(tasks), height=3.0)
    for axis, task_id in zip(axes.flat, tasks):
        arms_here = [
            a for a in ARMS if any(d["task"] == task_id and d["arm"] == a for d in decisions)
        ]
        for position, arm in enumerate(arms_here):
            group = [d for d in decisions if d["task"] == task_id and d["arm"] == arm]
            bottom = 0
            row = {"task": task_id, "arm": arm, "decisions": len(group)}
            for outcome, (_, color) in OUTCOME_STYLE.items():
                count = sum(d["outcome"] == outcome for d in group)
                share = count / len(group)
                row[outcome] = count
                if count:
                    # Touching fills are separated by a surface gap, not a border.
                    axis.bar(
                        position,
                        share,
                        bottom=bottom,
                        width=0.42,
                        color=color,
                        edgecolor=SURFACE,
                        linewidth=1.5,
                    )
                bottom += share
            row["exact_repeats"] = sum(bool(d["exact_repeat"]) for d in group)
            table.append(row)
        axis.set_xticks(range(len(arms_here)))
        axis.set_xticklabels(["Random" if a == "random" else a for a in arms_here])
        axis.set_ylim(0, 1)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
        axis.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        axis.set_ylabel("Share of decisions")
        axis.grid(axis="x", visible=False)
        axis.set_title(task_id)
    for axis in axes.flat[len(tasks) :]:
        axis.set_visible(False)
    figure.tight_layout(rect=(0, 0.04, 1, 0.9))
    figure.legend(
        handles=[Patch(color=c, label=label) for label, c in OUTCOME_STYLE.values()],
        loc="upper left",
        bbox_to_anchor=(0.008, 1.0),
        ncol=5,
        labelcolor=INK_2,
        handlelength=1.2,
    )
    outputs = _save(figure, output, "fig4_experiment_outcomes")
    plt.close(figure)
    return outputs, table


def figure_behavior(plt, decisions: list[dict], output: Path) -> tuple[list[str], list[dict]]:
    """What each arm tries, and whether its changes help the classes that were weakest."""
    tasks = list(dict.fromkeys(d["task"] for d in decisions))
    table: list[dict] = []
    if not tasks:
        return [], table
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list(
        "share", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
    )
    figure, axes = _axes(plt, len(tasks), rows=2, height=2.8)
    groups = list(KNOB_GROUPS)
    group_ticks = {
        "Optimization": "Optim-\nization",
        "Regularization": "Regular-\nization",
        "Architecture": "Archi-\ntecture",
        "Input handling": "Input\nhandling",
        "Class balance": "Class\nbalance",
    }
    for column, task_id in enumerate(tasks):
        members = [d for d in decisions if d["task"] == task_id]
        arms_here = [a for a in ARMS if any(d["arm"] == a for d in members)]
        grid = np.zeros((len(arms_here), len(groups)))
        for i, arm in enumerate(arms_here):
            group = [d for d in members if d["arm"] == arm]
            for j, name in enumerate(groups):
                grid[i, j] = np.mean([bool(d[f"changed:{name}"]) for d in group])
            changed = [d["number_changed_from_best"] for d in group]
            weak = [
                d["weak_class_recall_change"]
                for d in group
                if d["weak_class_recall_change"] is not None
            ]
            overall = [d["accuracy_change"] for d in group if d["accuracy_change"] is not None]
            table.append(
                {
                    "task": task_id,
                    "arm": arm,
                    "decisions": len(group),
                    "mean_fields_changed_from_best": float(np.mean(changed)),
                    **{f"share_changing:{n}": grid[i, j] for j, n in enumerate(groups)},
                    "exact_repeat_rate": float(np.mean([bool(d["exact_repeat"]) for d in group])),
                    "failure_rate": float(np.mean([d["outcome"] == "failure" for d in group])),
                    "built_on_best_rate": float(np.mean([bool(d["built_on_best"]) for d in group])),
                    "plan_mentions_class_rate": (
                        float(np.mean([d["plan_mentions_class_name"] for d in group]))
                        if group[0]["plan_mentions_class_name"] is not None
                        else None
                    ),
                    "mean_weak_class_recall_change": float(np.mean(weak)) if weak else None,
                    "mean_accuracy_change": float(np.mean(overall)) if overall else None,
                }
            )
        axis = axes[0][column]
        axis.imshow(grid, cmap=ramp, vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(range(len(groups)))
        axis.set_xticklabels([group_ticks[g] for g in groups], fontsize=7.5, color=INK_2)
        axis.set_yticks(range(len(arms_here)))
        axis.set_yticklabels(["Random" if a == "random" else a for a in arms_here])
        axis.grid(False)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                axis.text(
                    j,
                    i,
                    f"{grid[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#ffffff" if grid[i, j] > 0.45 else INK,
                )
        axis.set_title(task_id)
        axis = axes[1][column]
        for position, arm in enumerate(arms_here):
            style = ARM_STYLE[arm]
            values = [
                d["weak_class_recall_change"]
                for d in members
                if d["arm"] == arm and d["weak_class_recall_change"] is not None
            ]
            if not values:
                continue
            spread = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else [0]
            axis.scatter(
                position + np.asarray(spread),
                values,
                s=22,
                marker=style["marker"],
                color=style["color"],
                edgecolor=SURFACE,
                linewidth=0.5,
                zorder=3,
            )
            axis.hlines(
                np.mean(values),
                position - 0.26,
                position + 0.26,
                color=INK_2,
                linewidth=1.5,
                zorder=4,
            )
        axis.axhline(0, color=AXIS, linewidth=0.75)
        axis.set_xticks(range(len(arms_here)))
        axis.set_xticklabels(["Random" if a == "random" else a for a in arms_here])
        axis.set_xlim(-0.6, len(arms_here) - 0.4)
        axis.set_ylabel("Weakest-class recall change")
        axis.grid(axis="x", visible=False)
        axis.set_title(task_id)
    # Row headers sit in their own bands, clear of the panel titles beneath them.
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.4)
    top_row, bottom_row = axes[0][0].get_position(), axes[1][0].get_position()
    for y, text in (
        (
            top_row.y1 + 0.052,
            "What each arm tries: share of decisions changing each kind of setting",
        ),
        (
            bottom_row.y1 + 0.052,
            (
                "Whether it helps the weakest classes: each decision versus its best-so-far "
                "model (line: mean; compare arms, weak classes rise by chance too)"
            ),
        ),
    ):
        figure.text(0.008, y, text, color=INK, fontsize=9.5, fontweight="bold")
    outputs = _save(figure, output, "fig5_agent_behavior")
    plt.close(figure)
    return outputs, table


def figure_parameters(plt, runs: list[Run], output: Path) -> tuple[list[str], list[dict]]:
    """Score against model size for every completed experiment: higher and further left wins."""
    tasks = list(dict.fromkeys(r.task_id for r in runs))
    table: list[dict] = []
    figure, axes = _axes(plt, len(tasks))
    arms_seen = []
    for axis, task_id in zip(axes.flat, tasks):
        members = [r for r in runs if r.task_id == task_id]
        for arm in ARMS:
            xs, ys = [], []
            for run in [r for r in members if r.arm == arm]:
                for record in run.records:
                    result = record.result
                    if result.status == "completed" and result.parameter_count:
                        xs.append(result.parameter_count)
                        ys.append(result.metrics[run.task.metric])
                        table.append(
                            {
                                "task": task_id,
                                "seed": run.seed,
                                "arm": arm,
                                "trial": record.trial_id,
                                "parameters": result.parameter_count,
                                "score": result.metrics[run.task.metric],
                                "compute_seconds": result.compute_seconds,
                            }
                        )
            if not xs:
                continue
            if arm not in arms_seen:
                arms_seen.append(arm)
            style = ARM_STYLE[arm]
            axis.scatter(
                xs,
                ys,
                s=30,
                marker=style["marker"],
                color=style["color"],
                edgecolor=SURFACE,
                linewidth=0.5,
                zorder=3,
            )
        axis.set_xscale("log")
        # One very poor experiment would otherwise squash every useful difference at the top.
        scores = [row["score"] for row in table if row["task"] == task_id]
        if scores:
            top, floor = max(scores), max(scores) - 0.06
            axis.set_ylim(max(floor, min(scores)) - 0.002, top + 0.004)
            hidden = sum(score < floor for score in scores)
            if hidden:
                axis.annotate(
                    f"{hidden} experiments below {floor:.2f} are off this scale",
                    (0.01, 0.03),
                    xycoords="axes fraction",
                    color=INK_2,
                    fontsize=8,
                )
        axis.set_title(_panel_title(members[0]))
        axis.set_xlabel("Model parameters")
        axis.set_ylabel("Validation score")
    for axis in axes.flat[len(tasks) :]:
        axis.set_visible(False)
    figure.tight_layout(rect=(0, 0.05, 1, 0.92))
    _arm_legend(figure, arms_seen)
    _note(figure, "Up and to the left is better: a higher score from a smaller model.")
    outputs = _save(figure, output, "fig6_score_vs_parameters")
    plt.close(figure)
    return outputs, table


def figure_timeline(
    plt, runs: list[Run], decisions: list[dict], task_id: str, seed: int, output: Path
):
    """One complete case: what each arm tried, in order, and how each attempt turned out."""
    members = [r for r in runs if r.task_id == task_id and r.seed == seed]
    if not members:
        return []
    from matplotlib.patches import FancyBboxPatch, Patch

    arms_here = [a for a in ARMS if any(r.arm == a for r in members)]
    columns = max(len(r.records) for r in members)
    if not columns:
        return []
    figure, axis = plt.subplots(figsize=(1.75 * columns + 1.1, 0.72 * len(arms_here) + 1.25))
    for row, arm in enumerate(arms_here):
        run = next(r for r in members if r.arm == arm)
        first = run.records[0].result.metrics.get(run.task.metric) if run.records else None
        for column in range(columns):
            if column == 0:
                color, lines = "#f0efec", ["baseline", f"{first:.4f}" if first is not None else "—"]
                ink = INK
            else:
                match = [
                    d
                    for d in decisions
                    if d["task"] == task_id
                    and d["seed"] == seed
                    and d["arm"] == arm
                    and d["decision"] == column
                ]
                if not match:
                    continue
                decision = match[0]
                color = OUTCOME_STYLE[decision["outcome"]][1]
                changed = [
                    FIELD_SHORT.get(f, f)
                    for f in decision["fields_changed_from_best"].split(";")
                    if f
                ]
                label = ", ".join(changed[:2]) + (
                    f" +{len(changed) - 2}" if len(changed) > 2 else ""
                )
                if decision["score"] is None:
                    ended = "cut off" if decision["outcome"] == "budget_cut" else decision["status"]
                    lines = [label or "no recipe", ended]
                else:
                    lines = [label or "no change", f"{decision['score']:.4f}"]
                ink = "#ffffff" if decision["outcome"] in ("failure", "improvement") else INK
            axis.add_patch(
                FancyBboxPatch(
                    (column + 0.04, row + 0.06),
                    0.92,
                    0.88,
                    boxstyle="round,pad=0,rounding_size=0.05",
                    facecolor=color,
                    edgecolor="none",
                )
            )
            axis.text(
                column + 0.5,
                row + 0.37,
                lines[0],
                ha="center",
                va="center",
                fontsize=7.5,
                color=ink,
            )
            axis.text(
                column + 0.5,
                row + 0.68,
                lines[1],
                ha="center",
                va="center",
                fontsize=8,
                color=ink,
                fontweight="bold",
            )
    axis.set_xlim(0, columns)
    axis.set_ylim(len(arms_here), 0)
    axis.set_xticks([c + 0.5 for c in range(columns)])
    axis.set_xticklabels(["Experiment 1"] + [str(c + 1) for c in range(1, columns)])
    axis.set_yticks([r + 0.5 for r in range(len(arms_here))])
    axis.set_yticklabels([ARM_LABELS[a] for a in arms_here])
    axis.grid(False)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    run = members[0]
    axis.set_title(
        f"{task_id}, seed {seed}: settings changed from best so far, and resulting "
        f"{METRIC_LABELS.get(run.task.metric, run.task.metric)}",
        pad=30,
    )
    axis.legend(
        handles=[Patch(color=c, label=label) for label, c in OUTCOME_STYLE.values()],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=5,
        labelcolor=INK_2,
        handlelength=1.2,
        borderaxespad=0.2,
    )
    outputs = _save(figure, output, f"timeline_{task_id}_seed-{seed}")
    plt.close(figure)
    return outputs


def contrast_rows(runs: list[Run]) -> tuple[list[dict], dict]:
    """Paired by task and seed. Descriptive differences with per-seed values shown."""
    finished = {(r.task_id, r.seed, r.arm): r for r in runs if r.summary}
    tasks = list(dict.fromkeys(r.task_id for r in runs))
    rows, checks = [], {}
    for task_id in tasks:
        seeds = sorted({r.seed for r in runs if r.task_id == task_id})
        for seed in seeds:
            cell = {a: finished.get((task_id, seed, a)) for a in "ABCD"}
            if all(cell.values()):
                try:
                    compare_design_runs({a: cell[a].summary for a in "ABCD"}, cell["A"].task)
                    checks[f"{task_id}.seed-{seed}"] = "matched"
                except ValueError as exc:
                    checks[f"{task_id}.seed-{seed}"] = str(exc)
        for name, after, before, question in CONTRASTS:
            pairs = [
                (finished[(task_id, s, after)], finished[(task_id, s, before)])
                for s in seeds
                if (task_id, s, after) in finished and (task_id, s, before) in finished
            ]
            if not pairs:
                continue
            gains = [a.summary.best_score - b.summary.best_score for a, b in pairs]
            limit = pairs[0][0].task.limits

            def capped(run, field, cap):
                value = getattr(run.summary, field)
                return value if value is not None else cap

            compute = [
                capped(a, "compute_to_target", limit.total_compute_seconds)
                - capped(b, "compute_to_target", limit.total_compute_seconds)
                for a, b in pairs
            ]
            attempts = [
                a.summary.trials_to_target - b.summary.trials_to_target
                for a, b in pairs
                if a.summary.trials_to_target is not None and b.summary.trials_to_target is not None
            ]
            rows.append(
                {
                    "task": task_id,
                    "contrast": name,
                    "question": question,
                    "seeds": len(pairs),
                    "best_score_difference_mean": float(np.mean(gains)),
                    "best_score_difference_by_seed": ";".join(f"{g:+.5f}" for g in gains),
                    "seeds_favoring_change": sum(g > 0 for g in gains),
                    "capped_compute_to_target_difference_mean": float(np.mean(compute)),
                    "experiments_to_target_difference_mean_where_both_reached": (
                        float(np.mean(attempts)) if attempts else None
                    ),
                    "best_model_parameters_difference_mean": float(
                        np.mean(
                            [
                                (_best_model_parameters(a) or 0) - (_best_model_parameters(b) or 0)
                                for a, b in pairs
                            ]
                        )
                    ),
                    # A miss and a hit on the last attempt share the same capped value.
                    "seeds_reaching_target_with_change": sum(
                        a.summary.reached_target for a, _ in pairs
                    ),
                    "seeds_reaching_target_without_change": sum(
                        b.summary.reached_target for _, b in pairs
                    ),
                    # Those seeds cross at the first attempt in every arm: their cost contrast
                    # is zero by construction, not evidence of no effect.
                    "seeds_with_baseline_already_at_target": sum(
                        a.summary.trials_to_target == 1 for a, _ in pairs
                    ),
                }
            )
        complete = [s for s in seeds if all((task_id, s, a) in finished for a in "ABCD")]
        if complete:
            interaction = [
                (
                    finished[(task_id, s, "D")].summary.best_score
                    - finished[(task_id, s, "B")].summary.best_score
                )
                - (
                    finished[(task_id, s, "C")].summary.best_score
                    - finished[(task_id, s, "A")].summary.best_score
                )
                for s in complete
            ]
            rows.append(
                {
                    "task": task_id,
                    "contrast": "interaction",
                    "question": "Does the reviewer help more when it has error reports? (D-B)-(C-A)",
                    "seeds": len(complete),
                    "best_score_difference_mean": float(np.mean(interaction)),
                    "best_score_difference_by_seed": ";".join(f"{g:+.5f}" for g in interaction),
                }
            )
    return rows, checks


def _best_model_parameters(run: Run) -> int | None:
    """Size of the validation-selected model; ties go to the earliest, as in final scoring."""
    best, size = None, None
    for record in run.records:
        if record.result.status == "completed":
            score = record.result.metrics[run.task.metric]
            if best is None or score > best:
                best, size = score, record.result.parameter_count
    return size


def _count(run: Run, event: str) -> int:
    return sum(e["event"] == event for e in run.events)


def write_figures(study_dir: Path, out: Path | None = None) -> dict:
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rsi-ml-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    study_dir = Path(study_dir).resolve()
    output = Path(out).resolve() if out else study_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    _style(plt)
    study, started, missing = load_runs(study_dir)
    # Figures show finished runs only: holding a partial run flat would read as a plateau.
    runs = [run for run in started if run.summary]
    if not runs:
        return {"figures": [], "missing": missing, "note": "No agent run has finished."}
    decisions = [row for run in runs for row in decision_rows(run)]
    figures = []
    for x_axis, table_name in (("compute", "quality_vs_compute.csv"), ("experiments", None)):
        paths, table = figure_quality(plt, runs, output, x_axis=x_axis)
        figures += paths
        if table_name:
            _write_csv(output / table_name, table)
    paths, table = figure_cost_to_target(plt, runs, output)
    figures += paths
    _write_csv(output / "cost_to_target.csv", table)
    paths, table = figure_outcomes(plt, decisions, output)
    figures += paths
    _write_csv(output / "experiment_outcomes.csv", table)
    paths, table = figure_behavior(plt, decisions, output)
    figures += paths
    _write_csv(output / "agent_behavior.csv", table)
    paths, table = figure_parameters(plt, runs, output)
    figures += paths
    _write_csv(output / "score_vs_parameters.csv", table)
    for task_id in study["task_order"]:
        for seed in study["tasks"][task_id]["seeds"]:
            figures += figure_timeline(plt, runs, decisions, task_id, seed, output)
    _write_csv(output / "decisions.csv", decisions)
    contrasts, checks = contrast_rows(runs)
    _write_csv(output / "contrasts.csv", contrasts)
    costs = [
        {
            "task": r.task_id,
            "seed": r.seed,
            "arm": r.arm,
            "stop_reason": r.summary.stop_reason,
            "trials": r.summary.total_trials,
            "best_score": r.summary.best_score,
            "compute_seconds": r.summary.total_compute_seconds,
            "diagnostic_seconds": r.summary.diagnostic_seconds,
            "api_calls": r.summary.usage.calls,
            "api_transient_retries": r.summary.usage.transient_retries,
            "input_tokens": r.summary.usage.input_tokens,
            "output_tokens": r.summary.usage.output_tokens,
            "api_cost_usd": r.summary.usage.cost_usd,
            # Whether each arm actually received its treatments. A lost reviewer or report
            # would otherwise read as a null effect.
            "best_model_parameters": _best_model_parameters(r),
            "recipe_repairs": _count(r, "recipe_repair_requested"),
            "reviews_recorded": sum(record.review is not None for record in r.records),
            "review_failures": _count(r, "review_failed"),
            "reports_delivered": _count(r, "diagnostic_report"),
            "report_failures": _count(r, "diagnostics_failed"),
            "planning_failures": sum(
                record.result.status == "rejected" and record.recipe is None for record in r.records
            ),
            "agent_stops_ignored": _count(r, "agent_stop_ignored"),
            "recoveries": _count(r, "interrupted_operation_recovered"),
        }
        for r in runs
        if r.summary
    ]
    _write_csv(output / "runs.csv", costs)
    final = study_dir / "final_test.json"
    if final.exists():
        _write_csv(output / "final_test.csv", read_json(final)["scored"])
    receipt = {
        "study_id": study["study_id"],
        "figures": figures,
        "tables": sorted(p.name for p in output.glob("*.csv")),
        "matched_cases": checks,
        "missing_or_unfinished": missing,
        "interpretation": "Descriptive paired comparisons on validation scores; test scores are separate.",
    }
    write_json(output / "figures.json", receipt)
    return receipt
