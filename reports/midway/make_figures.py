"""Replot saved study means for the midway report; no ML code or experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
# Supports the course workspace and the standalone repository copy.
PROJECT = next(
    candidate
    for parent in ROOT.parents
    for candidate in (parent, parent / "ml-agent")
    if (candidate / "studies/mlp-covertype/figures/abcd_curves.csv").is_file()
)
OUTPUT = ROOT / "images"
SOURCES = (
    ("mlp-covertype", 168, "compute_minutes", 60.0),
    ("mlp-covertype-600s", 600, "compute_seconds", 1.0),
)
STYLES = {
    "A": {"color": "#1764a5", "linestyle": "-", "label": "A: agent alone"},
    "B": {"color": "#d45b12", "linestyle": "-", "label": "B: + reports"},
    "C": {"color": "#1764a5", "linestyle": "--", "label": "C: + reviewer"},
    "D": {"color": "#d45b12", "linestyle": "--", "label": "D: reports + reviewer"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), sharey=True)
    provenance = []
    for index, (axis, (study, budget, time_column, factor)) in enumerate(
        zip(axes, SOURCES)
    ):
        source = (
            PROJECT / "studies" / study / "figures" / "abcd_curves.csv"
        )
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for arm, style in STYLES.items():
            selected = [row for row in rows if row["arm"] == arm]
            times = [float(row[time_column]) * factor for row in selected]
            means = [
                100 * float(row["mean_best_validation_accuracy"]) for row in selected
            ]
            axis.step(times, means, where="post", linewidth=1.25, **style)
        axis.set_xlim(0, budget)
        axis.set_ylim(80, 96)
        axis.set_yticks([80, 84, 88, 92, 96])
        axis.set_xticks([0, 60, 120, 168] if budget == 168 else [0, 200, 400, 600])
        axis.set_title(f"({'ab'[index]}) {budget}-second budget", pad=5)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.5, zorder=0)
        axis.tick_params(length=3, width=0.6)
        provenance.append(
            {
                "file": source.relative_to(PROJECT).as_posix(),
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
                "rows": len(rows),
                "time_column": time_column,
                "seconds_per_input_unit": factor,
                "score_column": "mean_best_validation_accuracy",
                "score_display_multiplier": 100,
                "visible_x_limits_seconds": [0, budget],
                "visible_y_limits_percent": [80, 96],
            }
        )
    axes[0].set_ylabel("Best validation accuracy (%)")
    figure.supxlabel("Charged compute (seconds)", y=0.035, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        fontsize=7.4,
        columnspacing=1.05,
        handlelength=2.15,
        handletextpad=0.45,
    )
    figure.subplots_adjust(left=0.105, right=0.96, bottom=0.22, top=0.77, wspace=0.14)
    paths = [OUTPUT / "agent_design_curves.pdf", OUTPUT / "agent_design_curves.png"]
    figure.savefig(paths[0], metadata={"Title": "Covertype agent-design comparisons"})
    figure.savefig(paths[1], dpi=300)
    plt.close(figure)
    record = {
        "purpose": "Report visualization of existing study outputs; no experiment reruns",
        "inputs": provenance,
        "aggregation": (
            "No aggregation change. Each CSV already contains the saved mean best-validation "
            "curve over three seeds, using the original study's incumbent interpolation. "
            "Rows and their ordering are preserved. Only units change: minutes to seconds "
            "where needed, and accuracy fractions to percentages."
        ),
        "rendering": (
            "Post-step curves, shared 80-96 percent y axis. The 168-second panel clips the "
            "display at its nominal budget; source rows extending slightly beyond it remain "
            "unchanged. No values are filled, smoothed, resampled, or fitted."
        ),
        "figure_inches": [5.5, 2.25],
        "script": {"file": "make_figures.py", "sha256": sha256(Path(__file__))},
        "outputs": [
            {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in paths
        ],
    }
    (OUTPUT / "figure-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
