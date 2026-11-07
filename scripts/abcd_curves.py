"""The one plot for an architecture: four A-D curves of the task metric against compute.

X is cumulative training + analysis time. Y is the best validation score so far, averaged
over seeds. Reads the table a finished study already wrote, so it trains nothing:

    python scripts/abcd_curves.py studies/mlp-covertype/figures

The note states the highest curve and its gap as measured. It makes no claim that the gap
is larger than noise. Random search is left out unless --with-random asks for it, which writes
separate files so the A-D plot stays as it is.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from rsi_ml.figures import (
    ARM_LABELS,
    ARM_STYLE,
    INK,
    INK_2,
    SURFACE,
    _arm_legend,
    _mean_step,
    _save,
    _style,
    _time_unit,
)

ARMS = ("A", "B", "C", "D")
MIN_LABEL_GAP = 0.05  # Share of the axis height between end labels, so close curves stay readable.


def load_traces(
    table: Path, arms: tuple[str, ...] = ARMS
) -> tuple[str, dict[str, list[tuple[list[float], list[float]]]]]:
    with table.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["arm"] in arms and row["best_score"]]
    tasks = {row["task"] for row in rows}
    if len(tasks) != 1:
        raise ValueError(f"Expected one task in {table}, found {sorted(tasks)}.")
    traces: dict[str, list[tuple[list[float], list[float]]]] = {}
    for arm in arms:
        for seed in sorted({row["seed"] for row in rows if row["arm"] == arm}, key=int):
            mine = sorted(
                (row for row in rows if row["arm"] == arm and row["seed"] == seed),
                key=lambda row: int(row["attempt"]),
            )
            traces.setdefault(arm, []).append(
                (
                    [float(row["cumulative_compute_seconds"]) for row in mine],
                    [float(row["best_score"]) for row in mine],
                )
            )
    return tasks.pop(), traces


def spread_labels(ends: dict[str, float], low: float, high: float) -> dict[str, float]:
    """Push crowded end labels apart just enough to read, keeping their order.

    Each crowded group stays centered on its curves, so a label does not land beside the end
    of a different curve.
    """
    gap = MIN_LABEL_GAP * (high - low)

    def positions(group: list[str]) -> list[float]:
        first = sum(ends[arm] for arm in group) / len(group) - gap * (len(group) - 1) / 2
        return [first + gap * index for index in range(len(group))]

    groups: list[list[str]] = []
    for arm in sorted(ends, key=lambda name: (ends[name], name)):
        group = [arm]
        while groups and positions(groups[-1])[-1] + gap > positions(group)[0]:
            group = groups.pop() + group
        groups.append(group)
    placed = {arm: spot for group in groups for arm, spot in zip(group, positions(group))}
    overflow = max(0.0, max(placed.values()) - high)
    return {arm: spot - overflow for arm, spot in placed.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", type=Path, help="A study's figures directory.")
    parser.add_argument("--out", type=Path, help="Where to write; defaults to the same directory.")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument(
        "--x-max",
        type=float,
        help="Draw X in seconds up to this budget; a finished run holds its best score to there.",
    )
    parser.add_argument(
        "--with-random",
        action="store_true",
        help="Add random search as a fifth curve and write separate files.",
    )
    parser.add_argument(
        "--random-note",
        help="Caveat printed after random search's score; use with --with-random.",
    )
    args = parser.parse_args()
    if args.random_note and not args.with_random:
        parser.error("--random-note needs --with-random.")
    output = args.out or args.figures
    output.mkdir(parents=True, exist_ok=True)
    arms = (*ARMS, "random") if args.with_random else ARMS
    stem = "abcd_random_curves" if args.with_random else "abcd_curves"

    task, traces = load_traces(args.figures / "quality_vs_compute.csv", arms)
    missing = [arm for arm in arms if arm not in traces]
    if missing:
        raise ValueError(f"No results yet for: {missing}.")
    seeds = {len(traces[arm]) for arm in arms}
    if len(seeds) != 1:
        raise ValueError(f"Designs have different seed counts: {sorted(seeds)}.")

    longest = max(xs[-1] for arm in arms for xs, _ in traces[arm])
    scale, unit = (1.0, "seconds") if args.x_max else _time_unit(longest)
    curves = {
        arm: _mean_step([([x / scale for x in xs], ys) for xs, ys in traces[arm]]) for arm in arms
    }
    if args.x_max:
        # A last attempt can pass the budget by a fraction of a second as it is cut off.
        curves = {
            arm: ([*(min(t, args.x_max) for t in times), args.x_max], [*means, means[-1]])
            for arm, (times, means) in curves.items()
        }

    _style(plt)
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for arm, (times, means) in curves.items():
        style = ARM_STYLE[arm]
        axis.step(
            times,
            means,
            where="post",
            color=style["color"],
            linewidth=1.8,
            linestyle=style["line"],
            solid_capstyle="round",
        )
        axis.plot(
            times[-1],
            means[-1],
            linestyle="none",
            marker=style["marker"],
            markersize=5.5,
            color=style["color"],
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            zorder=3,
            clip_on=False,
        )

    every_value = [value for _, means in curves.values() for value in means]
    low, high = min(every_value), max(every_value)
    margin = max(0.001, (high - low) * 0.08)
    axis.set_ylim(low - margin, high + margin)
    axis.set_xlim(0, args.x_max)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=None))
    ends = {arm: means[-1] for arm, (_, means) in curves.items()}
    for arm, position in spread_labels(ends, low - margin, high + margin).items():
        axis.annotate(
            "Random" if arm == "random" else arm,
            (curves[arm][0][-1], position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
            annotation_clip=False,
        )

    axis.set_title(f"{task.capitalize()}: {args.metric} by agent design")
    axis.set_ylabel(f"Best validation {args.metric} so far")
    axis.set_xlabel(f"Compute used, training + analysis ({unit})")

    ranked = sorted(ends, key=lambda arm: ends[arm], reverse=True)
    leader, second = ranked[0], ranked[1]
    note = (
        f"Highest: {ARM_LABELS[leader]}, {ends[leader]:.2%}. "
        f"Next: {ARM_LABELS[second]}, {ends[second]:.2%} "
        f"({(ends[leader] - ends[second]) * 100:.2f} points lower). "
        f"Mean of {seeds.pop()} seeds."
    )
    if args.with_random:
        note += f"\nRandom search: {ends['random']:.2%}."
        if args.random_note:
            note += f" {args.random_note}"
    bottom = 0.09 if args.with_random else 0.05
    right = 0.91 if args.with_random else 0.97  # The word "Random" needs more room than a letter.
    figure.tight_layout(rect=(0, bottom, right, 0.92))
    _arm_legend(figure, list(arms))
    figure.text(0.012, 0.012, note, ha="left", va="bottom", color=INK_2, fontsize=8.5)
    paths = _save(figure, output, f"fig_{stem}")
    plt.close(figure)

    with (output / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "arm", f"compute_{unit}", f"mean_best_validation_{args.metric}"])
        for arm, (times, means) in curves.items():
            writer.writerows([task, arm, time, mean] for time, mean in zip(times, means))
    print(note)
    print(*paths, sep="\n")


if __name__ == "__main__":
    main()
