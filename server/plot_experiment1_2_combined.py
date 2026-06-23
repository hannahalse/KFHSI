import argparse
import os
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_index_time_series import (
    DATA_ROOT,
    DEFAULT_SCAN_YEAR,
    INDEX_CONFIGS,
    LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    build_scan_folder_lookup,
    build_series,
    load_time_series_rows,
    style_time_axis,
)


EXPERIMENTS = ("Experiment1", "Experiment2")
POSITION_COLORS = {
    1: "tab:red",
    2: "tab:orange",
    3: "tab:purple",
}
STRESS_POSITIONS = {1, 2}
CONTROL_POSITIONS = {3}
PART_LABEL_Y = 0.52
LIGHT_LEGEND_Y = -0.42
LIGHT_REGIMES = {
    "250": {"label": "250 PPFD", "color": "#eeeeee"},
    "400": {"label": "400 PPFD", "color": "#d9ecff"},
    "450": {"label": "450 PPFD", "color": "#eadcf8"},
    "500": {"label": "500 PPFD", "color": "#dcf2d8"},
    "600": {"label": "600 PPFD", "color": "#ffe5c2"},
}
LIGHT_SPAN_ALPHA = 0.8
LIGHT_BAND_ALPHA = 0.8
COMBINED_LEGEND_FONTSIZE = LEGEND_FONTSIZE + 1
LIGHT_LEGEND_FONTSIZE = 11


def default_output_dir():
    return os.path.join(
        DATA_ROOT,
        "Experiment1_Experiment2",
        "time_series_plots",
        "combined_position_mean_comparison",
    )


def csv_path_for_position(experiment, position):
    exp_number = re.fullmatch(r"Experiment(\d+)", experiment).group(1)
    return os.path.join(
        DATA_ROOT,
        experiment,
        f"masked_index_time_seriesExp{exp_number}Pos{position}.csv",
    )


def treatment_label(position):
    if position in STRESS_POSITIONS:
        return "stress"
    if position in CONTROL_POSITIONS:
        return "control"
    return "unassigned"


def load_combined_groups(default_year):
    groups = []
    for experiment in EXPERIMENTS:
        scan_lookup = build_scan_folder_lookup(os.path.join(DATA_ROOT, experiment))
        for position in (1, 2, 3):
            csv_path = csv_path_for_position(experiment, position)
            if not os.path.exists(csv_path):
                print(f"Skipping missing CSV: {csv_path}")
                continue

            rows = load_time_series_rows(
                csv_path,
                default_year=default_year,
                scan_lookup=scan_lookup,
            )
            if not rows:
                print(f"Skipping empty CSV: {csv_path}")
                continue

            groups.append(
                {
                    "experiment": experiment,
                    "position": position,
                    "csv_path": csv_path,
                    "rows": rows,
                }
            )

    if not groups:
        raise RuntimeError("No Experiment1 or Experiment2 position CSV rows were loaded.")
    return groups


def experiment_separator_time(groups):
    periods = experiment_periods(groups)
    if periods is None:
        return None

    _exp1_first, exp1_last, exp2_first, _exp2_last = periods
    return exp1_last + (exp2_first - exp1_last) / 2


def experiment_periods(groups):
    exp1_times = [
        row["scan_datetime"]
        for group in groups
        if group["experiment"] == "Experiment1"
        for row in group["rows"]
    ]
    exp2_times = [
        row["scan_datetime"]
        for group in groups
        if group["experiment"] == "Experiment2"
        for row in group["rows"]
    ]
    if not exp1_times or not exp2_times:
        return None

    return min(exp1_times), max(exp1_times), min(exp2_times), max(exp2_times)


def times_for(groups, experiment, month=None, day=None, since_day=None):
    times = []
    for group in groups:
        if group["experiment"] != experiment:
            continue
        for row in group["rows"]:
            scan_time = row["scan_datetime"]
            if month is not None and scan_time.month != month:
                continue
            if day is not None and scan_time.day != day:
                continue
            if since_day is not None and scan_time.day < since_day:
                continue
            times.append(scan_time)
    return times


def midpoint(times):
    if not times:
        return None
    return min(times) + (max(times) - min(times)) / 2


def boundary_between(left_times, right_times):
    if not left_times or not right_times:
        return None
    left_edge = max(left_times)
    right_edge = min(right_times)
    return left_edge + (right_edge - left_edge) / 2


def part2_light_change_time(groups):
    may8_times = times_for(groups, "Experiment2", month=5, day=8)
    may9_times = times_for(groups, "Experiment2", month=5, day=9)
    if not may8_times or not may9_times:
        return None
    last_may8 = max(may8_times)
    first_may9 = min(may9_times)
    return last_may8 + (first_may9 - last_may8) / 2


def light_regime_spans(groups):
    periods = experiment_periods(groups)
    separator_time = experiment_separator_time(groups)
    light_change_time = part2_light_change_time(groups)
    if periods is None or separator_time is None:
        return []

    exp1_first, _exp1_last, _exp2_first, exp2_last = periods
    exp1_day27 = times_for(groups, "Experiment1", month=4, day=27)
    exp1_day28 = times_for(groups, "Experiment1", month=4, day=28)
    exp1_day29 = times_for(groups, "Experiment1", month=4, day=29)
    exp1_day30 = times_for(groups, "Experiment1", month=4, day=30)

    exp1_400_start = boundary_between(exp1_day27, exp1_day28)
    exp1_450_start = boundary_between(exp1_day28, exp1_day29)
    exp1_500_start = boundary_between(exp1_day29, exp1_day30)
    exp1_250_return = datetime(exp1_first.year, 5, 1)

    exp2_day4 = times_for(groups, "Experiment2", month=5, day=4)
    exp2_500_times = [
        scan_time
        for day in range(5, 9)
        for scan_time in times_for(groups, "Experiment2", month=5, day=day)
    ]
    exp2_500_start = boundary_between(exp2_day4, exp2_500_times)

    candidate_spans = [
        ("250", exp1_first, exp1_400_start),
        ("400", exp1_400_start, exp1_450_start),
        ("450", exp1_450_start, exp1_500_start),
        ("500", exp1_500_start, exp1_250_return),
        ("250", exp1_250_return, separator_time),
        ("250", separator_time, exp2_500_start),
    ]
    if light_change_time is not None:
        candidate_spans.extend(
            [
                ("500", exp2_500_start, light_change_time),
                ("600", light_change_time, exp2_last),
            ]
        )
    else:
        candidate_spans.append(("500", exp2_500_start, exp2_last))

    spans = []
    for regime, start, end in candidate_spans:
        if start is None or end is None or end <= start:
            continue
        spans.append((regime, start, end))
    return spans


def line_style(group):
    position = group["position"]
    linestyle = "--" if position in CONTROL_POSITIONS else "-"
    return {
        "color": POSITION_COLORS.get(position),
        "linestyle": linestyle,
        "marker": "o",
        "linewidth": 2.0,
        "markersize": 4,
    }


def line_label(group):
    position = group["position"]
    return f"Position {position} ({treatment_label(position)})"


def add_experiment_separator(ax, separator_time, periods=None, add_labels=False):
    if separator_time is None:
        return
    ax.axvline(
        separator_time,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.65,
    )
    if not add_labels or periods is None:
        return

    exp1_first, exp1_last, exp2_first, exp2_last = periods
    exp1_mid = exp1_first + (exp1_last - exp1_first) / 2
    exp2_mid = exp2_first + (exp2_last - exp2_first) / 2
    for x, label in ((exp1_mid, "Part 1"), (exp2_mid, "Part 2")):
        ax.text(
            x,
            PART_LABEL_Y,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=LABEL_FONTSIZE,
            fontweight="bold",
            color="black",
            clip_on=False,
        )


def add_light_annotations(ax, groups, add_labels=False):
    for regime, start, end in light_regime_spans(groups):
        ax.axvspan(
            start,
            end,
            color=LIGHT_REGIMES[regime]["color"],
            alpha=LIGHT_SPAN_ALPHA,
            linewidth=0,
            zorder=0,
        )


def draw_annotation_band(ax, groups):
    periods = experiment_periods(groups)
    separator_time = experiment_separator_time(groups)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if separator_time is not None:
        ax.axvline(
            separator_time,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.65,
        )

    if periods is not None:
        exp1_first, exp1_last, exp2_first, exp2_last = periods
        part_labels = (
            (exp1_first + (exp1_last - exp1_first) / 2, "Part 1"),
            (exp2_first + (exp2_last - exp2_first) / 2, "Part 2"),
        )
        for x, label in part_labels:
            ax.text(
                x,
                PART_LABEL_Y,
                label,
                ha="center",
                va="center",
                fontsize=LABEL_FONTSIZE,
                fontweight="bold",
                color="black",
            )

    legend_order = ("250", "500", "400", "600", "450")
    handles = [
        Patch(
            facecolor=LIGHT_REGIMES[regime]["color"],
            edgecolor="0.65",
            alpha=LIGHT_BAND_ALPHA,
            label=LIGHT_REGIMES[regime]["label"],
        )
        for regime in legend_order
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(-0.02, LIGHT_LEGEND_Y),
        bbox_transform=ax.transAxes,
        ncols=3,
        frameon=False,
        fontsize=LIGHT_LEGEND_FONTSIZE,
        handlelength=1.4,
        columnspacing=1.1,
    )


def deduplicated_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), **kwargs)


def plot_index(groups, prefix, index_label, out_path):
    fig, (annotation_ax, ax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [0.55, 5]},
    )
    separator_time = experiment_separator_time(groups)

    for group in groups:
        series = build_series(group["rows"], prefix)
        ax.plot(
            series["times"],
            series["mean"],
            label=line_label(group),
            **line_style(group),
        )

    add_experiment_separator(ax, separator_time)
    add_light_annotations(ax, groups)
    draw_annotation_band(annotation_ax, groups)
    ax.set_ylabel(f"{index_label} mean", fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("Scan date", fontsize=LABEL_FONTSIZE)
    style_time_axis(ax)
    deduplicated_legend(ax, fontsize=COMBINED_LEGEND_FONTSIZE, ncols=2)
    fig.tight_layout(h_pad=0.2)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overview(groups, active_configs, out_path):
    fig_height = max(8, 2.8 * len(active_configs))
    fig, axes = plt.subplots(
        nrows=len(active_configs) + 1,
        ncols=1,
        figsize=(12, fig_height + 0.9),
        sharex=True,
        gridspec_kw={"height_ratios": [0.55] + [2.8] * len(active_configs)},
    )
    annotation_ax = axes[0]
    data_axes = axes[1:]
    if len(active_configs) == 1:
        data_axes = [axes[1]]

    separator_time = experiment_separator_time(groups)
    for ax, (prefix, index_label, _color) in zip(data_axes, active_configs):
        for group in groups:
            series = build_series(group["rows"], prefix)
            ax.plot(
                series["times"],
                series["mean"],
                label=line_label(group),
                **line_style(group),
            )

        add_experiment_separator(ax, separator_time)
        add_light_annotations(ax, groups)
        ax.set_ylabel(index_label, fontsize=LABEL_FONTSIZE)
        style_time_axis(ax)

    draw_annotation_band(annotation_ax, groups)
    deduplicated_legend(data_axes[0], fontsize=COMBINED_LEGEND_FONTSIZE, ncols=2)
    data_axes[-1].set_xlabel("Scan date", fontsize=LABEL_FONTSIZE)
    for ax in data_axes:
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    fig.tight_layout(h_pad=0.25)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Experiment1 and Experiment2 position means on shared axes."
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir(),
        help="Directory for saved combined plots.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_SCAN_YEAR,
        help=f"Year used when parsing scan names (default: {DEFAULT_SCAN_YEAR}).",
    )
    parser.add_argument(
        "--no-overview",
        action="store_true",
        help="Only save one plot per index, not the stacked overview figure.",
    )
    args = parser.parse_args()

    groups = load_combined_groups(default_year=args.year)
    os.makedirs(args.output_dir, exist_ok=True)

    for prefix, index_label, _color in INDEX_CONFIGS:
        plot_index(
            groups,
            prefix,
            index_label,
            os.path.join(args.output_dir, f"{prefix}_mean_experiment1_experiment2.png"),
        )

    if not args.no_overview:
        plot_overview(
            groups,
            INDEX_CONFIGS,
            os.path.join(args.output_dir, "index_mean_experiment1_experiment2_overview.png"),
        )

    print(
        f"Saved {len(INDEX_CONFIGS)} combined index plot(s) "
        f"for {len(groups)} position series to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
