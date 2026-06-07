import argparse
import os
import re

# How to run:
#   cd /Users/hannahalse/Desktop/KFHSI/server
#
# Experiment 3, plants 1-3 stressed and plants 4-5 control:
#   python3 plot_experiment3_all_plant_means.py
#
# Experiment 2, positions 1-2 stressed and position 3 control:
#   python3 plot_experiment3_all_plant_means.py --experiment Experiment2
#
# Experiment 1, positions 1-2 stressed and position 3 control:
#   python3 plot_experiment3_all_plant_means.py --experiment Experiment1
#
# The plots are saved under:
#   ../edge/data/<Experiment>/time_series_plots/

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
    discover_csv_paths,
    load_time_series_rows,
    style_time_axis,
)


DEFAULT_EXPERIMENT = "Experiment3"
EXPERIMENT_PRESETS = {
    "Experiment1": {
        "csv_pattern": "masked_index_time_seriesExp1Pos*.csv",
        "group_label": "Position",
        "file_token": "positions",
        "output_suffix": "all_position_mean_comparison",
        "stress_groups": "1,2",
        "control_groups": "3",
    },
    "Experiment2": {
        "csv_pattern": "masked_index_time_seriesExp2Pos*.csv",
        "group_label": "Position",
        "file_token": "positions",
        "output_suffix": "all_position_mean_comparison",
        "stress_groups": "1,2",
        "control_groups": "3",
    },
    "Experiment3": {
        "csv_pattern": "masked_index_time_seriesExp3Plant*.csv",
        "group_label": "Plant",
        "file_token": "plants",
        "output_suffix": "all_plant_mean_comparison",
        "stress_groups": "1,2,3",
        "control_groups": "4,5",
    },
}

GROUP_STYLE_COLORS = {
    1: "tab:red",
    2: "tab:orange",
    3: "tab:purple",
    4: "tab:blue",
    5: "tab:cyan",
}


def experiment_preset(experiment):
    if experiment not in EXPERIMENT_PRESETS:
        available = ", ".join(sorted(EXPERIMENT_PRESETS))
        raise ValueError(
            f"No defaults configured for {experiment!r}. "
            f"Available experiments: {available}. "
            "Use --csv-pattern, --stress-groups, and --control-groups for custom data."
        )
    return EXPERIMENT_PRESETS[experiment]


def default_csv_dir(experiment):
    return os.path.join(DATA_ROOT, experiment)


def default_output_dir(csv_dir, preset):
    return os.path.join(
        csv_dir,
        "time_series_plots",
        preset["output_suffix"],
    )


def parse_group_number(csv_path):
    match = re.search(r"(?:Plant|Pos)(\d+)", os.path.basename(csv_path))
    if not match:
        raise ValueError(f"Could not parse group number from CSV name: {csv_path}")
    return int(match.group(1))


def parse_group_set(value):
    if value is None or str(value).strip() == "":
        return set()
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def treatment_label(group_number, stress_groups, control_groups):
    if group_number in stress_groups:
        return "stress"
    if group_number in control_groups:
        return "control"
    return "unassigned"


def group_line_style(group_number, stress_groups, control_groups):
    treatment = treatment_label(group_number, stress_groups, control_groups)
    if treatment == "control":
        linestyle = "--"
    elif treatment == "stress":
        linestyle = "-"
    else:
        linestyle = ":"

    return {
        "color": GROUP_STYLE_COLORS.get(group_number, None),
        "linestyle": linestyle,
        "linewidth": 2.0,
        "marker": "o",
        "markersize": 4,
    }


def load_group_rows(csv_paths, default_year, scan_lookup):
    group_rows = []
    for csv_path in csv_paths:
        group_number = parse_group_number(csv_path)
        rows = load_time_series_rows(
            csv_path,
            default_year=default_year,
            scan_lookup=scan_lookup,
        )
        if rows:
            group_rows.append(
                {
                    "group_number": group_number,
                    "csv_path": csv_path,
                    "rows": rows,
                }
            )

    group_rows.sort(key=lambda item: item["group_number"])
    return group_rows


def plot_index_for_all_groups(
    group_rows,
    prefix,
    index_label,
    out_path,
    group_label,
    stress_groups,
    control_groups,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    for item in group_rows:
        group_number = item["group_number"]
        series = build_series(item["rows"], prefix)
        treatment = treatment_label(group_number, stress_groups, control_groups)
        style = group_line_style(group_number, stress_groups, control_groups)
        ax.plot(
            series["times"],
            series["mean"],
            label=f"{group_label} {group_number} ({treatment})",
            **style,
        )

    ax.set_title(f"{index_label} mean by {group_label.lower()}", fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(f"{index_label} mean", fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("Scan date", fontsize=LABEL_FONTSIZE)
    style_time_axis(ax)
    ax.legend(fontsize=LEGEND_FONTSIZE, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overview_for_all_groups(
    group_rows,
    active_configs,
    out_path,
    group_label,
    stress_groups,
    control_groups,
):
    fig_height = max(8, 2.8 * len(active_configs))
    fig, axes = plt.subplots(
        nrows=len(active_configs),
        ncols=1,
        figsize=(12, fig_height),
        sharex=True,
    )
    if len(active_configs) == 1:
        axes = [axes]

    for ax, (prefix, index_label, _color) in zip(axes, active_configs):
        for item in group_rows:
            group_number = item["group_number"]
            series = build_series(item["rows"], prefix)
            treatment = treatment_label(group_number, stress_groups, control_groups)
            style = group_line_style(group_number, stress_groups, control_groups)
            ax.plot(
                series["times"],
                series["mean"],
                label=f"{group_label} {group_number} ({treatment})",
                **style,
            )

        ax.set_ylabel(index_label, fontsize=LABEL_FONTSIZE)
        ax.set_title(f"{index_label} mean", fontsize=TITLE_FONTSIZE)
        style_time_axis(ax)

    axes[0].legend(fontsize=LEGEND_FONTSIZE, ncols=2)
    axes[-1].set_xlabel("Scan date", fontsize=LABEL_FONTSIZE)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean-only experiment group comparisons on shared axes."
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Experiment preset to use (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--csv-dir",
        default=None,
        help="Directory searched for CSVs (default: edge/data/<experiment>).",
    )
    parser.add_argument(
        "--csv-pattern",
        default=None,
        help="Filename pattern for CSVs (default depends on --experiment).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for saved plots (default: <csv-dir>/time_series_plots/<comparison>).",
    )
    parser.add_argument(
        "--scan-root",
        default=None,
        help="Directory searched for scan_* folders (default: --csv-dir).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_SCAN_YEAR,
        help=f"Year used when parsing scan names (default: {DEFAULT_SCAN_YEAR})",
    )
    parser.add_argument(
        "--group-label",
        default=None,
        help="Label used in legends, for example Plant or Position (default depends on --experiment).",
    )
    parser.add_argument(
        "--stress-groups",
        "--stress-plants",
        dest="stress_groups",
        default=None,
        help="Comma-separated stress group numbers (default depends on --experiment).",
    )
    parser.add_argument(
        "--control-groups",
        "--control-plants",
        dest="control_groups",
        default=None,
        help="Comma-separated control group numbers (default depends on --experiment).",
    )
    parser.add_argument(
        "--no-overview",
        action="store_true",
        help="Only write one plot per index, not the stacked overview plot.",
    )
    args = parser.parse_args()

    preset = experiment_preset(args.experiment)
    csv_dir = args.csv_dir or default_csv_dir(args.experiment)
    csv_pattern = args.csv_pattern or preset["csv_pattern"]
    output_dir = args.output_dir or default_output_dir(csv_dir, preset)
    group_label = args.group_label or preset["group_label"]
    file_token = preset["file_token"]
    stress_groups = parse_group_set(args.stress_groups or preset["stress_groups"])
    control_groups = parse_group_set(args.control_groups or preset["control_groups"])

    csv_paths = discover_csv_paths(csv_dir, pattern=csv_pattern)
    scan_lookup = build_scan_folder_lookup(args.scan_root or csv_dir)
    group_rows = load_group_rows(
        csv_paths,
        default_year=args.year,
        scan_lookup=scan_lookup,
    )
    if not group_rows:
        raise RuntimeError("No rows were loaded from the selected CSV files.")

    os.makedirs(output_dir, exist_ok=True)

    for prefix, index_label, _color in INDEX_CONFIGS:
        out_path = os.path.join(output_dir, f"{prefix}_mean_all_{file_token}.png")
        plot_index_for_all_groups(
            group_rows,
            prefix,
            index_label,
            out_path,
            group_label=group_label,
            stress_groups=stress_groups,
            control_groups=control_groups,
        )

    if not args.no_overview:
        plot_overview_for_all_groups(
            group_rows,
            INDEX_CONFIGS,
            os.path.join(output_dir, f"index_mean_all_{file_token}_overview.png"),
            group_label=group_label,
            stress_groups=stress_groups,
            control_groups=control_groups,
        )

    print(
        f"Saved {len(INDEX_CONFIGS)} mean comparison plot(s) "
        f"for {len(group_rows)} group(s) to {output_dir}"
    )


if __name__ == "__main__":
    main()
