import argparse
import csv
import os
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "edge", "data")
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "masked_index_time_seriesExp2Pos2.csv")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, "time_series_plots")
DEFAULT_SCAN_YEAR = datetime.now().year

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 12

INDEX_CONFIGS = [
    ("ndvi", "NDVI", "tab:green"),
    ("pri", "PRI", "tab:orange"),
    ("cri", "CRI", "tab:blue"),
    ("sipi", "SIPI", "tab:olive"),
    ("psri", "PSRI", "tab:red"),
]


SCAN_NAME_PATTERN = re.compile(
    r"^scan_(?P<day>\d{1,2})(?P<month>[A-Za-z]+)_(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})$"
)

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def parse_scan_datetime(scan_name, default_year=DEFAULT_SCAN_YEAR):
    """
    Parse scan names like scan_27April_15:26:44 into a datetime.
    """
    match = SCAN_NAME_PATTERN.match(scan_name)
    if not match:
        raise ValueError(f"Could not parse scan datetime from scan name: {scan_name}")

    month_name = match.group("month").lower()
    if month_name not in MONTH_MAP:
        raise ValueError(f"Unsupported month name in scan name: {scan_name}")

    return datetime(
        year=int(default_year),
        month=MONTH_MAP[month_name],
        day=int(match.group("day")),
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
        second=int(match.group("second")),
    )


def load_time_series_rows(csv_path, default_year=DEFAULT_SCAN_YEAR):
    """
    Load the master CSV and sort rows by scan timestamp.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["scan_datetime"] = parse_scan_datetime(row["scan_name"], default_year=default_year)
            for key in (
                "mask_pixel_count",
                "mask_coverage_pct",
                "pri_valid_pixel_count",
                "pri_mean",
                "pri_median",
                "pri_std",
                "cri_valid_pixel_count",
                "cri_mean",
                "cri_median",
                "cri_std",
                "ndvi_valid_pixel_count",
                "ndvi_mean",
                "ndvi_median",
                "ndvi_std",
                "sipi_valid_pixel_count",
                "sipi_mean",
                "sipi_median",
                "sipi_std",
                "psri_valid_pixel_count",
                "psri_mean",
                "psri_median",
                "psri_std",
            ):
                row[key] = parse_optional_float(row.get(key))
            rows.append(row)

    rows.sort(key=lambda row: row["scan_datetime"])
    return rows


def parse_optional_float(value):
    """
    Convert a CSV cell to float, returning NaN for missing values.
    """
    if value is None:
        return np.nan
    value = str(value).strip()
    if value == "":
        return np.nan
    return float(value)


def build_series(rows, prefix):
    """
    Extract one index series from the loaded rows.
    """
    return {
        "times": [row["scan_datetime"] for row in rows],
        "scan_names": [row["scan_name"] for row in rows],
        "mean": np.asarray([row[f"{prefix}_mean"] for row in rows], dtype=np.float32),
        "median": np.asarray([row[f"{prefix}_median"] for row in rows], dtype=np.float32),
        "std": np.asarray([row[f"{prefix}_std"] for row in rows], dtype=np.float32),
        "valid_pixel_count": np.asarray([row[f"{prefix}_valid_pixel_count"] for row in rows], dtype=np.float32),
        "mask_pixel_count": np.asarray([row["mask_pixel_count"] for row in rows], dtype=np.float32),
        "mask_coverage_pct": np.asarray([row["mask_coverage_pct"] for row in rows], dtype=np.float32),
    }


def has_index_data(rows, prefix):
    """
    Return True if at least one row contains finite data for this index.
    """
    mean_key = f"{prefix}_mean"
    if not rows:
        return False
    return any(np.isfinite(row.get(mean_key, np.nan)) for row in rows)


def style_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    for label in ax.get_xticklabels():
        label.set_rotation(0)


def plot_single_index(series, index_label, color, out_path):
    """
    Save one report-ready time-series plot for a single index.
    """
    times = series["times"]
    mean_values = series["mean"]
    median_values = series["median"]
    std_values = series["std"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, mean_values, "-", color=color, linewidth=2.0, label="Mean")
    ax.scatter(times, median_values, color="black", s=30, label="Median", zorder=3)
    ax.fill_between(
        times,
        mean_values - std_values,
        mean_values + std_values,
        color=color,
        alpha=0.18,
        label="Mean ± std",
    )

    ax.set_title(f"{index_label} over time", fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(index_label, fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("Scan time", fontsize=LABEL_FONTSIZE)
    style_time_axis(ax)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overview(rows, out_path, active_configs):
    """
    Save one figure with all available indices as stacked time-series plots.
    """
    fig_height = max(8, 2.8 * len(active_configs))
    fig, axes = plt.subplots(nrows=len(active_configs), ncols=1, figsize=(12, fig_height), sharex=True)
    if len(active_configs) == 1:
        axes = [axes]

    for ax, (prefix, label, color) in zip(axes, active_configs):
        series = build_series(rows, prefix)
        times = series["times"]
        mean_values = series["mean"]
        median_values = series["median"]
        std_values = series["std"]

        ax.plot(times, mean_values, "-", color=color, linewidth=2.0, label="Mean")
        ax.scatter(times, median_values, color="black", s=24, label="Median", zorder=3)
        ax.fill_between(
            times,
            mean_values - std_values,
            mean_values + std_values,
            color=color,
            alpha=0.18,
            label="Mean ± std",
        )
        ax.set_ylabel(label, fontsize=LABEL_FONTSIZE)
        ax.set_title(f"{label} over time", fontsize=TITLE_FONTSIZE)
        style_time_axis(ax)
        ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)

    axes[-1].set_xlabel("Scan time", fontsize=LABEL_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mask_support(rows, out_path):
    """
    Save one figure showing how the plant mask size evolves over time.
    """
    times = [row["scan_datetime"] for row in rows]
    pixel_counts = np.asarray([row["mask_pixel_count"] for row in rows], dtype=np.float32)
    coverage = np.asarray([row["mask_coverage_pct"] for row in rows], dtype=np.float32)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex=True)

    axes[0].plot(times, pixel_counts, "-", color="tab:red", linewidth=2.0)
    axes[0].set_title("Plant mask size over time", fontsize=TITLE_FONTSIZE)
    axes[0].set_ylabel("Plant pixels", fontsize=LABEL_FONTSIZE)
    style_time_axis(axes[0])

    axes[1].plot(times, coverage, "-", color="tab:purple", linewidth=2.0)
    axes[1].set_ylabel("Coverage (%)", fontsize=LABEL_FONTSIZE)
    axes[1].set_xlabel("Scan time", fontsize=LABEL_FONTSIZE)
    axes[1].set_title("Plant mask coverage over time", fontsize=TITLE_FONTSIZE)
    style_time_axis(axes[1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot time-series for all available indices from the master masked-index CSV."
    )
    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
        help=f"Path to the master CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_SCAN_YEAR,
        help=f"Year used when parsing scan names like scan_27April_15:26:44 (default: {DEFAULT_SCAN_YEAR})",
    )
    args = parser.parse_args()

    rows = load_time_series_rows(args.csv_path, default_year=args.year)
    if not rows:
        raise RuntimeError(f"No rows found in CSV: {args.csv_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    active_configs = [config for config in INDEX_CONFIGS if has_index_data(rows, config[0])]
    if not active_configs:
        raise RuntimeError(f"No finite index data found in CSV: {args.csv_path}")

    plot_overview(rows, os.path.join(args.output_dir, "index_time_series_overview.png"), active_configs)
    for prefix, label, color in active_configs:
        plot_single_index(
            build_series(rows, prefix),
            label,
            color,
            os.path.join(args.output_dir, f"{prefix}_time_series.png"),
        )
    plot_mask_support(rows, os.path.join(args.output_dir, "mask_time_series.png"))

    print(f"Time-series plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
