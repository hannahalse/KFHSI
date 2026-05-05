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
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "masked_index_time_series.csv")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, "time_series_plots")
DEFAULT_SCAN_YEAR = datetime.now().year


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
            ):
                row[key] = float(row[key])
            rows.append(row)

    rows.sort(key=lambda row: row["scan_datetime"])
    return rows


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


def style_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.grid(True, alpha=0.3)
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

    ax.set_title(f"{index_label} over time")
    ax.set_ylabel(index_label)
    ax.set_xlabel("Scan time")
    style_time_axis(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overview(rows, out_path):
    """
    Save one figure with NDVI, PRI, and CRI as stacked time-series plots.
    """
    configs = [
        ("ndvi", "NDVI", "tab:green"),
        ("pri", "PRI", "tab:orange"),
        ("cri", "CRI", "tab:blue"),
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
    for ax, (prefix, label, color) in zip(axes, configs):
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
        ax.set_ylabel(label)
        ax.set_title(f"{label} over time")
        style_time_axis(ax)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Scan time")
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
    axes[0].set_title("Plant mask size over time")
    axes[0].set_ylabel("Plant pixels")
    style_time_axis(axes[0])

    axes[1].plot(times, coverage, "-", color="tab:purple", linewidth=2.0)
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_xlabel("Scan time")
    axes[1].set_title("Plant mask coverage over time")
    style_time_axis(axes[1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot NDVI, PRI, and CRI time-series from the master masked-index CSV."
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

    plot_overview(rows, os.path.join(args.output_dir, "index_time_series_overview.png"))
    plot_single_index(
        build_series(rows, "ndvi"),
        "NDVI",
        "tab:green",
        os.path.join(args.output_dir, "ndvi_time_series.png"),
    )
    plot_single_index(
        build_series(rows, "pri"),
        "PRI",
        "tab:orange",
        os.path.join(args.output_dir, "pri_time_series.png"),
    )
    plot_single_index(
        build_series(rows, "cri"),
        "CRI",
        "tab:blue",
        os.path.join(args.output_dir, "cri_time_series.png"),
    )
    plot_mask_support(rows, os.path.join(args.output_dir, "mask_time_series.png"))

    print(f"Time-series plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
