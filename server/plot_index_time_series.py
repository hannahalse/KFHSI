import argparse
import csv
import fnmatch
import os
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(BASE_DIR, "edge", "data")
DEFAULT_CSV_DIR = os.path.join(DATA_ROOT, "Experiment1")
DEFAULT_CSV_PATTERN = "masked_index_time_series*.csv"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_CSV_DIR, "time_series_plots")
DEFAULT_SCAN_YEAR = datetime.now().year

TITLE_FONTSIZE = 23
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 13

INDEX_CONFIGS = [
    ("ndvi", "NDVI", "tab:green"),
    ("pri", "PRI", "tab:orange"),
    ("cri1", "CRI1", "tab:blue"),
    ("cri2", "CRI2", "tab:cyan"),
    ("sipi", "SIPI", "tab:olive"),
    ("psri", "PSRI", "tab:red"),
]

# Explicit scan exclusions for index-specific outliers.
# These exclusions affect plotting only, not the underlying CSV values.
EXCLUDED_SCANS_BY_INDEX = {
    "sipi": {
        #"scan_04May_11:53:02",
        #"scan_12May_13:36:09",
    },
    "psri": {
        #"scan_04May_11:53:02",
        #"scan_12May_13:36:09",
    },
}


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


def discover_csv_paths(csv_dir, pattern=DEFAULT_CSV_PATTERN):
    """
    Find all master time-series CSV files below csv_dir.
    """
    csv_dir = os.path.abspath(csv_dir)
    if not os.path.isdir(csv_dir):
        raise NotADirectoryError(f"CSV directory not found: {csv_dir}")

    csv_paths = []
    for current_root, dirnames, filenames in os.walk(csv_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            if fnmatch.fnmatch(filename, pattern):
                csv_paths.append(os.path.join(current_root, filename))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matching {pattern!r} found below {csv_dir}")
    return csv_paths


def build_scan_folder_lookup(scan_root):
    """
    Map scan folder names to real paths, including nested day folders.
    """
    scan_root = os.path.abspath(scan_root)
    if not os.path.isdir(scan_root):
        raise NotADirectoryError(f"Scan root not found: {scan_root}")

    lookup = {}
    for current_root, dirnames, _filenames in os.walk(scan_root):
        dirnames.sort()
        scan_dirnames = [dirname for dirname in dirnames if SCAN_NAME_PATTERN.match(dirname)]
        for dirname in scan_dirnames:
            lookup.setdefault(dirname, os.path.join(current_root, dirname))

        # Scan folders contain many frames; do not descend into them while searching.
        dirnames[:] = [dirname for dirname in dirnames if dirname not in scan_dirnames]
    return lookup


def resolve_scan_folder(row, scan_lookup):
    """
    Replace stale scan_folder values with the matching discovered scan_* path.
    """
    scan_folder = row.get("scan_folder", "")
    if scan_folder and os.path.isdir(scan_folder):
        return scan_folder

    scan_name = row.get("scan_name", "")
    return scan_lookup.get(scan_name, scan_folder)


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


def load_time_series_rows(csv_path, default_year=DEFAULT_SCAN_YEAR, scan_lookup=None):
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
            if scan_lookup is not None and "scan_folder" in row:
                row["scan_folder"] = resolve_scan_folder(row, scan_lookup)
            for key in (
                "mask_pixel_count",
                "mask_coverage_pct",
                "pri_valid_pixel_count",
                "pri_mean",
                "pri_median",
                "pri_std",
                "cri1_valid_pixel_count",
                "cri1_mean",
                "cri1_median",
                "cri1_std",
                "cri2_valid_pixel_count",
                "cri2_mean",
                "cri2_median",
                "cri2_std",
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
    excluded_scans = EXCLUDED_SCANS_BY_INDEX.get(prefix, set())
    filtered_rows = [row for row in rows if row["scan_name"] not in excluded_scans]

    return {
        "times": [row["scan_datetime"] for row in filtered_rows],
        "scan_names": [row["scan_name"] for row in filtered_rows],
        "mean": np.asarray([row[f"{prefix}_mean"] for row in filtered_rows], dtype=np.float32),
        "median": np.asarray([row[f"{prefix}_median"] for row in filtered_rows], dtype=np.float32),
        "std": np.asarray([row[f"{prefix}_std"] for row in filtered_rows], dtype=np.float32),
        "valid_pixel_count": np.asarray([row[f"{prefix}_valid_pixel_count"] for row in filtered_rows], dtype=np.float32),
        "mask_pixel_count": np.asarray([row["mask_pixel_count"] for row in filtered_rows], dtype=np.float32),
        "mask_coverage_pct": np.asarray([row["mask_coverage_pct"] for row in filtered_rows], dtype=np.float32),
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
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

    ax.set_ylabel(index_label, fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("Scan date", fontsize=LABEL_FONTSIZE)
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
        style_time_axis(ax)
        ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)

    axes[-1].set_xlabel("Scan date", fontsize=LABEL_FONTSIZE)
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
    axes[0].set_ylabel("Plant pixels", fontsize=LABEL_FONTSIZE)
    style_time_axis(axes[0])

    axes[1].plot(times, coverage, "-", color="tab:purple", linewidth=2.0)
    axes[1].set_ylabel("Coverage (%)", fontsize=LABEL_FONTSIZE)
    axes[1].set_xlabel("Scan date", fontsize=LABEL_FONTSIZE)
    style_time_axis(axes[1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def output_name_for_csv(csv_path):
    """
    Convert a CSV filename to a safe output subfolder name.
    """
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return name or "time_series"


def plot_time_series_csv(csv_path, output_dir, default_year=DEFAULT_SCAN_YEAR, scan_lookup=None):
    """
    Generate all time-series plots for one master CSV.
    """
    rows = load_time_series_rows(csv_path, default_year=default_year, scan_lookup=scan_lookup)
    if not rows:
        raise RuntimeError(f"No rows found in CSV: {csv_path}")

    os.makedirs(output_dir, exist_ok=True)

    active_configs = [config for config in INDEX_CONFIGS if has_index_data(rows, config[0])]
    if not active_configs:
        raise RuntimeError(f"No finite index data found in CSV: {csv_path}")

    plot_overview(rows, os.path.join(output_dir, "index_time_series_overview.png"), active_configs)
    for prefix, label, color in active_configs:
        plot_single_index(
            build_series(rows, prefix),
            label,
            color,
            os.path.join(output_dir, f"{prefix}_time_series.png"),
        )
    plot_mask_support(rows, os.path.join(output_dir, "mask_time_series.png"))

    return {
        "csv_path": csv_path,
        "output_dir": output_dir,
        "row_count": len(rows),
        "index_count": len(active_configs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot time-series for all available indices from masked-index CSV files."
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to one master CSV. If omitted, all matching CSVs below --csv-dir are plotted.",
    )
    parser.add_argument(
        "--csv-dir",
        default=DEFAULT_CSV_DIR,
        help=f"Directory searched for master CSVs (default: {DEFAULT_CSV_DIR})",
    )
    parser.add_argument(
        "--csv-pattern",
        default=DEFAULT_CSV_PATTERN,
        help=f"Filename pattern used with --csv-dir (default: {DEFAULT_CSV_PATTERN})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--scan-root",
        default=None,
        help="Directory searched for scan_* folders (default: CSV directory or --csv-dir).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_SCAN_YEAR,
        help=f"Year used when parsing scan names like scan_27April_15:26:44 (default: {DEFAULT_SCAN_YEAR})",
    )
    args = parser.parse_args()

    if args.csv_path:
        csv_paths = [os.path.abspath(args.csv_path)]
        default_scan_root = os.path.dirname(csv_paths[0])
    else:
        csv_paths = discover_csv_paths(args.csv_dir, pattern=args.csv_pattern)
        default_scan_root = args.csv_dir

    scan_root = args.scan_root or default_scan_root
    scan_lookup = build_scan_folder_lookup(scan_root)
    multiple_csvs = len(csv_paths) > 1

    for csv_path in csv_paths:
        output_dir = args.output_dir
        if multiple_csvs:
            output_dir = os.path.join(args.output_dir, output_name_for_csv(csv_path))
        result = plot_time_series_csv(
            csv_path,
            output_dir,
            default_year=args.year,
            scan_lookup=scan_lookup,
        )
        print(
            f"{os.path.basename(csv_path)}: "
            f"{result['row_count']} row(s), "
            f"{result['index_count']} index plot(s), "
            f"saved to {result['output_dir']}"
        )


if __name__ == "__main__":
    main()
