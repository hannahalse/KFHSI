import argparse
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from experiment_scan_groups import (
    DEFAULT_EXPERIMENT,
    PLANT_SCANS_BY_EXPERIMENT,
    POSITION_SCANS_BY_EXPERIMENT,
    discover_ordinal_scan_groups,
    discover_scan_names,
    experiment_dir,
    scan_folder_for_name,
    selected_position_keys,
    sort_position_keys,
)
from visualization import process_scan


def experiment_csv_token(experiment):
    match = re.fullmatch(r"Experiment(\d+)", experiment)
    if match:
        return f"Exp{match.group(1)}"
    return re.sub(r"[^A-Za-z0-9]+", "", experiment) or "Experiment"


def csv_path_for_group(output_dir, experiment, group_label):
    return os.path.join(
        output_dir,
        f"masked_index_time_series{experiment_csv_token(experiment)}{group_label}.csv",
    )


def corrected_cube_path(scan_folder):
    return os.path.join(scan_folder, "cube_ZXnm_corrected.npz")


def remove_existing_csv(csv_path, fresh, dry_run):
    if not fresh or not os.path.exists(csv_path):
        return

    if dry_run:
        print(f"Would remove existing CSV: {csv_path}")
        return

    os.remove(csv_path)
    print(f"Removed existing CSV: {csv_path}")


def run_scan_group(
    label,
    scan_names,
    data_dir,
    csv_path,
    replace_existing_scan_row,
    print_diagnostic,
    fresh,
    dry_run,
):
    print(f"\n{label}: {len(scan_names)} scan(s)")
    print(f"Output CSV: {csv_path}")
    remove_existing_csv(csv_path, fresh=fresh, dry_run=dry_run)

    for scan_name in scan_names:
        scan_folder = scan_folder_for_name(data_dir, scan_name)
        npz_path = corrected_cube_path(scan_folder)
        if not os.path.exists(npz_path):
            print(f"Skipping {scan_name}: missing {npz_path}")
            continue

        print(f"Processing {scan_name}")
        if dry_run:
            continue

        try:
            process_scan(
                scan_folder=scan_folder,
                csv_path=csv_path,
                show_plots=False,
                append_to_master_csv=True,
                replace_existing_scan_row=replace_existing_scan_row,
                print_diagnostic=print_diagnostic,
            )
        except Exception as exc:
            print(f"Failed {scan_name}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run visualization summaries for configured experiment scan groups."
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Experiment folder below edge/data (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--position",
        default="all",
        help="Configured position/plant key to process, or 'all' (default: all).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Scan directory. Defaults to edge/data/<experiment>.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output CSVs. Defaults to the selected data directory.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Ignore configured positions and process every scan_* folder in the data directory into one CSV.",
    )
    parser.add_argument(
        "--plant-groups",
        action="store_true",
        help=(
            "Discover day*/scan_* folders and group them by scan order "
            "as Plant1, Plant2, etc."
        ),
    )
    parser.add_argument(
        "--plant-list",
        action="store_true",
        help="Use manually configured plant groups from PLANT_SCANS_BY_EXPERIMENT.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove the output CSV before processing so stale rows cannot remain.",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Append rows without replacing an existing row for the same scan folder.",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Print the max-reflectance diagnostic for every scan. Slower because indices are calculated twice.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which scans would be processed without calculating anything.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir or experiment_dir(args.experiment)
    output_dir = args.output_dir or data_dir

    os.makedirs(output_dir, exist_ok=True)

    grouping_modes = [args.discover, args.plant_groups, args.plant_list]
    if sum(bool(mode) for mode in grouping_modes) > 1:
        raise ValueError("Use only one of --discover, --plant-groups, or --plant-list.")

    if args.discover:
        csv_path = csv_path_for_group(output_dir, args.experiment, "All")
        run_scan_group(
            label=f"{args.experiment} discovered scans",
            scan_names=discover_scan_names(data_dir),
            data_dir=data_dir,
            csv_path=csv_path,
            replace_existing_scan_row=not args.no_replace,
            print_diagnostic=args.diagnostic,
            fresh=args.fresh,
            dry_run=args.dry_run,
        )
        return

    if args.plant_list:
        plant_scans = PLANT_SCANS_BY_EXPERIMENT.get(args.experiment, {})
        if not plant_scans:
            raise ValueError(
                f"No PLANT_SCANS_BY_EXPERIMENT entry defined for {args.experiment!r}."
            )

        if args.position == "all":
            plants = sort_position_keys(plant_scans.keys())
        else:
            if args.position not in plant_scans:
                available = ", ".join(sort_position_keys(plant_scans.keys()))
                raise ValueError(
                    f"Plant {args.position!r} is not defined for {args.experiment!r}. "
                    f"Available plants: {available}"
                )
            plants = [args.position]

        for plant in plants:
            csv_path = csv_path_for_group(output_dir, args.experiment, f"Plant{plant}")
            run_scan_group(
                label=f"{args.experiment} plant {plant}",
                scan_names=plant_scans[plant],
                data_dir=data_dir,
                csv_path=csv_path,
                replace_existing_scan_row=not args.no_replace,
                print_diagnostic=args.diagnostic,
                fresh=args.fresh,
                dry_run=args.dry_run,
            )
        return

    if args.plant_groups:
        plant_scans = discover_ordinal_scan_groups(data_dir)
        if not plant_scans:
            raise ValueError(f"No day*/scan_* groups found in {data_dir}")

        if args.position == "all":
            plants = sort_position_keys(plant_scans.keys())
        else:
            if args.position not in plant_scans:
                available = ", ".join(sort_position_keys(plant_scans.keys()))
                raise ValueError(
                    f"Plant {args.position!r} is not available for {args.experiment!r}. "
                    f"Available plants: {available}"
                )
            plants = [args.position]

        for plant in plants:
            csv_path = csv_path_for_group(output_dir, args.experiment, f"Plant{plant}")
            run_scan_group(
                label=f"{args.experiment} plant {plant}",
                scan_names=plant_scans[plant],
                data_dir=data_dir,
                csv_path=csv_path,
                replace_existing_scan_row=not args.no_replace,
                print_diagnostic=args.diagnostic,
                fresh=args.fresh,
                dry_run=args.dry_run,
            )
        return

    position_scans = POSITION_SCANS_BY_EXPERIMENT.get(args.experiment, {})
    positions = selected_position_keys(args.experiment, args.position)

    for position in positions:
        csv_path = csv_path_for_group(output_dir, args.experiment, f"Pos{position}")
        run_scan_group(
            label=f"{args.experiment} position {position}",
            scan_names=position_scans[position],
            data_dir=data_dir,
            csv_path=csv_path,
            replace_existing_scan_row=not args.no_replace,
            print_diagnostic=args.diagnostic,
            fresh=args.fresh,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
