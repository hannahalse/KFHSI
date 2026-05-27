import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from Generate_cube import generate_cube
from experiment_scan_groups import (
    DEFAULT_EXPERIMENT,
    POSITION_SCANS_BY_EXPERIMENT,
    discover_scan_names,
    experiment_dir,
    scan_folder_for_name,
    selected_position_keys,
)


def corrected_cube_path(scan_folder):
    return os.path.join(scan_folder, "cube_ZXnm_corrected.npz")


def has_scan_images(scan_folder):
    if not os.path.isdir(scan_folder):
        return False
    return any(
        name.lower().endswith(".png") and name.upper().startswith("X")
        for name in os.listdir(scan_folder)
    )


def run_scan_group(label, scan_names, data_dir, skip_existing, dry_run):
    print(f"\n{label}: {len(scan_names)} scan(s)")

    for scan_name in scan_names:
        scan_folder = scan_folder_for_name(data_dir, scan_name)
        npz_path = corrected_cube_path(scan_folder)

        if not os.path.isdir(scan_folder):
            print(f"Skipping {scan_name}: missing scan folder {scan_folder}")
            continue

        if not has_scan_images(scan_folder):
            print(f"Skipping {scan_name}: no X*_Z*.png scan images found")
            continue

        if os.path.exists(npz_path) and skip_existing:
            print(f"Skipping {scan_name}: existing {npz_path}")
            continue

        action = "Would generate" if dry_run else "Generating"
        if os.path.exists(npz_path):
            action = "Would overwrite" if dry_run else "Overwriting"
        print(f"{action} corrected cube for {scan_name}")
        if dry_run:
            continue

        try:
            _cube, saved_path = generate_cube(scan_folder)
            print(f"Saved {saved_path}")
        except Exception as exc:
            print(f"Failed {scan_name}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate corrected cubes for configured experiment scan groups."
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Experiment folder below edge/data (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--position",
        default="all",
        help="Configured position key to process, or 'all' (default: all).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Scan directory. Defaults to edge/data/<experiment>.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Ignore configured positions and process every scan_* folder in the data directory.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scans that already have cube_ZXnm_corrected.npz. By default, existing cubes are overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which cubes would be generated without writing files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir or experiment_dir(args.experiment)

    if args.discover:
        scan_names = discover_scan_names(data_dir)
        run_scan_group(
            label=f"{args.experiment} discovered scans",
            scan_names=scan_names,
            data_dir=data_dir,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        return

    position_scans = POSITION_SCANS_BY_EXPERIMENT.get(args.experiment, {})
    positions = selected_position_keys(args.experiment, args.position)

    for position in positions:
        run_scan_group(
            label=f"{args.experiment} position {position}",
            scan_names=position_scans[position],
            data_dir=data_dir,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
