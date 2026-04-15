import argparse
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "kfhsi-mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(BASE_DIR, "edge", "data")
DEFAULT_WAVELENGTHS = (450.0, 550.0, 660.0, 800.0)
PNG_PATTERN = re.compile(r"X(\d+)[_-]Z(\d+)\.png$", re.IGNORECASE)
EPS = 1e-6


@dataclass
class Finding:
    level: str
    check: str
    message: str


class Report:
    def __init__(self):
        self.findings = []

    def add(self, level, check, message):
        self.findings.append(Finding(level=level, check=check, message=message))

    def info(self, check, message):
        self.add("INFO", check, message)

    def warn(self, check, message):
        self.add("WARN", check, message)

    def fail(self, check, message):
        self.add("FAIL", check, message)

    def has_failures(self):
        return any(f.level == "FAIL" for f in self.findings)

    def counts(self):
        summary = {"FAIL": 0, "WARN": 0, "INFO": 0}
        for finding in self.findings:
            summary[finding.level] += 1
        return summary

    def render(self):
        lines = []
        for finding in self.findings:
            lines.append(f"[{finding.level}] {finding.check}: {finding.message}")
        summary = self.counts()
        lines.append("")
        lines.append(
            "Summary: "
            f"{summary['FAIL']} failure(s), {summary['WARN']} warning(s), {summary['INFO']} info item(s)"
        )
        return "\n".join(lines)


class CubeNM:
    def __init__(self, data, wavs_nm):
        self.data = data
        self.wavs_nm = np.asarray(wavs_nm, dtype=np.float32)
        self.min_nm = float(self.wavs_nm[0])
        self.max_nm = float(self.wavs_nm[-1])

    @property
    def shape(self):
        return self.data.shape

    def nearest_band_index(self, wavelength_nm):
        nm = float(wavelength_nm)
        idx = int(np.searchsorted(self.wavs_nm, nm, side="left"))
        if idx == len(self.wavs_nm):
            return len(self.wavs_nm) - 1
        if idx > 0 and abs(self.wavs_nm[idx] - nm) > abs(self.wavs_nm[idx - 1] - nm):
            return idx - 1
        return idx

    def band(self, wavelength_nm):
        return self.data[:, :, :, self.nearest_band_index(wavelength_nm)]


def find_last_modified_folder(data_dir=DATA_DIR, prefix="scan_"):
    scan_folders = [
        name
        for name in os.listdir(data_dir)
        if name.startswith(prefix) and os.path.isdir(os.path.join(data_dir, name))
    ]
    if not scan_folders:
        raise FileNotFoundError(f"No scan folders found in {data_dir}")

    scan_folders.sort(
        key=lambda name: os.path.getmtime(os.path.join(data_dir, name)),
        reverse=True,
    )
    return os.path.join(data_dir, scan_folders[0])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a generated hyperspectral cube and save inspection outputs."
    )
    parser.add_argument(
        "--scan-folder",
        default=None,
        help="Scan folder containing raw X*_Z*.png files and cube_ZXnm_corrected.npz. Defaults to the latest scan under edge/data.",
    )
    parser.add_argument(
        "--npz-path",
        default=None,
        help="Path to cube_ZXnm_corrected.npz. Defaults to <scan-folder>/cube_ZXnm_corrected.npz.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for validation outputs. Defaults to <scan-folder>/validation.",
    )
    parser.add_argument(
        "--y",
        type=int,
        default=None,
        help="Y row to use for 2D visualizations. Defaults to the middle row.",
    )
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        default=list(DEFAULT_WAVELENGTHS),
        help="Wavelength slices to save for inspection.",
    )
    parser.add_argument(
        "--max-alignment-shift",
        type=int,
        default=12,
        help="Maximum X-shift to search when estimating adjacent-row alignment.",
    )
    return parser.parse_args()


def scan_image_rows(scan_folder):
    rows = defaultdict(list)
    matched_files = []

    for filename in os.listdir(scan_folder):
        match = PNG_PATTERN.match(filename)
        if not match:
            continue
        x_pos = int(match.group(1))
        z_pos = int(match.group(2))
        full_path = os.path.join(scan_folder, filename)
        rows[z_pos].append((x_pos, full_path))
        matched_files.append(filename)

    return rows, sorted(rows.keys()), matched_files


def validate_scan_folder(scan_folder, report):
    rows, z_positions, matched_files = scan_image_rows(scan_folder)
    if not rows:
        report.fail("scan_folder", f"No files matching X*_Z*.png were found in {scan_folder}")
        return None

    report.info(
        "scan_folder",
        f"Found {len(matched_files)} scan image(s) across {len(z_positions)} Z row(s)",
    )

    duplicate_positions = []
    row_counts = {}
    x_sets_by_z = {}
    frame_shapes = set()
    unreadable_files = []

    for z_pos in z_positions:
        seen_x = set()
        for x_pos, path in sorted(rows[z_pos], key=lambda item: item[0]):
            if x_pos in seen_x:
                duplicate_positions.append((z_pos, x_pos, os.path.basename(path)))
            seen_x.add(x_pos)

            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                unreadable_files.append(path)
                continue
            frame_shapes.add(image.shape)

        row_counts[z_pos] = len(rows[z_pos])
        x_sets_by_z[z_pos] = tuple(sorted(item[0] for item in rows[z_pos]))

    if duplicate_positions:
        sample = ", ".join(
            f"Z={z}, X={x} ({name})" for z, x, name in duplicate_positions[:5]
        )
        report.fail("scan_duplicates", f"Duplicate X positions detected: {sample}")
    else:
        report.info("scan_duplicates", "No duplicate X positions detected within any Z row")

    if unreadable_files:
        sample = ", ".join(os.path.basename(path) for path in unreadable_files[:5])
        report.fail("scan_read", f"Unreadable scan frame(s): {sample}")
    else:
        report.info("scan_read", "All scan frames were readable")

    unique_counts = sorted(set(row_counts.values()))
    if len(unique_counts) == 1:
        report.info(
            "scan_row_counts",
            f"All Z rows contain {unique_counts[0]} X positions",
        )
    else:
        report.fail(
            "scan_row_counts",
            "Inconsistent X counts per Z row: "
            + ", ".join(f"Z={z}:{count}" for z, count in row_counts.items()),
        )

    reference_x_set = next(iter(x_sets_by_z.values()))
    mismatched_rows = [
        z_pos for z_pos, x_values in x_sets_by_z.items() if x_values != reference_x_set
    ]
    if mismatched_rows:
        report.fail(
            "scan_x_pattern",
            "Rows do not share the same X positions: "
            + ", ".join(str(z_pos) for z_pos in mismatched_rows[:10]),
        )
    else:
        report.info(
            "scan_x_pattern",
            f"All rows share the same X-position pattern ({len(reference_x_set)} positions)",
        )

    if len(frame_shapes) == 1:
        shape = next(iter(frame_shapes))
        report.info("scan_frame_shape", f"All scan frames have shape {shape}")
    else:
        report.fail(
            "scan_frame_shape",
            "Inconsistent scan frame shapes: "
            + ", ".join(str(shape) for shape in sorted(frame_shapes)),
        )

    return {
        "rows": rows,
        "z_positions": z_positions,
        "row_counts": row_counts,
        "reference_x_set": reference_x_set,
        "frame_shape": next(iter(frame_shapes)) if len(frame_shapes) == 1 else None,
    }


def load_cube(npz_path):
    with np.load(npz_path) as data:
        if "cube" not in data or "wavs_nm" not in data:
            raise RuntimeError("NPZ file must contain 'cube' and 'wavs_nm'")

        payload = {
            "cube": data["cube"],
            "wavs_nm": data["wavs_nm"],
        }
        for key in ("Zs", "xslice_start", "xslice_stop"):
            if key in data:
                payload[key] = data[key]
    return payload


def validate_cube_payload(payload, scan_info, report):
    cube_data = payload["cube"]
    wavs_nm = np.asarray(payload["wavs_nm"], dtype=np.float32)

    if cube_data.ndim != 4:
        report.fail("cube_shape", f"Cube must be 4D (Z, X, Y, W), got shape {cube_data.shape}")
        return None

    z_count, x_count, y_count, w_count = cube_data.shape
    report.info("cube_shape", f"Cube shape is {cube_data.shape}")

    if w_count != len(wavs_nm):
        report.fail(
            "cube_wavelength_axis",
            f"Cube spectral dimension {w_count} does not match wavs_nm length {len(wavs_nm)}",
        )
    else:
        report.info(
            "cube_wavelength_axis",
            f"Cube contains {w_count} spectral band(s) from {wavs_nm[0]:.2f} to {wavs_nm[-1]:.2f} nm",
        )

    if scan_info is not None:
        if z_count != len(scan_info["z_positions"]):
            report.fail(
                "cube_z_count",
                f"Cube has {z_count} Z layers, but scan folder has {len(scan_info['z_positions'])}",
            )
        else:
            report.info("cube_z_count", "Cube Z count matches the scan folder")

        raw_x_count = len(scan_info["reference_x_set"])
        if x_count > raw_x_count:
            report.fail(
                "cube_x_count",
                f"Cube X count {x_count} is larger than raw scan X count {raw_x_count}",
            )
        else:
            report.info(
                "cube_x_count",
                f"Cube X count is {x_count}; raw scan X count before crop is {raw_x_count}",
            )

        frame_shape = scan_info["frame_shape"]
        if frame_shape is not None:
            expected_y = frame_shape[0]
            if y_count != expected_y:
                report.fail(
                    "cube_y_count",
                    f"Cube Y count {y_count} does not match raw frame height {expected_y}",
                )
            else:
                report.info("cube_y_count", "Cube Y count matches the raw frame height")

    if "Zs" in payload:
        saved_z = np.asarray(payload["Zs"]).astype(int).tolist()
        if scan_info is not None and saved_z != scan_info["z_positions"]:
            report.fail("cube_z_metadata", "Saved Z positions do not match the scan folder ordering")
        else:
            report.info("cube_z_metadata", f"Saved Z metadata contains {len(saved_z)} position(s)")

    start = None
    stop = None
    if "xslice_start" in payload and "xslice_stop" in payload:
        start = int(np.asarray(payload["xslice_start"]).item())
        stop = int(np.asarray(payload["xslice_stop"]).item())
        if start >= 0 and stop >= 0:
            expected_x = stop - start
            if expected_x != x_count:
                report.fail(
                    "cube_xslice",
                    f"xslice metadata [{start}:{stop}] implies width {expected_x}, but cube width is {x_count}",
                )
            else:
                report.info(
                    "cube_xslice",
                    f"xslice metadata [{start}:{stop}] matches the saved cube width",
                )

    nan_count = int(np.isnan(cube_data).sum())
    inf_count = int(np.isinf(cube_data).sum())
    if nan_count or inf_count:
        report.fail("cube_finite", f"Cube contains {nan_count} NaN value(s) and {inf_count} inf value(s)")
    else:
        report.info("cube_finite", "Cube contains no NaN or inf values")

    finite_mask = np.isfinite(cube_data)
    if np.any(finite_mask):
        report.info(
            "cube_stats",
            "Cube intensity stats: "
            f"min={float(np.min(cube_data[finite_mask])):.3f}, "
            f"max={float(np.max(cube_data[finite_mask])):.3f}, "
            f"mean={float(np.mean(cube_data[finite_mask])):.3f}",
        )

    if len(wavs_nm) < 2:
        report.fail("wavelength_monotonic", "wavs_nm must contain at least two entries")
    elif not np.all(np.isfinite(wavs_nm)):
        report.fail("wavelength_monotonic", "wavs_nm contains non-finite values")
    elif not np.all(np.diff(wavs_nm) > 0):
        report.fail("wavelength_monotonic", "wavs_nm is not strictly increasing")
    else:
        band_spacing = np.diff(wavs_nm)
        report.info(
            "wavelength_monotonic",
            f"Wavelength axis is strictly increasing with median spacing {float(np.median(band_spacing)):.3f} nm",
        )

    return CubeNM(cube_data.astype(np.float32, copy=False), wavs_nm)


def best_profile_shift(profile_a, profile_b, max_shift):
    best_shift = 0
    best_corr = -np.inf

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            a = profile_a[shift:]
            b = profile_b[: len(profile_b) - shift]
        else:
            offset = -shift
            a = profile_a[: len(profile_a) - offset]
            b = profile_b[offset:]

        if len(a) < 3 or len(b) < 3:
            continue

        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        corr = -np.inf if denom <= EPS else float(np.dot(a, b) / denom)
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    return best_shift, best_corr


def validate_adjacent_alignment(cube, max_shift, report):
    profiles = cube.data.mean(axis=(2, 3))
    shifts = []
    corrs = []

    for z_idx in range(profiles.shape[0] - 1):
        shift, corr = best_profile_shift(profiles[z_idx], profiles[z_idx + 1], max_shift=max_shift)
        shifts.append(shift)
        corrs.append(corr)

    if not shifts:
        report.warn("alignment", "Cube has fewer than two Z rows; skipping adjacent-row alignment check")
        return

    median_abs_shift = float(np.median(np.abs(shifts)))
    mean_corr = float(np.mean(corrs))
    report.info(
        "alignment",
        "Adjacent-row alignment estimate: "
        f"median |shift|={median_abs_shift:.2f} px, "
        f"max |shift|={max(abs(shift) for shift in shifts)} px, "
        f"mean corr={mean_corr:.3f}",
    )

    if median_abs_shift > 1.5:
        report.warn(
            "alignment",
            "Adjacent Z rows still look offset in X. Inspect the saved slice images for a residual zig-zag.",
        )

    if np.mean(np.abs(shifts)) > 2.0 and len(shifts) >= 4:
        even_median = float(np.median(shifts[::2]))
        odd_median = float(np.median(shifts[1::2]))
        if even_median * odd_median < 0:
            report.warn(
                "alignment_pattern",
                "Adjacent-row shifts alternate in sign, which may indicate residual snake-pattern misalignment.",
            )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_image(image):
    lo, hi = np.percentile(image, (1, 99))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def select_y(cube, requested_y):
    if requested_y is None:
        return cube.shape[2] // 2
    if requested_y < 0 or requested_y >= cube.shape[2]:
        raise ValueError(f"y must be between 0 and {cube.shape[2] - 1}, got {requested_y}")
    return requested_y


def save_rgb_image(cube, out_path, y):
    target_wavelengths = {"R": 660.0, "G": 550.0, "B": 460.0}
    channels = []
    actual = {}

    for channel_name in ("R", "G", "B"):
        idx = cube.nearest_band_index(target_wavelengths[channel_name])
        actual[channel_name] = float(cube.wavs_nm[idx])
        channel = cube.data[:, :, y, idx]
        channels.append(normalize_image(channel))

    rgb = np.stack(channels, axis=-1)
    plt.imsave(out_path, rgb)
    return actual


def save_wavelength_slice(cube, requested_nm, out_path, y):
    idx = cube.nearest_band_index(requested_nm)
    actual_nm = float(cube.wavs_nm[idx])
    image = cube.data[:, :, y, idx]
    normalized = normalize_image(image)

    plt.figure(figsize=(8, 6))
    plt.imshow(normalized, cmap="gray", aspect="auto")
    plt.xlabel("X position")
    plt.ylabel("Z position")
    plt.title(f"Slice at {actual_nm:.1f} nm (requested {requested_nm:.1f} nm)")
    plt.colorbar(label="Normalized intensity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return actual_nm


def build_sample_points(cube, y):
    z_count, x_count, _, _ = cube.shape
    z_mid = z_count // 2
    x_mid = x_count // 2
    z_step = 1 if z_count > 1 else 0
    x_step = 1 if x_count > 1 else 0

    candidates = [
        ("center", z_mid, x_mid, y),
        ("left", z_mid, max(0, x_mid - x_step), y),
        ("right", z_mid, min(x_count - 1, x_mid + x_step), y),
        ("up", max(0, z_mid - z_step), x_mid, y),
        ("down", min(z_count - 1, z_mid + z_step), x_mid, y),
    ]

    unique = []
    seen = set()
    for label, z_pos, x_pos, y_pos in candidates:
        key = (z_pos, x_pos, y_pos)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, z_pos, x_pos, y_pos))
    return unique


def save_sample_spectra(cube, out_path, y, report):
    sample_points = build_sample_points(cube, y)
    plt.figure(figsize=(10, 6))
    center_spec = None

    for label, z_pos, x_pos, y_pos in sample_points:
        spectrum = cube.data[z_pos, x_pos, y_pos, :]
        if label == "center":
            center_spec = spectrum
        plt.plot(cube.wavs_nm, spectrum, label=f"{label}: Z={z_pos}, X={x_pos}, Y={y_pos}")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Sample spectra from the cube")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    if center_spec is not None and len(sample_points) > 1:
        correlations = []
        center_centered = center_spec - np.mean(center_spec)
        center_norm = np.linalg.norm(center_centered)
        for label, z_pos, x_pos, y_pos in sample_points:
            if label == "center":
                continue
            spectrum = cube.data[z_pos, x_pos, y_pos, :]
            spectrum_centered = spectrum - np.mean(spectrum)
            denom = center_norm * np.linalg.norm(spectrum_centered)
            corr = np.nan if denom <= EPS else float(np.dot(center_centered, spectrum_centered) / denom)
            correlations.append((label, corr))

        if correlations:
            report.info(
                "spectra_similarity",
                "Center-spectrum correlation to nearby samples: "
                + ", ".join(f"{label}={corr:.3f}" for label, corr in correlations),
            )


def save_mean_spectrum(cube, out_path):
    mean_spectrum = cube.data.mean(axis=(0, 1, 2))
    plt.figure(figsize=(10, 6))
    plt.plot(cube.wavs_nm, mean_spectrum, color="black")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean intensity")
    plt.title("Mean spectrum across the full cube")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_report(report, out_path):
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(report.render())
        handle.write("\n")


def main():
    args = parse_args()

    scan_folder = args.scan_folder or find_last_modified_folder()
    npz_path = args.npz_path or os.path.join(scan_folder, "cube_ZXnm_corrected.npz")
    out_dir = args.out_dir or os.path.join(scan_folder, "validation")

    report = Report()
    report.info("paths", f"Using scan folder: {scan_folder}")
    report.info("paths", f"Using cube file: {npz_path}")
    report.info("paths", f"Saving validation outputs to: {out_dir}")

    if not os.path.isdir(scan_folder):
        report.fail("paths", f"Scan folder does not exist: {scan_folder}")
        print(report.render())
        return 1

    if not os.path.isfile(npz_path):
        report.fail("paths", f"Cube file does not exist: {npz_path}")
        print(report.render())
        return 1

    ensure_dir(out_dir)

    scan_info = validate_scan_folder(scan_folder, report)

    try:
        payload = load_cube(npz_path)
    except Exception as error:
        report.fail("cube_load", str(error))
        print(report.render())
        write_report(report, os.path.join(out_dir, "validation_report.txt"))
        return 1

    cube = validate_cube_payload(payload, scan_info, report)
    if cube is None:
        print(report.render())
        write_report(report, os.path.join(out_dir, "validation_report.txt"))
        return 1

    try:
        y = select_y(cube, args.y)
    except Exception as error:
        report.fail("y_selection", str(error))
        print(report.render())
        write_report(report, os.path.join(out_dir, "validation_report.txt"))
        return 1

    report.info("y_selection", f"Using Y={y} for 2D inspection outputs")
    validate_adjacent_alignment(cube, max_shift=args.max_alignment_shift, report=report)

    rgb_actual = save_rgb_image(cube, os.path.join(out_dir, "rgb_validation.png"), y=y)
    report.info(
        "rgb_validation",
        "Saved RGB reconstruction using nearest bands "
        f"R={rgb_actual['R']:.1f} nm, G={rgb_actual['G']:.1f} nm, B={rgb_actual['B']:.1f} nm",
    )

    actual_slices = []
    for wavelength in args.wavelengths:
        filename = f"slice_{int(round(wavelength))}nm.png"
        actual_nm = save_wavelength_slice(
            cube,
            requested_nm=wavelength,
            out_path=os.path.join(out_dir, filename),
            y=y,
        )
        actual_slices.append((wavelength, actual_nm))

    report.info(
        "slice_validation",
        "Saved wavelength slices: "
        + ", ".join(f"{requested:.1f}->{actual:.1f} nm" for requested, actual in actual_slices),
    )

    save_sample_spectra(cube, os.path.join(out_dir, "sample_spectra.png"), y=y, report=report)
    report.info("sample_spectra", "Saved sample spectra plot")

    save_mean_spectrum(cube, os.path.join(out_dir, "mean_spectrum.png"))
    report.info("mean_spectrum", "Saved mean spectrum plot")

    report_path = os.path.join(out_dir, "validation_report.txt")
    write_report(report, report_path)

    print(report.render())
    print(f"\nValidation report written to: {report_path}")
    return 1 if report.has_failures() else 0


if __name__ == "__main__":
    raise SystemExit(main())
