import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_raw_image(raw_path, flip_x=False):
    image = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read raw image: {raw_path}")

    image = image.astype(np.float32)
    if flip_x:
        image = np.fliplr(image)
    return image


def _resolve_y_roi(height, y_roi):
    if y_roi is None:
        return 0, height

    if len(y_roi) != 2:
        raise ValueError("y_roi must be a tuple/list like (y0, y1)")

    y0 = max(0, int(y_roi[0]))
    y1 = min(height, int(y_roi[1]))
    if y1 <= y0:
        raise ValueError(f"Invalid y_roi={y_roi} for image height {height}")
    return y0, y1


def extract_spectrum_from_row(
    raw_path,
    row,
    start_nm=400.0,
    end_nm=800.0,
    flip_x=False,
    show_plot=False,
):
    """
    Read a raw image and return one row as a spectrum.

    This helper uses a simple linear wavelength axis for quick inspection only.
    The smile estimator itself works in pixel coordinates.
    """
    image = load_raw_image(raw_path, flip_x=flip_x)
    height, width = image.shape
    if not (0 <= row < height):
        raise ValueError(f"Row {row} is out of range [0, {height - 1}]")

    wavelengths = np.linspace(start_nm, end_nm, width, dtype=np.float32)
    spectrum = image[row, :].astype(np.float32)

    if show_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(width), spectrum)
        plt.xlabel("Pixel")
        plt.ylabel("Intensity")
        plt.title(f"Spectrum from raw image (row={row})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return wavelengths, spectrum


def _find_local_peaks(profile, min_height, min_distance_pixels):
    if profile.ndim != 1:
        raise ValueError("profile must be 1D")

    if len(profile) < 3:
        return np.array([], dtype=int)

    candidates = np.flatnonzero(
        (profile[1:-1] > profile[:-2])
        & (profile[1:-1] >= profile[2:])
        & (profile[1:-1] >= min_height)
    ) + 1

    if len(candidates) == 0:
        return np.array([], dtype=int)

    min_distance_pixels = max(1, int(min_distance_pixels))
    order = np.argsort(profile[candidates])[::-1]
    selected = []

    for idx in candidates[order]:
        if all(abs(idx - kept) >= min_distance_pixels for kept in selected):
            selected.append(int(idx))

    selected.sort()
    return np.array(selected, dtype=int)


def locate_peaks(spec, wavs=None, noise_level=20, min_distance_pixels=2):
    """
    Locate peaks in a 1D spectrum.

    If wavs is provided, the second returned array contains wavelengths.
    Otherwise it contains pixel indices as float32.
    """
    spec = np.asarray(spec, dtype=np.float32)
    peaks_idx = _find_local_peaks(
        spec,
        min_height=float(noise_level),
        min_distance_pixels=min_distance_pixels,
    )

    if wavs is None:
        peak_axis = peaks_idx.astype(np.float32)
    else:
        wavs = np.asarray(wavs, dtype=np.float32)
        if len(wavs) != len(spec):
            raise ValueError("wavs and spec must have the same length")
        peak_axis = wavs[peaks_idx]

    peak_vals = spec[peaks_idx]

    print(f"Found {len(peaks_idx)} peaks above noise level {noise_level}.")
    for axis_value, value in zip(peak_axis, peak_vals):
        label = "pixel" if wavs is None else "nm"
        print(f"{label} = {axis_value:.2f}, intensity = {value:.1f}")

    return peaks_idx, peak_axis, peak_vals


def detect_spectral_lines(image, peak_thresh_rel=0.3, y_roi=None, min_distance_pixels=5):
    """
    Detect bright spectral lines from the y-summed profile of a calibration image.
    """
    height, _ = image.shape
    y0, y1 = _resolve_y_roi(height, y_roi)
    profile = image[y0:y1, :].sum(axis=0).astype(np.float32)

    max_value = float(profile.max())
    if max_value <= 0:
        raise RuntimeError("Image profile is empty or zero within the selected ROI.")

    threshold = peak_thresh_rel * max_value
    peak_xs = _find_local_peaks(
        profile,
        min_height=threshold,
        min_distance_pixels=min_distance_pixels,
    )
    if len(peak_xs) == 0:
        raise RuntimeError("No spectral lines found. Try lowering peak_thresh_rel.")

    return profile, peak_xs, (y0, y1)


def _windowed_centroid(row_profile, center_x, window_half_width):
    width = len(row_profile)
    center_idx = int(round(float(center_x)))
    x_start = max(0, center_idx - window_half_width)
    x_end = min(width, center_idx + window_half_width + 1)

    window = row_profile[x_start:x_end].astype(np.float32)
    total = float(window.sum())
    if total <= 0:
        return None

    xs_local = np.arange(x_start, x_end, dtype=np.float32)
    centroid = float(np.dot(xs_local, window) / total)
    return centroid


def _track_line_from_reference_row(image, peak_x, y0, y1, window_half_width, reference_row):
    tracked = {}

    center_x = _windowed_centroid(image[reference_row, :], peak_x, window_half_width)
    if center_x is None:
        return None

    current_x = center_x
    for y in range(reference_row, y0 - 1, -1):
        centroid = _windowed_centroid(image[y, :], current_x, window_half_width)
        if centroid is not None:
            tracked[y] = centroid
            current_x = centroid

    current_x = center_x
    for y in range(reference_row + 1, y1):
        centroid = _windowed_centroid(image[y, :], current_x, window_half_width)
        if centroid is not None:
            tracked[y] = centroid
            current_x = centroid

    if not tracked:
        return None

    ys = np.array(sorted(tracked.keys()), dtype=np.float32)
    xs = np.array([tracked[int(y)] for y in ys], dtype=np.float32)
    return ys, xs


def _fit_polynomial_track(ys, xs, degree):
    fit_degree = min(int(degree), len(ys) - 1)
    if fit_degree < 1:
        return None
    return np.polyfit(ys.astype(np.float32), xs.astype(np.float32), fit_degree)


def _interpolate_displacement_field(source_line_xs, target_line_xs, width):
    x_grid = np.arange(width, dtype=np.float32)
    line_count, row_count = source_line_xs.shape
    if line_count < 2:
        raise RuntimeError("Need at least two tracked spectral lines to interpolate a full correction field.")

    dx_roi = np.zeros((row_count, width), dtype=np.float32)

    for row_idx in range(row_count):
        source_x = source_line_xs[:, row_idx].astype(np.float32)
        delta_x = target_line_xs.astype(np.float32) - source_x

        order = np.argsort(source_x)
        source_sorted = source_x[order]
        delta_sorted = delta_x[order]

        unique_source, unique_indices = np.unique(source_sorted, return_index=True)
        delta_unique = delta_sorted[unique_indices]

        dx_roi[row_idx, :] = np.interp(
            x_grid,
            unique_source,
            delta_unique,
            left=float(delta_unique[0]),
            right=float(delta_unique[-1]),
        )

    return dx_roi


def estimate_smile_correction_field_from_raw(
    raw_path,
    peak_thresh_rel=0.3,
    window_half_width=5,
    y_roi=None,
    min_distance_pixels=5,
    line_fit_degree=2,
    reference_row=None,
    flip_x=False,
):
    """
    Estimate a smile correction field from a calibration image with narrow vertical lines.

    Returns a dictionary containing:
      - dx_field: full-image forward x displacement field
      - dx_roi: displacement field within the fitted y ROI
      - source_line_xs_roi: fitted line centers per line and row
      - target_line_xs: x positions that each line is corrected toward

    The field is a forward displacement: x_corrected = x_raw + dx_field[y, x_raw].
    """
    image = load_raw_image(raw_path, flip_x=flip_x)
    height, width = image.shape

    profile, peak_xs, (y0, y1) = detect_spectral_lines(
        image,
        peak_thresh_rel=peak_thresh_rel,
        y_roi=y_roi,
        min_distance_pixels=min_distance_pixels,
    )

    if reference_row is None:
        reference_row = (y0 + y1 - 1) // 2
    elif not (y0 <= int(reference_row) < y1):
        raise ValueError(f"reference_row must be within [{y0}, {y1 - 1}]")
    else:
        reference_row = int(reference_row)

    y_eval_roi = np.arange(y0, y1, dtype=np.float32)
    line_results = []
    source_line_xs = []
    target_line_xs = []

    for line_index, peak_x in enumerate(peak_xs):
        tracked = _track_line_from_reference_row(
            image,
            peak_x=peak_x,
            y0=y0,
            y1=y1,
            window_half_width=window_half_width,
            reference_row=reference_row,
        )
        if tracked is None:
            continue

        ys, xs = tracked
        min_samples = max(7, int(line_fit_degree) + 1)
        if len(xs) < min_samples:
            continue

        coeffs = _fit_polynomial_track(ys, xs, degree=line_fit_degree)
        if coeffs is None:
            continue

        fitted_xs_roi = np.polyval(coeffs, y_eval_roi).astype(np.float32)
        target_x = float(np.polyval(coeffs, reference_row))

        linear_coeffs = np.polyfit(y_eval_roi, fitted_xs_roi, 1)
        linear_fit_roi = np.polyval(linear_coeffs, y_eval_roi).astype(np.float32)
        smile_residual_roi = fitted_xs_roi - linear_fit_roi

        source_line_xs.append(fitted_xs_roi)
        target_line_xs.append(target_x)
        line_results.append(
            {
                "line_index": int(line_index),
                "peak_x0": int(peak_x),
                "ys": ys,
                "xs": xs,
                "fit_coeffs": tuple(float(v) for v in coeffs),
                "fit_xs_roi": fitted_xs_roi,
                "target_x": target_x,
                "smile_roi": smile_residual_roi,
                "max_abs_smile_pix": float(np.max(np.abs(smile_residual_roi))),
            }
        )

    if len(line_results) < 2:
        raise RuntimeError("Need at least two valid tracked lines to estimate a smile correction field.")

    source_line_xs_roi = np.stack(source_line_xs, axis=0).astype(np.float32)
    target_line_xs = np.array(target_line_xs, dtype=np.float32)
    dx_roi = _interpolate_displacement_field(source_line_xs_roi, target_line_xs, width=width)

    dx_field = np.zeros((height, width), dtype=np.float32)
    dx_field[y0:y1, :] = dx_roi
    dx_field[:y0, :] = dx_roi[0, :]
    dx_field[y1:, :] = dx_roi[-1, :]

    return {
        "raw_path": raw_path,
        "image": image,
        "image_shape": (height, width),
        "profile": profile,
        "peak_xs": peak_xs.astype(np.int32),
        "y_roi": (int(y0), int(y1)),
        "reference_row": int(reference_row),
        "fit_y_roi": y_eval_roi.astype(np.float32),
        "line_results": line_results,
        "source_line_xs_roi": source_line_xs_roi,
        "target_line_xs": target_line_xs,
        "dx_roi": dx_roi,
        "dx_field": dx_field,
    }


def quantify_smile_from_raw(
    raw_path,
    peak_thresh_rel=0.3,
    window_half_width=5,
    y_roi=None,
    nm_per_pixel=None,
    show_debug=True,
    min_distance_pixels=5,
    line_fit_degree=2,
    reference_row=None,
    flip_x=False,
):
    """
    Quantify smile from a raw calibration image with bright vertical lines.

    This preserves the old high-level behavior, but now uses polynomial-smoothed
    line tracks from the field estimator.
    """
    field = estimate_smile_correction_field_from_raw(
        raw_path=raw_path,
        peak_thresh_rel=peak_thresh_rel,
        window_half_width=window_half_width,
        y_roi=y_roi,
        min_distance_pixels=min_distance_pixels,
        line_fit_degree=line_fit_degree,
        reference_row=reference_row,
        flip_x=flip_x,
    )

    results = {}
    for line_number, line in enumerate(field["line_results"]):
        smile = line["smile_roi"]
        entry = {
            "line_x0": int(line["peak_x0"]),
            "ys": field["fit_y_roi"],
            "xs": line["fit_xs_roi"],
            "fit_coeffs": line["fit_coeffs"],
            "smile": smile,
            "max_smile_pix": float(np.max(smile)),
            "min_smile_pix": float(np.min(smile)),
            "max_abs_smile_pix": float(np.max(np.abs(smile))),
        }

        if nm_per_pixel is not None:
            entry["max_smile_nm"] = entry["max_smile_pix"] * nm_per_pixel
            entry["min_smile_nm"] = entry["min_smile_pix"] * nm_per_pixel
            entry["max_abs_smile_nm"] = entry["max_abs_smile_pix"] * nm_per_pixel

        results[line_number] = entry

    if show_debug:
        plot_smile_field_diagnostics(field)

    return results


def save_smile_correction_field(output_path, field):
    np.savez_compressed(
        output_path,
        dx_field=field["dx_field"],
        dx_roi=field["dx_roi"],
        peak_xs=field["peak_xs"],
        fit_y_roi=field["fit_y_roi"],
        source_line_xs_roi=field["source_line_xs_roi"],
        target_line_xs=field["target_line_xs"],
        y0=field["y_roi"][0],
        y1=field["y_roi"][1],
        reference_row=field["reference_row"],
        image_height=field["image_shape"][0],
        image_width=field["image_shape"][1],
    )


def plot_smile_field_diagnostics(field):
    image = field["image"]
    y0, y1 = field["y_roi"]

    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap="gray", aspect="auto")
    for line in field["line_results"]:
        plt.plot(line["xs"], line["ys"], ".", markersize=1.5)
        plt.plot(line["fit_xs_roi"], field["fit_y_roi"], "-", linewidth=1.0)
    plt.axhline(field["reference_row"], color="tab:red", linewidth=0.8, linestyle="--")
    plt.xlabel("x (dispersion)")
    plt.ylabel("y (slit)")
    plt.title("Tracked spectral lines and polynomial fits")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.imshow(
        field["dx_roi"],
        cmap="coolwarm",
        aspect="auto",
        origin="upper",
        extent=(0, image.shape[1], y1, y0),
    )
    plt.colorbar(label="Forward x shift [pixels]")
    plt.xlabel("x (dispersion)")
    plt.ylabel("y (slit)")
    plt.title("Estimated smile correction field")
    plt.tight_layout()
    plt.show()

    first_line = field["line_results"][0]
    plt.figure(figsize=(6, 4))
    plt.plot(field["fit_y_roi"], first_line["smile_roi"], ".-")
    plt.axhline(0.0, color="k", linewidth=0.7)
    plt.xlabel("Row (y)")
    plt.ylabel("Smile residual [pixels]")
    plt.title("Smile residual for first fitted line")
    plt.tight_layout()
    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate a smile correction field from a calibration image with narrow spectral lines."
    )
    parser.add_argument("raw_path", help="Path to the calibration image")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional .npz path where the estimated correction field will be saved",
    )
    parser.add_argument(
        "--y-roi",
        nargs=2,
        type=int,
        metavar=("Y0", "Y1"),
        default=None,
        help="Optional y-range to use for line tracking",
    )
    parser.add_argument(
        "--reference-row",
        type=int,
        default=None,
        help="Reference row the spectral lines are corrected toward. Defaults to ROI midpoint.",
    )
    parser.add_argument("--peak-thresh-rel", type=float, default=0.3)
    parser.add_argument("--min-peak-distance", type=int, default=5)
    parser.add_argument("--window-half-width", type=int, default=5)
    parser.add_argument("--line-fit-degree", type=int, default=2)
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--show-debug", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()

    field = estimate_smile_correction_field_from_raw(
        raw_path=args.raw_path,
        peak_thresh_rel=args.peak_thresh_rel,
        window_half_width=args.window_half_width,
        y_roi=args.y_roi,
        min_distance_pixels=args.min_peak_distance,
        line_fit_degree=args.line_fit_degree,
        reference_row=args.reference_row,
        flip_x=args.flip_x,
    )

    max_abs_smiles = [line["max_abs_smile_pix"] for line in field["line_results"]]
    print(f"Tracked {len(field['line_results'])} line(s)")
    print(f"Reference row: {field['reference_row']}")
    print(f"Max |smile| across fitted lines: {max(max_abs_smiles):.3f} pixels")
    print(
        "Displacement field stats: "
        f"min={float(np.min(field['dx_roi'])):.3f} px, "
        f"max={float(np.max(field['dx_roi'])):.3f} px"
    )

    if args.out:
        save_smile_correction_field(args.out, field)
        print(f"Saved smile correction field to {args.out}")

    if args.show_debug:
        plot_smile_field_diagnostics(field)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
