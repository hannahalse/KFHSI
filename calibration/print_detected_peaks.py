#!/usr/bin/env python3
"""
Preprocess Ar/Hg calibration images to match scan frames, extract spectra,
detect peaks, and print the peak pixel lists.

This is intended for manual calibration workflows where the user wants to
copy the detected peak positions into another tool such as GeoGebra.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from Ar_wavelength_calibr import (
    ar_original_name,
    hg_original_name,
    base_dir,
    flip_original_x,
    roi_top,
    roi_bottom,
    binning_factor,
    y_roi,
    min_height_rel,
    min_distance,
    prominence_rel,
    load_gray,
    preprocess,
    integrated_spectrum,
    subtract_ar_from_hg_images,
)


TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 13


def detect_peaks(spec):
    spec = np.asarray(spec, dtype=np.float32)
    max_val = float(spec.max())
    if max_val <= 0:
        raise ValueError("Spectrum has no positive signal (max <= 0).")

    peaks_idx, _ = find_peaks(
        spec,
        height=min_height_rel * max_val,
        distance=min_distance,
        prominence=prominence_rel * max_val,
    )
    peak_vals = spec[peaks_idx]
    order = np.argsort(peaks_idx)
    return peaks_idx[order], peak_vals[order]


def print_peak_list(name, peaks_idx, peak_vals):
    print(f"\n{name}: Found {len(peaks_idx)} peaks")
    for i, (pixel, intensity) in enumerate(zip(peaks_idx, peak_vals), start=1):
        print(f"{i:2d}: pixel={int(pixel):4d}, intensity={float(intensity):.1f}")

    pixel_list = ", ".join(str(int(pixel)) for pixel in peaks_idx)
    print(f"{name} peak pixels = [{pixel_list}]")


def plot_spectrum(name, spectrum, peaks_idx):
    plt.figure(figsize=(10, 4))
    plt.plot(spectrum, linewidth=1.5, label=name)
    if len(peaks_idx) > 0:
        plt.plot(peaks_idx, spectrum[peaks_idx], "x", markersize=7, label="Peaks")
    plt.title(f"{name} spectrum with detected peaks", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Pixel", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Integrated intensity", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.show()


def save_peaks_csv(csv_path, ar_peaks_idx, ar_peak_vals, hg_peaks_idx, hg_peak_vals):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "rank", "pixel", "intensity"])
        writer.writeheader()

        for rank, (pixel, intensity) in enumerate(zip(ar_peaks_idx, ar_peak_vals), start=1):
            writer.writerow(
                {
                    "source": "Ar",
                    "rank": rank,
                    "pixel": int(pixel),
                    "intensity": float(intensity),
                }
            )

        for rank, (pixel, intensity) in enumerate(zip(hg_peaks_idx, hg_peak_vals), start=1):
            writer.writerow(
                {
                    "source": "Hg_only",
                    "rank": rank,
                    "pixel": int(pixel),
                    "intensity": float(intensity),
                }
            )

    print(f"\nSaved peaks CSV: {os.path.abspath(csv_path)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print detected peak pixels for preprocessed Ar and Hg calibration images."
    )
    parser.add_argument(
        "--ar-path",
        default=os.path.join(base_dir, ar_original_name),
        help="Path to the original Ar calibration image.",
    )
    parser.add_argument(
        "--hg-path",
        default=os.path.join(base_dir, hg_original_name),
        help="Path to the original Hg calibration image.",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable horizontal flipping before preprocessing.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots of the spectra with detected peaks.",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional path to save the detected peaks as CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    do_flip = not args.no_flip if flip_original_x else False

    ar_orig = load_gray(args.ar_path, flip_x=do_flip)
    hg_orig = load_gray(args.hg_path, flip_x=do_flip)

    ar_pre = preprocess(ar_orig)
    hg_pre = preprocess(hg_orig)

    print("Preprocess:")
    print(f"  Ar: {ar_orig.shape} -> {ar_pre.shape} (flip_x={do_flip})")
    print(f"  Hg: {hg_orig.shape} -> {hg_pre.shape} (flip_x={do_flip})")
    print(
        "  Settings:"
        f" roi_top={roi_top}, roi_bottom={roi_bottom}, binning_factor={binning_factor}, y_roi={y_roi}"
    )

    ar_spec = integrated_spectrum(ar_pre, y_roi)
    _, hg_only_spec, _, _, dx, alpha = subtract_ar_from_hg_images(
        hg_img=hg_pre,
        ar_img=ar_pre,
        y_roi_local=y_roi,
        show_debug=False,
    )

    print(f"  Hg-Ar alignment shift: {dx} px")
    print(f"  Hg-Ar scale alpha:     {alpha:.4f}")

    ar_peaks_idx, ar_peak_vals = detect_peaks(ar_spec)
    hg_peaks_idx, hg_peak_vals = detect_peaks(hg_only_spec)

    print_peak_list("Ar", ar_peaks_idx, ar_peak_vals)
    print_peak_list("Hg only", hg_peaks_idx, hg_peak_vals)

    if args.save_csv:
        save_peaks_csv(
            args.save_csv,
            ar_peaks_idx,
            ar_peak_vals,
            hg_peaks_idx,
            hg_peak_vals,
        )

    if args.plot:
        plot_spectrum("Ar", ar_spec, ar_peaks_idx)
        plot_spectrum("Hg only", hg_only_spec, hg_peaks_idx)


if __name__ == "__main__":
    main()
