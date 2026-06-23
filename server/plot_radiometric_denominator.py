import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from radiometric_calibration import load_reference_source, wavelength_axis


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WHITE_REFERENCE_SOURCE = os.path.join(BASE_DIR, "calibration", "whiteReference")
DARK_REFERENCE_SOURCE = os.path.join(BASE_DIR, "calibration", "darkReference")
WL_MIN = 380.0
WL_MAX = 820.0
EPS = 1e-6

LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 20


def mean_reference_spectrum(reference_source, flip_x=True, wl_min=WL_MIN, wl_max=WL_MAX):
    """
    Load one reference source and return its mean spectrum over image height.

    The source can be either:
    - one PNG file
    - a directory of PNGs, in which case the mean image is used
    """
    image = load_reference_source(reference_source, flip_x=flip_x)
    wavs_nm = wavelength_axis(image.shape[1]).astype(np.float32)

    mask = (wavs_nm >= wl_min) & (wavs_nm <= wl_max)
    if not np.any(mask):
        raise RuntimeError(
            f"No wavelengths remain after clipping to {wl_min:.1f}-{wl_max:.1f} nm"
        )

    spectrum = image[:, mask].mean(axis=0).astype(np.float32)
    return wavs_nm[mask], spectrum


def compute_denominator(white_source, dark_source, flip_x=True, wl_min=WL_MIN, wl_max=WL_MAX):
    """
    Compute the mean white-dark denominator used in radiometric correction.
    """
    wavs_white, white_spectrum = mean_reference_spectrum(
        white_source,
        flip_x=flip_x,
        wl_min=wl_min,
        wl_max=wl_max,
    )
    wavs_dark, dark_spectrum = mean_reference_spectrum(
        dark_source,
        flip_x=flip_x,
        wl_min=wl_min,
        wl_max=wl_max,
    )

    if len(wavs_white) != len(wavs_dark) or not np.allclose(wavs_white, wavs_dark):
        raise RuntimeError("White and dark reference spectra do not share the same wavelength axis")

    denominator = white_spectrum - dark_spectrum
    return wavs_white, white_spectrum, dark_spectrum, denominator


def plot_denominator(wavs_nm, denominator, white_source, dark_source, out_path=None):
    """
    Plot the white-dark denominator spectrum and its inverse.
    """
    inverse_denominator = 1.0 / (denominator + EPS)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

    axes[0].plot(wavs_nm, denominator, color="tab:green", linewidth=2.0, label="White - dark")
    axes[0].set_ylabel("Denominator (a.u.)", fontsize=LABEL_FONTSIZE)
    axes[0].tick_params(axis="both", labelsize=TICK_FONTSIZE)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=LEGEND_FONTSIZE)

    axes[1].plot(wavs_nm, inverse_denominator, color="tab:red", linewidth=2.0, label="1 / (white - dark)")
    axes[1].set_xlabel("Wavelength (nm)", fontsize=LABEL_FONTSIZE)
    axes[1].set_ylabel("Inverse denominator", fontsize=LABEL_FONTSIZE)
    axes[1].tick_params(axis="both", labelsize=TICK_FONTSIZE)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=LEGEND_FONTSIZE)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200)
        print(f"Saved denominator plot to {out_path}")

    plt.show()

    print(f"White reference source: {white_source}")
    print(f"Dark reference source:  {dark_source}")
    print(f"Denominator min={denominator.min():.4f}, max={denominator.max():.4f}, mean={denominator.mean():.4f}")
    print(
        f"Inverse denominator min={inverse_denominator.min():.4f}, "
        f"max={inverse_denominator.max():.4f}, mean={inverse_denominator.mean():.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot the white-dark denominator using the same reference sources as server/visualization.py."
    )
    parser.add_argument(
        "--white-source",
        default=WHITE_REFERENCE_SOURCE,
        help=f"White reference source (default: {WHITE_REFERENCE_SOURCE})",
    )
    parser.add_argument(
        "--dark-source",
        default=DARK_REFERENCE_SOURCE,
        help=f"Dark reference source (default: {DARK_REFERENCE_SOURCE})",
    )
    parser.add_argument(
        "--no-flip-x",
        action="store_true",
        help="Disable horizontal flipping before reference extraction.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional PNG output path for the denominator plot.",
    )
    args = parser.parse_args()

    wavs_nm, _, _, denominator = compute_denominator(
        args.white_source,
        args.dark_source,
        flip_x=not args.no_flip_x,
    )
    plot_denominator(
        wavs_nm,
        denominator,
        white_source=args.white_source,
        dark_source=args.dark_source,
        out_path=args.save_path,
    )


if __name__ == "__main__":
    main()
