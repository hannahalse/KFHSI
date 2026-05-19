import argparse
import re
from pathlib import Path
from datetime import datetime

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from wavelength_calibr import wavelength_axis


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = BASE_DIR / "whiteReference"

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 12

SNAPSHOT_NAME_PATTERN = re.compile(
    r"snapshot_camera_hsi_snapshot_(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<year>\d{4}), "
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})\.png$"
)


def parse_snapshot_datetime(image_path: Path):
    """
    Parse the acquisition timestamp from filenames like:
    snapshot_camera_hsi_snapshot_15.5.2026, 11_04_29.png
    """
    match = SNAPSHOT_NAME_PATTERN.match(image_path.name)
    if not match:
        raise ValueError(f"Could not parse timestamp from filename: {image_path.name}")

    return datetime(
        year=int(match.group("year")),
        month=int(match.group("month")),
        day=int(match.group("day")),
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
        second=int(match.group("second")),
    )


def extract_mean_spectrum(image_path: Path, flip_x: bool = True, wl_min: float = 380.0, wl_max: float = 820.0):
    """
    Load one raw grayscale image and extract the mean spectrum across image height.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image = image.astype(np.float32)
    if flip_x:
        image = np.fliplr(image)

    spectrum = image.mean(axis=0)
    wavs_nm = wavelength_axis(image.shape[1]).astype(np.float32)

    mask = (wavs_nm >= wl_min) & (wavs_nm <= wl_max)
    if not np.any(mask):
        raise RuntimeError(
            f"No wavelengths remain after clipping to {wl_min:.1f}-{wl_max:.1f} nm"
        )

    return wavs_nm[mask], spectrum[mask]


def load_spectra_from_folder(image_dir: Path, flip_x: bool = True, wl_min: float = 380.0, wl_max: float = 820.0):
    """
    Load all PNG spectra from one folder.
    """
    image_paths = sorted(
        (path for path in image_dir.iterdir() if path.suffix.lower() == ".png"),
        key=parse_snapshot_datetime,
    )
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {image_dir}")

    spectra = []
    wavs_ref = None
    for image_path in image_paths:
        wavs_nm, spectrum = extract_mean_spectrum(
            image_path,
            flip_x=flip_x,
            wl_min=wl_min,
            wl_max=wl_max,
        )
        if wavs_ref is None:
            wavs_ref = wavs_nm
        elif len(wavs_nm) != len(wavs_ref) or not np.allclose(wavs_nm, wavs_ref):
            raise RuntimeError(f"Wavelength axis mismatch for image: {image_path}")

        spectra.append(spectrum)

    return image_paths, wavs_ref, np.stack(spectra, axis=0)


def plot_white_reference_spectra(
    wavs_nm,
    spectra,
    title: str,
    out_path: Path | None = None,
):
    """
    Plot all spectra faintly and the mean spectrum on top.
    """
    mean_spectrum = np.mean(spectra, axis=0)
    num_spectra = len(spectra)
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=1, vmax=max(1, num_spectra))

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, spectrum in enumerate(spectra, start=1):
        ax.plot(
            wavs_nm,
            spectrum,
            linewidth=1.2,
            alpha=0.9,
            color=cmap(norm(idx)),
        )

    ax.plot(wavs_nm, mean_spectrum, color="tab:blue", linewidth=2.5, label="Mean spectrum")
    ax.set_xlabel("Wavelength (nm)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Intensity (a.u.)", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Image number / acquisition order", fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Extract spectra from all white-reference PNGs, compute their mean, and plot them."
    )
    parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_IMAGE_DIR),
        help=f"Folder containing white-reference PNGs (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--no-flip-x",
        action="store_true",
        help="Disable horizontal flipping before spectrum extraction.",
    )
    parser.add_argument(
        "--wl-min",
        type=float,
        default=380.0,
        help="Minimum wavelength to keep (default: 380.0)",
    )
    parser.add_argument(
        "--wl-max",
        type=float,
        default=820.0,
        help="Maximum wavelength to keep (default: 820.0)",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional output PNG path for the plot.",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    image_paths, wavs_nm, spectra = load_spectra_from_folder(
        image_dir=image_dir,
        flip_x=not args.no_flip_x,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
    )

    mean_spectrum = np.mean(spectra, axis=0)
    print(f"Loaded {len(image_paths)} white-reference images from {image_dir}")
    print(f"Wavelength range: {wavs_nm[0]:.2f} to {wavs_nm[-1]:.2f} nm")
    print(f"Mean spectrum intensity range: {mean_spectrum.min():.2f} to {mean_spectrum.max():.2f}")
    print("\nImage numbering by acquisition time:")
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"  {idx}: {image_path.name}")

    plot_white_reference_spectra(
        wavs_nm,
        spectra,
        title=f"White-reference spectra and mean ({len(image_paths)} images)",
        out_path=Path(args.save_path).resolve() if args.save_path else None,
    )


if __name__ == "__main__":
    main()
