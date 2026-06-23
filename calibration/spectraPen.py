import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SPECTRA = (
    (BASE_DIR / "SpectraPenData_10Wbulbs.spec", "10 W bulbs"),
    (BASE_DIR / "SpectraPenData_20Wbulb.spec", "20 W bulbs"),
)
DEFAULT_SAVE_PATH = BASE_DIR / "spectrapen_10W_20W_raw.png"

LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 18


def extract_all_json_blocks(data):
    json_blocks = []
    i = 0

    while i < len(data):
        if data[i] == ord("{"):
            depth = 1
            start = i
            i += 1

            while i < len(data) and depth > 0:
                if data[i] == ord("{"):
                    depth += 1
                elif data[i] == ord("}"):
                    depth -= 1
                i += 1

            if depth == 0:
                try:
                    block = data[start:i].decode("utf-8")
                    json_blocks.append(json.loads(block))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
        else:
            i += 1

    return json_blocks


def read_spectrapen_measurement(spec_path, wl_min=400.0, wl_max=850.0):
    with open(spec_path, "rb") as f:
        data = f.read()

    marker = b"Measurement1\x00"
    marker_index = data.find(marker)
    if marker_index == -1:
        raise RuntimeError(f'Cannot find "Measurement1" in {spec_path}')

    count_offset = marker_index + len(marker) + 1 + 3 + 8
    n_points = int.from_bytes(data[count_offset:count_offset + 4], "little")

    values_offset = count_offset + 4
    values = np.frombuffer(
        data[values_offset:values_offset + 4 * n_points],
        dtype="<u4",
    ).astype(np.float32)

    json_blocks = extract_all_json_blocks(data)
    if not json_blocks:
        raise RuntimeError(f"No JSON metadata block found in {spec_path}")

    sconst = json_blocks[0]["device"]["sconst"]
    pixel_index = np.arange(n_points, dtype=np.float32)
    wavelength = np.zeros_like(pixel_index)

    for power, coefficient in enumerate(sconst):
        wavelength += coefficient * (pixel_index ** power)

    keep = (wavelength >= wl_min) & (wavelength <= wl_max)
    if not np.any(keep):
        raise RuntimeError(
            f"No wavelengths remain for {spec_path} after clipping to {wl_min:.1f}-{wl_max:.1f} nm"
        )

    return wavelength[keep], values[keep]


def plot_spectra(spectra, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(10, 6))

    for spec_path, label in spectra:
        wavelength, values = read_spectrapen_measurement(spec_path)
        ax.plot(
            wavelength,
            values,
            linewidth=2.2,
            label=label,
        )

    ax.set_xlabel("Wavelength (nm)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Relative intensity (a.u.)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved spectra plot to {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SpectraPen spectra for the 10 W and 20 W bulb measurements."
    )
    parser.add_argument(
        "--save-path",
        default=str(DEFAULT_SAVE_PATH),
        help=f"Output PNG path (default: {DEFAULT_SAVE_PATH})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening the interactive plot window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_spectra(
        DEFAULT_SPECTRA,
        save_path=Path(args.save_path).resolve() if args.save_path else None,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
