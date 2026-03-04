#!/usr/bin/env python3
import argparse
import numpy as np

#Running this file results in for LEFT CHAMBER
# A = 0.7241145833
# B = 288.45625

#def pixel_to_nm(px):
#    return A * px + B

#def nm_to_px(nm):
#    return int(round((nm - B) / A))

# Edit these if anchors are updated.
DEFAULT_ANCHOR_PX = np.array([216.0, 504.0, 600.0], dtype=float) #From find_peak_pixel.py
DEFAULT_ANCHOR_NM = np.array([445.48, 650.95, 724.77], dtype=float) #From spectraPen.py


def fit_linear(anchor_px: np.ndarray, anchor_nm: np.ndarray) -> tuple[float, float]:
    """Least-squares fit: nm = a*px + b"""
    if anchor_px.size != anchor_nm.size or anchor_px.size < 2:
        raise ValueError("anchor_px and anchor_nm must have same length and contain at least 2 points.")
    a, b = np.polyfit(anchor_px.astype(float), anchor_nm.astype(float), 1)
    return float(a), float(b)


def px_to_nm(px, a: float, b: float):
    px = np.asarray(px, dtype=float)
    return a * px + b


def nm_to_px(nm, a: float, b: float):
    nm = np.asarray(nm, dtype=float)
    if a == 0:
        raise ValueError("Invalid calibration: slope a is 0.")
    return (nm - b) / a


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build linear pixel->nm calibration from fixed anchors. Prints fit, residuals, and optional conversions."
    )
    parser.add_argument("--px", type=float, default=None, help="Convert this pixel index to nm using the fit.")
    parser.add_argument("--nm", type=float, default=None, help="Convert this wavelength (nm) to pixel using the fit.")
    args = parser.parse_args()

    anchor_px = DEFAULT_ANCHOR_PX
    anchor_nm = DEFAULT_ANCHOR_NM

    a, b = fit_linear(anchor_px, anchor_nm)

    print("Anchors:")
    for p, w in zip(anchor_px, anchor_nm):
        print(f"  px={p:.1f} -> nm={w:.1f}")

    print(f"\nLinear calibration:")
    print(f"  nm = {a:.10f} * px + {b:.10f}")

    pred = px_to_nm(anchor_px, a, b)
    err = pred - anchor_nm
    rms = float(np.sqrt(np.mean(err**2)))

    print("\nAnchor residuals (pred - true) [nm]:", np.round(err, 3).tolist())
    print("RMS [nm]:", rms)

    if args.px is not None:
        print(f"\nConvert px -> nm:")
        print(f"  px={args.px:.3f} -> nm={float(px_to_nm(args.px, a, b)):.3f}")

    if args.nm is not None:
        print(f"\nConvert nm -> px:")
        print(f"  nm={args.nm:.3f} -> px={float(nm_to_px(args.nm, a, b)):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())