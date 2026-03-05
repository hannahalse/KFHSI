#!/usr/bin/env python3
"""
This script uses: 
- INPUTS (originals):  ar_200ms.png, hg_200ms.png
- Flips originals horizontally (spectral axis) before any processing
- Preprocess: crop rows [roi_top:roi_bottom], bin along x (binning_factor)
- Process: integrate spectrum, align Ar->Hg, estimate alpha, subtract alpha*Ar from Hg
- Detect peaks and plot spectra
- Fit linear + quadratic calibration using your chosen pixel_points and nm_points
- Saves outputs
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

#Set to True to save debug images
save_debug_images = False 

# Base directory 
base_dir = os.path.dirname(os.path.abspath(__file__))

# Debug folder
debug_dir = os.path.join(base_dir, "debug_calib")
if save_debug_images:
    os.makedirs(debug_dir, exist_ok=True)

# -----------------------------
# CONFIG
# -----------------------------
# Originals (these must exist in same folder as this script)
ar_original_name = "ar_200ms.png"
hg_original_name = "hg_200ms.png"

# Flip originals horizontally
flip_original_x = True

# Preprocess settings (must match camera config)
roi_top = 248
roi_bottom = 803
binning_factor = 2

# Spectrum integration ROI inside the preprocessed image
y_roi = (0, 555)

# Alignment search window (pixels)
max_shift = 10

# Peak detection parameters. How many pixels we are willing to move the spectrums to get them to overlap. 
#The other ones - what counts as a peak. 12% of max to be a peak, 10 pixels in distance from each other, and prominence lets ut know how much a peak is sticking up from the neighbours. 
min_height_rel = 0.12
min_distance = 10
prominence_rel = 0.03

# Peaks (pixels) selected (from Ar aligned) for calibration (Found by running the script and looking at the printed peaks).
pixel_points = np.array([564, 579, 608, 625, 643, 661, 731, 754, 777, 883], dtype=float)

# Corresponding known wavelengths (from Maries thesis)(nm)
nm_points = np.array([696.54, 706.72, 727.29, 738.40, 750.93, 763.51, 810.95, 826.45, 841.64, 912.30], dtype=float)


# -----------------------------
# CALIBRATION FUNCTIONS (fitted functions) 
# -----------------------------
#From previous runs. If anything in script changes - change these too.  
def pixel_to_nm_linear(pixel):
    return 0.6773347176 * pixel + 315.1920552541


def pixel_to_nm_quadratic(pixel):
    return -5.6694394352e-05 * pixel**2 + 0.7583713843 * pixel + 286.8160173070


def wavelength_axis(width: int):
    p = np.arange(width, dtype=float)
    return pixel_to_nm_quadratic(p)


# -----------------------------
# I/O + PREPROCESS
# -----------------------------
def load_gray(path: str, flip_x: bool = False) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read: {path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32)

    if flip_x:
        image = np.fliplr(image)

    return image


def bin_image_x(image: np.ndarray, factor: int) -> np.ndarray:
    h, w = image.shape
    new_w = w // factor
    image = image[:, :new_w * factor]  # trim to multiple
    binned = image.reshape(h, new_w, factor).mean(axis=2)
    return binned


def preprocess(image: np.ndarray) -> np.ndarray:
    # Crop in Y (rows)
    cropped = image[roi_top:roi_bottom, :]
    # Bin in X (spectral axis)
    binned = bin_image_x(cropped, binning_factor)
    return binned


def save_u8(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, np.clip(image, 0, 255).astype(np.uint8))


# -----------------------------
# SPECTRUM + ALIGN + SUBTRACT
# -----------------------------
def integrated_spectrum(image: np.ndarray, y_roi_local):
    y0, y1 = y_roi_local
    if y1 is None:
        y1 = image.shape[0]
    return image[y0:y1, :].sum(axis=0)


def find_shift(spec_ref: np.ndarray, spec_to_align: np.ndarray, max_shift_local: int = 10) -> int:
    corr = correlate(spec_ref, spec_to_align, mode="full")
    center = len(corr) // 2
    search = corr[center - max_shift_local: center + max_shift_local + 1]
    dx = int(np.argmax(search) - max_shift_local)
    return dx


def estimate_alpha(spec_hg: np.ndarray, spec_ar: np.ndarray, mask=None) -> float:
    if mask is None:
        mask = np.ones_like(spec_hg, dtype=bool)
    num = np.sum(spec_hg[mask] * spec_ar[mask])
    den = np.sum(spec_ar[mask] ** 2)
    return float(num / den) if den > 0 else 0.0

def subtract_ar_from_hg_images(hg_img: np.ndarray, ar_img: np.ndarray, y_roi_local, max_shift_local=10, show_debug=True):
    spec_hg = integrated_spectrum(hg_img, y_roi_local)
    spec_ar = integrated_spectrum(ar_img, y_roi_local)

    dx = find_shift(spec_hg, spec_ar, max_shift_local=max_shift_local)

    spec_ar_shifted = np.roll(spec_ar, dx)
    ar_img_shifted = np.roll(ar_img, dx, axis=1)

    alpha = estimate_alpha(spec_hg, spec_ar_shifted)

    spec_hg_only = np.clip(spec_hg - alpha * spec_ar_shifted, 0, None)
    hg_img_only = np.clip(hg_img - alpha * ar_img_shifted, 0, None)

    if show_debug:
        print(f"Alignment dx = {dx} pixels")
        print(f"Scaling alpha = {alpha:.4f}")

        plt.figure(figsize=(10, 4))
        plt.plot(spec_hg, label="Hg (mixed)")
        plt.plot(alpha * spec_ar_shifted, label="α · Ar (aligned)")
        plt.plot(spec_hg_only, label="Hg only", linewidth=2)
        plt.legend()
        plt.xlabel("Pixel (dispersion)")
        plt.ylabel("Integrated intensity")
        plt.title("Hg − α·Ar (capped at 0)")
        plt.tight_layout()
        plt.show()

    return hg_img_only, spec_hg_only, ar_img_shifted, spec_ar_shifted, dx, alpha


# -----------------------------
# PEAK DETECTION + PLOTTING
# -----------------------------
def find_peaks_in_spectrum(spec: np.ndarray):
    spec = np.asarray(spec, dtype=np.float32)
    max_val = float(spec.max())
    if max_val <= 0:
        raise ValueError("Spectrum has no positive signal (max <= 0).")

    height = min_height_rel * max_val
    prominence = prominence_rel * max_val

    peaks_idx, props = find_peaks(
        spec,
        height=height,
        distance=min_distance,
        prominence=prominence
    )

    peak_vals = spec[peaks_idx]
    order = np.argsort(peak_vals)[::-1]
    return peaks_idx[order], peak_vals[order], props


def print_peaks(name: str, peaks_idx: np.ndarray, peak_vals: np.ndarray, max_print: int = 25):
    print(f"\n{name}: Found {len(peaks_idx)} peaks.")
    for i, (p, v) in enumerate(zip(peaks_idx[:max_print], peak_vals[:max_print]), start=1):
        print(f"{i:2d}: pixel={int(p):4d}, intensity={float(v):.1f}")


def plot_spectrum_with_peaks(spec: np.ndarray, peaks_idx: np.ndarray, title: str):
    plt.figure(figsize=(10, 4))
    plt.plot(spec, label=title)
    if len(peaks_idx) > 0:
        plt.plot(peaks_idx, spec[peaks_idx], "x", label="Peaks")
    plt.xlabel("Pixel")
    plt.ylabel("Integrated intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# FIT CALIBRATION
# -----------------------------
def fit_linear_and_quadratic(pixel: np.ndarray, nm: np.ndarray):
    c1 = np.polyfit(pixel, nm, 1)
    nm1 = np.polyval(c1, pixel)
    rms1 = float(np.sqrt(np.mean((nm1 - nm) ** 2)))
    max_err1 = float(np.max(np.abs(nm1 - nm)))
    end_errs1 = (float(nm1[0] - nm[0]), float(nm1[-1] - nm[-1]))

    c2 = np.polyfit(pixel, nm, 2)
    nm2 = np.polyval(c2, pixel)
    rms2 = float(np.sqrt(np.mean((nm2 - nm) ** 2)))
    max_err2 = float(np.max(np.abs(nm2 - nm)))

    return (c1, rms1, max_err1, end_errs1), (c2, rms2, max_err2), nm1, nm2


def print_fit_and_errors(pixel: np.ndarray, nm: np.ndarray, coeff2: np.ndarray):
    nm_fit = np.polyval(coeff2, pixel)
    error = nm_fit - nm
    max_error = float(np.max(np.abs(error)))
    rms_error = float(np.sqrt(np.mean(error ** 2)))

    a, b, c = coeff2
    print("\Calibration:")
    print(f"nm = {a:.10e} * p^2 + {b:.10f} * p + {c:.10f}")
    print(f"Quadratic max |error| = {max_error:.3f} nm")

    print("\nError per point (nm):")
    for p, real, pred in zip(pixel, nm, nm_fit):
        print(f"pixel {p:4.0f}:  real={real:8.2f}   pred={pred:8.2f}   err={pred-real:6.3f}")

    print(f"\nRMS error: {rms_error:.3f} nm")


def plot_calibration(pixel: np.ndarray, nm: np.ndarray, coeff2: np.ndarray):
    xx = np.linspace(float(pixel.min()) - 50, float(pixel.max()) + 50, 500)
    plt.figure(figsize=(6, 4))
    plt.scatter(pixel, nm, label="Measured lines")
    plt.plot(xx, np.polyval(coeff2, xx), label="Calibration fit")
    plt.xlabel("Pixel")
    plt.ylabel("Wavelength (nm)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
def main():
    # ---- 0) Load ORIGINALS ----
    ar_path = os.path.join(base_dir, ar_original_name)
    hg_path = os.path.join(base_dir, hg_original_name)

    ar_orig = load_gray(ar_path, flip_x=flip_original_x)
    hg_orig = load_gray(hg_path, flip_x=flip_original_x)

    print(f"Loaded originals:")
    print(f"  ar: {ar_original_name} shape={ar_orig.shape} flip_x={flip_original_x}")
    print(f"  hg: {hg_original_name} shape={hg_orig.shape} flip_x={flip_original_x}")
    

    # ---- 1) Preprocess (crop y + bin x) ----
    ar_pre = preprocess(ar_orig)
    hg_pre = preprocess(hg_orig)

    print(f"\nPreprocess:")
    print(f"  ar: {ar_orig.shape} -> {(roi_bottom-roi_top, ar_orig.shape[1])} -> {ar_pre.shape}")
    print(f"  hg: {hg_orig.shape} -> {(roi_bottom-roi_top, hg_orig.shape[1])} -> {hg_pre.shape}")

    # ---- 2) Spectra + Align + Subtract ----
    # 2.1 Ar spectrum (raw / preprocessed)
    ar_spec = integrated_spectrum(ar_pre, y_roi)
    ar_peaks_idx, ar_peak_vals, _ = find_peaks_in_spectrum(ar_spec)
    print_peaks("Ar (preprocessed)", ar_peaks_idx, ar_peak_vals)
    plot_spectrum_with_peaks(ar_spec, ar_peaks_idx, "Ar (preprocessed)")

    # 2.2 Hg-only (Hg - alpha*Ar aligned)
    hg_only_img, hg_only_spec, ar_img_aligned, ar_spec_aligned, dx, alpha = subtract_ar_from_hg_images(
        hg_img=hg_pre,
        ar_img=ar_pre,
        y_roi_local=y_roi,
        max_shift_local=max_shift,
        show_debug=True,
    )

    hg_peaks_idx, hg_peak_vals, _ = find_peaks_in_spectrum(hg_only_spec)
    print_peaks("Hg only", hg_peaks_idx, hg_peak_vals)
    plot_spectrum_with_peaks(hg_only_spec, hg_peaks_idx, "Hg only")

    ar_aligned_peaks_idx, ar_aligned_peak_vals, _ = find_peaks_in_spectrum(ar_spec_aligned)
    print_peaks("Ar (aligned)", ar_aligned_peaks_idx, ar_aligned_peak_vals)
    plot_spectrum_with_peaks(ar_spec_aligned, ar_aligned_peaks_idx, "Ar (aligned)")


    if save_debug_images:
        ar_flip_dbg = os.path.join(debug_dir, "ar_200ms_flipped_debug.png")
        hg_flip_dbg = os.path.join(debug_dir, "hg_200ms_flipped_debug.png")
        save_u8(ar_flip_dbg, ar_orig)
        save_u8(hg_flip_dbg, hg_orig)
        
        ar_pre_name = f"ar_200ms_roi{roi_top}-{roi_bottom}_bin{binning_factor}.png"
        hg_pre_name = f"hg_200ms_roi{roi_top}-{roi_bottom}_bin{binning_factor}.png"
        save_u8(os.path.join(debug_dir, ar_pre_name), ar_pre)
        save_u8(os.path.join(debug_dir, hg_pre_name), hg_pre)

        save_u8(os.path.join(debug_dir, "hg_only_u8_debug.png"), hg_only_img)
        save_u8(os.path.join(debug_dir, "ar_aligned_u8_debug.png"), ar_img_aligned)   
        print(f"Size of hg_only_img (H,W): {hg_only_img.shape}")
        print(f"Size of ar_aligned_img (H,W): {ar_img_aligned.shape}") 

    # ---- 2.3 Sanity: wavelength axis ----
    # (Only meaningful for widths that exist in your current processed data)
    nm_axis = wavelength_axis(hg_only_img.shape[1])
    print("\nWavelength axis sanity:")
    print("  nm_axis[0], nm_axis[-1] =", float(nm_axis[0]), float(nm_axis[-1]))

    # ---- 3) Fit calibration (same as fit_calibration.py) ----
    (c1, rms1, max_err1, end_errs1), (c2, rms2, max_err2), _, _ = fit_linear_and_quadratic(pixel_points, nm_points)

    print("\n--- Fit results ---")
    print("Linear:     nm = {:.10f}*p + {:.10f}   RMS = {:.3f} nm".format(c1[0], c1[1], rms1))
    print("Linear max |error| =", max_err1)
    print("Linear end errors:", end_errs1[0], end_errs1[1])

    print("Quadratic:  nm = {:.10e}*p^2 + {:.10f}*p + {:.10f}   RMS = {:.3f} nm".format(
        c2[0], c2[1], c2[2], rms2
    ))

    print_fit_and_errors(pixel_points, nm_points, c2)
    plot_calibration(pixel_points, nm_points, c2)


if __name__ == "__main__":
    main()
    
    #--------- TESTING ----------
    
    # Load test image
    test_path = os.path.join(base_dir, "660nmLC30int.png")
    image = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("Could not load test image")

    image = np.fliplr(image)
    spectrum = image.sum(axis=0)
    nm_axis = wavelength_axis(image.shape[1])
    
    peak_i = int(np.argmax(spectrum))
    w = 10
    lo = max(0, peak_i - w)
    hi = min(len(spectrum), peak_i + w + 1)

    centroid_nm = float(np.sum(nm_axis[lo:hi] * spectrum[lo:hi]) / np.sum(spectrum[lo:hi]))
    print("Peak centroid (nm):", centroid_nm)

    # ---- Klipp til 400–800 nm ----
    mask = (nm_axis >= 400) & (nm_axis <= 800)

    nm_axis_clip = nm_axis[mask]
    spectrum_clip = spectrum[mask]

    plt.figure(figsize=(8,4))
    plt.plot(nm_axis_clip, spectrum_clip)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Integrated intensity")
    plt.title("Test image spectrum with calibrated wavelength axis (400–800 nm)")
    plt.tight_layout()
    plt.show()
    # ---------------------------------