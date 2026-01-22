import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks




def extract_spectrum_from_row(raw_path, row, start_nm=400.0, end_nm=800.0, show_plot=False):
    """
    Reads a raw image, retrieves spectrum from one row, and maps it to wavelengths.

    Pratameters
    raw_path: path to the raw image 
    start_nm: starting wavelength (nm)
    end_nm: ending wavelength (nm)
    show_plot: if True, plots the spectrum. Wavelength vs intensity.

    Returns
    wavelents: wavelengths in nm
    spectrum: intensity values for the selected row

    """
    img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read raw image: {raw_path}")

    H, W = img.shape
    if not (0 <= row < H):
        raise ValueError(f"Row {row} is out of range [0, {H-1}]")
    print(f"Raw image shape: H={H}, W={W}")
    wavelengths = np.linspace(start_nm, end_nm, W, dtype=np.float32)
    spectrum = img[row, :].astype(np.float32)
    x = np.arange(spectrum.size)


    if show_plot:
        plt.figure(figsize=(8, 4))
        #plt.plot(wavelengths, spectrum)
        plt.plot(x, spectrum)
        #plt.xlabel("Wavelength (nm)")
        plt.xlabel("Pixel")
        plt.ylabel("Intensity")
        plt.title(f"Spectrum from raw image (row={row})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return wavelengths, spectrum



def quantify_smile_from_raw(
    raw_path,
    peak_thresh_rel=0.3,
    window_half_width=5,
    y_roi=None,
    nm_per_pixel=None,
    show_debug=True,
):
    """
    Kvantifiserer smile-effekt fra et rått spektralbilde med vertikale linjer.

    raw_path        : sti til kalibreringsbilde (gråskala PNG/TIFF)
    peak_thresh_rel : relativ terskel (0–1) for å finne linjer i x-profilen
    window_half_width : halv bredde for vindu rundt hver linje når centroid beregnes
    y_roi           : (y0, y1) eller None for å bruke hele høyden
    nm_per_pixel    : hvis kjent, brukes til å rapportere smile i nm i tillegg
    show_debug      : plott linjesentre og smile-kurve for første linje

    return:
        results: dict[line_index] -> info om smile for hver spektrallinje
    """

    # Read image
    img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {raw_path}")
    H, W = img.shape
    print(f"Loaded image {raw_path} with shape H={H}, W={W}")

    if y_roi is None:
        y0, y1 = 0, H
    else:
        y0, y1 = max(0, y_roi[0]), min(H, y_roi[1])

    # Find spectral lines as peaks in the integrated profile along x
    profile = img[y0:y1, :].sum(axis=0).astype(np.float32)
    max_val = profile.max()
    thresh = peak_thresh_rel * max_val

    peak_xs = []
    for x in range(1, W - 1):
        if profile[x] > profile[x-1] and profile[x] > profile[x+1] and profile[x] > thresh:
            peak_xs.append(x)
    peak_xs = np.array(peak_xs, dtype=int)

    if len(peak_xs) == 0:
        raise RuntimeError("No spectral lines found – try lowering peak_thresh_rel.")

    print(f"Found {len(peak_xs)} spectral lines at x ≈ {peak_xs.tolist()}")

    results = {}

    #For each line: find center as function of y 
    for li, x0 in enumerate(peak_xs):
        ys = []
        xs_centers = []

        for y in range(y0, y1):
            x_start = max(0, x0 - window_half_width)
            x_end = min(W, x0 + window_half_width + 1)
            line_profile = img[y, x_start:x_end].astype(np.float32)
            s = line_profile.sum()
            if s <= 0:
                continue

            xs_local = np.arange(x_start, x_end, dtype=np.float32)
            centroid = (xs_local * line_profile).sum() / s
            ys.append(y)
            xs_centers.append(centroid)

        if len(xs_centers) < 5:
            continue

        ys = np.array(ys, dtype=np.float32)
        xs_centers = np.array(xs_centers, dtype=np.float32)

        # Fit x = a*y + b and calculate smile 
        coeffs = np.polyfit(ys, xs_centers, 1)  # x_fit = a*y + b
        x_fit = np.polyval(coeffs, ys)
        smile = xs_centers - x_fit

        max_smile_pix = float(smile.max())
        min_smile_pix = float(smile.min())
        max_abs_smile_pix = float(np.max(np.abs(smile)))

        entry = {
            "line_x0": int(x0),
            "ys": ys,
            "xs": xs_centers,
            "fit_coeffs": tuple(coeffs),
            "smile": smile,
            "max_smile_pix": max_smile_pix,
            "min_smile_pix": min_smile_pix,
            "max_abs_smile_pix": max_abs_smile_pix,
        }

        if nm_per_pixel is not None:
            entry["max_smile_nm"] = max_smile_pix * nm_per_pixel
            entry["min_smile_nm"] = min_smile_pix * nm_per_pixel
            entry["max_abs_smile_nm"] = max_abs_smile_pix * nm_per_pixel

        results[li] = entry

    if not results:
        raise RuntimeError("No valid lines with enough samples for smile fit.")

    # 5) Debug-plott
    if show_debug:
        # plott linjesentre oppå bildet
        plt.figure(figsize=(10, 4))
        plt.imshow(img, cmap="gray", aspect="auto")
        for li, r in results.items():
            plt.plot(r["xs"], r["ys"], ".", markersize=1, label=f"line {li}")
        plt.ylim(248, 803)  # merk: vi bruker invert_yaxis ellers
        plt.gca().invert_yaxis()
        plt.xlabel("x (dispersion)")
        plt.ylabel("y (slit)")
        plt.title("Detected line centers")
        plt.legend(markerscale=4, fontsize=8)
        plt.tight_layout()
        plt.show()

        # plott smile for første linje
        first_key = sorted(results.keys())[0]
        r0 = results[first_key]
        plt.figure(figsize=(6, 4))
        plt.plot(r0["ys"], r0["smile"], ".-")
        plt.axhline(0, color="k", linewidth=0.5)
        plt.xlabel("Row (y)")
        plt.ylabel("Smile [pixels]")
        plt.title(f"Smile for line {first_key}")
        plt.tight_layout()
        plt.show()

        # plott smile for nittende linje
        nineteenth_key = sorted(results.keys())[18]
        r0 = results[nineteenth_key]
        plt.figure(figsize=(6, 4))
        plt.plot(r0["ys"], r0["smile"], ".-")
        plt.axhline(0, color="k", linewidth=0.5)
        plt.xlabel("Row (y)")
        plt.ylabel("Smile [pixels]")
        plt.title(f"Smile for line {nineteenth_key}")
        plt.tight_layout()
        plt.show()

        print(f"Example line {first_key}: max |smile| = {r0['max_abs_smile_pix']:.3f} px")

    return results


def locate_peaks(spec, noise_level = 20, min_distance_pixels = 2):
    """
    Locate peaks in a spectrum above a certain noise level and minimum distance.

    Parameters:
    spec: 1D numpy array of intensity values
    noise_level: minimum height for a peak to be considered valid
    min_distance_pixels: minimum distance between peaks in pixels

    Returns:
    peak_wavs: wavelengths of detected peaks
    peak_vals: intensity values of detected peaks
    """
    peaks_idx, props = find_peaks(
        spec,
        height=noise_level,
        distance=min_distance_pixels
    )
    peak_wavs = wavs[peaks_idx]
    peak_vals = spec[peaks_idx]
    
    print(f"Found {len(peaks_idx)} peaks above noise level {noise_level}.")
    for w, v in zip(peak_wavs, peak_vals):
        print(f"λ = {w:.2f} nm, intensity = {v:.1f}")
    return peaks_idx, peak_wavs, peak_vals, props



wavs, spec = extract_spectrum_from_row(
    raw_path="/Users/hannahalse/KFSpectra/smile_keystone/calibration_data/hg_200ms.png",
    row=608,
    start_nm=400.0,
    end_nm=800.0,
    show_plot=True
)


peaks_idx, peak_wavs, peak_vals, props = locate_peaks(spec)

strong_idx = np.argsort(peak_vals)[-10:]
for idx in strong_idx:
    print("pixel:", peaks_idx[idx], "| intensity:", peak_vals[idx])




results = quantify_smile_from_raw(
    raw_path="/Users/hannahalse/KFSpectra/smile_keystone/calibration_data/hg_200ms.png",
    peak_thresh_rel=0.3,
    window_half_width=5,
    y_roi=(260, 803),        # eller f.eks. (100, 500) hvis du vil klippe bort topp/bunn
    nm_per_pixel=None,      
    show_debug=True,
)

# F.eks. samlet tall for hele bildet:
max_abs_smiles = [r["max_abs_smile_pix"] for r in results.values()]
print("Max |smile| over alle linjer:", max(max_abs_smiles), "piksler")
