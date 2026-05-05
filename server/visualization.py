import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from Generate_cube import CubeNM

from cube_visuals import (
    visualise_spectrum_at,
    visualise_wavelength_slice,
    reconstruct_rgb_image,
    visualise_spectrum_before_after_gaussian,
    visualise_raw_image_spectrum,
    visualise_white_dark_difference,
)

from indices import (
    calculate_ndvi,
    calculate_cri,
    calculate_pri,
)

from Generate_cube import (
    find_last_modified_folder,
)

from radiometric_calibration import (
    radiometric_correct_cube, 
    save_corrected_cube
)

#KFHSI
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "edge", "data")

scan_folder = os.path.join(DATA_DIR, "scan_30April_09:46:32")  # Choose one specific folder for now
npz_path = os.path.join(scan_folder, "cube_ZXnm_corrected.npz")

data = np.load(npz_path)
cube = CubeNM(data["cube"], data["wavs_nm"])

white_path = os.path.join(BASE_DIR, "server", "whiteReferenceInChamber10W.png")
dark_path = os.path.join(BASE_DIR, "server", "darkReference.png")

# How to collapse the Y dimension when making 2D maps/RGB views.
# Options:
#   "slice"             -> use one Y row
#   "mean"              -> average over all Y rows
#   "central_band_mean" -> average over a central Y band
Y_REDUCTION_MODE = "central_band_mean"
Y_BAND_HALF_HEIGHT = 20
NDVI_MASK_THRESHOLD = 0.35
APPEND_TO_MASTER_CSV = True
MASTER_SUMMARY_CSV = os.path.join(DATA_DIR, "masked_index_time_series.csv")
REPLACE_EXISTING_SCAN_ROW = True


# -------- Calibration --------
A = 0.7241145833
B = 288.45625

def px_to_nm(px):
    """Convert pixel index → wavelength (nm)."""
    return A * px + B

def wavelength_axis(width):
    """Return wavelength for every pixel in the image."""
    pixels = np.arange(width)
    return px_to_nm(pixels)

#Extracts the spectrum from the whole image, by averaging across the vertical axis.
def extract_spectrum_from_raw(image_path, flip_x=True, plot=True):
    """
    Load raw grayscale image, optionally flip horizontally,
    extract mean spectrum across vertical axis,
    and generate wavelength axis.

    Returns:
        wavs_nm : ndarray (W,)
        spectrum : ndarray (W,)
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read {image_path}")

    img = img.astype(np.float32)

    # Flip horizontally if needed
    if flip_x:
        img = np.fliplr(img)

    # Get image width
    H, W = img.shape

    # Extract mean spectrum across height (collapse Y-direction)
    spectrum = img.mean(axis=0)

    # Generate wavelength axis
    wavs_nm = wavelength_axis(W)

    # Optional plotting
    if plot:
        plt.figure()
        plt.plot(wavs_nm, spectrum)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (a.u.)")
        plt.title("Extracted spectrum from {}".format(os.path.basename(image_path)))
        plt.show()

    return wavs_nm, spectrum


def plot_white_dark_difference(white_path, dark_path, flip_x=True, plot=False):
    wavs, diff, _, _ = visualise_white_dark_difference(
        white_path,
        dark_path,
        flip_x=flip_x,
        plot=plot,
    )
    return wavs, diff


def get_central_y_band(y_size, center_y=None, half_height=20):
    if center_y is None:
        center_y = y_size // 2
    y0 = max(0, int(center_y) - int(half_height))
    y1 = min(y_size, int(center_y) + int(half_height) + 1)
    if y1 <= y0:
        raise ValueError(f"Invalid Y band: ({y0}, {y1})")
    return y0, y1


def reduce_y_dimension(volume, mode="slice", y=None, band_half_height=20):
    """
    Reduce a (Z, X, Y) volume to a (Z, X) map for visualization/summary.
    """
    if volume.ndim == 2:
        return volume, "2D map"
    if volume.ndim != 3:
        raise ValueError(f"Expected 2D or 3D data, got shape {volume.shape}")

    y_size = volume.shape[2]
    y_middle = y_size // 2 if y is None else int(y)

    if mode == "slice":
        reduced = volume[:, :, y_middle]
        label = f"y={y_middle}"
    elif mode == "mean":
        reduced = np.nanmean(volume, axis=2)
        label = "mean over all Y"
    elif mode == "central_band_mean":
        y0, y1 = get_central_y_band(y_size, center_y=y_middle, half_height=band_half_height)
        reduced = np.nanmean(volume[:, :, y0:y1], axis=2)
        label = f"mean over y={y0}:{y1}"
    else:
        raise ValueError(f"Unsupported Y reduction mode: {mode}")

    return reduced, label


def summarize_index_volume(index_data, name="Index"):
    """
    Print summary statistics for the full 3D index volume before any Y reduction.
    """
    print(f"{name} full cube min:  {np.nanmin(index_data):.4f}")
    print(f"{name} full cube max:  {np.nanmax(index_data):.4f}")
    print(f"{name} full cube mean: {np.nanmean(index_data):.4f}")
    print(f"{name} full cube std:  {np.nanstd(index_data):.4f}")


def create_ndvi_mask_3d(ndvi, threshold=0.35):
    """
    Create a full 3D plant mask from NDVI with shape (Z, X, Y).
    """
    if ndvi.ndim != 3:
        raise ValueError(f"Expected NDVI volume with shape (Z, X, Y), got {ndvi.shape}")
    return ndvi > threshold


def summarize_mask_3d(mask_3d, name="Plant mask"):
    """
    Print coverage statistics for a full 3D boolean mask.
    """
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected 3D mask with shape (Z, X, Y), got {mask_3d.shape}")

    pixel_count = int(np.count_nonzero(mask_3d))
    total_count = int(mask_3d.size)
    coverage_pct = 100.0 * pixel_count / total_count if total_count else 0.0

    print(f"{name} full cube pixel count: {pixel_count}")
    print(f"{name} full cube coverage:    {coverage_pct:.2f}%")


def summarize_mask_2d(mask_2d, name="Plant mask"):
    """
    Print coverage statistics for a 2D boolean mask.
    """
    if mask_2d.ndim != 2:
        raise ValueError(f"Expected 2D mask with shape (Z, X), got {mask_2d.shape}")

    pixel_count = int(np.count_nonzero(mask_2d))
    total_count = int(mask_2d.size)
    coverage_pct = 100.0 * pixel_count / total_count if total_count else 0.0

    print(f"{name} pixel count: {pixel_count}")
    print(f"{name} coverage:    {coverage_pct:.2f}%")
    return {
        "pixel_count": pixel_count,
        "coverage_pct": coverage_pct,
    }


def summarize_masked_index_volume(index_data, mask_3d, name="Index"):
    """
    Print summary statistics for a full 3D index volume after applying a 3D mask.
    """
    if index_data.shape != mask_3d.shape:
        raise ValueError(
            f"index_data and mask_3d must have the same shape, got {index_data.shape} and {mask_3d.shape}"
        )

    masked = np.where(mask_3d, index_data, np.nan)
    #print(f"{name} plant-only full cube min:  {np.nanmin(masked):.4f}")
    #print(f"{name} plant-only full cube max:  {np.nanmax(masked):.4f}")
    #print(f"{name} plant-only full cube mean: {np.nanmean(masked):.4f}")
    #print(f"{name} plant-only full cube std:  {np.nanstd(masked):.4f}")
    return masked


def show_index_map(index_data, name, y_mode="slice", y=None, band_half_height=20, cmap="RdYlGn", vmin=None, vmax=None):
    """
    Show a 2D map from a 3D spectral index array with shape (Z, X, Y).
    """
    index_map, y_label = reduce_y_dimension(
        index_data,
        mode=y_mode,
        y=y,
        band_half_height=band_half_height,
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(index_map, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(label=name)
    plt.title(f"{name} map ({y_label})")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.tight_layout()
    plt.show()

    #print(f"{name} reduced map ({y_label}) min:  {np.nanmin(index_map):.4f}")
    #print(f"{name} reduced map ({y_label}) max:  {np.nanmax(index_map):.4f}")
    #print(f"{name} reduced map ({y_label}) mean: {np.nanmean(index_map):.4f}")
    return index_map


def create_ndvi_mask(ndvi, threshold=0.35, y_mode="slice", y=None, band_half_height=20):
    """
    Create binary plant mask from NDVI.

    Parameters:
        ndvi: ndarray with shape (Z, X, Y) or already reduced (Z, X)
        threshold: NDVI threshold for plant segmentation

    Returns:
        mask: boolean 2D ndarray with shape (Z, X)
    """
    ndvi_2d, _ = reduce_y_dimension(
        ndvi,
        mode=y_mode,
        y=y,
        band_half_height=band_half_height,
    )
    return ndvi_2d > threshold


def apply_mask_to_index(index_data, mask):
    """
    Set background pixels to NaN using a boolean mask.

    index_data and mask must have the same shape.
    """
    masked = np.where(mask, index_data, np.nan)
    return masked


def summarize_masked_index(index_data, name="Index"):
    """
    Print summary statistics for masked index data.
    Assumes background is NaN.
    """
    values = np.asarray(index_data[np.isfinite(index_data)], dtype=np.float32)
    valid_count = int(values.size)

    if valid_count == 0:
        print(f"{name} has no valid plant pixels")
        return {
            "valid_pixel_count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
        }

    mean_value = float(np.mean(values))
    median_value = float(np.median(values))
    std_value = float(np.std(values))

    print(f"{name} valid pixel count: {valid_count}")
    print(f"{name} mean:              {mean_value:.4f}")
    print(f"{name} median:            {median_value:.4f}")
    print(f"{name} std:               {std_value:.4f}")
    return {
        "valid_pixel_count": valid_count,
        "mean": mean_value,
        "median": median_value,
        "std": std_value,
    }


def update_master_summary_csv(
    csv_path,
    scan_folder,
    y_label,
    mask_summary,
    index_summaries,
    y_reduction_mode,
    y_band_half_height,
    ndvi_threshold,
    white_path,
    dark_path,
    replace_existing_scan_row=True,
):
    """
    Save thesis-facing masked index summaries to one master CSV with one row per scan.
    """
    fieldnames = [
        "scan_name",
        "scan_folder",
        "analysis_y_reduction",
        "y_reduction_mode",
        "y_band_half_height",
        "ndvi_mask_threshold",
        "white_reference",
        "dark_reference",
        "mask_pixel_count",
        "mask_coverage_pct",
        "pri_valid_pixel_count",
        "pri_mean",
        "pri_median",
        "pri_std",
        "cri_valid_pixel_count",
        "cri_mean",
        "cri_median",
        "cri_std",
        "ndvi_valid_pixel_count",
        "ndvi_mean",
        "ndvi_median",
        "ndvi_std",
    ]

    row = {
        "scan_name": os.path.basename(scan_folder),
        "scan_folder": scan_folder,
        "analysis_y_reduction": y_label,
        "y_reduction_mode": y_reduction_mode,
        "y_band_half_height": y_band_half_height,
        "ndvi_mask_threshold": ndvi_threshold,
        "white_reference": os.path.basename(white_path),
        "dark_reference": os.path.basename(dark_path),
        "mask_pixel_count": mask_summary["pixel_count"],
        "mask_coverage_pct": mask_summary["coverage_pct"],
        "pri_valid_pixel_count": index_summaries["PRI"]["valid_pixel_count"],
        "pri_mean": index_summaries["PRI"]["mean"],
        "pri_median": index_summaries["PRI"]["median"],
        "pri_std": index_summaries["PRI"]["std"],
        "cri_valid_pixel_count": index_summaries["CRI"]["valid_pixel_count"],
        "cri_mean": index_summaries["CRI"]["mean"],
        "cri_median": index_summaries["CRI"]["median"],
        "cri_std": index_summaries["CRI"]["std"],
        "ndvi_valid_pixel_count": index_summaries["NDVI"]["valid_pixel_count"],
        "ndvi_mean": index_summaries["NDVI"]["mean"],
        "ndvi_median": index_summaries["NDVI"]["median"],
        "ndvi_std": index_summaries["NDVI"]["std"],
    }

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    if replace_existing_scan_row:
        existing_rows = [existing_row for existing_row in existing_rows if existing_row.get("scan_folder") != scan_folder]

    existing_rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    print(f"Master index summary CSV updated: {csv_path}")


def show_mask(mask_2d, title="Plant mask"):
    """
    Show 2D mask.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_2d, cmap="gray", aspect="auto")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.tight_layout()
    plt.show()


def overlay_mask_on_rgb(rgb_image, mask_2d, alpha=0.35):
    """
    Overlay binary mask on RGB image for visual inspection.

    rgb_image: (Z, X, 3), uint8 or float
    mask_2d: (Z, X), bool
    """
    rgb = rgb_image.astype(np.float32).copy()
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    overlay = rgb.copy()
    overlay[mask_2d] = [1.0, 0.0, 0.0]  # red overlay on plant pixels

    blended = (1 - alpha) * rgb + alpha * overlay

    plt.figure(figsize=(8, 6))
    plt.imshow(blended, aspect="auto")
    plt.title("RGB with plant mask overlay")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cube_reflectance = radiometric_correct_cube(cube, white_path=white_path, dark_path=dark_path, flip_x=True)
    
    ndvi = calculate_ndvi(cube_reflectance)
    pri = calculate_pri(cube_reflectance)
    cri = calculate_cri(cube_reflectance)
    
    # Supplementary whole-cube summaries are left here commented out.
    # summarize_index_volume(ndvi, name="NDVI")
    # summarize_index_volume(pri, name="PRI")
    # summarize_index_volume(cri, name="CRI")
    # plant_mask_3d = create_ndvi_mask_3d(ndvi, threshold=0.35)
    # summarize_mask_3d(plant_mask_3d, name="NDVI plant mask")
    # summarize_masked_index_volume(ndvi, plant_mask_3d, name="NDVI")
    # summarize_masked_index_volume(pri, plant_mask_3d, name="PRI")
    # summarize_masked_index_volume(cri, plant_mask_3d, name="CRI")

    y_middle = ndvi.shape[2] // 2

    if Y_REDUCTION_MODE == "slice":
        rgb_image, rgb_path = reconstruct_rgb_image(
            cube_reflectance,
            out_path=os.path.join(current_dir, "rgb_image.png"),
            y=y_middle,
        )
    elif Y_REDUCTION_MODE == "mean":
        rgb_image, rgb_path = reconstruct_rgb_image(
            cube_reflectance,
            out_path=os.path.join(current_dir, "rgb_image.png"),
            y=None,
            aggregate="mean",
        )
    elif Y_REDUCTION_MODE == "central_band_mean":
        y_range = get_central_y_band(
            ndvi.shape[2],
            center_y=y_middle,
            half_height=Y_BAND_HALF_HEIGHT,
        )
        rgb_image, rgb_path = reconstruct_rgb_image(
            cube_reflectance,
            out_path=os.path.join(current_dir, "rgb_image.png"),
            y=None,
            y_range=y_range,
            aggregate="mean",
        )
    else:
        raise ValueError(f"Unsupported Y reduction mode: {Y_REDUCTION_MODE}")

    # Step 1: create 2D plant mask from NDVI
    ndvi_2d, ndvi_label = reduce_y_dimension(
        ndvi,
        mode=Y_REDUCTION_MODE,
        y=y_middle,
        band_half_height=Y_BAND_HALF_HEIGHT,
    )
    pri_2d, _ = reduce_y_dimension(
        pri,
        mode=Y_REDUCTION_MODE,
        y=y_middle,
        band_half_height=Y_BAND_HALF_HEIGHT,
    )
    cri_2d, _ = reduce_y_dimension(
        cri,
        mode=Y_REDUCTION_MODE,
        y=y_middle,
        band_half_height=Y_BAND_HALF_HEIGHT,
    )

    ndvi_mask_2d = create_ndvi_mask(
        ndvi,
        threshold=NDVI_MASK_THRESHOLD,
        y_mode=Y_REDUCTION_MODE,
        y=y_middle,
        band_half_height=Y_BAND_HALF_HEIGHT,
    )

    print(f"Analysis Y reduction: {ndvi_label}")
    mask_summary = summarize_mask_2d(ndvi_mask_2d, name=f"NDVI plant mask ({ndvi_label})")

    # Step 2: show NDVI and mask
    show_index_map(
        ndvi,
        "NDVI",
        y_mode=Y_REDUCTION_MODE,
        y=y_middle,
        band_half_height=Y_BAND_HALF_HEIGHT,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
    )
    show_mask(ndvi_mask_2d, title=f"NDVI plant mask ({ndvi_label})")

    # Step 3: overlay mask on RGB
    overlay_mask_on_rgb(rgb_image, ndvi_mask_2d)

    # Step 4: apply mask to 2D slices of PRI and CRI
    pri_masked = apply_mask_to_index(pri_2d, ndvi_mask_2d)
    cri_masked = apply_mask_to_index(cri_2d, ndvi_mask_2d)
    ndvi_masked = apply_mask_to_index(ndvi_2d, ndvi_mask_2d)
  

    # Step 5: show masked maps
    plt.figure(figsize=(8, 6))
    plt.imshow(pri_masked, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(label="PRI")
    plt.title(f"Masked PRI ({ndvi_label})")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(cri_masked, cmap="viridis", aspect="auto")
    plt.colorbar(label="CRI")
    plt.title(f"Masked CRI ({ndvi_label})")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.tight_layout()
    plt.show()

    # Step 6: thesis-facing quantitative outputs from the central-band masked maps
    pri_summary = summarize_masked_index(pri_masked, name=f"PRI masked ({ndvi_label})")
    cri_summary = summarize_masked_index(cri_masked, name=f"CRI masked ({ndvi_label})")
    ndvi_summary = summarize_masked_index(ndvi_masked, name=f"NDVI masked ({ndvi_label})")

    if APPEND_TO_MASTER_CSV:
        update_master_summary_csv(
            csv_path=MASTER_SUMMARY_CSV,
            scan_folder=scan_folder,
            y_label=ndvi_label,
            mask_summary=mask_summary,
            index_summaries={
                "PRI": pri_summary,
                "CRI": cri_summary,
                "NDVI": ndvi_summary,
            },
            y_reduction_mode=Y_REDUCTION_MODE,
            y_band_half_height=Y_BAND_HALF_HEIGHT,
            ndvi_threshold=NDVI_MASK_THRESHOLD,
            white_path=white_path,
            dark_path=dark_path,
            replace_existing_scan_row=REPLACE_EXISTING_SCAN_ROW,
        )
    
    #Step 7: Visualise spectrum before and after Gaussian smoothing for one pixel
    visualise_spectrum_before_after_gaussian(cube_reflectance, z=24, x=17, y=y_middle,)
    visualise_spectrum_at(cube_reflectance, z=28, x=11, y=y_middle)
    #visualise_raw_image_spectrum(dark_path, flip_x=True)
    #visualise_raw_image_spectrum(white_path, flip_x=True)
    #visualise_raw_image_spectrum(dark_path, flip_x=True)
    #visualise_white_dark_difference(white_path, dark_path, flip_x=True, plot=True)
