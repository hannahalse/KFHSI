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
    calculate_cri1,
    calculate_cri2,
    calculate_pri,
    calculate_sipi,
    calculate_psri,
    MAX_STABLE_REFLECTANCE,
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
DATA_DIR = os.path.join(BASE_DIR, "edge", "data", "Experiment3", "day6")  # Change to "Experiment" for all data
MASTER_CSV = os.path.join(BASE_DIR, "edge", "data", "Experiment3") 

"""#Experiment 2
#Pos 1: scan_04May_11:01:08
#Pos 2: scan_04May_11:53:02
#Pos 3: scan_04May_13:00:39
#Pos 1: scan_05May_11:58:21
#Pos 2: scan_05May_12:50:07
#Pos 3: scan_05May_13:39:47
#Pos 1: scan_06May_11:10:08
#Pos 2: scan_06May_11:59:40
#Pos 3: scan_06May_12:48:43
#Pos 1: scan_07May_12:03:15
#Pos 2: scan_07May_12:51:47
#Pos 3: scan_07May_13:44:14
#Pos 1: scan_08May_12:09:29
#Pos 2: scan_08May_12:58:07
#Pos 3: scan_08May_13:49:28
#Pos 1: scan_09May_12:17:32
#Pos 2: scan_09May_13:09:07
#Pos 3: scan_09May_14:03:51
#Pos 1: scan_11May_09:20:25
#Pos 2: scan_11May_10:09:11
#Pos 3: scan_11May_10:57:40
#Pos 1: scan_12May_10:58:19
#Pos 2: scan_12May_12:45:45
#Pos 3: scan_12May_13:36:09
#Pos 1: scan_13May_10:07:42
#Pos 2: scan_13May_10:56:34
#Pos 3: scan_13May_11:46:46


#For the new calibration function
#Pos 1: scan_04May_11:01:08
#Pos 1: scan_05May_11:58:21
#Pos 1: scan_06May_11:10:08
#Pos 1: scan_07May_12:03:15
#Pos 1: scan_08May_12:09:29
#Pos 1: scan_09May_12:17:32
#Pos 1: scan_11May_09:20:25
#Pos 1: scan_12May_10:58:19
#Pos 1: scan_13May_10:07:42"""


#scan_04June_08:42:18
#scan_04June_09:58:07
#scan_04June_10:55:04
#scan_04June_11:51:11
#scan_04June_12:47:06


#Kjør med 28 for å få plantepixelfigur
DEFAULT_SCAN_FOLDER = os.path.join(DATA_DIR, "scan_27May_08:42:56")  # Choose one specific folder for now

white_path = os.path.join(BASE_DIR, "calibration", "whiteReference")
dark_path = os.path.join(BASE_DIR, "calibration", "darkReference")

# How to collapse the Y dimension when making 2D maps/RGB views.
# Options:
#   "slice"             -> use one Y row
#   "mean"              -> average over all Y rows
#   "central_band_mean" -> average over a central Y band
Y_REDUCTION_MODE = "central_band_mean"
Y_BAND_HALF_HEIGHT = 20
# COMPARISON_BAND_HALF_HEIGHTS = (10, 20, 40, 60, 100)
NDVI_MASK_THRESHOLD = 0.03
REFERENCE_MAX_REFLECTANCE = 2.0
#--------------CHANGE THIS TO ADD TO THE CSV FILE!!!!!!--------------
APPEND_TO_MASTER_CSV = False
MASTER_SUMMARY_CSV = os.path.join(MASTER_CSV, "masked_index_time_seriesExp3Plant5.csv")
REPLACE_EXISTING_SCAN_ROW = False

TITLE_FONTSIZE = 26
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 18
COLORBAR_LABEL_FONTSIZE = 18


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
    cbar = plt.colorbar()
    cbar.set_label(name, fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    plt.xlabel("X", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Z", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
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


def calculate_index_volumes(cube_reflectance, max_reflectance=MAX_STABLE_REFLECTANCE):
    """
    Calculate all index volumes with the same upper reflectance quality limit.
    """
    return {
        "NDVI": calculate_ndvi(cube_reflectance, max_reflectance=max_reflectance),
        "PRI": calculate_pri(cube_reflectance, max_reflectance=max_reflectance),
        "CRI1": calculate_cri1(cube_reflectance, max_reflectance=max_reflectance),
        "CRI2": calculate_cri2(cube_reflectance, max_reflectance=max_reflectance),
        "SIPI": calculate_sipi(cube_reflectance, max_reflectance=max_reflectance),
        "PSRI": calculate_psri(cube_reflectance, max_reflectance=max_reflectance),
    }


def masked_count_and_stats(index_map, plant_mask):
    values = np.asarray(index_map[np.isfinite(index_map) & plant_mask], dtype=np.float32)
    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def print_max_reflectance_diagnostic(
    analysis_index_volumes,
    reference_index_volumes,
    y_middle,
    y_mode,
    band_half_height,
    ndvi_threshold,
    analysis_max_reflectance,
    reference_max_reflectance,
):
    """
    Compare current quality filtering with the previous near-clip threshold.
    """
    analysis_ndvi_2d, y_label = reduce_y_dimension(
        analysis_index_volumes["NDVI"],
        mode=y_mode,
        y=y_middle,
        band_half_height=band_half_height,
    )
    reference_ndvi_2d, _ = reduce_y_dimension(
        reference_index_volumes["NDVI"],
        mode=y_mode,
        y=y_middle,
        band_half_height=band_half_height,
    )

    analysis_mask = analysis_ndvi_2d > ndvi_threshold
    reference_mask = reference_ndvi_2d > ndvi_threshold
    analysis_mask_count = int(np.count_nonzero(analysis_mask))
    reference_mask_count = int(np.count_nonzero(reference_mask))
    removed_mask_count = int(np.count_nonzero(reference_mask & ~analysis_mask))
    removed_mask_pct = 100.0 * removed_mask_count / reference_mask_count if reference_mask_count else 0.0

    print("\nMax reflectance quality check")
    print(f"Analysis Y reduction: {y_label}")
    print(f"Current max reflectance:  {analysis_max_reflectance:.2f}")
    print(f"Reference max reflectance: {reference_max_reflectance:.2f}")
    print(
        "NDVI plant mask pixels: "
        f"{analysis_mask_count}/{reference_mask_count} kept, "
        f"{removed_mask_count} removed ({removed_mask_pct:.1f}%)"
    )

    for index_name in ("NDVI", "PRI", "CRI1", "CRI2", "SIPI", "PSRI"):
        analysis_map, _ = reduce_y_dimension(
            analysis_index_volumes[index_name],
            mode=y_mode,
            y=y_middle,
            band_half_height=band_half_height,
        )
        reference_map, _ = reduce_y_dimension(
            reference_index_volumes[index_name],
            mode=y_mode,
            y=y_middle,
            band_half_height=band_half_height,
        )

        analysis_stats = masked_count_and_stats(analysis_map, analysis_mask)
        reference_stats = masked_count_and_stats(reference_map, reference_mask)
        removed_count = max(0, reference_stats["count"] - analysis_stats["count"])
        removed_pct = 100.0 * removed_count / reference_stats["count"] if reference_stats["count"] else 0.0
        mean_change = analysis_stats["mean"] - reference_stats["mean"]
        median_change = analysis_stats["median"] - reference_stats["median"]

        print(
            f"{index_name}: "
            f"{analysis_stats['count']}/{reference_stats['count']} pixels kept, "
            f"{removed_count} removed ({removed_pct:.1f}%), "
            f"mean {reference_stats['mean']:.4f} -> {analysis_stats['mean']:.4f} "
            f"(change {mean_change:+.4f}), "
            f"median {reference_stats['median']:.4f} -> {analysis_stats['median']:.4f} "
            f"(change {median_change:+.4f})"
        )


# def print_half_height_mean_comparison(index_volumes, y_middle, half_heights, ndvi_threshold):
#     """
#     Print masked mean values for several central-band half-heights.
#
#     The NDVI mask is rebuilt separately for each half-height so the comparison
#     reflects the full thesis-facing workflow for that reduction size.
#     """
#     print("\nCentral-band mean comparison:")
#     for half_height in half_heights:
#         ndvi_2d, y_label = reduce_y_dimension(
#             index_volumes["NDVI"],
#             mode="central_band_mean",
#             y=y_middle,
#             band_half_height=half_height,
#         )
#         mask_2d = ndvi_2d > ndvi_threshold
#         pixel_count = int(np.count_nonzero(mask_2d))
#
#         print(f"  half_height={half_height} ({y_label}), plant pixels={pixel_count}")
#
#         for index_name, volume in index_volumes.items():
#             reduced_2d, _ = reduce_y_dimension(
#                 volume,
#                 mode="central_band_mean",
#                 y=y_middle,
#                 band_half_height=half_height,
#             )
#             masked = np.where(mask_2d, reduced_2d, np.nan)
#             if np.isfinite(masked).any():
#                 mean_value = float(np.nanmean(masked))
#                 print(f"    {index_name} mean: {mean_value:.4f}")
#             else:
#                 print(f"    {index_name} mean: NaN")


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
        "cri1_valid_pixel_count",
        "cri1_mean",
        "cri1_median",
        "cri1_std",
        "cri2_valid_pixel_count",
        "cri2_mean",
        "cri2_median",
        "cri2_std",
        "ndvi_valid_pixel_count",
        "ndvi_mean",
        "ndvi_median",
        "ndvi_std",
        "sipi_valid_pixel_count",
        "sipi_mean",
        "sipi_median",
        "sipi_std",
        "psri_valid_pixel_count",
        "psri_mean",
        "psri_median",
        "psri_std",
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
        "cri1_valid_pixel_count": index_summaries["CRI1"]["valid_pixel_count"],
        "cri1_mean": index_summaries["CRI1"]["mean"],
        "cri1_median": index_summaries["CRI1"]["median"],
        "cri1_std": index_summaries["CRI1"]["std"],
        "cri2_valid_pixel_count": index_summaries["CRI2"]["valid_pixel_count"],
        "cri2_mean": index_summaries["CRI2"]["mean"],
        "cri2_median": index_summaries["CRI2"]["median"],
        "cri2_std": index_summaries["CRI2"]["std"],
        "ndvi_valid_pixel_count": index_summaries["NDVI"]["valid_pixel_count"],
        "ndvi_mean": index_summaries["NDVI"]["mean"],
        "ndvi_median": index_summaries["NDVI"]["median"],
        "ndvi_std": index_summaries["NDVI"]["std"],
        "sipi_valid_pixel_count": index_summaries["SIPI"]["valid_pixel_count"],
        "sipi_mean": index_summaries["SIPI"]["mean"],
        "sipi_median": index_summaries["SIPI"]["median"],
        "sipi_std": index_summaries["SIPI"]["std"],
        "psri_valid_pixel_count": index_summaries["PSRI"]["valid_pixel_count"],
        "psri_mean": index_summaries["PSRI"]["mean"],
        "psri_median": index_summaries["PSRI"]["median"],
        "psri_std": index_summaries["PSRI"]["std"],
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
    plt.xlabel("X", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Z", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
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
    plt.xlabel("X", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Z", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.show()

def load_scan_cube(scan_folder):
    npz_path = os.path.join(scan_folder, "cube_ZXnm_corrected.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Corrected cube not found: {npz_path}")
    data = np.load(npz_path)
    return CubeNM(data["cube"], data["wavs_nm"])


def show_masked_index_map(index_map, name, y_label, cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(index_map, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label(name, fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    plt.xlabel("X", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Z", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.show()


def process_scan(
    scan_folder,
    csv_path=MASTER_SUMMARY_CSV,
    show_plots=True,
    append_to_master_csv=APPEND_TO_MASTER_CSV,
    replace_existing_scan_row=REPLACE_EXISTING_SCAN_ROW,
    print_diagnostic=True,
    y_reduction_mode=Y_REDUCTION_MODE,
    y_band_half_height=Y_BAND_HALF_HEIGHT,
    ndvi_threshold=NDVI_MASK_THRESHOLD,
    max_reflectance=MAX_STABLE_REFLECTANCE,
    reference_max_reflectance=REFERENCE_MAX_REFLECTANCE,
):
    """
    Process one scan folder and optionally append its summaries to a master CSV.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cube = load_scan_cube(scan_folder)
    cube_reflectance = radiometric_correct_cube(cube, white_path=white_path, dark_path=dark_path, flip_x=True)

    index_volumes = calculate_index_volumes(cube_reflectance, max_reflectance=max_reflectance)
    ndvi = index_volumes["NDVI"]
    pri = index_volumes["PRI"]
    cri1 = index_volumes["CRI1"]
    cri2 = index_volumes["CRI2"]
    sipi = index_volumes["SIPI"]
    psri = index_volumes["PSRI"]

    y_middle = ndvi.shape[2] // 2
    if print_diagnostic:
        reference_index_volumes = calculate_index_volumes(
            cube_reflectance,
            max_reflectance=reference_max_reflectance,
        )
        print_max_reflectance_diagnostic(
            analysis_index_volumes=index_volumes,
            reference_index_volumes=reference_index_volumes,
            y_middle=y_middle,
            y_mode=y_reduction_mode,
            band_half_height=y_band_half_height,
            ndvi_threshold=ndvi_threshold,
            analysis_max_reflectance=max_reflectance,
            reference_max_reflectance=reference_max_reflectance,
        )

    rgb_image = None
    if show_plots:
        if y_reduction_mode == "slice":
            rgb_image, _ = reconstruct_rgb_image(
                cube_reflectance,
                out_path=os.path.join(current_dir, "rgb_image.png"),
                y=y_middle,
            )
        elif y_reduction_mode == "mean":
            rgb_image, _ = reconstruct_rgb_image(
                cube_reflectance,
                out_path=os.path.join(current_dir, "rgb_image.png"),
                y=None,
                aggregate="mean",
            )
        elif y_reduction_mode == "central_band_mean":
            y_range = get_central_y_band(
                ndvi.shape[2],
                center_y=y_middle,
                half_height=y_band_half_height,
            )
            rgb_image, _ = reconstruct_rgb_image(
                cube_reflectance,
                out_path=os.path.join(current_dir, "rgb_image.png"),
                y=None,
                y_range=y_range,
                aggregate="mean",
            )
        else:
            raise ValueError(f"Unsupported Y reduction mode: {y_reduction_mode}")

    ndvi_2d, ndvi_label = reduce_y_dimension(
        ndvi,
        mode=y_reduction_mode,
        y=y_middle,
        band_half_height=y_band_half_height,
    )
    pri_2d, _ = reduce_y_dimension(
        pri,
        mode=y_reduction_mode,
        y=y_middle,
        band_half_height=y_band_half_height,
    )
    cri1_2d, _ = reduce_y_dimension(
        cri1,
        mode=y_reduction_mode,
        y=y_middle,
        band_half_height=y_band_half_height,
    )
    cri2_2d, _ = reduce_y_dimension(
        cri2,
        mode=y_reduction_mode,
        y=y_middle,
        band_half_height=y_band_half_height,
    )
    sipi_2d, _ = reduce_y_dimension(
        sipi,
        mode=y_reduction_mode,
        y=y_middle,
        band_half_height=y_band_half_height,
    )
    psri_2d, _ = reduce_y_dimension(
        psri,
        mode=y_reduction_mode,
        y=y_middle,
        band_half_height=y_band_half_height,
    )

    ndvi_mask_2d = create_ndvi_mask(
        ndvi,
        threshold=ndvi_threshold,
        y_mode=y_reduction_mode,
        y=y_middle,
        band_half_height=y_band_half_height,
    )

    print(f"Analysis Y reduction: {ndvi_label}")
    mask_summary = summarize_mask_2d(ndvi_mask_2d, name=f"NDVI plant mask ({ndvi_label})")

    if show_plots:
        show_index_map(
            ndvi,
            "NDVI",
            y_mode=y_reduction_mode,
            y=y_middle,
            band_half_height=y_band_half_height,
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
        )
        show_mask(ndvi_mask_2d, title=f"NDVI plant mask ({ndvi_label})")
        overlay_mask_on_rgb(rgb_image, ndvi_mask_2d)

    pri_masked = apply_mask_to_index(pri_2d, ndvi_mask_2d)
    cri1_masked = apply_mask_to_index(cri1_2d, ndvi_mask_2d)
    cri2_masked = apply_mask_to_index(cri2_2d, ndvi_mask_2d)
    ndvi_masked = apply_mask_to_index(ndvi_2d, ndvi_mask_2d)
    sipi_masked = apply_mask_to_index(sipi_2d, ndvi_mask_2d)
    psri_masked = apply_mask_to_index(psri_2d, ndvi_mask_2d)

    if show_plots:
        show_masked_index_map(pri_masked, "PRI", ndvi_label, cmap="RdYlGn", vmin=-1, vmax=1)
        show_masked_index_map(cri1_masked, "CRI1", ndvi_label)
        show_masked_index_map(cri2_masked, "CRI2", ndvi_label)
        show_masked_index_map(sipi_masked, "SIPI", ndvi_label)
        show_masked_index_map(psri_masked, "PSRI", ndvi_label)

    pri_summary = summarize_masked_index(pri_masked, name=f"PRI masked ({ndvi_label})")
    cri1_summary = summarize_masked_index(cri1_masked, name=f"CRI1 masked ({ndvi_label})")
    cri2_summary = summarize_masked_index(cri2_masked, name=f"CRI2 masked ({ndvi_label})")
    ndvi_summary = summarize_masked_index(ndvi_masked, name=f"NDVI masked ({ndvi_label})")
    sipi_summary = summarize_masked_index(sipi_masked, name=f"SIPI masked ({ndvi_label})")
    psri_summary = summarize_masked_index(psri_masked, name=f"PSRI masked ({ndvi_label})")

    index_summaries = {
        "PRI": pri_summary,
        "CRI1": cri1_summary,
        "CRI2": cri2_summary,
        "NDVI": ndvi_summary,
        "SIPI": sipi_summary,
        "PSRI": psri_summary,
    }

    if append_to_master_csv:
        update_master_summary_csv(
            csv_path=csv_path,
            scan_folder=scan_folder,
            y_label=ndvi_label,
            mask_summary=mask_summary,
            index_summaries=index_summaries,
            y_reduction_mode=y_reduction_mode,
            y_band_half_height=y_band_half_height,
            ndvi_threshold=ndvi_threshold,
            white_path=white_path,
            dark_path=dark_path,
            replace_existing_scan_row=replace_existing_scan_row,
        )

    if show_plots:
        visualise_spectrum_before_after_gaussian(cube_reflectance, z=38, x=20, y=y_middle)
        visualise_spectrum_at(cube_reflectance, z=38, x=20, y=y_middle)

    return {
        "scan_folder": scan_folder,
        "csv_path": csv_path,
        "mask_summary": mask_summary,
        "index_summaries": index_summaries,
    }


if __name__ == "__main__":
    process_scan(DEFAULT_SCAN_FOLDER)
