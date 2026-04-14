import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from Generate_cube import CubeNM

from cube_visuals import (
    visualise_spectrum_at,
    visualise_wavelength_slice,
    reconstruct_rgb_image,
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

scan_folder = os.path.join(BASE_DIR, "edge", "data", "scan_26March_13:22:22")  # Choose one specific folder for now
npz_path = os.path.join(scan_folder, "cube_ZXnm_corrected.npz")

data = np.load(npz_path)
cube = CubeNM(data["cube"], data["wavs_nm"])

white_path = os.path.join(BASE_DIR, "server", "whiteReference.png")
dark_path = os.path.join(BASE_DIR, "server", "darkReference.png")


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


def plot_white_dark_difference(white_path, dark_path, flip_x=True):
    wavs, white_spec = extract_spectrum_from_raw(white_path, flip_x=flip_x)
    _, dark_spec = extract_spectrum_from_raw(dark_path, flip_x=flip_x)

    if len(white_spec) != len(dark_spec):
        raise RuntimeError("White and dark spectra have different lengths.")

    diff = white_spec - dark_spec

    # Limit to relevant spectral area (380–820 nm)
    mask = (wavs >= 380) & (wavs <= 820)

    plt.figure(figsize=(8, 5))
    plt.plot(wavs[mask], diff[mask])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity difference (a.u.)")
    plt.title("White minus dark spectrum (380–820 nm)")
    plt.tight_layout()
    plt.show()

    return wavs, diff


def show_index_map(index_data, name, y=None, cmap="RdYlGn", vmin=None, vmax=None):
    """
    Show a 2D map from a 3D spectral index array with shape (Z, X, Y).
    """
    if y is None:
        y = index_data.shape[2] // 2

    index_map = index_data[:, :, y]

    plt.figure(figsize=(8, 6))
    plt.imshow(index_map, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(label=name)
    plt.title(f"{name} map at y={y}")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.tight_layout()
    plt.show()

    print(f"{name} min: {np.nanmin(index_data):.4f}")
    print(f"{name} max: {np.nanmax(index_data):.4f}")
    print(f"{name} mean: {np.nanmean(index_data):.4f}")


if __name__ == "__main__":
    
    cube_reflectance = radiometric_correct_cube(cube, white_path=white_path, dark_path=dark_path, flip_x=True)

    #output_path = os.path.join(scan_folder, "cube_ZXnm_radiometric.npz")
    #save_corrected_cube(output_path, cube_reflectance)
    ndvi = calculate_ndvi(cube_reflectance)
    #pri = calculate_pri(cube_reflectance)
    #cri = calculate_cri(cube_reflectance)

    #print(f"NDVI shape: {ndvi.shape}")
    #print(f"PRI shape: {pri.shape}")
    #print(f"CRI shape: {cri.shape}")

    y_middle = ndvi.shape[2] // 2
    #show_index_map(ndvi, "NDVI", y=y_middle, cmap="RdYlGn", vmin=-1, vmax=1)
    #show_index_map(pri, "PRI", y=y_middle, cmap="RdYlGn", vmin=-1, vmax=1)
    #show_index_map(cri, "CRI", y=y_middle, cmap="viridis")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reconstruct_rgb_image(cube_reflectance, out_path=os.path.join(current_dir, "rgb_image.png"), y=y_middle)
    
    visualise_spectrum_at(cube_reflectance, z=11, x=9, y=y_middle)