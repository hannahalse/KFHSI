import os, re
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt

from cube_visuals import (
    visualise_spectrum_at,
    visualise_wavelength_slice,
    reconstruct_rgb_image,
)

from SpectralTools import (
    calculate_nir_red_indices,
    calculate_ndvi,
)

#from calibration.wavelength_calibr import (
#    wavelength_axis,
#)

A = 0.7241145833
B = 288.45625

def px_to_nm(px):
    "Convert pixel index → wavelength (nm)"
    return A * px + B


def wavelength_axis(width):
    "Return wavelength for every pixel in the image"
    pixels = np.arange(width)
    return px_to_nm(pixels)


# ----------------- CONFIG -----------------
wl_min = 380.0
wl_max = 820.0
SHIFT         = -5 
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR      = os.path.join(BASE_DIR, "edge", "data")

# -------------------------------------------

def find_last_modified_folder(data_dir=DATA_DIR, prefix="scan_"):
    """
    Finds the latest modified folder in data_dir with the given prefix.
    Returns the full path to the folder.
    """
    scan_folders = [f for f in os.listdir(data_dir)
                    if f.startswith(prefix) and os.path.isdir(os.path.join(data_dir, f))]
    if not scan_folders:
        raise FileNotFoundError(f"No scan folders found in: {data_dir}")

    scan_folders.sort(
        key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
        reverse=True
    )

    latest_scan = scan_folders[0]
    return os.path.join(data_dir, latest_scan)

def sort_images(scan_folder):
    """
    Scan a folder for images named like 'X<number>_Z<number>.png', group them by Z-position, and return:
      - rows: dict[z] -> list of (x, full_path)
      - Zs: sorted list of all Z-positions
    """
    pat = re.compile(r"X(\d+)[_-]Z(\d+)\.png", re.IGNORECASE)
    rows = defaultdict(list)  # Z is key. z -> list[(x, path)]

    for f in os.listdir(scan_folder):
        # Only consider PNG files starting with X
        if not (f.lower().endswith(".png") and f.upper().startswith("X")):
            continue

        m = pat.search(f)
        if not m:
            continue

        x10 = int(m.group(1))   # X index
        z10 = int(m.group(2))   # Z index
        rows[z10].append((x10, os.path.join(scan_folder, f))) #For each z-pos, have list of (x-pos, filepath)

    if not rows:
        raise RuntimeError(f"No X*_Z*.png images found in {scan_folder}.")

    Zs = sorted(rows.keys())  # sorted Z positions (top → bottom)
    return rows, Zs

class CubeNM:
    def __init__(self, data, wavs_nm):
        """
        data: ndarray (Z, X, Y, W) where axis=3 corresponds to wavelengths
        wavs_nm: ndarray (W,) with wavelengths in nm
        """
        self.data = data
        self.wavs_nm = wavs_nm.astype(np.float32)
        self.min_nm = float(wavs_nm[0])
        self.max_nm = float(wavs_nm[-1])

    def __getitem__(self, idx):
        # Support cube[z, x, y, nm] with nm in wavelength range
        if isinstance(idx, tuple) and len(idx) == 4 and isinstance(idx[3],(int, float, np.integer, np.floating)):
            z, x, y, nm = idx
            nm = float(nm)
            if nm < self.min_nm or nm > self.max_nm:
                raise IndexError(f"nm index {nm} out of range [{self.min_nm}, {self.max_nm}]")
            k = int(np.searchsorted(self.wavs_nm, nm, side='left'))
            if k == len(self.wavs_nm):
                k -= 1
            elif k > 0 and abs(self.wavs_nm[k] - nm) > abs(self.wavs_nm[k-1] - nm):
                k -= 1
            return self.data[z, x, y, k]
        return self.data[idx]

    @property
    def shape(self):
        return self.data.shape

    def numpy(self):
        return self.data

def build_cube(rows, Zs, scan_folder):
    """
    Build a hyperspectral cube from grouped scan rows.

    Parameters:
        rows: dict[z] -> list of (x, full_path)
        Zs: sorted list of Z positions (keys from rows)
        scan_folder: folder where the X*_Z*.png images are stored (used for saving npz)
        start_nm: start wavelength 
        end_nm: end wavelength 

    Returns:
        cube_nm: ndarray with shape (Z, X, Y, W)
        wavs:    1D ndarray with wavelengths in nm (length = W)
        npz_path: path to the saved .npz file
    """
    # ---- Read first frame to get H, W and wavelength grid ----
    first_z = Zs[0]
    first_x_sorted = sorted(rows[first_z], key=lambda t: t[0])
    Xc = len(first_x_sorted)  # number of X positions in each Z row

    sample_img = cv2.imread(first_x_sorted[0][1], cv2.IMREAD_GRAYSCALE)
    if sample_img is None:
        raise RuntimeError(f"Could not read {first_x_sorted[0][1]}")
    H, W = sample_img.shape
    print(f"Frame size: H={H}, W={W}")

    #------- PLACEHOLDER -------
    #wavs = np.linspace(start_nm, end_nm, W, dtype=np.float32)  # wavelength grid
    #print(f"Spectral grid: {len(wavs)} bands from {start_nm} nm to {end_nm} nm")
    #------- PLACEHOLDER -------

    wavs = wavelength_axis(W).astype(np.float32)
    #Clipping the cube: 
    mask = (wavs >= wl_min) & (wavs <= wl_max)
    keep_idx = np.where(mask)[0]
    wavs = wavs[mask]
    print(f"Clipped spectral range: {wavs[0]:.1f} nm → {wavs[-1]:.1f} nm ({len(wavs)} bands)")

    Zc = len(Zs)
    cube_nm = np.zeros((Zc, Xc, H, len(wavs)), dtype=np.float32)  # empty cube [Z, X, Y, W]

    # ---- Fill the cube (Z, X, Y, W) ----
    for zi, z in enumerate(Zs):
        x_paths = sorted(rows[z], key=lambda t: int(t[0]))
        if len(x_paths) != Xc:
            print(f"Row Z={z} has {len(x_paths)} X positions (expected {Xc}).")

        for xi, (_x, path) in enumerate(x_paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Could not read {path}")
            if img.shape != (H, W):
                raise RuntimeError(f"Inconsistent frame size at {path}: {img.shape} vs {(H, W)}")

            # Store the full image for this (Z, X) position
            #Flip the image horizontally
            #img = np.fliplr(img)   
            img_f = img.astype(np.float32)
            cube_nm[zi, xi, :, :] = img_f[:, keep_idx]
            #cube_nm[zi, xi, :, :] = img.astype(np.float32)

    print("Built cube_nm with shape (Z, X, Y, wavelength):", cube_nm.shape)
    print("Cube wavelengths:", wavs[0], wavs[-1])

    # ---- Save to disk ----
    npz_path = os.path.join(scan_folder, "cube_ZXnm.npz")
    np.savez_compressed(npz_path, cube=cube_nm, wavs_nm=wavs, Zs=np.array(Zs, dtype=np.int32))
    print("Saved cube to:", npz_path)

    return cube_nm, wavs, npz_path

def build_alternating_shifts(Zc, plus_first=True):
    """
    Returns an array of alternating +SHIFT and -SHIFT for Zc layers.
    
    """
    s = np.zeros(Zc, dtype=int)
    for zi in range(Zc):
        if plus_first:
            s[zi] = SHIFT if (zi % 2 == 0) else -SHIFT
        else:
            s[zi] = -SHIFT if (zi % 2 == 0) else SHIFT
    return s

def apply_integer_x_shifts(cube_nm, shifts, pad_value=np.nan):
    """
    Creates a new cube with integer X-shifts applied per Z-layer.
    Returns: shifted cube (with NaN in padded areas).
    """
    Zc, Xc, Yc, Wc = cube_nm.shape
    out = np.full_like(cube_nm, pad_value)
    for zi, s in enumerate(shifts):
        if s == 0:
            out[zi] = cube_nm[zi]
        elif s > 0:
            # Move towards right: data moves outward; left filled with NaN
            out[zi, s:, :, :] = cube_nm[zi, :Xc - s, :, :]
        else:
            s2 = -s
            # Move towards left
            out[zi, :Xc - s2, :, :] = cube_nm[zi, s2:, :, :]
    return out

def crop_valid_overlap(shifted):
    """
    Finds the overlapping valid X-interval (all rows non-NaN), and crops the cube accordingly.
    Returns: cropped cube and slice object for X dimension.
    """
    Zc, Xc, Yc, Wc = shifted.shape
    mask = ~np.isnan(shifted[:, :, 0, 0])   # (Z, X)
    left = 0
    right = Xc
    # Iterate through each Z row to find the first valid column
    for zi in range(Zc):
        row = mask[zi]
        if row.any():
            first = np.argmax(row)
            if first > left:
                left = first
    # Iterate through each Z row to find the last valid column
    for zi in range(Zc):
        row = mask[zi]
        if row.any():
            last = Xc - np.argmax(row[::-1])
            if last < right:
                right = last
    xs = slice(left, right) # Valid X range
    return shifted[:, xs, :, :], xs # Return cropped cube and X slice

if __name__ == "__main__":
    # ---- Finding the latest modified scan folder ----
    scan_folder = find_last_modified_folder()
    #To choose a specific scan folder use this: 
    #scan_folder = "/Users/hannahalse/KFSpectra/edge/data/scan_30October_15:21:15"
    print("Using this scan folder:", scan_folder)
    
    # ---- Building the cube from scan ----
    rows, Zs = sort_images(scan_folder)
    cube_nm, wavs, npz_path = build_cube(rows, Zs, scan_folder)

    # ---- Correcting snake-like X-offsets -------
    # 1) Building alternating +/- X-shifts per Z-row
    alt_shifts = build_alternating_shifts(Zc=len(Zs), plus_first=True)

    # 2) Applying shifts to the cube (NaNs at the edges)
    shifted_cube = apply_integer_x_shifts(cube_nm, alt_shifts, pad_value=np.nan)

    # 3) Cropping to the common valid X-overlap
    cube_nm_sym, xslice = crop_valid_overlap(shifted_cube)
    print("Sym-shifted & cropped cube shape:", cube_nm_sym.shape, "X slice:", xslice)

    # 4) Continue working with the corrected cube wrapped in CubeNM ---
    cube_nm = cube_nm_sym
    cube = CubeNM(cube_nm, wavs)
    print("Final cube shape (Z, X, Y, nm):", cube.shape)
    
    


    _, _, H, _ = cube_nm.shape
    y_middle = H // 2
    rgb_image_path = os.path.join(scan_folder, "reconstructed_rgb.png")
    rgb_image, out_path = reconstruct_rgb_image(cube, rgb_image_path, y=y_middle)
    
    """
    # ---- Visualisations ----
    #_, _, H, _ = cube_nm.shape
    #y_middle = H // 2

    #visualise_wavelength_slice(cube, 520, os.path.join(scan_folder, "wavelength_520nm.png"))
    #visualise_spectrum_at(cube, z=35, x=3, y=y_middle)
    #rgb_image_path = os.path.join(scan_folder, "reconstructed_rgb.png")
    #rgb_image, out_path = reconstruct_rgb_image(cube, rgb_image_path, y=y_middle)
    

    cube_file = "../edge/data/scan_16November_13:58:13/cube_ZXnm.npz"
    data = np.load(cube_file)
    cube = data["cube"]
    wavelengths = data["wavs_nm"]
    #print(f"Number of wavelengths {len(wavelengths)} from {wavelengths[0]} nm to {wavelengths[-1]} nm")
    #print(f"Cube shape: {cube.shape}, whare length of cube[3] is: {cube.shape[3]}")

    z, x = 18, 25  # velg en piksel midt på bladet
    spectrum = cube[z, x, 225, :]  # hvis form (Z, Y, X, B) og du tar Z=0

    plt.plot(wavelengths, spectrum)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity")
    plt.show()

    # ----- NDVI Calculation -----
    #red_idx, nir_idx = calculate_nir_red_indices(cube, wavelengths)
    #print("red_idx, nir_idx:", red_idx, nir_idx)
    #ndvi_image = calculate_ndvi(cube, red_idx, nir_idx)
    """


  
    
                                                 
