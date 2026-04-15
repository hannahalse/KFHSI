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

#from calibration.wavelength_calibr import (
#    wavelength_axis,
#)

#-------------TODO Cant seem to import this from calibration, so hardcoding here for now.----------------
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
FLIP_X = True  # Set to True if images need to be flipped horizontally

# -------------------------------------------

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

def build_raw_cube(rows, Zs):
    """
    Build the raw hyperspectral cube from grouped scan rows.

    Returns:
        cube_nm_raw: ndarray with shape (Z, X, Y, W)
        wavs: 1D ndarray with wavelengths in nm
    """
    # ----- Read first frame to get H, W and wavelength grid ----
    first_z = Zs[0]
    first_x_sorted = sorted(rows[first_z], key=lambda t: t[0])
    Xc = len(first_x_sorted)    #Number of X positions in each Z row

    sample_img = cv2.imread(first_x_sorted[0][1], cv2.IMREAD_GRAYSCALE)
    if sample_img is None:
        raise RuntimeError(f"Could not read {first_x_sorted[0][1]}")

    H, W = sample_img.shape
    print(f"Frame size: H={H}, W={W}")

    wavs = wavelength_axis(W).astype(np.float32)
    #Clipping the cube:
    mask = (wavs >= wl_min) & (wavs <= wl_max)
    keep_idx = np.where(mask)[0]
    wavs = wavs[mask]
    print(f"Clipped spectral range: {wavs[0]:.1f} nm → {wavs[-1]:.1f} nm ({len(wavs)} bands)")

    Zc = len(Zs)
    cube_nm_raw = np.zeros((Zc, Xc, H, len(wavs)), dtype=np.float32)    #Empty cube [Z, X, Y, W]

    #Fill the cube
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
            
            #FLIP THE IMAGE???? TODO
            img_f = img.astype(np.float32)
            if FLIP_X:
                img_f = np.fliplr(img_f)
                cube_nm_raw[zi, xi, :, :] = img_f[:, keep_idx]
            else: 
                cube_nm_raw[zi, xi, :, :] = img_f[:, keep_idx]

    print("Built raw cube with shape (Z, X, Y, wavelength):", cube_nm_raw.shape)
    print("Cube wavelengths:", wavs[0], wavs[-1])

    return cube_nm_raw, wavs

def generate_cube(scan_folder):
    """
    Full cube generation pipeline:
    1. sort scan images
    2. build raw cube
    3. apply snake-like X shifts
    4. crop to valid overlap
    5. wrap in CubeNM
    6. save corrected cube

    Returns:
        cube: CubeNM object
        npz_path: path to saved corrected cube
    """
    rows, Zs = sort_images(scan_folder)

    # ---- Build raw cube ----
    cube_nm_raw, wavs = build_raw_cube(rows, Zs)

    # ---- Correct snake-like X-offsets ----
    alt_shifts = build_alternating_shifts(Zc=len(Zs), plus_first=True)
    shifted_cube = apply_integer_x_shifts(cube_nm_raw, alt_shifts, pad_value=np.nan)
    cube_nm_corr, xslice = crop_valid_overlap(shifted_cube)

    print("Sym-shifted & cropped cube shape:", cube_nm_corr.shape, "X slice:", xslice)

    # ---- Wrap corrected cube ----
    cube = CubeNM(cube_nm_corr, wavs)
    print("Final cube shape (Z, X, Y, nm):", cube.shape)

    # ---- Save corrected cube ----
    npz_path = os.path.join(scan_folder, "cube_ZXnm_corrected.npz")
    np.savez_compressed(
        npz_path,
        cube=cube.data,
        wavs_nm=cube.wavs_nm,
        Zs=np.array(Zs, dtype=np.int32),
        xslice_start=-1 if xslice.start is None else xslice.start,
        xslice_stop=-1 if xslice.stop is None else xslice.stop,
    )
    print("Saved corrected cube to:", npz_path)

    return cube, npz_path

if __name__ == "__main__":
    #scan_folder = find_last_modified_folder()
    #Choosing one specific folder: 
    scan_folder = os.path.join(BASE_DIR, "edge", "data", "scan_26March_13:22:22")
    
    print("Using this scan folder:", scan_folder)

    cube, npz_path = generate_cube(scan_folder)
    print("Cube generation complete. Cube shape:", cube.shape)
    