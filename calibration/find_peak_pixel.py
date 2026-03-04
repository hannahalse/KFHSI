#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import numpy as np

#To run: .\find_peak_pixel.py 450nmLC.png 660nmLC30int.png 735nmLC.png --flip --band_half_height 0 --smooth 1
#Prints the index of the pixel where the peak is located. This pixel index is later used as anchor in the calibration in TODO. 
#For the left chamber, the following pixel indices has been found: pixel_idx = [216, 504, 600] for the theoretical peaks at 450, 660 and 735 nm (Heliospectra). 
#For the right chamber, the following pixel indices has been found: TODO


#Load image  
def load_grayscale_image(image_path: str) -> np.ndarray:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, but got shape={image.shape}")

    return image


#Band_half_height = 0: only middle row. If >0: average over multiple rows
def extract_spectrum_profile(image: np.ndarray, band_half_height: int = 0) -> np.ndarray:
    height, _ = image.shape
    mid = height // 2

    y0 = max(0, mid - band_half_height)
    y1 = min(height, mid + band_half_height + 1)

    band = image[y0:y1, :].astype(np.float64)
    profile = band.mean(axis=0)

    #Returns the 1D array of the spectrum profile (average intensity across the band for each column)
    return profile

#Moving average smoothing - yields stable peak detection. 
def smooth_profile(profile: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return profile
    if window % 2 == 0:
        raise ValueError("Smoothing window must be an odd number.")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(profile, kernel, mode="same")

#Finds the peak pixel where the profile is largest. Returns the index of the pixel. 
def find_peak_pixel(profile: np.ndarray) -> int:
    return int(np.argmax(profile))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finn peak-pixel i rådatabilder og print resultat (for å lage anchor_px)."
    )
    parser.add_argument("images", nargs="+", help="En eller flere .png-filer.")
    parser.add_argument("--flip", action="store_true", help="Flip horisontalt før analyse.")
    parser.add_argument("--band_half_height", type=int, default=0,
                        help="0 = kun midterste rad. >0 = snitt over flere rader rundt midten.")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Moving average smoothing (oddetall). 1 = ingen smoothing.")
    args = parser.parse_args()

    results = []
    for image_path in args.images:
        image = load_grayscale_image(image_path)
        if args.flip:
            image = cv2.flip(image, 1)

        profile = extract_spectrum_profile(image, band_half_height=args.band_half_height)

        if args.smooth > 1:
            profile = smooth_profile(profile, window=args.smooth)

        peak_px = find_peak_pixel(profile)
        peak_val = float(profile[peak_px])

        base = os.path.basename(image_path)
        print(f"{base}: peak_px={peak_px} (value={peak_val:.1f})")




if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Feil: {error}", file=sys.stderr)
        raise SystemExit(1)