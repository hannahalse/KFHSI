import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from indices import DEFAULT_BANDPASS_NM, gaussian_smooth_spectrum


A = 0.7241145833
B = 288.45625


def px_to_nm(px):
    return A * px + B


def wavelength_axis(width):
    pixels = np.arange(width)
    return px_to_nm(pixels)

def visualise_spectrum_at(cube, z, x, y, bandpass_nm=DEFAULT_BANDPASS_NM):
    """
    Visualize the Gaussian-smoothed spectrum from a specific position and row.
    
    Parameters:
        cube: CubeNM object with shape (Z, X, Y, W)
        z: Z position
        x: X position  
        y: Y position (row number in the image)
        bandpass_nm: Gaussian smoothing bandpass in nm
    """
    raw_spectrum = np.asarray(cube[z, x, y, :], dtype=np.float32)
    smoothed_spectrum = gaussian_smooth_spectrum(
        raw_spectrum,
        cube.wavs_nm,
        bandpass_nm=bandpass_nm,
    )

    print(f"Cube shape: {cube.shape}")
    print(f"Spectrum at Z={z}, X={x}, Y={y} (Gaussian-smoothed, {bandpass_nm:.1f} nm FWHM):")
    
    plt.figure(figsize=(10, 6))
    plt.plot(
        cube.wavs_nm,
        smoothed_spectrum,
        label=f"Z={z}, X={x}, Y-row={y}",
        color="tab:blue",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"Smoothed spectrum at position (Z={z}, X={x}, Y={y})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def extract_spectrum_from_raw_image(image_path, flip_x=True, wl_min=380.0, wl_max=820.0):
    """
    Read a raw grayscale image and extract its mean spectrum across image height.
    The returned spectrum is clipped to the requested wavelength range.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read raw image: {image_path}")

    image = image.astype(np.float32)
    if flip_x:
        image = np.fliplr(image)

    spectrum = image.mean(axis=0)
    wavs_nm = wavelength_axis(image.shape[1]).astype(np.float32)

    mask = (wavs_nm >= wl_min) & (wavs_nm <= wl_max)
    if not np.any(mask):
        raise RuntimeError(
            f"No wavelengths remain after clipping to {wl_min:.1f}-{wl_max:.1f} nm"
        )

    return wavs_nm[mask], spectrum[mask]


def visualise_raw_image_spectrum(image_path, flip_x=True, wl_min=380.0, wl_max=820.0):
    """
    Extract and plot the mean spectrum from any raw grayscale image.
    """
    wavs_nm, spectrum = extract_spectrum_from_raw_image(
        image_path,
        flip_x=flip_x,
        wl_min=wl_min,
        wl_max=wl_max,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(wavs_nm, spectrum, color="tab:blue")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(
        f"Raw image spectrum: {os.path.basename(image_path)} ({wl_min:.0f}-{wl_max:.0f} nm)"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return wavs_nm, spectrum


def visualise_white_dark_difference(
    white_path,
    dark_path,
    flip_x=True,
    wl_min=380.0,
    wl_max=820.0,
    plot=True,
):
    """
    Plot the white-dark reference difference used as the radiometric denominator.
    """
    wavs_nm, white_spectrum = extract_spectrum_from_raw_image(
        white_path,
        flip_x=flip_x,
        wl_min=wl_min,
        wl_max=wl_max,
    )
    wavs_nm_dark, dark_spectrum = extract_spectrum_from_raw_image(
        dark_path,
        flip_x=flip_x,
        wl_min=wl_min,
        wl_max=wl_max,
    )

    if len(wavs_nm) != len(wavs_nm_dark) or not np.allclose(wavs_nm, wavs_nm_dark):
        raise RuntimeError("White and dark spectra do not share the same wavelength axis")

    denominator = white_spectrum - dark_spectrum

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(wavs_nm, denominator, color="tab:green")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("White - dark (a.u.)")
        plt.title(
            f"Reference denominator: {os.path.basename(white_path)} - {os.path.basename(dark_path)}"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return wavs_nm, denominator, white_spectrum, dark_spectrum


def visualise_spectrum_before_after_gaussian(cube, z, x, y, bandpass_nm=DEFAULT_BANDPASS_NM, out_path=None,):
    """
    Plot one pixel spectrum before and after Gaussian spectral smoothing.

    The smoothing is applied only along the wavelength axis, so no spatial
    neighbors are mixed into the spectrum.
    """
    raw_spectrum = np.asarray(cube[z, x, y, :], dtype=np.float32)
    smoothed_spectrum = gaussian_smooth_spectrum(raw_spectrum, cube.wavs_nm, bandpass_nm=bandpass_nm,)

    plt.figure(figsize=(10, 6))
    plt.plot(cube.wavs_nm, raw_spectrum, label="Original spectrum", color="tab:blue", alpha=0.6,)
    plt.plot(cube.wavs_nm, smoothed_spectrum, label=f"Gaussian-smoothed ({bandpass_nm:.1f} nm FWHM)", color="tab:orange", linewidth=2.0,)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"Spectrum at position (Z={z}, X={x}, Y={y})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Spectrum comparison saved to {out_path}")
    else:
        plt.show()


def visualise_wavelength_slice(cube, wavelength_nm, out_path, y=None, aggregate="mean"):
    """
    Visualize a single wavelength slice from the cube and save as PNG.

    Parameters:
        cube: CubeNM object with shape (Z, X, Y, W)
        wavelength_nm: wavelength in nm to visualize
        out_path: output PNG path
        y: if not None, use one specific Y-row -> image shape (Z, X)
        aggregate: if y is None, reduce over Y using this method ("mean" supported)
    """
    # Extract 3D slice at requested wavelength
    slice_3d = cube[:, :, :, wavelength_nm]   # shape (Z, X, Y)

    # Choose how to reduce Y
    if y is not None:
        img = slice_3d[:, :, y]   # one specific row
    else:
        if aggregate == "mean":
            img = slice_3d.mean(axis=2)
        else:
            raise ValueError(f"Unsupported aggregate: {aggregate}")

    # Normalize for visualization
    lo, hi = np.percentile(img, (1, 99))
    img_n = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

    # Plot and save
    plt.figure(figsize=(8, 6))
    plt.imshow(img_n, cmap="gray", aspect="auto")
    plt.xlabel("X position")
    plt.ylabel("Z position")
    plt.title(f"Slice at {float(wavelength_nm):.1f} nm")
    plt.colorbar(label="Normalized intensity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Wavelength ~{float(wavelength_nm):.1f} nm saved to {out_path}")
    
def reconstruct_rgb_image(cube, out_path, y=None, y_range=None, aggregate="mean"):
    """
    Reconstruct and save an RGB image from the cube using specific wavelengths for R, G, B.

    Parameters:
        cube: CubeNM object with shape (Z, X, Y, W)
        out_path: full path to where the RGB PNG should be saved
        y: if not None -> use this Y-row (int) to build the image (Z, X, 3)
        y_range: optional tuple (y0, y1) for aggregating only a Y subrange
        aggregate: how to reduce over Y if y is None ("mean" supported)

    Returns:
        rgb_image_uint8: RGB image as uint8 array (Z, X, 3)
        out_path: the path where the image was saved
    """

    # Target wavelengths for R, G, B channels
    r_wavelength = 660  # Red
    g_wavelength = 550  # Green
    b_wavelength = 460  # Blue
    
    # Find indices for these wavelengths
    r_idx = int(np.searchsorted(cube.wavs_nm, r_wavelength))
    g_idx = int(np.searchsorted(cube.wavs_nm, g_wavelength))
    b_idx = int(np.searchsorted(cube.wavs_nm, b_wavelength))

    #print(f"RGB wavelengths: R={cube.wavs_nm[r_idx]:.1f} nm, " f"G={cube.wavs_nm[g_idx]:.1f} nm, B={cube.wavs_nm[b_idx]:.1f} nm")

    # Extract channels from raw data: (Z, X, Y)
    r_channel = cube.data[:, :, :, r_idx]
    g_channel = cube.data[:, :, :, g_idx]
    b_channel = cube.data[:, :, :, b_idx]

    # Normalize each channel to 0–1
    def normalize_channel(ch):
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            return (ch - ch_min) / (ch_max - ch_min)
        return ch

    r_norm = normalize_channel(r_channel)
    g_norm = normalize_channel(g_channel)
    b_norm = normalize_channel(b_channel)

    # Decide how to handle Y-dimension
    if y is not None:
        # Use a single Y-row
        r2 = r_norm[:, :, y]
        g2 = g_norm[:, :, y]
        b2 = b_norm[:, :, y]
    else:
        # Aggregate over Y
        if y_range is not None:
            y0, y1 = int(y_range[0]), int(y_range[1])
            r_norm = r_norm[:, :, y0:y1]
            g_norm = g_norm[:, :, y0:y1]
            b_norm = b_norm[:, :, y0:y1]

        if aggregate == "mean":
            r2 = r_norm.mean(axis=2)
            g2 = g_norm.mean(axis=2)
            b2 = b_norm.mean(axis=2)
        else:
            raise ValueError(f"Unsupported aggregate: {aggregate}")

    # Stack into RGB image (Z, X, 3)
    rgb_image = np.stack([r2, g2, b2], axis=-1)
    rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

    # Save to disk
    plt.imsave(out_path, rgb_image_uint8)

    return rgb_image_uint8, out_path
