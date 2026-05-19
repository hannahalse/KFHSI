import os
import numpy as np
import cv2

from Generate_cube import CubeNM

EPS = 1e-6
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WHITE_PATH = os.path.join(BASE_DIR, "server", "whiteReferenceInChamber10W.png")
DEFAULT_DARK_PATH = os.path.join(BASE_DIR, "server", "darkReference1.png")

# Samme kalibrering som i resten av prosjektet - Får ikke til å importere. 
A = 0.7241145833
B = 288.45625

def px_to_nm(px):
    """Convert pixel index to wavelength (nm)."""
    return A * px + B

def wavelength_axis(width):
    """Return wavelength for every pixel in the image."""
    pixels = np.arange(width)
    return px_to_nm(pixels)

def load_reference_image(image_path, flip_x=True):
    """
    Load a raw grayscale reference image and optionally flip it horizontally.

    Returns:
        image: float32 array with shape (Y, W)
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read reference image: {image_path}")

    image = image.astype(np.float32)

    if flip_x:
        image = np.fliplr(image)

    return image


def load_mean_reference_image(image_dir, flip_x=True):
    """
    Load all PNG reference images from a directory, optionally flip them,
    and return their pixelwise mean image.

    Returns:
        mean_image: float32 array with shape (Y, W)
    """
    if not os.path.isdir(image_dir):
        raise RuntimeError(f"Reference directory does not exist: {image_dir}")

    image_paths = sorted(
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith(".png")
    )
    if not image_paths:
        raise RuntimeError(f"No PNG reference images found in directory: {image_dir}")

    images = []
    shape_ref = None
    for image_path in image_paths:
        image = load_reference_image(image_path, flip_x=flip_x)
        if shape_ref is None:
            shape_ref = image.shape
        elif image.shape != shape_ref:
            raise RuntimeError(
                f"Reference image shape mismatch in {image_dir}: "
                f"{image_path} has shape {image.shape}, expected {shape_ref}"
            )
        images.append(image)

    mean_image = np.mean(np.stack(images, axis=0), axis=0).astype(np.float32)
    return mean_image


def load_reference_source(reference_source, flip_x=True):
    """
    Load either one reference image or the mean of all PNGs in a directory.
    """
    if os.path.isdir(reference_source):
        return load_mean_reference_image(reference_source, flip_x=flip_x)
    return load_reference_image(reference_source, flip_x=flip_x)

def clip_reference_to_cube_wavelengths(reference_image, cube_wavs_nm):
    """
    Clip reference image in spectral direction so it matches cube wavelengths.

    Parameters:
        reference_image: ndarray with shape (Y, W_raw)
        cube_wavs_nm: ndarray with wavelengths already used in the cube

    Returns:
        clipped_reference: ndarray with shape (Y, W_cube)
    """
    _, width = reference_image.shape
    raw_wavs = wavelength_axis(width)

    mask = (raw_wavs >= cube_wavs_nm[0]) & (raw_wavs <= cube_wavs_nm[-1])
    clipped_reference = reference_image[:, mask]

    if clipped_reference.shape[1] != len(cube_wavs_nm):
        raise RuntimeError(
            f"Reference width after clipping ({clipped_reference.shape[1]}) "
            f"does not match cube wavelength axis ({len(cube_wavs_nm)})."
        )

    return clipped_reference

def radiometric_correct_cube(
    cube,
    white_path=DEFAULT_WHITE_PATH,
    dark_path=DEFAULT_DARK_PATH,
    flip_x=True,
    clip_min=0.0,
    clip_max=2.0,
):
    """
    Apply radiometric correction to a hyperspectral cube using raw white and dark references.

    Formula:
        corrected = (image - dark) / (white - dark + EPS)

    Parameters:
        cube: CubeNM object with shape (Z, X, Y, W)
        white_path: path to raw white reference image
        dark_path: path to raw dark reference image
        flip_x: whether to flip reference images horizontally
        clip_min: minimum output value after correction
        clip_max: maximum output value after correction

    Returns:
        corrected_cube: CubeNM object with corrected data
    """
    if not isinstance(cube, CubeNM):
        raise TypeError("cube must be a CubeNM object")

    cube_data = cube.data.astype(np.float32)
    Zc, Xc, Yc, Wc = cube_data.shape

    white_raw = load_reference_source(white_path, flip_x=flip_x)
    dark_raw = load_reference_source(dark_path, flip_x=flip_x)

    white = clip_reference_to_cube_wavelengths(white_raw, cube.wavs_nm)
    dark = clip_reference_to_cube_wavelengths(dark_raw, cube.wavs_nm)

    if white.shape != (Yc, Wc):
        raise RuntimeError(f"White reference shape {white.shape} does not match cube frame shape {(Yc, Wc)}")
    if dark.shape != (Yc, Wc):
        raise RuntimeError(f"Dark reference shape {dark.shape} does not match cube frame shape {(Yc, Wc)}")

    denominator = white - dark

    if np.any(denominator <= 0):
        print("[WARNING] Some white-dark values are <= 0. These may give unstable correction.")

    corrected_data = (cube_data - dark[None, None, :, :]) / (denominator[None, None, :, :] + EPS)
    corrected_data = np.clip(corrected_data, clip_min, clip_max).astype(np.float32)

    corrected_cube = CubeNM(corrected_data, cube.wavs_nm.copy())

    print(f"[INFO] Using white reference source: {white_path}")
    print(f"[INFO] Using dark reference source: {dark_path}")
    print(f"[INFO] Radiometric correction complete. Corrected cube shape: {corrected_cube.shape}")
    print(f"[INFO] Corrected cube min={np.nanmin(corrected_data):.4f}, max={np.nanmax(corrected_data):.4f}, mean={np.nanmean(corrected_data):.4f}")

    return corrected_cube

def save_corrected_cube(output_path, corrected_cube):
    """
    Save corrected cube to compressed npz file.
    """
    np.savez_compressed(
        output_path,
        cube=corrected_cube.data,
        wavs_nm=corrected_cube.wavs_nm
    )
    print(f"[INFO] Saved corrected cube to: {output_path}")
    
