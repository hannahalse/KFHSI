import numpy as np

EPS = 1e-6
DEFAULT_BANDPASS_NM = 5.0
GAUSSIAN_CUTOFF_SIGMA = 3.0


def _require_spectral_cube(cube):
    if not hasattr(cube, "data") or not hasattr(cube, "wavs_nm"):
        raise TypeError("cube must provide .data and .wavs_nm for spectral averaging")


def _gaussian_weighted_band(cube, target_nm, bandpass_nm=DEFAULT_BANDPASS_NM) -> np.ndarray:
    """
    Estimate one wavelength band using a Gaussian spectral response.

    The optical setup has roughly 5 nm spectral bandpass, while the saved cube
    is sampled much more densely. This helper keeps the full sampled cube, but
    estimates reflectance at one target wavelength from a local weighted average
    over neighboring spectral samples.
    """
    _require_spectral_cube(cube)

    if bandpass_nm <= 0:
        raise ValueError("bandpass_nm must be > 0")

    wavs_nm = np.asarray(cube.wavs_nm, dtype=np.float32)
    cube_data = np.asarray(cube.data, dtype=np.float32)

    sigma_nm = float(bandpass_nm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    distances_nm = wavs_nm - float(target_nm)
    mask = np.abs(distances_nm) <= (GAUSSIAN_CUTOFF_SIGMA * sigma_nm)

    if not np.any(mask):
        idx = int(np.argmin(np.abs(distances_nm)))
        return cube_data[:, :, :, idx].astype(np.float32)

    local_distances = distances_nm[mask]
    weights = np.exp(-0.5 * (local_distances / sigma_nm) ** 2).astype(np.float32)
    weights /= np.sum(weights)

    weighted_band = np.tensordot(cube_data[:, :, :, mask], weights, axes=([-1], [0]))
    return weighted_band.astype(np.float32)


def gaussian_smooth_spectrum(spectrum, wavs_nm, bandpass_nm=DEFAULT_BANDPASS_NM) -> np.ndarray:
    """
    Smooth a single spectrum with the same Gaussian bandpass model used for index extraction.

    This is useful for visualizing what the 5 nm optical response does to one
    pixel spectrum without mixing any spatial neighbors.
    """
    if bandpass_nm <= 0:
        raise ValueError("bandpass_nm must be > 0")

    spectrum = np.asarray(spectrum, dtype=np.float32)
    wavs_nm = np.asarray(wavs_nm, dtype=np.float32)

    if spectrum.ndim != 1 or wavs_nm.ndim != 1:
        raise ValueError("spectrum and wavs_nm must both be 1D arrays")
    if len(spectrum) != len(wavs_nm):
        raise ValueError("spectrum and wavs_nm must have the same length")

    sigma_nm = float(bandpass_nm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    cutoff_nm = GAUSSIAN_CUTOFF_SIGMA * sigma_nm
    smoothed = np.empty_like(spectrum, dtype=np.float32)

    for idx, center_nm in enumerate(wavs_nm):
        distances_nm = wavs_nm - center_nm
        mask = np.abs(distances_nm) <= cutoff_nm

        local_distances = distances_nm[mask]
        local_weights = np.exp(-0.5 * (local_distances / sigma_nm) ** 2).astype(np.float32)
        local_weights /= np.sum(local_weights)

        smoothed[idx] = np.dot(spectrum[mask], local_weights)

    return smoothed


def calculate_ndvi(cube, bandpass_nm=DEFAULT_BANDPASS_NM) -> np.ndarray:
    """
    Calculate NDVI from a hyperspectral cube using bandpass-aware spectral averaging.

    Parameters:
        cube: Hyperspectral cube-like object with .data and .wavs_nm,
              with effective shape (Z, X, Y, W).
        bandpass_nm: effective optical bandpass in nm used for spectral averaging.

    Returns:
        NDVI array with shape (Z, X, Y).
    """
    red = _gaussian_weighted_band(cube, 660, bandpass_nm=bandpass_nm)
    nir = _gaussian_weighted_band(cube, 800, bandpass_nm=bandpass_nm)

    ndvi = (nir - red) / (nir + red + EPS)
    ndvi = ndvi.astype(np.float32)

    print(f"NDVI mean = {np.nanmean(ndvi):.4f}")
    return ndvi


def calculate_cri(cube, bandpass_nm=DEFAULT_BANDPASS_NM) -> np.ndarray:
    """
    Calculate CRI from a hyperspectral cube using bandpass-aware spectral averaging.

    Parameters:
        cube: Hyperspectral cube-like object with .data and .wavs_nm,
              with effective shape (Z, X, Y, W).
        bandpass_nm: effective optical bandpass in nm used for spectral averaging.

    Returns:
        CRI array with shape (Z, X, Y).
    """
    p510 = _gaussian_weighted_band(cube, 510, bandpass_nm=bandpass_nm)
    p550 = _gaussian_weighted_band(cube, 550, bandpass_nm=bandpass_nm)

    cri = np.full_like(p510, np.nan, dtype=np.float32)
    mask = (p510 > EPS) & (p550 > EPS)

    cri[mask] = (1.0 / p510[mask]) - (1.0 / p550[mask])
    print(f"CRI mean = {np.nanmean(cri):.4f}") # Global mean CRI value across the cube
    return cri


def calculate_pri(cube, bandpass_nm=DEFAULT_BANDPASS_NM) -> np.ndarray:
    """
    Calculate PRI from a hyperspectral cube using bandpass-aware spectral averaging.

    Parameters:
        cube: Hyperspectral cube-like object with .data and .wavs_nm,
              with effective shape (Z, X, Y, W).
        bandpass_nm: effective optical bandpass in nm used for spectral averaging.

    Returns:
        PRI array with shape (Z, X, Y).
    """
    p531 = _gaussian_weighted_band(cube, 531, bandpass_nm=bandpass_nm)
    p570 = _gaussian_weighted_band(cube, 570, bandpass_nm=bandpass_nm)

    pri = (p531 - p570) / (p531 + p570 + EPS)
    pri = pri.astype(np.float32)

    print(f"PRI mean = {np.nanmean(pri):.4f}") # Global mean PRI value across the cube
    return pri
