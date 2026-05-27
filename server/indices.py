import numpy as np

EPS = 1e-6
DEFAULT_BANDPASS_NM = 5.0
GAUSSIAN_CUTOFF_SIGMA = 3.0

# Quality thresholds applied before index summaries. These are deliberately
# global so every plant and scan is filtered with the same rule.
MIN_STABLE_REFLECTANCE = 0.02
MAX_STABLE_REFLECTANCE = 1.2
MIN_STABLE_DENOMINATOR = 0.02


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
    min_nm = float(wavs_nm[0])
    max_nm = float(wavs_nm[-1])
    if float(target_nm) < min_nm or float(target_nm) > max_nm:
        raise ValueError(
            f"Target wavelength {float(target_nm):.1f} nm is outside cube range "
            f"{min_nm:.1f}-{max_nm:.1f} nm"
        )

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


def _valid_reflectance_bands(
    *bands,
    min_reflectance=MIN_STABLE_REFLECTANCE,
    max_reflectance=MAX_STABLE_REFLECTANCE,
) -> np.ndarray:
    """
    Return pixels with finite, non-clipped reflectance in every required band.

    Very low reflectance makes reciprocal and ratio indices unstable, while
    values at the radiometric clipping ceiling are not reliable quantitative
    reflectance measurements.
    """
    if not bands:
        raise ValueError("At least one reflectance band is required")

    valid = np.ones_like(bands[0], dtype=bool)
    for band in bands:
        band = np.asarray(band, dtype=np.float32)
        valid &= np.isfinite(band)
        valid &= band > float(min_reflectance)
        valid &= band <= float(max_reflectance)
    return valid


def _valid_denominator(denominator, min_abs_value=MIN_STABLE_DENOMINATOR) -> np.ndarray:
    denominator = np.asarray(denominator, dtype=np.float32)
    return np.isfinite(denominator) & (np.abs(denominator) >= float(min_abs_value))


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


def calculate_ndvi(
    cube,
    bandpass_nm=DEFAULT_BANDPASS_NM,
    min_reflectance=0.0,
    max_reflectance=MAX_STABLE_REFLECTANCE,
    min_denominator=MIN_STABLE_DENOMINATOR,
) -> np.ndarray:
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

    denominator = nir + red
    valid = _valid_reflectance_bands(
        red,
        nir,
        min_reflectance=min_reflectance,
        max_reflectance=max_reflectance,
    )
    valid &= _valid_denominator(denominator, min_abs_value=min_denominator)

    ndvi = np.full_like(red, np.nan, dtype=np.float32)
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]

    # Whole-cube mean print disabled; thesis-facing summaries are handled in visualization.py.
    return ndvi


def calculate_cri1(
    cube,
    bandpass_nm=DEFAULT_BANDPASS_NM,
    min_reflectance=MIN_STABLE_REFLECTANCE,
    max_reflectance=MAX_STABLE_REFLECTANCE,
) -> np.ndarray:
    """
    Calculate CRI1 from a hyperspectral cube using bandpass-aware spectral averaging.

    Parameters:
        cube: Hyperspectral cube-like object with .data and .wavs_nm,
              with effective shape (Z, X, Y, W).
        bandpass_nm: effective optical bandpass in nm used for spectral averaging.

    Returns:
        CRI1 array with shape (Z, X, Y).
    """
    p510 = _gaussian_weighted_band(cube, 510, bandpass_nm=bandpass_nm)
    p550 = _gaussian_weighted_band(cube, 550, bandpass_nm=bandpass_nm)

    cri1 = np.full_like(p510, np.nan, dtype=np.float32)
    mask = _valid_reflectance_bands(
        p510,
        p550,
        min_reflectance=min_reflectance,
        max_reflectance=max_reflectance,
    )

    cri1[mask] = (1.0 / p510[mask]) - (1.0 / p550[mask])
    # Whole-cube mean print disabled; thesis-facing summaries are handled in visualization.py.
    return cri1


def calculate_cri2(
    cube,
    bandpass_nm=DEFAULT_BANDPASS_NM,
    min_reflectance=MIN_STABLE_REFLECTANCE,
    max_reflectance=MAX_STABLE_REFLECTANCE,
) -> np.ndarray:
    """
    Calculate CRI2 from a hyperspectral cube using bandpass-aware spectral averaging.

    CRI2 = 1/R510 - 1/R700
    """
    p510 = _gaussian_weighted_band(cube, 510, bandpass_nm=bandpass_nm)
    p700 = _gaussian_weighted_band(cube, 700, bandpass_nm=bandpass_nm)

    cri2 = np.full_like(p510, np.nan, dtype=np.float32)
    mask = _valid_reflectance_bands(
        p510,
        p700,
        min_reflectance=min_reflectance,
        max_reflectance=max_reflectance,
    )

    cri2[mask] = (1.0 / p510[mask]) - (1.0 / p700[mask])
    return cri2


def calculate_cri(
    cube,
    bandpass_nm=DEFAULT_BANDPASS_NM,
    min_reflectance=MIN_STABLE_REFLECTANCE,
    max_reflectance=MAX_STABLE_REFLECTANCE,
) -> np.ndarray:
    """
    Backward-compatible alias for CRI1.
    """
    return calculate_cri1(
        cube,
        bandpass_nm=bandpass_nm,
        min_reflectance=min_reflectance,
        max_reflectance=max_reflectance,
    )


def calculate_pri(
    cube,
    bandpass_nm=DEFAULT_BANDPASS_NM,
    min_reflectance=0.0,
    max_reflectance=MAX_STABLE_REFLECTANCE,
    min_denominator=MIN_STABLE_DENOMINATOR,
) -> np.ndarray:
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

    denominator = p531 + p570
    valid = _valid_reflectance_bands(
        p531,
        p570,
        min_reflectance=min_reflectance,
        max_reflectance=max_reflectance,
    )
    valid &= _valid_denominator(denominator, min_abs_value=min_denominator)

    pri = np.full_like(p531, np.nan, dtype=np.float32)
    pri[valid] = (p531[valid] - p570[valid]) / denominator[valid]

    # Whole-cube mean print disabled; thesis-facing summaries are handled in visualization.py.
    return pri


def calculate_sipi(
    cube,
    bandpass_nm=DEFAULT_BANDPASS_NM,
    min_reflectance=0.0,
    max_reflectance=MAX_STABLE_REFLECTANCE,
    min_denominator=MIN_STABLE_DENOMINATOR,
) -> np.ndarray:
    """
    Calculate SIPI from a hyperspectral cube using bandpass-aware spectral averaging.

    SIPI = (R800 - R445) / (R800 - R680)
    """
    r445 = _gaussian_weighted_band(cube, 445, bandpass_nm=bandpass_nm)
    r680 = _gaussian_weighted_band(cube, 680, bandpass_nm=bandpass_nm)
    r800 = _gaussian_weighted_band(cube, 800, bandpass_nm=bandpass_nm)

    denominator = r800 - r680
    valid = _valid_reflectance_bands(
        r445,
        r680,
        r800,
        min_reflectance=min_reflectance,
        max_reflectance=max_reflectance,
    )
    valid &= _valid_denominator(denominator, min_abs_value=min_denominator)

    sipi = np.full_like(r800, np.nan, dtype=np.float32)
    sipi[valid] = (r800[valid] - r445[valid]) / denominator[valid]
    return sipi


def calculate_psri(
    cube,
    bandpass_nm=DEFAULT_BANDPASS_NM,
    min_reflectance=0.0,
    max_reflectance=MAX_STABLE_REFLECTANCE,
    min_denominator=MIN_STABLE_DENOMINATOR,
) -> np.ndarray:
    """
    Calculate PSRI from a hyperspectral cube using bandpass-aware spectral averaging.

    PSRI = (R678 - R500) / R750
    """
    r500 = _gaussian_weighted_band(cube, 500, bandpass_nm=bandpass_nm)
    r678 = _gaussian_weighted_band(cube, 678, bandpass_nm=bandpass_nm)
    r750 = _gaussian_weighted_band(cube, 750, bandpass_nm=bandpass_nm)

    valid = _valid_reflectance_bands(
        r500,
        r678,
        r750,
        min_reflectance=min_reflectance,
        max_reflectance=max_reflectance,
    )
    valid &= _valid_denominator(r750, min_abs_value=min_denominator)

    psri = np.full_like(r750, np.nan, dtype=np.float32)
    psri[valid] = (r678[valid] - r500[valid]) / r750[valid]
    return psri
