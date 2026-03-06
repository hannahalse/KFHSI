import numpy as np 

#TODO: Currently OK, but calculates the NDVI for the whole 2D image. For later: see if I can implement PCA to remove background/soil. 
#TODO: Add function for black and white references. Maybe not in this file but in general. 
#TODO: Find which CRI variant I use (CRI1, CRI2) 
#TODO: make to reflectance not raw intensity. 

def calculate_ndvi(cube: CubeNM) -> np.ndarray:
    """
    Calculate NDVI from a hyperspectral cube.

    Args:
        cube: Hyperspectral cube with shape (Z, X, Y, W). 
    Returns:
        NDVI array with shape (Z, X, Y)
    """
    red = cube[:, :, :, 660].astype(np.float32) #red band at 660 nm
    nir = cube[:, :, :, 800].astype(np.float32) #NIR band at 800 nm
    
    ndvi = (nir - red) / (nir + red + 1e-6)  # Avoid division by zero
    print(f"NDVI mean = {np.nanmean(ndvi):.4f}") # Global mean PRI value across the cube
    return ndvi


def calculate_cri(cube: CubeNM) -> np.ndarray:
    """
    Calculate CRI from a hyperspectral cube.

    Args:
        cube: Hyperspectral cube with shape (Z, X, Y, W). 

    """

    p510 = cube[:, :, :, 510].astype(np.float32) #band at 510 nm
    p550 = cube[:, :, :, 550].astype(np.float32) #band at 550 nm

    cri = np.full_like(p510, np.nan, dtype=np.float32)

    eps = 1e-6
    mask = (p510 > eps) & (p550 > eps)

    cri[mask] = (1.0 / p510[mask]) - (1.0 / p550[mask])
    print(f"CRI mean = {np.nanmean(cri):.4f}") # Global mean PRI value across the cube

    return cri


def calculate_pri(cube: CubeNM) -> np.ndarray:
    """
    Calculate PRI from a hyperspectral cube.

    Args:
        cube: Hyperspectral cube with shape (Z, X, Y, W).

    """

    p531 = cube[:, :, :, 531].astype(np.float32) #band at 531 nm
    p570 = cube[:, :, :, 570].astype(np.float32) #band at 570 nm

    pri = (p531 - p570) / (p531 + p570 + 1e-6) # Avoid division by zero
    print (f"PRI mean = {np.nanmean(pri):.4f}") # Global mean PRI value across the cube
    return pri

