import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scan_folder = find_last_modified_folder()
    npz_path = os.path.join(scan_folder, "cube_ZXnm_corrected.npz")
    print(f"Loading cube from: {npz_path}")

    data = np.load(npz_path)
    cube = CubeNM(data["cube"], data["wavs_nm"])
    
    _, _, H, _ = cube.shape
    y_middle = H // 2

    reconstruct_rgb_image(cube, out_path=os.path.join(current_dir, "rgb_image.png"), y=y_middle)
    visualise_wavelength_slice(cube, wavelength_nm=520, out_path="slice_520nm_middle_row.png", y=y_middle)