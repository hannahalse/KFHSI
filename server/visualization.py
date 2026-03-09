import os, re
import numpy as np
import cv2
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
    npz_path = os.path.join(scan_folder, "cube_ZXnm.npz")
    print(f"Loading cube from: {npz_path}")

    data = np.load(npz_path)
    cube = CubeNM(data["cube"], data["wavs_nm"])

    reconstruct_rgb_image(
        cube,
        out_path=os.path.join(current_dir, "rgb_image.png")
    )
    #RESULT IS NOT SHIFTED
    