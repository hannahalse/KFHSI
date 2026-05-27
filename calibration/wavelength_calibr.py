import numpy as np

# Calibration constants from these anchors (Left chamber)
#anchor_px = [216.0, 504.0, 600.0]. From find_peak_pixel.py
#anchor_nm = [445.48, 650.95, 724.77]. From spectraPen.py

#Used in old results: 
A = 0.7241145833
B = 288.45625



def px_to_nm(px):
    "Convert pixel index → wavelength (nm)"
    return A * px + B


def nm_to_px(nm):
    "Convert wavelength (nm) → pixel index"
    return int(round((nm - B) / A))


def wavelength_axis(width):
    "Return wavelength for every pixel in the image"
    pixels = np.arange(width)
    return px_to_nm(pixels)