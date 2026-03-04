import cv2
import numpy as np
import matplotlib.pyplot as plt

from wavelength_calibr import wavelength_axis


# Load image
image = cv2.imread("450nmLC.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise RuntimeError("Could not load image")


# Flip if needed
image = cv2.flip(image, 1)


# Extract middle row
height, width = image.shape
row = height // 2
intensity = image[row, :].astype(float)


# Build wavelength axis
wl = wavelength_axis(width)


# ---- Clip to 350–850 nm ----
mask = (wl >= 350) & (wl <= 850)

wl_clip = wl[mask]
intensity_clip = intensity[mask]


# Optional: print peak
peak_px = np.argmax(intensity_clip)
peak_nm = wl_clip[peak_px]

print("Peak wavelength:", peak_nm)


# Plot
plt.plot(wl_clip, intensity_clip)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("Camera spectrum – 450 nm LED")

plt.grid(True)
plt.xlim(350, 850)

plt.show()