import numpy as np
import matplotlib.pyplot as plt

data = np.load("/Users/hannahalse/Desktop/KFHSi/edge/data/scan_16November_13:58:13/cube_ZXnm.npz")

cube = data["cube"]
wavs = data["wavs_nm"]

print("Cube shape:", cube.shape)
print("Wavelength range:", wavs[0], wavs[-1])

red_nm = 660
nir_nm = 800

red_idx = np.argmin(np.abs(wavs - red_nm))
nir_idx = np.argmin(np.abs(wavs - nir_nm))

print("Red band:", wavs[red_idx])
print("NIR band:", wavs[nir_idx])

z = cube.shape[0] // 2

red = cube[z, :, :, red_idx]
nir = cube[z, :, :, nir_idx]

ndvi = (nir - red) / (nir + red + 1e-6)

plt.figure(figsize=(6,5))
plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar(label="NDVI")
plt.title("NDVI map ")
plt.show()