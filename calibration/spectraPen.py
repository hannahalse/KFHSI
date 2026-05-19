import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys

filename = "SpectraPenData_20Wbulb.spec"  
#filename = sys.argv[1]   # f.eks SpectraPenData_450nmLC.spec

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13

with open(filename, "rb") as f:
    data = f.read()

def extract_all_json_blocks(data):
    json_blocks = []
    i = 0

    while i < len(data):
        if data[i] == ord("{"):
            depth = 1
            start = i
            i += 1

            while i < len(data) and depth > 0:
                if data[i] == ord("{"):
                    depth += 1
                elif data[i] == ord("}"):
                    depth -= 1
                i += 1

            if depth == 0:
                try:
                    block = data[start:i].decode("utf-8")
                    json_blocks.append(json.loads(block))
                except:
                    pass
        else:
            i += 1

    return json_blocks




"""
# --- Parse JSON metadata ---
start = data.find(b"{")
depth = 0
end = None
for i in range(start, len(data)):
    b = data[i]
    if b == ord("{"):
        depth += 1
    elif b == ord("}"):
        depth -= 1
        if depth == 0:
            end = i
            break
meta = json.loads(data[start:end + 1].decode("utf-8"))
print(json.dumps(meta, indent=2))
"""

# --- Find spectrum block ---
marker = b"Measurement1\x00"
m = data.find(marker)
if m == -1:
    raise RuntimeError('Cannot find "Measurement1" in the file.')

header_start = m
header_end = m + 120

print(data[header_start:header_end])

snippet = data[m:m+200]
print(snippet.decode("utf-8", errors="ignore"))


count_offset = m + len(marker) + 1 + 3 + 8
n_points = int.from_bytes(data[count_offset:count_offset + 4], "little")

values_offset = count_offset + 4
values = np.frombuffer(
    data[values_offset:values_offset + 4 * n_points], dtype="<u4"
)

json_blocks = extract_all_json_blocks(data)

for i, block in enumerate(json_blocks):
    print(f"\n--- JSON block {i} ---")
    print(json.dumps(block, indent=2))

meta = json_blocks[0]

# --- Pixel → wavelength ---
sconst = meta["device"]["sconst"]

idx = np.arange(n_points, dtype=float)
wavelength = np.zeros_like(idx)

for p, c in enumerate(sconst):
    wavelength += c * (idx ** p)


# --- Limit range ---
mask = (wavelength >= 400) & (wavelength <= 850)
wavelength = wavelength[mask]
values = values[mask]


"""
Comment out to remove peak detection and printing.
# --- Find peaks ---
# prominence styrer hvor "tydelig" en topp må være. Juster ved behov.
# distance (i antall punkter) hindrer at du får mange peaks tett i tett fra små svingninger.
peaks, props = find_peaks(values, prominence=2000, distance=3)

peak_wl = wavelength[peaks]
peak_val = values[peaks]


# --- Sort peaks ---
order = np.argsort(peak_val)[::-1]
peak_wl = peak_wl[order]
peak_val = peak_val[order]


# --- Print main peak ---
print("\nMain peak:")
print(f"{peak_wl[0]:.2f} nm  ({int(peak_val[0])} counts)")
"""

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(wavelength, values)

#Comment out if when no peaks. 
#plt.plot(peak_wl, peak_val, "x")

plt.xlabel("Wavelength (nm)", fontsize=LABEL_FONTSIZE)
plt.ylabel("Relative intensity (a.u.)", fontsize=LABEL_FONTSIZE)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)

plt.title("20 W halogen light bulbs", fontsize=TITLE_FONTSIZE)
plt.tight_layout()
plt.show()


#print("\nKalibreringskoeffisienter (sconst):")
#for i, c in enumerate(meta["device"]["sconst"]):
#    print(f"c{i} = {c}")
