# KFSpectra

This repository contains the Python code used to process the hyperspectral scan data in the thesis. The analysis workflow assembles scan images into hyperspectral cubes, applies spatial and radiometric corrections, calculates vegetation indices, creates plant masks, writes per-scan summary CSV files, and generates the time-series figures.

Raw experiment data and generated cubes are expected under `edge/data/`. These files are large and are normally kept outside the code submission. The summary CSV files can be included when the numerical results should be available without rerunning the full cube processing.

## Repository Structure

```text
calibration/      Calibration utilities, reference spectra, and SpectraPen plotting.
edge/             Camera/printer control code and local experiment data folder.
server/           Cube generation, radiometric correction, index calculation, and plotting.
utils/            General configuration and helper utilities.
```

The main analysis scripts are in `server/`.

## Environment Setup

The project uses Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If a prepared `.venv` already exists, use it directly:

```bash
source .venv/bin/activate
```

Most commands below are written using `../.venv/bin/python` from inside the `server/` or `calibration/` folders. If your virtual environment is active, `python` can be used instead.

## Expected Data Layout

The processing scripts expect data under:

```text
edge/data/
```

For Experiments 1 and 2, scan folders are expected directly below the experiment folder:

```text
edge/data/Experiment1/scan_27April_15:26:44/
edge/data/Experiment2/scan_04May_11:01:08/
```

For Experiment 3, scan folders are grouped by day:

```text
edge/data/Experiment3/day1/scan_22May_10:46:38/
edge/data/Experiment3/day2/scan_23May_08:48:39/
```

Each scan folder should contain raw scan images named like:

```text
X<number>_Z<number>.png
```

Generated cubes are saved in each scan folder as:

```text
cube_ZXnm_corrected.npz
```

In this filename, `corrected` refers to the spatial X-shift correction and cropping. Radiometric calibration is applied later during index calculation.

## Processing Pipeline

1. Raw scan images are sorted by X and Z scan coordinates.
2. The spectral pixel axis is converted to wavelength using the wavelength calibration.
3. Images are assembled into a four-dimensional cube with shape `(Z, X, Y, wavelength)`.
4. Alternating X-shifts are applied to compensate for the zigzag scanning offset.
5. The cube is cropped to the valid overlapping region and saved.
6. During visualization/index processing, the cube is radiometrically calibrated using white and dark reference images.
7. Required wavelength bands are extracted with Gaussian spectral weighting.
8. NDVI, PRI, CRI1, CRI2, SIPI, and PSRI are calculated voxel-wise.
9. The Y dimension is reduced to a 2D map. The thesis analysis uses the central-band mean with the middle row plus 20 rows above and below it.
10. A plant mask is created from NDVI and applied to all index maps.
11. Summary statistics are written to CSV: valid plant pixels, mean, median, and standard deviation.

The main processing settings are defined in `server/visualization.py`, including:

```text
Y_REDUCTION_MODE
Y_BAND_HALF_HEIGHT
NDVI_MASK_THRESHOLD
white_path
dark_path
```

## Scan Group Configuration

Experiment scan groups are defined in:

```text
server/experiment_scan_groups.py
```

This file controls which scans belong to each position or plant when running batch processing.

Experiments 1 and 2 use `POSITION_SCANS_BY_EXPERIMENT`.

Experiment 3 uses `PLANT_SCANS_BY_EXPERIMENT`.

## Generate Cubes

From the `server/` folder:

```bash
cd /path/to/KFHSI/server
```

Generate missing cubes for Experiment 1:

```bash
../.venv/bin/python batch_generate_cubes_experiment2.py --experiment Experiment1 --skip-existing
```

Generate missing cubes for Experiment 2:

```bash
../.venv/bin/python batch_generate_cubes_experiment2.py --experiment Experiment2 --skip-existing
```

For Experiment 3, the scan folders are nested by day. Run the generator per day using `--discover`, for example:

```bash
../.venv/bin/python batch_generate_cubes_experiment2.py --experiment Experiment3 --data-dir ../edge/data/Experiment3/day1 --discover --skip-existing
```

Repeat for the remaining `day*` folders if cubes need to be regenerated.

Use `--dry-run` first to inspect what would be generated without writing files:

```bash
../.venv/bin/python batch_generate_cubes_experiment2.py --experiment Experiment2 --dry-run
```

## Recalculate Index Summary CSV Files

The batch visualization script loads existing cubes, applies radiometric calibration, calculates indices, applies the plant mask, and writes summary CSV files.

Experiment 1:

```bash
../.venv/bin/python batch_visualization_experiment2.py --experiment Experiment1 --fresh
```

Experiment 2:

```bash
../.venv/bin/python batch_visualization_experiment2.py --experiment Experiment2 --fresh
```

Experiment 3 using the manually configured plant lists:

```bash
../.venv/bin/python batch_visualization_experiment2.py --experiment Experiment3 --plant-list --fresh
```

The output CSV files are saved in the corresponding experiment folder, for example:

```text
edge/data/Experiment3/masked_index_time_seriesExp3Plant1.csv
```

## Generate Individual Time-Series Plots

The time-series plotting script uses the already existing CSV files. It does not recalculate indices.

Experiment 3, Plants 1-5:

```bash
cd /path/to/KFHSI/server
../.venv/bin/python plot_index_time_series.py --csv-dir ../edge/data/Experiment3 --output-dir ../edge/data/Experiment3/time_series_plots --scan-root ../edge/data/Experiment3 --csv-pattern 'masked_index_time_seriesExp3Plant[1-5].csv'
```

This creates one output folder per plant:

```text
edge/data/Experiment3/time_series_plots/masked_index_time_seriesExp3Plant1/
```

Each folder contains individual index plots and a stacked overview plot:

```text
ndvi_time_series.png
pri_time_series.png
cri1_time_series.png
cri2_time_series.png
sipi_time_series.png
psri_time_series.png
index_time_series_overview.png
mask_time_series.png
```

## Generate All-Plant Comparison Plots

To compare all Experiment 3 plants in the same figure:

```bash
cd /path/to/KFHSI/server
../.venv/bin/python plot_experiment3_all_plant_means.py
```

Outputs are written to:

```text
edge/data/Experiment3/time_series_plots/all_plant_mean_comparison/
```

The stacked overview is:

```text
index_mean_all_plants_overview.png
```

## Generate Experiment 1 and 2 Combined Plots

To plot Experiment 1 and Experiment 2 together:

```bash
cd /path/to/KFHSI/server
../.venv/bin/python plot_experiment1_2_combined.py
```

Outputs are written to:

```text
edge/data/Experiment1_Experiment2/time_series_plots/combined_position_mean_comparison/
```

## Radiometric Denominator Plot

To plot the radiometric denominator, `white reference - dark reference`:

```bash
cd /path/to/KFHSI/server
../.venv/bin/python plot_radiometric_denominator.py
```

To save the figure:

```bash
../.venv/bin/python plot_radiometric_denominator.py --save-path ../calibration/radiometric_denominator.png
```

## SpectraPen Bulb Spectra

To plot the raw SpectraPen spectra for the 10 W and 20 W bulb measurements:

```bash
cd /path/to/KFHSI/calibration
../.venv/bin/python spectraPen.py
```

The plot is also saved as:

```text
calibration/spectrapen_10W_20W_raw.png
```

## Validate a Generated Cube

To inspect and validate a cube:

```bash
cd /path/to/KFHSI/server
../.venv/bin/python validate_cube.py --scan-folder ../edge/data/Experiment3/day16/scan_06June_16:24:55
```

Validation outputs are saved inside the selected scan folder.

## Notes For Code Submission

- Do not include the full `edge/data/` folder in a normal code-only submission unless the raw data are explicitly requested.
- If needed, include only the summary CSV files from `edge/data/Experiment*/`.
- Do not include `__pycache__/`, `.pyc`, `.DS_Store`, or temporary debug outputs.
- The plotting scripts regenerate figures from CSV files. They do not recalculate vegetation indices.
- To fully reproduce numerical results, regenerate cubes first if needed, then regenerate CSV summaries, then regenerate plots.
- The current analysis uses radiometric calibration during `visualization.py`, not during cube generation.
