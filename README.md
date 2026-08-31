# Ghana Cocoa Land-Use Change (2026)

Reproducible geospatial analysis of cocoa-related land-use and land-cover (LULC)
change in Ghana's **Western**, **Western North**, and **Ashanti** regions.

## Project goals

- Prepare administrative boundaries, cocoa layers, imagery, and reference LULC data.
- Produce harmonized LULC maps for the configured analysis years.
- Quantify class transitions, cocoa expansion or contraction, and forest conversion.
- Summarize results by region and export maps, tables, figures, and reports.

No source data are included in this scaffold. Add licensed datasets under `data/raw/`;
raw and derived geospatial files are excluded from Git by default.

## Structure

```text
config/                  Project and class definitions
data/raw/                Original inputs, separated by dataset and region
data/external/           Third-party supporting data
data/interim/            Temporary or cleaned intermediate data
data/processed/          Analysis-ready datasets
docs/                    Methods, data inventory, and metadata guidance
notebooks/               Exploratory analysis
outputs/                 Maps, figures, tables, reports, and logs
scripts/                 Command-line workflow entry points
src/ghana_cocoa_lulc/    Reusable Python package
tests/                   Automated tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/validate_inputs.py
```

Edit `config/project.yaml` to set years, CRS, input filenames, and output options.
See `docs/data_inventory.md` before adding data and `docs/methodology.md` for the
proposed workflow.

## Suggested execution order

1. `python scripts/validate_inputs.py`
2. `python scripts/01_prepare_boundaries.py`
3. Implement imagery preprocessing and classification for the selected platform.
4. `python scripts/02_calculate_change.py`
5. Review outputs and document accuracy/limitations.

## Download Dynamic World data

Authenticate Earth Engine, set `EARTHENGINE_PROJECT` in `.env` or your shell, then run:

```bash
python scripts/dynamicworld/download_dynamicworld_lulc.py \
  --all-configured-years \
  --ee-project YOUR_GOOGLE_CLOUD_PROJECT
```

The downloader reads `assets/aoi_bounding_box.gpkg` and writes tiled annual-mode
GeoTIFFs to `data/raw/dynamicworld/`. Use `--help` for single-year, grid-size,
retry, AOI-layer, and overwrite options.

Interrupted downloads are resumable: run the same command again and existing tiles
will be skipped. Slow Earth Engine computations use a 15-minute request deadline and
eight retries by default; these can be changed with `--ee-deadline-seconds` and
`--retries`.

## Export Sentinel-2 dry-season imagery

Create ten band-specific, tiled GeoTIFF exports of a cloud-masked DJF median
composite in Google Drive:

```bash
python scripts/sentinel2/download_sentinel2_djf.py \
  --year 2025 \
  --ee-project YOUR_GOOGLE_CLOUD_PROJECT
```

The season labeled 2025 covers December 2024 through February 2025. The script
reads the local project AOI, uses Cloud Score+ (`cs_cdf >= 0.60`), exports in
`EPSG:32630` at 10 m, and splits large bands into 4096-pixel GeoTIFF shards.


## Important safeguards

- Keep all datasets in a common projected CRS before area calculations.
- Record source, license, date, resolution, and preprocessing for every layer.
- Validate classifications independently for every mapped year.
- Treat cocoa probability products and mapped cocoa polygons as estimates, not
  automatically as ground truth.
