# Refactor Target

This file records the current known pipeline target before larger cleanup work.
Use it as a smoke-test reference while moving code into cleaner modules.

## Current Local Entry Point

Run from the repository root:

```bash
.venv/bin/python main.py
```

The current local configuration lives at the top of `main.py`.

## Current Smoke Input

Donor folder:

```text
/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241
```

Current filters in `main.py` select:

```text
sample1
image 5
```

Channel configuration:

```text
actin: Actin-FITC.tif
cd4: CD4-PerCP.tif
cd45ra_PacBlue: CD45RA-PacBlue.tif
cd19car: CD19CAR-AF647.tif
ccr7: CCR7-AF594.tif
```

## Expected Pipeline Artifacts

For each processed image folder:

```text
cell_rois/
raw_actin/
raw_ccr7/
raw_cd45ra/
padded_cells/
padded_ccr7/
padded_cd45ra/
processed_*.tif
*-measurements.csv
combined_measurements.csv
cellpose_segmentation_visualization.png
```

For each processed sample folder:

```text
combined_measurements.csv
```

For the donor folder:

```text
all_samples_combined.csv
all_samples_combined_classified.csv
cell_area_histogram.png
formatted_cells/
```

## Import Smoke Check

Run:

```bash
.venv/bin/python scripts/smoke_imports.py
```

This checks that the current main pipeline modules can still be imported after
refactors. It does not run segmentation, ImageJ processing, or CSV generation.

## Phase 2 Shared Types

New shared dataclasses live under `dlbcl_pipeline/`:

```text
dlbcl_pipeline/config.py
dlbcl_pipeline/constants.py
dlbcl_pipeline/model_folder_structure.py
dlbcl_pipeline/measurements/aggregation.py
dlbcl_pipeline/build_paths.py
dlbcl_pipeline/process_donor.py
dlbcl_pipeline/data_types.py
```

These files are scaffolding only. They define the shapes future scripts and
notebooks should pass around, but the existing `main.py` behavior is unchanged.

`build_paths.py` names common output locations. `model_folder_structure.py` turns a
`PipelineConfig` into a `DonorFolderStructure` future pipeline code can consume.
`measurements/aggregation.py` provides the clearer table-building API for image,
sample, donor, and response-group CSVs.
`process_donor.py` orchestrates image jobs and donor-level table building.

## Phase 8 Thin Scripts

Thin CLI wrappers live in `scripts/`:

```text
scripts/run_donor.py
scripts/combine_donor.py
scripts/plot_response_groups.py
```

They should stay small: parse CLI arguments, build config, call package APIs,
and exit with a status code.
