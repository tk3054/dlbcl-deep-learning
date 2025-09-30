# Cell Analysis Pipeline

Automated pipeline for cell segmentation, quality filtering, and multi-channel fluorescence analysis.

## Overview

This pipeline processes fluorescence microscopy images to:
1. Segment individual cells using watershed algorithm
2. Detect and filter cut-off cells using edge detection
3. Manual quality review with GUI
4. Extract measurements from multiple fluorescence channels

## Requirements

```bash
pip install numpy opencv-python pillow scikit-image scipy pandas
brew install python-tk@3.12  # For GUI
```

ImageJ/Fiji is also required for channel extraction.

## Pipeline Steps

### 1. Cell Segmentation
```bash
python segment_cells.py
```
- Performs watershed segmentation on masked images
- Exports individual cell ROIs and raw crops
- **Edit `SAMPLE_FOLDER` and `IMAGE_NUMBER` at the top to change inputs**

### 2. Pad Raw Crops (Optional)
```bash
python pad_raw_crops.py
```
- Pads cell crops to square images (64x64)
- Preserves aspect ratio with black padding
- **Edit `TARGET_SIZE` to change output dimensions**

### 3. Filter Bad Cells
```bash
python filter_bad_cells.py
```
- Detects cut-off cells using consecutive white pixel analysis
- Threshold: 50+ consecutive white pixels on any edge = CUT
- Saves analysis results to CSV

### 4. Manual Review
```bash
python manual_cell_reviewer.py
```
- GUI window for manual cell classification
- **Keyboard controls:**
  - ← Left Arrow: Mark as BAD
  - → Right Arrow: Mark as GOOD
  - B: Undo last classification
  - Q/Escape: Quit and save
- Saves classifications to `manual_classifications.csv`

### 5. Load ROIs in ImageJ
```
File > Open > load_rois.ijm
```
- Loads all cell ROIs onto original image
- **Edit `sampleFolder` and `imageNumber` at the top**

### 6. Extract Channel Measurements
```
File > Open > extract_channels.ijm
```
- Measures all channels: Actin-FITC, CD4-PerCP, CD45RA-AF647, CCR7-PE
- Saves measurements to individual CSV files per channel
- **Edit `sampleFolder` and `imageNumber` at the top**

## Configuration

All scripts have configurable parameters at the top:

### Python Scripts
```python
SAMPLE_FOLDER = "sample1"  # Options: "sample1", "sample2", "sample3"
IMAGE_NUMBER = "3"         # Options: "1", "2", "3", "4", etc.
```

### ImageJ Macros
```imagej
sampleFolder = "sample1";  // Options: "sample1", "sample2", "sample3"
imageNumber = "3";         // Options: "1", "2", "3", "4", etc.
```

## File Structure

```
/BASE_PATH/sample1/1/
  ├── *_raw.jpg              # Raw image
  ├── *_mask.jpg             # Binary mask
  ├── cell_rois/             # Individual cell ROI masks
  ├── raw_crops/             # Unpadded cell crops
  ├── padded_cells/          # Padded square crops (64x64)
  ├── original_image.tif     # Reference image
  ├── raw_crops_quality_analysis.csv
  ├── manual_classifications.csv
  ├── actin-fitc-measurements.csv
  ├── cd4-percp-measurements.csv
  ├── cd45ra-af647-measurements.csv
  └── ccr7-pe-measurements.csv
```

## Parameters

### Segmentation
- `MIN_DISTANCE = 12`: Minimum distance between cell centers
- `MIN_SIZE = 500`: Minimum cell area (pixels)
- `MAX_SIZE = 9000`: Maximum cell area (pixels)

### Cut Detection
- `CONSECUTIVE_THRESHOLD = 50`: Consecutive white pixels to consider as cut edge

### Padding
- `TARGET_SIZE = 64`: Size of square output images

## Notes

- Always run `segment_cells.py` first to generate ROIs and raw crops
- The pipeline assumes preprocessed images with `*_raw.jpg` and `*_mask.jpg` naming
- ImageJ macros require channels to be saved as separate TIF files in the sample directory
- Manual review results can be used to train automated classifiers