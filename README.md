# Cell Analysis Pipeline

A comprehensive automated pipeline for cell segmentation, fluorescence quantification, and analysis from microscopy images.

## 📋 Overview

This pipeline automates the complete workflow for analyzing fluorescent microscopy images:
1. Cell segmentation using Cellpose (deep learning)
2. Multi-channel fluorescence measurement
3. ROI extraction and visualization
4. Data aggregation and quality control

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n cell_analysis python=3.9
conda activate cell_analysis

# Install required packages
pip install cellpose numpy pandas pillow opencv-python scikit-image matplotlib
pip install pyimagej scyjava

# Install Fiji/ImageJ
# Download from: https://fiji.sc/
```

### 2. Configure Your Data

Edit `main.py` at the top:

```python
# Set your data directory
BASE_PATH = "/path/to/your/data"

# Select which samples to process
SAMPLES_TO_PROCESS = [1, 2, 3]  # or [1] for just sample1

# Configure your channel filenames (IMPORTANT!)
CHANNEL_CONFIG = {
    'actin': 'Actin-FITC.tif',           # Change to your actual filenames
    'cd4': 'CD4-PerCP.tif',
    'cd45ra_af647': 'CD45RA-AF647.tif',
    'cd45ra_sparkviolet': 'CD45RA-SparkViolet.tif',  # plots label this as "PacBlue"
    'cd19car': 'CD19CAR-AF647.tif',
    'ccr7': 'CCR7-PE.tif',
}
```

### 3. Run the Pipeline

```bash
python main.py
```

That's it! The pipeline will automatically:
- Discover all samples and images
- Process each image through the complete pipeline
- Combine results into a master CSV

## 📁 Required Directory Structure

Your data should be organized as:

```
BASE_PATH/
├── sample1/
│   ├── 1/
│   │   ├── Actin-FITC.tif
│   │   ├── CD4-PerCP.tif
│   │   ├── CD45RA-AF647.tif
│   │   └── ... (other channel files)
│   ├── 2/
│   └── 3/
├── sample2/
│   ├── 1/
│   ├── 2/
│   └── ...
└── sample3/
    └── ...
```

**Important Notes:**
- Sample folders MUST start with "sample" (e.g., sample1, sample2, sample3)
- Image folders can be ANY name (1, 2, 3, 44, 1[CAR], etc.)
- Each image folder should contain your channel TIF files

## 🔧 Configuration Options

### Channel Names

**This is the most important configuration!** If your channel files have different names, edit `CHANNEL_CONFIG` in `main.py`:

```python
CHANNEL_CONFIG = {
    'actin': 'YourActinFileName.tif',  # ← Change this!
    'cd4': 'YourCD4FileName.tif',      # ← And this!
    # ... etc
}
```

### Cellpose Parameters

Edit `PARAMS` in `process_single_image.py`:

```python
PARAMS = {
    'cellpose_model': 'cyto2',           # Options: 'cyto2', 'cyto', 'nuclei'
    'cellpose_diameter': 50,             # Cell diameter in pixels
    'cellpose_flow_threshold': 0.6,      # Higher = stricter (0.0-1.0)
    'cellpose_cellprob_threshold': 0.0,  # Higher = fewer cells
    'min_size': 500,                     # Minimum cell area (pixels)
    'max_size': 20000,                   # Maximum cell area (pixels)
    'cellpose_use_gpu': True,            # Use GPU if available
    'shrink_pixels': 8,                  # ROI shrinkage for cleaner boundaries
}
```

## 📊 Pipeline Steps

The pipeline executes these steps for each image:

### Step 1: Cell Segmentation (Cellpose)
- Uses deep learning to identify cell boundaries
- Creates ROI masks for each cell
- Exports raw crops for quality control

### Step 2: ROI Shrinking
- Shrinks ROI boundaries by N pixels
- Creates cleaner measurements by excluding edges

### Step 3: Channel Preprocessing
- Background subtraction
- Contrast enhancement
- Noise reduction (channel-specific)

### Step 4: JPG Duplicate Creation
- Creates JPG versions for visualization
- Preserves original TIF files

### Step 5: ROI Crop Extraction
Three versions created:
- `roi_crops_tif/` - Original TIF data (full dynamic range)
- `roi_crops_whiteBg/` - PNG with white background (verification)
- `roi_crops_blackBg/` - PNG with black background (analysis)

### Step 6: ROI Visualization
- Overlays ROI boundaries on channel images
- Creates multi-channel comparison views

### Step 7: Fluorescence Measurement
- Measures intensity in each channel for each cell
- Extracts: mean, **median**, std, min, max, integrated density
- Also measures: area, circularity, roundness, solidity

### Step 8: Data Combination
- Combines all channels into single CSV per image
- Merges all images into master CSV

## 📈 Output Files

### Per-Image Outputs

Each image folder will contain:

```
sample1/1/
├── cell_rois/                      # Original ROI masks
├── cell_rois_shrunk/               # Shrunk ROI masks (used for measurements)
├── roi_crops_tif/                  # TIF crops (original data)
├── roi_crops_whiteBg/              # PNG with white background (verification)
├── roi_crops_blackBg/              # PNG with black background
├── raw_crops_jpg/                  # Raw unpadded crops from segmentation
├── processed_*.tif                 # Preprocessed channel files
├── *_raw.jpg                       # JPG duplicates for visualization
├── cellpose_segmentation_visualization.png  # Segmentation overlay
├── actin-fitc-measurements.csv     # Individual channel measurements
├── cd4-percp-measurements.csv
├── ... (other channel CSVs)
└── combined_measurements.csv       # All channels merged
```

### Master Outputs

In `BASE_PATH/`:

```
BASE_PATH/
├── all_samples_combined.csv              # Master CSV (all cells, all samples)
├── all_samples_intensity_histograms.png  # Intensity distributions
└── cell_classifications.csv              # Quality control labels (if performed)
```

## 📊 Output CSV Format

`all_samples_combined.csv` contains:

| Column | Description |
|--------|-------------|
| `unique_id` | sample_image_cellid (e.g., sample1_2_45) |
| `sample` | Sample folder name |
| `image` | Image number |
| `cell_id` | Cell ID within image |
| `area` | Cell area (μm²) |
| `x`, `y` | Cell centroid position |
| `circ` | Circularity (0-1, 1=perfect circle) |
| `round` | Roundness |
| `solidity` | Solidity (convexity) |
| `ar` | Aspect ratio |
| `actin_mean` | Mean intensity in actin channel |
| `actin_median` | **Median intensity in actin channel** |
| `actin_std` | Standard deviation |
| `actin_min`, `actin_max` | Min/max intensities |
| `actin_intden` | Integrated density |
| `actin_rawintden` | Raw integrated density |
| `cd4_mean`, `cd4_median`, ... | Same for CD4 channel |
| ... | (repeat for all channels) |

**Note:** Median intensity is now included alongside mean for more robust measurements!

## 🛠️ Troubleshooting

### Error: "No image found"

```
ERROR: No image found in sample1/2/ (looked for Actin-FITC.tif or *_raw.jpg)
→ Please check channel names at the top of main.py (CHANNEL_CONFIG)
```

**Solution:** Your channel file has a different name. Check what files exist in your image folder and update `CHANNEL_CONFIG` in `main.py`.

### Error: "No module named 'cellpose'"

```bash
pip install cellpose
```

### GPU Issues

If Cellpose GPU fails, set in `process_single_image.py`:
```python
'cellpose_use_gpu': False,
```

### Memory Issues

Process fewer samples at once:
```python
SAMPLES_TO_PROCESS = [1]  # Process one at a time
```

### No Cells Detected

Try adjusting Cellpose parameters in `process_single_image.py`:
```python
PARAMS = {
    'cellpose_diameter': None,           # Let Cellpose auto-detect
    'cellpose_flow_threshold': 0.4,      # Lower = more permissive
    'cellpose_cellprob_threshold': -2.0, # Lower = more cells
    'min_size': 200,                     # Lower minimum
}
```

### ImageJ/PyImageJ Issues

If ImageJ initialization fails:
```bash
# Try reinstalling PyImageJ
pip uninstall pyimagej scyjava
pip install pyimagej scyjava
```

## 🔍 Quality Control

### Interactive Cell Sorter

Review and classify individual cells:

```bash
python quality_control/interactive_cell_sorter.py
```

**Keyboard shortcuts:**
- `← Left`: Mark as BAD
- `→ Right`: Mark as GOOD
- `1-6`: Specific classifications (GOOD, BAD, UNCERTAIN, DEBRIS, DOUBLET, EDGE)
- `Space`: Skip
- `B`: Undo
- `S`: Save
- `Q`: Quit and save

Classifications are saved to `cell_classifications.csv` and can be merged with the master dataset.

### Visualization Tools

Generate intensity distribution plots:
```bash
python produce_figures/plot_all_intensity_histograms.py
```

This creates histograms showing median intensity distributions across all cells for each channel.

## 📝 Advanced Usage

### Process Single Image

```python
from process_single_image import run_pipeline, PARAMS
from main import CHANNEL_CONFIG

result = run_pipeline(
    sample_folder="sample1",
    image_number="5",
    base_path="/path/to/data",
    params=PARAMS,
    channel_config=CHANNEL_CONFIG,
    verbose=True
)

if result['success']:
    print(f"Processed {result['results']['segmentation']['num_cells']} cells")
```

### Custom Channel Measurements

To add or modify channels:

1. Add the channel to `CHANNEL_CONFIG` in `main.py`
2. Edit `imageJ_scripts/extract_channels.ijm` to add measurement calls
3. Update `csvOps/combine_channel.py` if needed

### Modify Preprocessing

Each channel has different preprocessing in `imageJ_scripts/preprocess_channels.ijm`:
- **cd4**: Background subtraction (100px rolling)
- **cd45ra**: Gaussian blur + background subtraction
- **ccr7**: Median filter + background subtraction
- **Others**: Copied as-is

To change preprocessing for a channel, edit the `preprocessChannel()` function in `preprocess_channels.ijm`.

### ROI Crop Background Color

To change from white to black background (or vice versa), the pipeline creates both:
- `roi_crops_whiteBg/` - White padding for visual verification
- `roi_crops_blackBg/` - Black padding for analysis

Both are automatically generated during pipeline execution.

## 🎯 Key Features

### ✨ Centralized Configuration
- Edit channel names in ONE place (`main.py`)
- All scripts automatically adapt
- No need to edit 14+ files individually

### 🤖 Fully Automated
- Auto-discovers all samples and images
- Handles any image folder naming (1, 2, 44, 1[CAR], etc.)
- Processes everything end-to-end

### 📊 Robust Measurements
- Both mean and median intensities
- Multiple morphological features
- Shrunk ROIs for cleaner boundaries

### 🔍 Quality Control
- Interactive cell sorter GUI
- Visual verification tools
- Raw crops for manual inspection

### 💾 Organized Output
- Hierarchical directory structure
- Master combined CSV
- Individual channel CSVs preserved

## 🤝 Support

If you encounter issues:

1. **Check channel filenames** in `CHANNEL_CONFIG` match your actual files
2. **Verify directory structure** matches expected format (sample1, sample2, etc.)
3. **Review error messages** - they now point you to the exact configuration to fix
4. **Check dependencies** are installed (`pip list | grep cellpose`)

For bugs or feature requests, please open an issue.

## 📚 Citation

If you use this pipeline in your research, please cite:
- **Cellpose**: Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. Cellpose: a generalist algorithm for cellular segmentation. *Nat Methods* 18, 100–106 (2021).

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **Cellpose** for deep learning segmentation
- **ImageJ/Fiji** for image processing
- **PyImageJ** for Python-ImageJ integration
