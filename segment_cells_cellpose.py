#!/usr/bin/env python3
"""
Cell Segmentation with Cellpose
Uses Cellpose deep learning models for cell segmentation instead of watershed

Performs Cellpose segmentation on manually masked images and exports:
- Individual cell ROIs as binary masks
- Raw cell crops for quality filtering

Usage:
    python segment_cells_cellpose.py
    (Edit SAMPLE_FOLDER and IMAGE_NUMBER below to change inputs)
"""

import os
import sys
import numpy as np
from PIL import Image
from skimage import measure, io
from scipy import ndimage as ndi

# Try to import Cellpose at module level
try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    models = None


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Change these values to process different samples
SAMPLE_FOLDER = "sample1"  # Options: "sample1", "sample2", "sample3"
IMAGE_NUMBER = "5"         # Options: "1", "2", "3", "4", etc.

# Cellpose parameters
MODEL_TYPE = "cyto2"       # Options: "cyto2" (cells), "nuclei", "cpsam" (latest)
DIAMETER = None            # Auto-detect if None, or specify in pixels
FLOW_THRESHOLD = 0.4       # Higher = stricter (fewer masks)
CELLPROB_THRESHOLD = 0.0   # Higher = stricter (fewer masks)
USE_GPU = True             # Set to True if you have GPU (Mac MPS)

# Size filtering
MIN_SIZE = 500
MAX_SIZE = 9000


# ============================================================================
# AUTO-GENERATED PATHS (do not edit)
# ============================================================================

BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"
BASE_DIR = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"


# ============================================================================
# CELLPOSE SEGMENTATION FUNCTIONS
# ============================================================================

def load_image(image_path):
    """Load an image as numpy array"""
    img = Image.open(image_path)
    if img.mode == 'RGB':
        # Convert RGB to grayscale
        return np.array(img.convert('L'))
    return np.array(img)


def preprocess_for_cellpose(image):
    """Enhance dim cells before Cellpose segmentation using CLAHE"""
    import cv2

    # Normalize to 0-1 range
    img_norm = (image - image.min()) / (image.max() - image.min())

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    img_8bit = (img_norm * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_8bit)

    # Optional: Gaussian blur to reduce noise
    img_enhanced = cv2.GaussianBlur(img_enhanced, (3,3), 0)

    return img_enhanced


def segment_cells_with_cellpose(image, mask=None, model_type='cyto2', diameter=None,
                                flow_threshold=0.4, cellprob_threshold=0.0,
                                min_size=500, max_size=9000, use_gpu=False,
                                enhance_contrast=True, verbose=True):
    """
    Segment cells using Cellpose deep learning model

    Args:
        image: Input image (grayscale numpy array)
        mask: Optional binary mask to constrain segmentation region
        model_type: Cellpose model ('cyto2', 'nuclei', 'cpsam')
        diameter: Cell diameter in pixels (None = auto-detect)
        flow_threshold: Flow error threshold (0.4 default)
        cellprob_threshold: Cell probability threshold (0.0 default)
        min_size: Minimum cell area in pixels
        max_size: Maximum cell area in pixels
        use_gpu: Use GPU acceleration
        enhance_contrast: Apply CLAHE preprocessing to enhance dim cells
        verbose: Print progress

    Returns:
        valid_regions: List of regionprops objects
        valid_bboxes: List of bounding boxes
        labeled_mask: Labeled mask with cell IDs
    """
    if not CELLPOSE_AVAILABLE:
        raise ImportError(
            "Cellpose not installed. Install with: pip install cellpose"
        )

    if verbose:
        print(f"Initializing Cellpose model: {model_type}")
        print(f"  GPU enabled: {use_gpu}")

    # Preprocess image if requested
    if enhance_contrast:
        if verbose:
            print("Applying CLAHE contrast enhancement...")
        image = preprocess_for_cellpose(image)

    # Initialize Cellpose model
    model = models.CellposeModel(gpu=use_gpu, model_type=model_type)

    if verbose:
        print("Running Cellpose segmentation...")
        print(f"  Diameter: {'auto-detect' if diameter is None else f'{diameter} pixels'}")
        print(f"  Flow threshold: {flow_threshold}")
        print(f"  Cell probability threshold: {cellprob_threshold}")

    # Run Cellpose segmentation
    # channels=[0,0] means grayscale (first channel is the channel to segment, second is optional nuclear channel)
    result = model.eval(
        x=image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=[0, 0]  # Grayscale
    )

    # Handle different return formats (v4.0+ returns 3 values, older versions return 4)
    if len(result) == 4:
        masks, flows, styles, diams = result
    else:
        masks, flows, styles = result
        diams = diameter if diameter is not None else 30  # Default

    if verbose:
        detected_diameter = diams if diameter is None else diameter
        print(f"  Detected diameter: {detected_diameter:.1f} pixels")
        print(f"  Found {len(np.unique(masks)) - 1} initial masks")

    # If mask is provided, filter to only cells within mask region
    if mask is not None:
        binary_mask = mask > 0
        # Zero out any Cellpose masks outside the provided mask
        masks = masks * binary_mask

    # Get region properties
    regions = measure.regionprops(masks, intensity_image=image)

    # Filter by size
    valid_regions = []
    valid_bboxes = []

    # Create new labeled mask with only valid cells
    labeled_mask = np.zeros_like(masks)
    new_label = 1

    for region in regions:
        if min_size <= region.area <= max_size:
            # Add to valid list
            valid_regions.append(region)

            # Convert bbox to (x, y, w, h) format
            minr, minc, maxr, maxc = region.bbox
            x, y = minc, minr
            w, h = maxc - minc, maxr - minr
            valid_bboxes.append((x, y, w, h))

            # Add to new labeled mask
            labeled_mask[masks == region.label] = new_label
            new_label += 1

    if verbose:
        print(f"  After size filtering ({min_size}-{max_size} pixels): {len(valid_regions)} cells")

    return valid_regions, valid_bboxes, labeled_mask


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_cell_rois(image, labeled_mask, regions, base_dir):
    """Export individual cell ROIs as binary masks for ImageJ"""
    import warnings

    roi_dir = os.path.join(base_dir, 'cell_rois')
    os.makedirs(roi_dir, exist_ok=True)

    for i, region in enumerate(regions):
        # Create binary mask for this single cell
        single_cell_mask = (labeled_mask == (i + 1)).astype(np.uint8) * 255

        # Save each cell as separate binary image
        roi_path = os.path.join(roi_dir, f'cell_{i+1:02d}.tif')

        # Suppress low contrast warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(roi_path, single_cell_mask)

    # Also save original for reference
    original_path = os.path.join(base_dir, 'original_image.tif')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(original_path, image.astype(np.uint8))

    return roi_dir


def export_raw_crops(image, bboxes, base_dir):
    """Export raw (unpadded) cell crops for quality filtering"""
    raw_crops_dir = os.path.join(base_dir, 'raw_crops')
    os.makedirs(raw_crops_dir, exist_ok=True)

    for i, (x, y, w, h) in enumerate(bboxes):
        # Raw crop without padding
        cropped = image[y:y+h, x:x+w]

        if cropped.size == 0:
            print(f"⚠️  Skipping empty bbox {i+1}")
            continue

        # Save as numbered files for easier sorting
        save_path = os.path.join(raw_crops_dir, f'cell_{i+1:02d}.tif')
        io.imsave(save_path, cropped.astype(np.uint8))

    return raw_crops_dir


# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================

def segment_cells_cellpose(sample_folder, image_number, base_path,
                           model_type='cyto2', diameter=None,
                           flow_threshold=0.4, cellprob_threshold=0.0,
                           min_size=500, max_size=9000, use_gpu=False,
                           enhance_contrast=True, verbose=True):
    """
    Segment cells using Cellpose.

    This is the main function to be called from notebooks or other scripts.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        model_type: Cellpose model type ('cyto2', 'nuclei', 'cpsam')
        diameter: Cell diameter in pixels (None = auto-detect)
        flow_threshold: Flow error threshold (higher = stricter)
        cellprob_threshold: Cell probability threshold (higher = stricter)
        min_size: Minimum cell area in pixels
        max_size: Maximum cell area in pixels
        use_gpu: Use GPU acceleration
        enhance_contrast: Apply CLAHE preprocessing to enhance dim cells
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'num_cells': Number of cells segmented
            - 'roi_dir': Path to ROI directory
            - 'raw_crops_dir': Path to raw crops directory
            - 'detected_diameter': Detected cell diameter
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
    """
    from pathlib import Path

    # Build paths
    base_dir = f"{base_path}/{sample_folder}/{image_number}"

    # Find raw and mask files
    base_path_obj = Path(base_dir)
    raw_files = list(base_path_obj.glob("*_raw.jpg"))
    mask_files = list(base_path_obj.glob("*_mask.jpg"))

    if not raw_files:
        return {
            'success': False,
            'error': f"Raw image not found in {base_dir}",
            'num_cells': 0
        }

    if not mask_files:
        return {
            'success': False,
            'error': f"Mask image not found in {base_dir}",
            'num_cells': 0
        }

    raw_path = raw_files[0]
    mask_path = mask_files[0]

    if verbose:
        print("="*60)
        print("CELLPOSE CELL SEGMENTATION PIPELINE")
        print("="*60)
        print(f"Sample: {sample_folder}/{image_number}")
        print(f"Base directory: {base_dir}\n")
        print("Loading images...")
        print(f"  Raw: {raw_path.name}")
        print(f"  Mask: {mask_path.name}")

    # Load images
    image = load_image(str(raw_path))
    mask = load_image(str(mask_path))

    if verbose:
        print(f"  Loaded image: {image.shape}")
        print(f"  Loaded mask: {mask.shape}\n")

    # Perform Cellpose segmentation
    try:
        regions, bboxes, labeled_mask = segment_cells_with_cellpose(
            image, mask=mask,
            model_type=model_type,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            max_size=max_size,
            use_gpu=use_gpu,
            enhance_contrast=enhance_contrast,
            verbose=verbose
        )
    except Exception as e:
        return {
            'success': False,
            'error': f"Cellpose segmentation failed: {str(e)}",
            'num_cells': 0
        }

    if verbose:
        print(f"\nExporting cell ROIs...")

    # Export ROIs
    roi_dir = export_cell_rois(image, labeled_mask, regions, base_dir)

    if verbose:
        print(f"  Exported {len(regions)} cell ROIs to: {roi_dir}\n")
        print("Exporting raw crops...")

    # Export raw crops
    raw_crops_dir = export_raw_crops(image, bboxes, base_dir)

    if verbose:
        print(f"  Exported {len(bboxes)} raw crops to: {raw_crops_dir}\n")
        print("="*60)
        print("CELLPOSE SEGMENTATION COMPLETE")
        print("="*60)

    return {
        'success': True,
        'num_cells': len(regions),
        'roi_dir': roi_dir,
        'raw_crops_dir': raw_crops_dir,
        'base_dir': base_dir
    }


def visualize_segmentation(base_dir, num_cells):
    """Create visualization of segmentation results with boundaries"""
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    print("\nCreating visualization...")

    base_path = Path(base_dir)
    cellpose_roi_dir = base_path / "cell_rois_cellpose"

    # Load original image
    raw_files = list(base_path.glob("*_raw.jpg"))
    if not raw_files:
        print("⚠️  Could not find raw image for visualization")
        return

    raw_img = np.array(Image.open(raw_files[0]).convert('L'))

    # Load all ROI masks and combine
    roi_files = sorted(cellpose_roi_dir.glob("*.tif"))
    combined_mask = np.zeros_like(raw_img)

    for i, roi_file in enumerate(roi_files):
        roi_mask = np.array(Image.open(roi_file).convert('L'))
        combined_mask[roi_mask > 0] = i + 1

    # Create boundaries
    boundaries = find_boundaries(combined_mask, mode='inner')

    # Overlay boundaries on raw image
    overlay = np.stack([raw_img] * 3, axis=-1)
    overlay[boundaries] = [255, 0, 0]  # Red boundaries

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(combined_mask, cmap='nipy_spectral')
    axes[1].set_title(f'Cellpose Segmentation ({num_cells} cells)', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Cell Boundaries Overlay', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = base_path / "cellpose_segmentation_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved: {output_path}")


def main():
    """Run Cellpose segmentation with configuration from globals"""

    result = segment_cells_cellpose(
        sample_folder=SAMPLE_FOLDER,
        image_number=IMAGE_NUMBER,
        base_path=BASE_PATH,
        model_type=MODEL_TYPE,
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        use_gpu=USE_GPU,
        verbose=True
    )

    if not result['success']:
        print(f"\n❌ Segmentation failed: {result['error']}")
        sys.exit(1)

    # Create visualization
    visualize_segmentation(result['base_dir'], result['num_cells'])

    print("\n" + "="*60)
    print("✅ ALL DONE!")
    print("="*60)
    print(f"Segmented {result['num_cells']} cells")
    print(f"\nOutputs:")
    print(f"  • ROI masks: {result['roi_dir']}")
    print(f"  • Raw crops: {result['raw_crops_dir']}")
    print(f"  • Visualization: {result['base_dir']}/cellpose_segmentation_visualization.png")


if __name__ == "__main__":
    main()
