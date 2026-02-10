#!/usr/bin/env python3
"""
Cell Segmentation with Cellpose
Uses Cellpose deep learning models for cell segmentation instead of watershed

Performs Cellpose segmentation on manually masked images and exports:
- Individual cell ROIs as binary masks

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
IMAGE_NUMBER = "6"         # Options: "1", "2", "3", "4", etc.

# Cellpose parameters
MODEL_TYPE = "cyto2"       # Options: "cyto2" (cells), "nuclei", "cpsam" (latest)
DIAMETER = 50              # Auto-detect if None, or specify in pixels
FLOW_THRESHOLD = 0.4       # Higher = stricter (fewer masks)
CELLPROB_THRESHOLD = -2.0  # Higher = stricter (fewer masks), lower = more permissive
USE_GPU = True             # Set to True if you have GPU (Mac MPS)

# Size filtering
MIN_SIZE = 500
MAX_SIZE = 20000           # Increased for larger cells


# ============================================================================
# AUTO-GENERATED PATHS (do not edit)
# ============================================================================

BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"
BASE_DIR = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"


# ============================================================================
# CELLPOSE SEGMENTATION FUNCTIONS
# ============================================================================

def load_image(image_path):
    """Load an image as numpy array, preserving bit depth"""
    import cv2
    # Try to load with full bit depth first (for TIF files)
    img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Fall back to normal imread for JPEG/PNG
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        # Final fallback to PIL
        from PIL import Image
        pil_img = Image.open(image_path)
        if pil_img.mode == 'RGB':
            return np.array(pil_img.convert('L'))
        return np.array(pil_img)
    return img


def segment_cells_with_cellpose(image, mask=None, model_type='cyto2', diameter=None,
                                flow_threshold=0.4, cellprob_threshold=0.0,
                                min_size=500, max_size=9000, use_gpu=False,
                                verbose=True):
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

    # Apply Cellpose's native normalization (like GUI does)
    from cellpose import transforms

    # Ensure image has correct shape for normalization
    if len(image.shape) == 2:
        # Add channel dimension for grayscale images
        img_norm = transforms.normalize_img(image[..., np.newaxis], axis=-1)
        # Remove channel dimension after normalization
        img_norm = img_norm.squeeze()
    else:
        img_norm = transforms.normalize_img(image, axis=-1)

    if verbose:
        print("Applied Cellpose normalization")

    # Initialize Cellpose model
    model = models.CellposeModel(gpu=use_gpu, model_type=model_type)

    if verbose:
        print("Running Cellpose segmentation...")
        print(f"  Diameter: {'auto-detect' if diameter is None else f'{diameter} pixels'}")
        print(f"  Flow threshold: {flow_threshold}")
        print(f"  Cell probability threshold: {cellprob_threshold}")

    # Run Cellpose segmentation with GUI defaults
    result = model.eval(
        x=img_norm,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=None,    # Auto-detect (like GUI)
        rescale=None,     # Let Cellpose decide (like GUI)
        resample=True     # GUI default
    )

    # Handle different return formats (v4.0+ returns 3 values, older versions return 4)
    try:
        masks, flows, styles, diams = result
    except ValueError:
        # v4.0+ returns only 3 values (masks, flows, styles)
        masks, flows, styles = result
        # In v4.0+, diameter is in flows[3] if it was auto-detected
        if isinstance(flows, list) and len(flows) > 3:
            diams = flows[3]
        else:
            diams = diameter if diameter is not None else 30  # Default

    # Extract cell probability map (flows[2])
    cell_prob_map = None
    if isinstance(flows, list) and len(flows) > 2:
        cell_prob_map = flows[2]
    elif isinstance(flows, tuple) and len(flows) > 2:
        cell_prob_map = flows[2]

    if verbose:
        detected_diameter = diams if diameter is None else diameter
        print(f"  Detected diameter: {detected_diameter:.1f} pixels")
        print(f"  Found {len(np.unique(masks)) - 1} initial masks")
        if cell_prob_map is not None:
            print(f"  Extracted probability map: {cell_prob_map.shape}")

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

    return valid_regions, valid_bboxes, labeled_mask, cell_prob_map


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

    # Also save original for reference (preserve original bit depth)
    original_path = os.path.join(base_dir, 'original_image.tif')

    # Debug: print image info to verify correct image is being saved
    print(f"  Saving original image: {image.shape}, dtype: {image.dtype}, unique_values: {len(np.unique(image))}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Save with original dtype (uint8, uint16, etc.)
        io.imsave(original_path, image, check_contrast=False)

    return roi_dir


# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================

def segment_cells_cellpose(sample_folder, image_number, base_path,
                           model_type='cyto2', diameter=None,
                           flow_threshold=0.4, cellprob_threshold=0.0,
                           min_size=500, max_size=9000, use_gpu=False,
                           channel_config=None, verbose=True):
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
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'num_cells': Number of cells segmented
            - 'roi_dir': Path to ROI directory
            - 'detected_diameter': Detected cell diameter
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
    """
    from pathlib import Path

    # Build paths
    base_dir = f"{base_path}/{sample_folder}/{image_number}"

    # Use default channel config if not provided
    if channel_config is None:
        channel_config = {'actin': 'Actin-FITC.tif'}

    # Find image files (prefer original TIF over preprocessed JPEG)
    base_path_obj = Path(base_dir)

    # Look for original Actin TIF first (better quality)
    actin_filename = channel_config.get('actin', 'Actin-FITC.tif')
    actin_tif = base_path_obj / actin_filename
    raw_files = list(base_path_obj.glob("*_raw.jpg"))

    if actin_tif.exists():
        image_path = actin_tif
        image_source = "original TIF"
    elif raw_files:
        image_path = raw_files[0]
        image_source = "preprocessed JPEG"
    else:
        return {
            'success': False,
            'error': f"No image found in {base_dir} (looked for {actin_filename} or *_raw.jpg)\n→ Please check channel names at the top of main.py (CHANNEL_CONFIG)",
            'num_cells': 0
        }

    # Mask is optional for Cellpose
    mask_files = list(base_path_obj.glob("*_mask.jpg"))
    mask_path = mask_files[0] if mask_files else None

    # Load images
    image = load_image(str(image_path))
    mask = load_image(str(mask_path)) if mask_path else None

    if verbose:
        mask_info = f" (with mask)" if mask is not None else ""
        print(f"Segmenting cells with Cellpose{mask_info}...")

    # Perform Cellpose segmentation
    try:
        regions, bboxes, labeled_mask, cell_prob_map = segment_cells_with_cellpose(
            image, mask=mask,
            model_type=model_type,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            max_size=max_size,
            use_gpu=use_gpu,
            verbose=verbose
        )
    except Exception as e:
        return {
            'success': False,
            'error': f"Cellpose segmentation failed: {str(e)}",
            'num_cells': 0
        }

    # Save the probability map if available
    if cell_prob_map is not None:
        prob_map_path = base_path_obj / 'cellpose_prob_map.tif'
        try:
            io.imsave(prob_map_path, cell_prob_map, check_contrast=False)
            if verbose:
                print(f"  ✓ Saved probability map: {prob_map_path.name}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Failed to save probability map: {e}")

    # Export ROIs
    roi_dir = export_cell_rois(image, labeled_mask, regions, base_dir)

    if verbose:
        print(f"  ✓ Segmented {len(regions)} cells, exported ROIs")

    # Create visualization
    visualize_segmentation(base_dir, len(regions), channel_config)

    return {
        'success': True,
        'num_cells': len(regions),
        'roi_dir': roi_dir,
        'base_dir': base_dir,
        'cell_prob_map': cell_prob_map
    }


def visualize_segmentation(base_dir, num_cells, channel_config=None):
    """Create visualization of segmentation results with boundaries"""
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    import cv2

    # Use default channel config if not provided
    if channel_config is None:
        channel_config = {'actin': 'Actin-FITC.tif'}

    base_path = Path(base_dir)
    cellpose_roi_dir = base_path / "cell_rois"

    # Load original image (prefer TIF over JPEG for better quality)
    actin_filename = channel_config.get('actin', 'Actin-FITC.tif')
    actin_tif = base_path / actin_filename
    raw_files = list(base_path.glob("*_raw.jpg"))

    if actin_tif.exists():
        # Load TIF with full bit depth
        raw_img = cv2.imread(str(actin_tif), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        # Normalize to 8-bit for display
        raw_img = raw_img.astype(float)
        img_min, img_max = raw_img.min(), raw_img.max()
        if img_max > img_min:
            raw_img = (raw_img - img_min) / (img_max - img_min) * 255
        raw_img = raw_img.astype(np.uint8)
    elif raw_files:
        raw_img = np.array(Image.open(raw_files[0]).convert('L'))
    else:
        print("⚠️  Could not find image for visualization")
        return

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

    # Label each ROI using the same numbering as the ROI masks (cell_01, cell_02, ...)
    for region in measure.regionprops(combined_mask):
        if region.area == 0:
            continue
        y, x = region.centroid
        axes[1].text(
            x, y, f"{region.label:02d}",
            color="white", fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, linewidth=0)
        )

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
    visualize_segmentation(result['base_dir'], result['num_cells'], None)

    print("\n" + "="*60)
    print("✅ ALL DONE!")
    print("="*60)
    print(f"Segmented {result['num_cells']} cells")
    print(f"\nOutputs:")
    print(f"  • ROI masks: {result['roi_dir']}")
    print(f"  • Visualization: {result['base_dir']}/cellpose_segmentation_visualization.png")


if __name__ == "__main__":
    main()
