#!/usr/bin/env python3
"""
Cell Segmentation Pipeline Script
Converts cell_segmentation_pipeline.ipynb to standalone Python script

Performs watershed segmentation on manually masked images and exports:
- Individual cell ROIs as binary masks
- Raw cell crops for quality filtering

Usage:
    python run_cell_segmentation.py
    (Edit SAMPLE_FOLDER and IMAGE_NUMBER below to change inputs)
"""

import os
import sys
import numpy as np
from PIL import Image
from skimage import measure, io
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Change these values to process different samples
SAMPLE_FOLDER = "sample1"  # Options: "sample1", "sample2", "sample3"
IMAGE_NUMBER = "3"         # Options: "1", "2", "3", "4", etc.

# Segmentation parameters
MIN_DISTANCE = 12
MIN_SIZE = 500
MAX_SIZE = 9000


# ============================================================================
# AUTO-GENERATED PATHS (do not edit)
# ============================================================================

BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"
BASE_DIR = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class SegmentationConfig:
    """Configuration for cell segmentation"""
    def __init__(self):
        # Use global configuration
        self.base_dir = BASE_DIR

        # Find raw and mask files automatically
        self.raw_filename = self._find_file_pattern("*_raw.jpg")
        self.mask_filename = self._find_file_pattern("*_mask.jpg")

        # Segmentation parameters
        self.MIN_DISTANCE = MIN_DISTANCE
        self.MIN_SIZE = MIN_SIZE
        self.MAX_SIZE = MAX_SIZE

    def _find_file_pattern(self, pattern):
        """Find first file matching pattern in base_dir"""
        from pathlib import Path
        matches = list(Path(self.base_dir).glob(pattern))
        if matches:
            return matches[0].name
        return None


# ============================================================================
# SEGMENTATION FUNCTIONS
# ============================================================================

def load_image(image_path):
    """Load an image as grayscale numpy array"""
    return np.array(Image.open(image_path).convert('L'))


def split_touching_cells(mask, min_distance=12):
    """Use distance transform + watershed to split touching cells"""
    distance = ndi.distance_transform_edt(mask)

    # Get coordinates of local maxima
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask)

    # Create marker image from coordinates
    local_max = np.zeros_like(distance, dtype=bool)
    local_max[tuple(coords.T)] = True
    markers = ndi.label(local_max)[0]

    # Apply watershed
    labels = watershed(-distance, markers, mask=mask)
    return labels


def segment_cells_from_mask(image, mask, min_distance=12, min_size=500, max_size=9000):
    """Segment individual cells from a mask using watershed"""
    binary_mask = mask > 0

    # Split overlapping cells using watershed
    labeled_mask = split_touching_cells(binary_mask, min_distance=min_distance)

    regions = measure.regionprops(labeled_mask, intensity_image=image)

    # Filter by size
    valid_regions = []
    valid_bboxes = []

    for region in regions:
        if min_size <= region.area <= max_size:
            valid_regions.append(region)

            # Convert bbox to (x, y, w, h) format
            minr, minc, maxr, maxc = region.bbox
            x, y = minc, minr
            w, h = maxc - minc, maxr - minr
            valid_bboxes.append((x, y, w, h))

    return valid_regions, valid_bboxes, labeled_mask


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_cell_rois(image, labeled_mask, regions, base_dir):
    """Export individual cell ROIs as binary masks for ImageJ"""
    roi_dir = os.path.join(base_dir, 'cell_rois')
    os.makedirs(roi_dir, exist_ok=True)

    for i, region in enumerate(regions):
        # Create binary mask for this single cell
        single_cell_mask = (labeled_mask == region.label).astype(np.uint8) * 255

        # Save each cell as separate binary image
        roi_path = os.path.join(roi_dir, f'cell_{i+1:02d}.tif')

        # Suppress low contrast warnings
        import warnings
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
# MAIN PIPELINE
# ============================================================================

def run_segmentation(config):
    """Execute the complete segmentation pipeline"""

    print("="*60)
    print("CELL SEGMENTATION PIPELINE")
    print("="*60)
    print(f"Base directory: {config.base_dir}\n")

    # Check files exist
    raw_path = os.path.join(config.base_dir, config.raw_filename)
    mask_path = os.path.join(config.base_dir, config.mask_filename)

    if not os.path.exists(raw_path):
        print(f"❌ Error: Raw image not found: {raw_path}")
        sys.exit(1)
    if not os.path.exists(mask_path):
        print(f"❌ Error: Mask image not found: {mask_path}")
        sys.exit(1)

    # Load images
    print("Loading images...")
    print(f"  Raw: {config.raw_filename}")
    print(f"  Mask: {config.mask_filename}")

    image = load_image(raw_path)
    mask = load_image(mask_path)

    print(f"✅ Loaded image: {image.shape}")
    print(f"✅ Loaded mask: {mask.shape}\n")

    # Perform segmentation
    print("Performing watershed segmentation...")
    print(f"  Parameters: min_distance={config.MIN_DISTANCE}, min_size={config.MIN_SIZE}, max_size={config.MAX_SIZE}")

    regions, bboxes, labeled_mask = segment_cells_from_mask(
        image, mask,
        min_distance=config.MIN_DISTANCE,
        min_size=config.MIN_SIZE,
        max_size=config.MAX_SIZE
    )

    print(f"✅ Found {len(regions)} valid cells\n")

    # Export ROIs
    print("Exporting cell ROIs...")
    roi_dir = export_cell_rois(image, labeled_mask, regions, config.base_dir)
    print(f"✅ Exported {len(regions)} cell ROIs to: {roi_dir}\n")

    # Export raw crops
    print("Exporting raw crops...")
    raw_crops_dir = export_raw_crops(image, bboxes, config.base_dir)
    print(f"✅ Exported {len(bboxes)} raw crops to: {raw_crops_dir}\n")

    print("="*60)
    print("✅ SEGMENTATION COMPLETE")
    print("="*60)
    print("\n📋 Next steps:")
    print("  1. Run: python pad_raw_crops.py")
    print("  2. Run: python filter_bad_cells.py")
    print("  3. Run: python manual_cell_reviewer.py")
    print("  4. Run ImageJ macro: load_rois.ijm")
    print("  5. Run ImageJ macro: extract_channels.ijm\n")


def main():
    # Create configuration using global variables
    config = SegmentationConfig()

    # Run pipeline
    try:
        run_segmentation(config)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()