#!/usr/bin/env python3
"""
Pad Masked Cells to Square Images
Extracts cells using ROI masks and pads to square format (64x64)

This script uses the exact cell mask shapes (from cell_rois/) to extract
only the cell pixels from the original image, then pads to a square.

Usage:
    python pad_raw_crops.py
    (Edit SAMPLE_FOLDER and IMAGE_NUMBER below to change inputs)
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from skimage import io
from PIL import Image


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Change these values to process different samples
SAMPLE_FOLDER = "sample1"  # Options: "sample1", "sample2", "sample3"
IMAGE_NUMBER = "4"         # Options: "1", "2", "3", "4", etc.

# Padding parameters
TARGET_SIZE = 64  # Size of square output images


# ============================================================================
# AUTO-GENERATED PATHS (do not edit)
# ============================================================================

BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"
BASE_DIR = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"
ROI_DIR = f"{BASE_DIR}/cell_rois"
ORIGINAL_IMAGE = f"{BASE_DIR}/original_image.tif"
OUTPUT_DIR = f"{BASE_DIR}/padded_cells"


# ============================================================================
# EXTRACTION AND PADDING FUNCTIONS
# ============================================================================

def extract_masked_cell(original_image, roi_mask):
    """
    Extract cell from original image using ROI mask.
    Returns only the masked region cropped to bounding box.
    """
    # Convert mask to binary
    binary_mask = roi_mask > 0

    # Find bounding box of the mask
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if not rows.any() or not cols.any():
        return None, None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box
    cropped_image = original_image[rmin:rmax+1, cmin:cmax+1].copy()
    cropped_mask = binary_mask[rmin:rmax+1, cmin:cmax+1]

    # Apply mask - set non-cell pixels to black
    masked_cell = cropped_image.copy()
    masked_cell[~cropped_mask] = 0

    return masked_cell, cropped_mask


def pad_to_square(image, target_size):
    """
    Pad grayscale image to a square of size (target_size x target_size),
    preserving aspect ratio and centering the cell.
    """
    h, w = image.shape

    # Skip invalid images
    if h == 0 or w == 0:
        return None

    # Calculate padding
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        image,
        top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=0  # black padding
    )

    # Resize to (target_size, target_size) only if it exceeds
    if padded.shape[0] > target_size or padded.shape[1] > target_size:
        padded = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return padded


# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================

def pad_masked_cells(sample_folder, image_number, base_path, target_size=64, verbose=True):
    """
    Extract masked cells and pad to square format.

    This is the main function to be called from notebooks or other scripts.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        target_size: Size of square output (default: 64)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'num_padded': Number of cells padded
            - 'output_dir': Path to output directory
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
    """
    # Build paths
    base_dir = f"{base_path}/{sample_folder}/{image_number}"
    roi_dir = f"{base_dir}/cell_rois"
    original_image_path = f"{base_dir}/original_image.tif"
    output_dir = f"{base_dir}/padded_cells"

    if verbose:
        print("="*60)
        print("PAD MASKED CELLS TO SQUARE")
        print("="*60)
        print(f"Sample: {sample_folder}/{image_number}")
        print(f"ROI directory: {roi_dir}")
        print(f"Original image: {original_image_path}")
        print(f"Output directory: {output_dir}")
        print(f"Target size: {target_size}x{target_size}\n")

    # Check directories exist
    if not os.path.exists(roi_dir):
        return {
            'success': False,
            'error': f"ROI directory not found: {roi_dir}",
            'num_padded': 0
        }

    if not os.path.exists(original_image_path):
        return {
            'success': False,
            'error': f"Original image not found: {original_image_path}",
            'num_padded': 0
        }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load original image
    if verbose:
        print("Loading original image...")
    original_image = io.imread(original_image_path)
    if len(original_image.shape) == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    if verbose:
        print(f"  Loaded original image: {original_image.shape}\n")

    # Get all ROI mask files
    roi_path = Path(roi_dir)
    roi_files = sorted(roi_path.glob("*.tif"))

    if not roi_files:
        return {
            'success': False,
            'error': f"No .tif files found in {roi_dir}",
            'num_padded': 0
        }

    if verbose:
        print(f"Found {len(roi_files)} ROI masks to process\n")
        print("Processing...")

    # Process each ROI
    success_count = 0
    for roi_file in roi_files:
        # Load ROI mask
        roi_mask = io.imread(roi_file)
        if len(roi_mask.shape) == 3:
            roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_RGB2GRAY)

        # Extract masked cell from original image
        masked_cell, mask = extract_masked_cell(original_image, roi_mask)

        if masked_cell is None:
            if verbose:
                print(f"  WARNING: Skipping invalid ROI: {roi_file.name}")
            continue

        # Pad to square
        padded = pad_to_square(masked_cell, target_size)

        if padded is None:
            if verbose:
                print(f"  WARNING: Skipping invalid padded result: {roi_file.name}")
            continue

        # Normalize if needed
        if padded.dtype != np.uint8:
            padded = cv2.normalize(padded, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Save padded image
        output_path = os.path.join(output_dir, roi_file.name)
        io.imsave(output_path, padded)
        success_count += 1

    if verbose:
        print(f"\n  Successfully padded {success_count}/{len(roi_files)} cells")
        print(f"  Output saved to: {output_dir}\n")
        print("="*60)
        print("PADDING COMPLETE")
        print("="*60)

    return {
        'success': True,
        'num_padded': success_count,
        'output_dir': output_dir,
        'base_dir': base_dir
    }


def pad_masked_cells_standalone():
    """Standalone version using global config - for backward compatibility"""

    print("="*60)
    print("PAD MASKED CELLS TO SQUARE")
    print("="*60)
    print(f"ROI directory: {ROI_DIR}")
    print(f"Original image: {ORIGINAL_IMAGE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}\n")

    # Check directories exist
    if not os.path.exists(ROI_DIR):
        print(f"ERROR: ROI directory not found: {ROI_DIR}")
        print("   Please run segment_cells.py first!")
        sys.exit(1)

    if not os.path.exists(ORIGINAL_IMAGE):
        print(f"ERROR: Original image not found: {ORIGINAL_IMAGE}")
        print("   Please run segment_cells.py first!")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load original image
    print("Loading original image...")
    original_image = io.imread(ORIGINAL_IMAGE)
    if len(original_image.shape) == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    print(f"Loaded original image: {original_image.shape}\n")

    # Get all ROI mask files
    roi_path = Path(ROI_DIR)
    roi_files = sorted(roi_path.glob("*.tif"))

    if not roi_files:
        print(f"ERROR: No .tif files found in {ROI_DIR}")
        sys.exit(1)

    print(f"Found {len(roi_files)} ROI masks to process\n")
    print("Processing...")

    # Process each ROI
    success_count = 0
    for roi_file in roi_files:
        # Load ROI mask
        roi_mask = io.imread(roi_file)
        if len(roi_mask.shape) == 3:
            roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_RGB2GRAY)

        # Extract masked cell from original image
        masked_cell, mask = extract_masked_cell(original_image, roi_mask)

        if masked_cell is None:
            print(f"WARNING: Skipping invalid ROI: {roi_file.name}")
            continue

        # Pad to square
        padded = pad_to_square(masked_cell, TARGET_SIZE)

        if padded is None:
            print(f"WARNING: Skipping invalid padded result: {roi_file.name}")
            continue

        # Normalize if needed
        if padded.dtype != np.uint8:
            padded = cv2.normalize(padded, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Save padded image
        output_path = os.path.join(OUTPUT_DIR, roi_file.name)
        io.imsave(output_path, padded)
        success_count += 1

    print(f"\nSuccessfully padded {success_count}/{len(roi_files)} cells")
    print(f"Output saved to: {OUTPUT_DIR}\n")

    print("="*60)
    print("PADDING COMPLETE")
    print("="*60)


def main():
    try:
        pad_masked_cells_standalone()
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
