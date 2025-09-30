#!/usr/bin/env python3
"""
Pad Raw Crops to Square Images
Converts existing raw crop images to padded square format (64x64)

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


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Change these values to process different samples
SAMPLE_FOLDER = "sample1"  # Options: "sample1", "sample2", "sample3"
IMAGE_NUMBER = "3"         # Options: "1", "2", "3", "4", etc.

# Padding parameters
TARGET_SIZE = 64  # Size of square output images


# ============================================================================
# AUTO-GENERATED PATHS (do not edit)
# ============================================================================

BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"
BASE_DIR = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"
RAW_CROPS_DIR = f"{BASE_DIR}/raw_crops"
OUTPUT_DIR = f"{BASE_DIR}/padded_cells"


# ============================================================================
# PADDING FUNCTION
# ============================================================================

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
# MAIN PIPELINE
# ============================================================================

def pad_raw_crops():
    """Pad all raw crops to square format"""

    print("="*60)
    print("PAD RAW CROPS TO SQUARE")
    print("="*60)
    print(f"Input directory: {RAW_CROPS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}\n")

    # Check input directory exists
    if not os.path.exists(RAW_CROPS_DIR):
        print(f"ERROR: Raw crops directory not found: {RAW_CROPS_DIR}")
        print("   Please run segment_cells.py first!")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all raw crop files
    raw_crops_path = Path(RAW_CROPS_DIR)
    crop_files = sorted(raw_crops_path.glob("*.tif"))

    if not crop_files:
        print(f"ERROR: No .tif files found in {RAW_CROPS_DIR}")
        sys.exit(1)

    print(f"Found {len(crop_files)} raw crop files to process\n")
    print("Processing...")

    # Process each crop
    success_count = 0
    for crop_file in crop_files:
        # Load image
        image = io.imread(crop_file)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Pad to square
        padded = pad_to_square(image, TARGET_SIZE)

        if padded is None:
            print(f"WARNING: Skipping invalid crop: {crop_file.name}")
            continue

        # Normalize if needed
        if padded.dtype != np.uint8:
            padded = cv2.normalize(padded, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Save padded image
        output_path = os.path.join(OUTPUT_DIR, crop_file.name)
        io.imsave(output_path, padded)
        success_count += 1

    print(f"\nSuccessfully padded {success_count}/{len(crop_files)} crops")
    print(f"Output saved to: {OUTPUT_DIR}\n")

    print("="*60)
    print("PADDING COMPLETE")
    print("="*60)


def main():
    try:
        pad_raw_crops()
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
