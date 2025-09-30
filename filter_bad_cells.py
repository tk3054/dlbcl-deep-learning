#!/usr/bin/env python3
"""
Filter Bad Cells Script
Converts filter_bad_images.ipynb to standalone Python script

Analyzes ROI masks to detect cut-off cells using straight-line detection
(consecutive white pixels along borders indicate cutting)

Usage:
    python run_filter_bad_cells.py
    (Edit SAMPLE_FOLDER and IMAGE_NUMBER below to change inputs)
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from pathlib import Path


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Change these values to process different samples (must match segmentation script)
SAMPLE_FOLDER = "sample1"  # Options: "sample1", "sample2", "sample3"
IMAGE_NUMBER = "3"         # Options: "1", "2", "3", "4", etc.

# Detection parameters
CONSECUTIVE_THRESHOLD = 50  # Number of consecutive white pixels = cut


# ============================================================================
# AUTO-GENERATED PATHS (do not edit)
# ============================================================================

BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"
BASE_DIR = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class FilterConfig:
    """Configuration for bad cell filtering"""
    def __init__(self):
        # Use global configuration
        self.base_dir = BASE_DIR
        self.roi_dir = f"{self.base_dir}/cell_rois"
        self.raw_crops_dir = f"{self.base_dir}/raw_crops"
        self.consecutive_threshold = CONSECUTIVE_THRESHOLD


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def find_consecutive_white_pixels(border_pixels):
    """
    Find the maximum number of consecutive white (non-zero) pixels in a border
    """
    max_consecutive = 0
    current_consecutive = 0

    for pixel in border_pixels:
        if pixel > 0:  # White pixel
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive


def analyze_roi_mask(roi_mask_path, consecutive_threshold=50):
    """
    Analyze an ROI mask to detect if the cell is cut off by checking for
    straight lines of white pixels along borders (the signature of cropping)

    Args:
        roi_mask_path: Path to the ROI mask file
        consecutive_threshold: Number of consecutive white pixels to consider as cut

    Returns:
        dict with analysis results
    """
    # Load ROI mask
    roi_mask = np.array(Image.open(roi_mask_path).convert('L'))
    h, w = roi_mask.shape

    # Extract border pixels
    top_border = roi_mask[0, :]
    bottom_border = roi_mask[-1, :]
    left_border = roi_mask[:, 0]
    right_border = roi_mask[:, -1]

    # Check for consecutive white pixels on each border
    edge_info = {
        'top': find_consecutive_white_pixels(top_border),
        'bottom': find_consecutive_white_pixels(bottom_border),
        'left': find_consecutive_white_pixels(left_border),
        'right': find_consecutive_white_pixels(right_border)
    }

    # Determine which edges are cut
    is_cut_off = False
    cut_off_edges = []

    for edge, consecutive_count in edge_info.items():
        if consecutive_count >= consecutive_threshold:
            is_cut_off = True
            cut_off_edges.append(edge)

    # Count cells (should be 1 in ROI mask, but check anyway)
    binary_mask = roi_mask > 0
    labeled_mask = ndi.label(binary_mask)[0]
    cell_count = len(np.unique(labeled_mask)) - 1  # Subtract background

    return {
        'filename': os.path.basename(roi_mask_path),
        'cell_count': cell_count,
        'is_cut_off': is_cut_off,
        'cut_off_edges': cut_off_edges,
        'edge_consecutive_pixels': edge_info,
        'mask_shape': (h, w)
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_filtering(config):
    """Execute the complete filtering pipeline"""

    print("="*60)
    print("BAD CELL FILTERING PIPELINE")
    print("="*60)
    print(f"Base directory: {config.base_dir}")
    print(f"Threshold: {config.consecutive_threshold} consecutive pixels\n")

    # Check directories exist
    roi_path = Path(config.roi_dir)
    if not roi_path.exists():
        print(f"❌ Error: ROI directory not found: {config.roi_dir}")
        print("   Please run run_cell_segmentation.py first!")
        sys.exit(1)

    # Get all ROI files
    roi_files = sorted(roi_path.glob("*.tif"))
    if not roi_files:
        print(f"❌ Error: No ROI files found in {config.roi_dir}")
        sys.exit(1)

    print(f"Found {len(roi_files)} ROI files to analyze\n")

    # Analyze all ROIs
    print("Analyzing ROI masks...")
    print("-" * 100)
    print(f"{'Filename':<20} {'Cells':<6} {'Status':<10} {'Cut Edges':<15} {'Edge Pixels (T/B/L/R)':<25}")
    print("-" * 100)

    results = []
    for roi_file in roi_files:
        result = analyze_roi_mask(roi_file, consecutive_threshold=config.consecutive_threshold)
        results.append(result)

        status = "CUT" if result['is_cut_off'] else "WHOLE"
        edges = ", ".join(result['cut_off_edges']) if result['cut_off_edges'] else "none"

        # Show consecutive pixel counts for each edge
        edge_pixels = result['edge_consecutive_pixels']
        pixels_str = f"{edge_pixels['top']}/{edge_pixels['bottom']}/{edge_pixels['left']}/{edge_pixels['right']}"

        print(f"{result['filename']:<20} {result['cell_count']:<6} {status:<10} {edges:<15} {pixels_str:<25}")

    print("-" * 100)

    # Summary statistics
    whole_count = sum(1 for r in results if not r['is_cut_off'])
    cut_count = len(results) - whole_count
    single_cell_count = sum(1 for r in results if r['cell_count'] == 1)

    print(f"\nSummary: {whole_count} WHOLE, {cut_count} CUT, {single_cell_count} single-cell ROIs")
    print(f"Detection: Cells with {config.consecutive_threshold}+ consecutive white pixels on any border = CUT\n")

    # Save results to CSV
    print("Saving results...")
    df = pd.DataFrame(results)
    output_path = f"{config.base_dir}/raw_crops_quality_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Results saved to: {output_path}\n")

    # Display quality summary
    print("Quality Summary:")
    print(f"  WHOLE cells: {whole_count}")
    print(f"  CUT cells: {cut_count}")
    print(f"  Single-cell ROIs: {single_cell_count}\n")

    print("="*60)
    print("✅ FILTERING COMPLETE")
    print("="*60)
    print("\n📋 Next step:")
    print(f"  Run: python manual_cell_reviewer.py --sample {os.path.basename(config.base_dir)}\n")


def main():
    # Create configuration using global variables
    config = FilterConfig()

    # Run pipeline
    try:
        run_filtering(config)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()