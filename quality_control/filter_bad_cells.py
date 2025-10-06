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
import cv2


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
    Analyze an ROI crop to detect if the cell is cut off by checking for
    consecutive non-transparent pixels along the cell perimeter

    Args:
        roi_mask_path: Path to the ROI crop file (PNG with alpha channel)
        consecutive_threshold: Minimum consecutive pixels to consider as cut

    Returns:
        dict with analysis results
    """
    # Load PNG with alpha channel
    img = Image.open(roi_mask_path)

    # Extract alpha channel (transparency mask)
    if img.mode == 'RGBA':
        roi_mask = np.array(img)[:, :, 3]  # Alpha channel
    else:
        # Fallback to grayscale if no alpha
        roi_mask = np.array(img.convert('L'))

    h, w = roi_mask.shape

    # Create binary mask
    binary_mask = (roi_mask > 0).astype(np.uint8)

    # Find contours of the cell (CHAIN_APPROX_NONE keeps all perimeter pixels)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        # No cell found
        return {
            'filename': os.path.basename(roi_mask_path),
            'cell_count': 0,
            'is_cut_off': False,
            'cut_off_edges': [],
            'edge_consecutive_pixels': {'top': 0, 'bottom': 0, 'left': 0, 'right': 0},
            'mask_shape': (h, w),
            'max_straight_edge': 0
        }

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Flatten contour to get list of (x, y) coordinates
    perimeter_points = contour.reshape(-1, 2)

    # Find maximum consecutive straight pixels along perimeter
    max_consecutive = 0
    current_consecutive = 1
    edge_info = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}

    # Check for straight lines by looking at consecutive points
    for i in range(1, len(perimeter_points)):
        prev_pt = perimeter_points[i - 1]
        curr_pt = perimeter_points[i]

        # Check if points form a straight line (either horizontal or vertical)
        if prev_pt[0] == curr_pt[0] or prev_pt[1] == curr_pt[1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)

            # Track which edge
            if prev_pt[1] == curr_pt[1]:  # Horizontal line
                if curr_pt[1] < h / 2:
                    edge_info['top'] = max(edge_info['top'], current_consecutive)
                else:
                    edge_info['bottom'] = max(edge_info['bottom'], current_consecutive)
            else:  # Vertical line
                if curr_pt[0] < w / 2:
                    edge_info['left'] = max(edge_info['left'], current_consecutive)
                else:
                    edge_info['right'] = max(edge_info['right'], current_consecutive)
        else:
            current_consecutive = 1

    # Determine which edges are cut
    is_cut_off = max_consecutive >= consecutive_threshold
    cut_off_edges = []

    for edge, length in edge_info.items():
        if length >= consecutive_threshold:
            cut_off_edges.append(edge)

    # Count cells
    labeled_mask = ndi.label(binary_mask)[0]
    cell_count = len(np.unique(labeled_mask)) - 1  # Subtract background

    return {
        'filename': os.path.basename(roi_mask_path),
        'cell_count': cell_count,
        'is_cut_off': is_cut_off,
        'cut_off_edges': cut_off_edges,
        'edge_consecutive_pixels': {k: int(v) for k, v in edge_info.items()},
        'mask_shape': (h, w),
        'max_straight_edge': int(max_consecutive)
    }


# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================

def filter_bad_cells(sample_folder, image_number, base_path, consecutive_threshold=50, verbose=True):
    """
    Analyze and filter cut-off cells.

    This is the main function to be called from notebooks or other scripts.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        consecutive_threshold: Threshold for cut detection (default: 50)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'results': List of analysis results for each cell
            - 'whole_count': Number of whole cells
            - 'cut_count': Number of cut cells
            - 'output_csv': Path to output CSV file
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
    """
    # Build paths
    base_dir = f"{base_path}/{sample_folder}/{image_number}"
    roi_crops_dir = f"{base_dir}/roi_crops_whiteBg"

    if verbose:
        print("="*60)
        print("BAD CELL FILTERING PIPELINE")
        print("="*60)
        print(f"Sample: {sample_folder}/{image_number}")
        print(f"Base directory: {base_dir}")
        print(f"Using crops from: roi_crops_whiteBg")
        print(f"Threshold: {consecutive_threshold} consecutive pixels\n")

    # Check directories exist
    roi_path = Path(roi_crops_dir)
    if not roi_path.exists():
        return {
            'success': False,
            'error': f"ROI crops directory not found: {roi_crops_dir}. Run Cell 5c first!",
            'results': [],
            'whole_count': 0,
            'cut_count': 0
        }

    # Get all PNG crop files
    roi_files = sorted(roi_path.glob("*.png"))
    if not roi_files:
        return {
            'success': False,
            'error': f"No PNG crop files found in {roi_crops_dir}",
            'results': [],
            'whole_count': 0,
            'cut_count': 0
        }

    if verbose:
        print(f"Found {len(roi_files)} ROI files to analyze\n")
        print("Analyzing ROI masks...")
        print("-" * 100)
        print(f"{'Filename':<20} {'Cells':<6} {'Status':<10} {'Cut Edges':<15} {'Edge Pixels (T/B/L/R)':<25}")
        print("-" * 100)

    # Analyze all ROIs
    results = []
    for roi_file in roi_files:
        result = analyze_roi_mask(roi_file, consecutive_threshold=consecutive_threshold)
        results.append(result)

        if verbose:
            status = "CUT" if result['is_cut_off'] else "WHOLE"
            edges = ", ".join(result['cut_off_edges']) if result['cut_off_edges'] else "none"
            edge_pixels = result['edge_consecutive_pixels']
            pixels_str = f"{edge_pixels['top']}/{edge_pixels['bottom']}/{edge_pixels['left']}/{edge_pixels['right']}"
            print(f"{result['filename']:<20} {result['cell_count']:<6} {status:<10} {edges:<15} {pixels_str:<25}")

    if verbose:
        print("-" * 100)

    # Summary statistics
    whole_count = sum(1 for r in results if not r['is_cut_off'])
    cut_count = len(results) - whole_count
    single_cell_count = sum(1 for r in results if r['cell_count'] == 1)

    if verbose:
        print(f"\nSummary: {whole_count} WHOLE, {cut_count} CUT, {single_cell_count} single-cell ROIs")
        print(f"Detection: Cells with {consecutive_threshold}+ consecutive white pixels on any border = CUT\n")
        print("Saving results...")

    # Save results to CSV
    df = pd.DataFrame(results)

    # Add binary quality column: 0 = good (not cut off), 1 = bad (cut off)
    df['quality_binary'] = df['is_cut_off'].astype(int)

    output_path = f"{base_dir}/raw_crops_quality_analysis.csv"
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"  Results saved to: {output_path}\n")
        print("Quality Summary:")
        print(f"  WHOLE cells: {whole_count}")
        print(f"  CUT cells: {cut_count}")
        print(f"  Single-cell ROIs: {single_cell_count}\n")
        print("="*60)
        print("FILTERING COMPLETE")
        print("="*60)

    return {
        'success': True,
        'results': results,
        'whole_count': whole_count,
        'cut_count': cut_count,
        'output_csv': output_path,
        'base_dir': base_dir
    }


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