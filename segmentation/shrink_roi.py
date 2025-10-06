#!/usr/bin/env python3
"""
Shrink ROI Masks
Applies morphological erosion to shrink ROI masks by a specified number of pixels.
This helps avoid edge artifacts and improves signal quality.

Usage:
    python shrink_roi.py
    (Edit SAMPLE_FOLDER, IMAGE_NUMBER, and SHRINK_PIXELS below)
"""

import os
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import io
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import find_best_image_for_visualization


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Change these values to process different samples
SAMPLE_FOLDER = "sample1"  # Options: "sample1", "sample2", etc.
IMAGE_NUMBER = "2[CAR]"    # Options: "1", "2", "3", etc.

# Shrink amount (in pixels)
SHRINK_PIXELS = 10          # How many pixels to shrink inward

# Auto-generated paths
BASE_PATH = "/Users/taeeonkong/Desktop/2025 Fall Images/09-26-2025 DLBCL"
BASE_DIR = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"


# ============================================================================
# SHRINKING FUNCTIONS
# ============================================================================

def shrink_roi_mask(mask, shrink_pixels):
    """
    Shrink a binary ROI mask using morphological erosion.

    Args:
        mask: Binary mask (numpy array)
        shrink_pixels: Number of pixels to erode inward

    Returns:
        Shrunken binary mask
    """
    if shrink_pixels <= 0:
        return mask

    # Convert to binary
    binary_mask = mask > 0

    # Create circular structuring element
    # This ensures uniform shrinking in all directions
    y, x = np.ogrid[-shrink_pixels:shrink_pixels+1, -shrink_pixels:shrink_pixels+1]
    structuring_element = x**2 + y**2 <= shrink_pixels**2

    # Apply erosion
    eroded_mask = ndi.binary_erosion(binary_mask, structure=structuring_element)

    # Convert back to uint8 (0-255)
    return (eroded_mask * 255).astype(np.uint8)


def shrink_all_rois(base_dir, shrink_pixels, output_suffix="_shrunk", verbose=True):
    """
    Shrink all ROI masks in a directory.

    Args:
        base_dir: Base directory containing cell_rois folder
        shrink_pixels: Number of pixels to shrink
        output_suffix: Suffix for output directory name
        verbose: Print progress

    Returns:
        dict with keys:
            - 'success': Boolean
            - 'num_rois': Number of ROIs processed
            - 'output_dir': Path to output directory
            - 'error': Error message if failed
    """
    # Input and output directories
    roi_dir = os.path.join(base_dir, 'cell_rois')
    output_dir = os.path.join(base_dir, f'cell_rois{output_suffix}')

    # Check if input directory exists
    if not os.path.exists(roi_dir):
        return {
            'success': False,
            'error': f"ROI directory not found: {roi_dir}",
            'num_rois': 0
        }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of ROI files
    roi_files = sorted([f for f in os.listdir(roi_dir) if f.endswith('.tif')])

    if len(roi_files) == 0:
        return {
            'success': False,
            'error': f"No .tif files found in {roi_dir}",
            'num_rois': 0
        }

    if verbose:
        print(f"Shrinking {len(roi_files)} ROI masks by {shrink_pixels} pixels...")

    # Process each ROI
    processed_count = 0
    for roi_file in roi_files:
        # Load ROI mask
        input_path = os.path.join(roi_dir, roi_file)
        mask = np.array(Image.open(input_path).convert('L'))

        # Shrink the mask
        shrunken_mask = shrink_roi_mask(mask, shrink_pixels)

        # Skip if mask disappeared completely
        if np.sum(shrunken_mask) == 0:
            if verbose:
                print(f"  �  Skipping {roi_file}: mask too small, disappeared after shrinking")
            continue

        # Save shrunken mask
        output_path = os.path.join(output_dir, roi_file)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(output_path, shrunken_mask)

        processed_count += 1

    if verbose:
        print(f"Saved {processed_count} shrunken ROI masks to: {output_dir}")
        if processed_count < len(roi_files):
            print(f"{len(roi_files) - processed_count} masks were too small and were skipped")

    return {
        'success': True,
        'num_rois': processed_count,
        'output_dir': output_dir
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_roi_comparison(base_dir, image_number, verbose=True):
    """
    Create visualization comparing original vs shrunken ROIs.

    Args:
        base_dir: Base directory containing ROI folders and original image
        image_number: Image number (e.g., "1[CAR]", "2[CAR]")
        verbose: Print progress

    Returns:
        Path to saved visualization
    """
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from skimage import morphology

    base_path = Path(base_dir)

    # Use utility function to find the best image
    image_path = find_best_image_for_visualization(base_dir, image_number, verbose=verbose)

    if image_path is None:
        if verbose:
            print("Could not find image for visualization")
        return None

    # Load the image
    if image_path.suffix.lower() == '.tif':
        # Load TIF with full bit depth
        import cv2
        raw_img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            # Fallback to PIL
            raw_img = np.array(Image.open(image_path).convert('L'))
        else:
            # Normalize to 8-bit for display if it's 16-bit
            if raw_img.dtype == np.uint16:
                raw_img = raw_img.astype(float)
                img_min, img_max = raw_img.min(), raw_img.max()
                if img_max > img_min:
                    raw_img = (raw_img - img_min) / (img_max - img_min) * 255
                raw_img = raw_img.astype(np.uint8)
    else:
        # Load JPG/PNG with PIL
        raw_img = np.array(Image.open(image_path).convert('L'))

    # Load original ROIs
    original_roi_dir = base_path / "cell_rois"
    shrunken_roi_dir = base_path / "cell_rois_shrunk"

    if not original_roi_dir.exists() or not shrunken_roi_dir.exists():
        if verbose:
            print("⚠️  ROI directories not found for visualization")
        return None

    # Combine original ROIs
    original_roi_files = sorted(original_roi_dir.glob("*.tif"))
    original_combined = np.zeros_like(raw_img)
    for i, roi_file in enumerate(original_roi_files):
        roi_mask = np.array(Image.open(roi_file).convert('L'))
        pixels_in_roi = np.sum(roi_mask > 0)
        original_combined[roi_mask > 0] = i + 1
        if verbose:
            print(f"  Original {roi_file.name}: {pixels_in_roi} pixels, label {i+1}")

    # Combine shrunken ROIs
    shrunken_roi_files = sorted(shrunken_roi_dir.glob("*.tif"))
    shrunken_combined = np.zeros_like(raw_img)
    for i, roi_file in enumerate(shrunken_roi_files):
        roi_mask = np.array(Image.open(roi_file).convert('L'))
        pixels_in_roi = np.sum(roi_mask > 0)
        shrunken_combined[roi_mask > 0] = i + 1
        if verbose:
            print(f"  Shrunken {roi_file.name}: {pixels_in_roi} pixels, label {i+1}")

    # Check which labels are actually in the combined masks
    if verbose:
        original_labels = np.unique(original_combined)[1:]  # Exclude 0
        shrunken_labels = np.unique(shrunken_combined)[1:]  # Exclude 0
        print(f"\nOriginal combined mask has labels: {original_labels.tolist()}")
        print(f"Shrunken combined mask has labels: {shrunken_labels.tolist()}")
        if len(original_labels) < len(original_roi_files):
            print(f"  WARNING: Expected {len(original_roi_files)} labels but only have {len(original_labels)} - ROIs are overlapping!")

    # Create boundaries for each label individually
    original_boundaries = np.zeros_like(raw_img, dtype=bool)
    for label in original_labels:
        mask = (original_combined == label)
        eroded = morphology.binary_erosion(mask)
        boundary = mask & ~eroded
        original_boundaries |= boundary

    shrunken_boundaries = np.zeros_like(raw_img, dtype=bool)
    for label in shrunken_labels:
        mask = (shrunken_combined == label)
        eroded = morphology.binary_erosion(mask)
        boundary = mask & ~eroded
        shrunken_boundaries |= boundary

    # Create figure
    plt.figure(figsize=(16, 16))

    # Original ROIs
    plt.subplot(2, 2, 1)
    plt.imshow(raw_img, cmap='gray')
    # Overlay red boundaries
    overlay = np.zeros((*raw_img.shape, 4))
    overlay[original_boundaries] = [1, 0, 0, 1]  # Red
    plt.imshow(overlay)
    plt.title(f'Original ROIs - Red ({len(original_labels)} ROIs)', fontsize=14)
    plt.axis('off')

    # Shrunken ROIs
    plt.subplot(2, 2, 2)
    plt.imshow(raw_img, cmap='gray')
    # Overlay green boundaries
    overlay = np.zeros((*raw_img.shape, 4))
    overlay[shrunken_boundaries] = [0, 1, 0, 1]  # Green
    plt.imshow(overlay)
    plt.title(f'Shrunken ROIs - Green ({len(shrunken_labels)} ROIs)', fontsize=14)
    plt.axis('off')

    # Both overlaid
    plt.subplot(2, 2, 3)
    plt.imshow(raw_img, cmap='gray')
    # Overlay both boundaries
    overlay = np.zeros((*raw_img.shape, 4))
    overlay[original_boundaries] = [1, 0, 0, 1]   # Red
    overlay[shrunken_boundaries] = [0, 1, 0, 1]   # Green
    plt.imshow(overlay)
    plt.title('Both Overlaid (Red=Original, Green=Shrunken)', fontsize=14)
    plt.axis('off')

    # Side-by-side masks
    plt.subplot(2, 2, 4)
    plt.imshow(raw_img, cmap='gray')
    plt.imshow(original_combined, cmap='Reds', alpha=0.3, vmin=0, vmax=len(original_roi_files))
    plt.imshow(shrunken_combined, cmap='Greens', alpha=0.3, vmin=0, vmax=len(shrunken_roi_files))
    plt.title('Mask Comparison (Red=Original, Green=Shrunken)', fontsize=14)
    plt.axis('off')

    plt.tight_layout()

    # Save figure
    output_path = base_path / "roi_shrink_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"✓ Visualization saved: {output_path}")

    return str(output_path)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run ROI shrinking with configuration from globals"""
    import sys

    print("\n" + "="*60)
    print("ROI SHRINKING")
    print("="*60)
    print(f"Sample: {SAMPLE_FOLDER}/{IMAGE_NUMBER}")
    print(f"Shrink amount: {SHRINK_PIXELS} pixels")
    print("="*60 + "\n")

    result = shrink_all_rois(
        base_dir=BASE_DIR,
        shrink_pixels=SHRINK_PIXELS,
        output_suffix="_shrunk",
        verbose=True
    )

    if not result['success']:
        print(f"\nERROR: Shrinking failed: {result['error']}")
        sys.exit(1)

    # Create visualization
    print("\nCreating visualization...")
    viz_path = visualize_roi_comparison(BASE_DIR, IMAGE_NUMBER, verbose=True)

    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print(f"Processed {result['num_rois']} ROIs")
    print(f"Output: {result['output_dir']}")
    if viz_path:
        print(f"Visualization: {viz_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
