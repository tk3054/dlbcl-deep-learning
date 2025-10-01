#!/usr/bin/env python3
"""
Extract ROI Crops from Original Image
Extracts each cell from the original Actin-FITC image using ROI masks.
Output is the exact bounding box shape (not padded to square).
"""

import numpy as np
from PIL import Image
from pathlib import Path
import cv2


def extract_roi_crops(sample_folder, image_number, base_path, source_image="Actin-FITC.tif", output_dir_name="roi_crops", use_transparency=False, white_background=False, roi_dir_name=None, verbose=True):
    """
    Extract cell crops from original image using ROI masks.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        source_image: Source image filename (default: "Actin-FITC.tif")
        output_dir_name: Output directory name (default: "roi_crops")
        use_transparency: If True, save as PNG with transparent background (default: False)
        roi_dir_name: ROI directory name (default: auto-detect - prefers Cellpose)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'num_extracted': Number of cells extracted
            - 'output_dir': Path to output directory
    """
    # Build paths
    base_dir = Path(f"{base_path}/{sample_folder}/{image_number}")

    if roi_dir_name is None:
        roi_dir_name = "cell_rois"

    roi_dir = base_dir / roi_dir_name
    source_path = base_dir / source_image
    output_dir = base_dir / output_dir_name

    if verbose:
        print("="*60)
        print("EXTRACT ROI CROPS FROM ORIGINAL IMAGE")
        print("="*60)
        print(f"Sample: {sample_folder}/{image_number}")
        print(f"Source image: {source_image}")
        print(f"Output directory: {output_dir}")
        print("="*60)

    # Check if directories/files exist
    if not roi_dir.exists():
        return {
            'success': False,
            'error': f"ROI directory not found: {roi_dir}",
            'num_extracted': 0,
            'output_dir': str(output_dir)
        }

    if not source_path.exists():
        return {
            'success': False,
            'error': f"Source image not found: {source_path}",
            'num_extracted': 0,
            'output_dir': str(output_dir)
        }

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Load source image
    if verbose:
        print(f"\nLoading source image: {source_path.name}")

    source_img = np.array(Image.open(source_path))

    if verbose:
        print(f"  Image shape: {source_img.shape}")
        print(f"  Image dtype: {source_img.dtype}")

    # Get all ROI mask files
    roi_files = sorted(roi_dir.glob("*.tif"))

    if verbose:
        print(f"\nFound {len(roi_files)} ROI masks to process\n")

    # Process each ROI
    extracted_count = 0

    for roi_file in roi_files:
        if verbose:
            print(f"Processing {roi_file.name}...")

        # Load ROI mask
        roi_mask = np.array(Image.open(roi_file).convert('L'))

        # Find bounding box of the mask
        # Get coordinates where mask is white (non-zero)
        coords = np.where(roi_mask > 0)

        if len(coords[0]) == 0:
            if verbose:
                print(f"  ⚠️  Empty mask, skipping")
            continue

        # Get bounding box
        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1

        # Crop the source image to bounding box
        cropped_img = source_img[y_min:y_max, x_min:x_max]

        # Crop the mask to same bounding box
        cropped_mask = roi_mask[y_min:y_max, x_min:x_max]

        if use_transparency:
            # Create image with transparent background
            # Convert grayscale to RGB if needed
            if len(cropped_img.shape) == 2:
                rgb_crop = np.stack([cropped_img] * 3, axis=-1)
            else:
                rgb_crop = cropped_img

            # Normalize to 8-bit if needed for better visualization
            if rgb_crop.dtype == np.uint16:
                # Scale 16-bit to 8-bit for visualization
                rgb_crop = (rgb_crop / 256).astype(np.uint8)

            # Create RGBA image
            rgba_crop = np.zeros((*rgb_crop.shape[:2], 4), dtype=np.uint8)
            rgba_crop[:, :, :3] = rgb_crop  # RGB channels
            rgba_crop[:, :, 3] = (cropped_mask > 0).astype(np.uint8) * 255  # Alpha channel

            # Save as PNG with transparency (for viewing only - not for analysis!)
            output_filename = roi_file.stem + "_crop.png"
            output_path = output_dir / output_filename
            Image.fromarray(rgba_crop, mode='RGBA').save(output_path)
        else:
            # Apply mask to image (set background to 0)
            masked_crop = cropped_img.copy()
            masked_crop[cropped_mask == 0] = 0

            # Save as TIFF to preserve dynamic range
            output_filename = roi_file.stem + "_crop.tif"
            output_path = output_dir / output_filename
            Image.fromarray(masked_crop).save(output_path)

        if verbose:
            if use_transparency:
                print(f"  ✓ Saved: {output_filename} (shape: {rgba_crop.shape})")
            else:
                print(f"  ✓ Saved: {output_filename} (shape: {masked_crop.shape})")

        extracted_count += 1

    if verbose:
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Extracted {extracted_count} cell crops")
        print(f"Output directory: {output_dir}")

    return {
        'success': True,
        'num_extracted': extracted_count,
        'output_dir': str(output_dir)
    }


if __name__ == "__main__":
    # Configuration
    SAMPLE_FOLDER = "sample1"
    IMAGE_NUMBER = "5"
    BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"

    result = extract_roi_crops(
        sample_folder=SAMPLE_FOLDER,
        image_number=IMAGE_NUMBER,
        base_path=BASE_PATH,
        verbose=True
    )

    if result['success']:
        print(f"\n✓ Successfully extracted {result['num_extracted']} cells")
    else:
        print(f"\n✗ Error: {result['error']}")
