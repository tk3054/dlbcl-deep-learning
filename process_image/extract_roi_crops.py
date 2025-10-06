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


def extract_roi_crops(sample_folder, image_number, base_path, source_image="Actin-FITC.tif", output_dir_name="roi_crops", use_transparency=False, background_color=0, roi_dir_name=None, pad_to_size=255, save_tif_blackbg=True, verbose=True):
    """
    Extract cell crops from original image using ROI masks.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        source_image: Source image filename (default: "Actin-FITC.tif")
        output_dir_name: Output directory name (default: "roi_crops")
        use_transparency: If True, save as PNG with background_color padding (default: False)
        background_color: Background color for transparency mode, 0=black, 255=white (default: 0)
        roi_dir_name: ROI directory name (default: auto-detect - prefers Cellpose)
        pad_to_size: Pad crops to this size (default: 255, set to None to disable padding)
        save_tif_blackbg: If True, also save TIF versions with black background (default: True)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'num_extracted': Number of cells extracted
            - 'output_dir': Path to output directory
            - 'tif_output_dir': Path to TIF output directory (if save_tif_blackbg=True)
    """
    # Build paths
    base_dir = Path(f"{base_path}/{sample_folder}/{image_number}")

    if roi_dir_name is None:
        roi_dir_name = "cell_rois"

    roi_dir = base_dir / roi_dir_name
    source_path = base_dir / source_image
    output_dir = base_dir / output_dir_name

    # Create TIF black bg output directory if requested
    if save_tif_blackbg:
        tif_output_dir = base_dir / "tif_roi_crops_blackbg"
        tif_output_dir.mkdir(exist_ok=True)
    else:
        tif_output_dir = None

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
    source_img = np.array(Image.open(source_path))

    # Get all ROI mask files
    roi_files = sorted(roi_dir.glob("*.tif"))

    if verbose:
        print(f"Extracting ROI crops from {source_image} → {output_dir_name}/ ({len(roi_files)} cells)...")

    # Process each ROI
    extracted_count = 0

    for roi_file in roi_files:
        # Load ROI mask
        roi_mask = np.array(Image.open(roi_file).convert('L'))

        # Find bounding box of the mask
        # Get coordinates where mask is white (non-zero)
        coords = np.where(roi_mask > 0)

        if len(coords[0]) == 0:
            if verbose:
                print(f"  ⚠️  Empty mask in {roi_file.name}, skipping")
            continue

        # Get bounding box
        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1

        # Crop the source image to bounding box
        cropped_img = source_img[y_min:y_max, x_min:x_max]

        # Crop the mask to same bounding box
        cropped_mask = roi_mask[y_min:y_max, x_min:x_max]

        # Pad to square size if requested
        if pad_to_size is not None:
            h, w = cropped_img.shape[:2]
            if h > pad_to_size or w > pad_to_size:
                if verbose:
                    print(f"  ⚠️  Warning: {roi_file.name} is {w}x{h}, larger than {pad_to_size}x{pad_to_size}. Skipping padding.")
            else:
                # Calculate padding
                pad_top = (pad_to_size - h) // 2
                pad_bottom = pad_to_size - h - pad_top
                pad_left = (pad_to_size - w) // 2
                pad_right = pad_to_size - w - pad_left

                # Pad the cropped image with black (0)
                if len(cropped_img.shape) == 2:
                    cropped_img = np.pad(cropped_img, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)
                else:
                    cropped_img = np.pad(cropped_img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=0)

                # Pad the mask as well
                cropped_mask = np.pad(cropped_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)

        if use_transparency:
            # Create image with background color (white or black)
            # Convert grayscale to RGB if needed
            if len(cropped_img.shape) == 2:
                rgb_crop = np.stack([cropped_img] * 3, axis=-1)
            else:
                rgb_crop = cropped_img

            # Normalize to 8-bit if needed for better visualization
            if rgb_crop.dtype == np.uint16:
                # Scale 16-bit to 8-bit for visualization (preserve intensity distribution)
                img_min, img_max = rgb_crop.min(), rgb_crop.max()
                if img_max > img_min:
                    rgb_crop = ((rgb_crop.astype(float) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    rgb_crop = np.zeros_like(rgb_crop, dtype=np.uint8)
            elif rgb_crop.dtype != np.uint8:
                rgb_crop = rgb_crop.astype(np.uint8)

            # Create RGB image filled with background color
            bg_crop = np.full((*rgb_crop.shape[:2], 3), background_color, dtype=np.uint8)

            # Copy cell pixels onto background using mask
            mask_3d = np.stack([cropped_mask > 0] * 3, axis=-1)
            bg_crop[mask_3d] = rgb_crop[mask_3d]

            # Save as PNG (for viewing only - not for analysis!)
            output_filename = roi_file.stem + "_crop.png"
            output_path = output_dir / output_filename
            Image.fromarray(bg_crop, mode='RGB').save(output_path)
        else:
            # Apply mask to image (set background to 0)
            masked_crop = cropped_img.copy()
            masked_crop[cropped_mask == 0] = 0

            # Save as TIFF to preserve dynamic range
            output_filename = roi_file.stem + "_crop.tif"
            output_path = output_dir / output_filename
            Image.fromarray(masked_crop).save(output_path)

        # Also save TIF with black background if requested
        if save_tif_blackbg:
            # Apply mask to image (set background to 0)
            tif_masked_crop = cropped_img.copy()
            tif_masked_crop[cropped_mask == 0] = 0

            # Save as TIFF to preserve dynamic range
            tif_output_filename = roi_file.stem + "_crop.tif"
            tif_output_path = tif_output_dir / tif_output_filename
            Image.fromarray(tif_masked_crop).save(tif_output_path)

        extracted_count += 1

    if verbose:
        print(f"  ✓ Extracted {extracted_count} crops")
        if save_tif_blackbg:
            print(f"  ✓ Also saved TIF versions in tif_roi_crops_blackbg/")

    result = {
        'success': True,
        'num_extracted': extracted_count,
        'output_dir': str(output_dir)
    }

    if save_tif_blackbg:
        result['tif_output_dir'] = str(tif_output_dir)

    return result


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
