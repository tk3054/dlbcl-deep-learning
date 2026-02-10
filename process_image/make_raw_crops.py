#!/usr/bin/env python3
"""
Create raw (unpadded, unmasked) crops using ROI masks for bounding boxes.
"""

from pathlib import Path
import numpy as np
from PIL import Image


def make_raw_crops(sample_folder, image_number, base_path,
                   source_image="Actin-FITC.tif",
                   roi_dir_name="cell_rois",
                   output_dir_name="raw_crops",
                   background="white",
                   verbose=True):
    """
    Create raw crops using ROI masks to define bounding boxes.

    Args:
        sample_folder: Sample folder name (e.g., "sample1")
        image_number: Image number within sample (e.g., "1")
        base_path: Base directory path (e.g., "/path/to/data")
        source_image: Source image filename (default: "Actin-FITC.tif")
        roi_dir_name: ROI directory name (default: "cell_rois")
        output_dir_name: Output directory name (default: "raw_crops")
        background: Background for non-cell pixels ("white", "black", or "transparent")
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'num_extracted': Number of crops extracted
            - 'output_dir': Path to output directory
    """
    base_dir = Path(f"{base_path}/{sample_folder}/{image_number}")
    roi_dir = base_dir / roi_dir_name
    source_path = base_dir / source_image
    output_dir = base_dir / output_dir_name

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

    output_dir.mkdir(exist_ok=True)

    source_img = np.array(Image.open(source_path))
    roi_files = sorted(roi_dir.glob("*.tif"))

    if verbose:
        print(f"Extracting raw crops from {source_image} -> {output_dir_name}/ ({len(roi_files)} cells)...")

    extracted_count = 0
    for roi_file in roi_files:
        roi_mask = np.array(Image.open(roi_file).convert('L'))
        coords = np.where(roi_mask > 0)

        if len(coords[0]) == 0:
            if verbose:
                print(f"  Skipping empty mask: {roi_file.name}")
            continue

        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1

        cropped_img = source_img[y_min:y_max, x_min:x_max]
        cropped_mask = roi_mask[y_min:y_max, x_min:x_max] > 0

        if background == "transparent":
            # For transparency, write RGBA with alpha from the mask
            if len(cropped_img.shape) == 2:
                rgb_crop = np.stack([cropped_img] * 3, axis=-1)
            else:
                rgb_crop = cropped_img

            if rgb_crop.dtype != np.uint8:
                img_min, img_max = rgb_crop.min(), rgb_crop.max()
                if img_max > img_min:
                    rgb_crop = ((rgb_crop.astype(float) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    rgb_crop = np.zeros_like(rgb_crop, dtype=np.uint8)

            alpha = (cropped_mask.astype(np.uint8) * 255)
            rgba = np.dstack([rgb_crop, alpha])
            output_img = Image.fromarray(rgba, mode="RGBA")
        else:
            if background == "white":
                if np.issubdtype(cropped_img.dtype, np.integer):
                    bg_value = np.iinfo(cropped_img.dtype).max
                else:
                    bg_value = 1.0
            else:
                bg_value = 0

            masked_crop = np.full_like(cropped_img, bg_value)
            masked_crop[cropped_mask] = cropped_img[cropped_mask]
            output_img = Image.fromarray(masked_crop)

        output_filename = roi_file.stem + "_raw.tif"
        output_path = output_dir / output_filename
        output_img.save(output_path)
        extracted_count += 1

    if verbose:
        print(f"  Extracted {extracted_count} raw crops")

    return {
        'success': True,
        'num_extracted': extracted_count,
        'output_dir': str(output_dir)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create raw crops from ROI masks.")
    parser.add_argument("--base-path", required=True, help="Base directory path")
    parser.add_argument("--sample-folder", required=True, help="Sample folder name")
    parser.add_argument("--image-number", required=True, help="Image number within sample")
    parser.add_argument("--source-image", default="Actin-FITC.tif", help="Source image filename")
    parser.add_argument("--roi-dir-name", default="cell_rois", help="ROI directory name")
    parser.add_argument("--output-dir-name", default="raw_crops", help="Output directory name")
    parser.add_argument("--background", default="white",
                        choices=["white", "black", "transparent"],
                        help="Background for non-cell pixels")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    args = parser.parse_args()

    result = make_raw_crops(
        sample_folder=args.sample_folder,
        image_number=args.image_number,
        base_path=args.base_path,
        source_image=args.source_image,
        roi_dir_name=args.roi_dir_name,
        output_dir_name=args.output_dir_name,
        background=args.background,
        verbose=not args.quiet
    )

    if result["success"]:
        print(f"Done: {result['num_extracted']} crops in {result['output_dir']}")
    else:
        print(f"Error: {result['error']}")
