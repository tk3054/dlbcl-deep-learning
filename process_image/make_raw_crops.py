#!/usr/bin/env python3
"""
Create raw (unpadded, unmasked) crops using ROI masks for bounding boxes.
"""

from pathlib import Path
import numpy as np
import tifffile


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
        background: Background mode for non-cell pixels
            ("white", "black", "transparent", or "original")
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

    source_img = tifffile.imread(source_path)
    if source_img.ndim > 3:
        source_img = np.squeeze(source_img)
    roi_files = sorted(roi_dir.glob("*.tif"))

    if verbose:
        print(f"Extracting raw crops from {source_image} -> {output_dir_name}/ ({len(roi_files)} cells)...")

    extracted_count = 0
    for roi_file in roi_files:
        roi_mask = tifffile.imread(roi_file)
        if roi_mask.ndim > 2:
            roi_mask = np.squeeze(roi_mask)
        if roi_mask.ndim == 3:
            roi_mask = roi_mask[..., 0]
        coords = np.where(roi_mask > 0)

        if len(coords[0]) == 0:
            if verbose:
                print(f"  Skipping empty mask: {roi_file.name}")
            continue

        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1

        cropped_img = source_img[y_min:y_max, x_min:x_max]
        cropped_mask = roi_mask[y_min:y_max, x_min:x_max] > 0

        if background == "original":
            # Exact bounding-box crop with original pixel values untouched.
            output_arr = cropped_img
        elif background == "transparent":
            # Keep grayscale sources single-channel so ImageJ does not split RGB.
            if len(cropped_img.shape) == 2 or (len(cropped_img.shape) == 3 and cropped_img.shape[2] == 1):
                masked_crop = np.zeros_like(cropped_img)
                masked_crop[cropped_mask] = cropped_img[cropped_mask]
                output_arr = masked_crop
            else:
                # For true color sources, keep RGB values and add alpha.
                rgb_crop = cropped_img[:, :, :3].copy()
                rgb_crop[~cropped_mask] = 0

                if np.issubdtype(rgb_crop.dtype, np.integer):
                    alpha_max = np.iinfo(rgb_crop.dtype).max
                else:
                    alpha_max = 1.0
                alpha = np.where(cropped_mask, alpha_max, 0).astype(rgb_crop.dtype)
                output_arr = np.dstack([rgb_crop, alpha])
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
            output_arr = masked_crop

        output_filename = roi_file.stem + "_raw.tif"
        output_path = output_dir / output_filename
        tifffile.imwrite(output_path, output_arr)
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
                        choices=["white", "black", "transparent", "original"],
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
