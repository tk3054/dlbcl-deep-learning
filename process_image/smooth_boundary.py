#!/usr/bin/env python3
"""
Visualize Gaussian-smoothed ROI weighting and centered 224x224 crops.
"""

from pathlib import Path
import argparse

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

SAMPLE_FOLDER = "sample1"
IMAGE_NUMBER = "1"
BASE_PATH = '/Users/taeeonkong/Desktop/DL Project/responder/01-06-2026 DLBCL 118867'

PARAMS = {
    "source_image": "processed_Actin-FITC.tif",
    "roi_dir_name": "cell_rois",
    "output_dir_name": "visualize smoothing",
    "sigma_values": [1, 2, 3, 4, 5],
    "target_size": 224,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_centered_crop(image, center_y, center_x, target_size, fill_value=0.0):
    """
    Extract a fixed-size crop centered on the requested point.

    Pads with fill_value when the crop extends beyond the image boundary.
    """
    target_size = int(target_size)
    half_size = target_size // 2

    y_start = int(round(center_y)) - half_size
    x_start = int(round(center_x)) - half_size
    y_end = y_start + target_size
    x_end = x_start + target_size

    src_y_start = max(0, y_start)
    src_x_start = max(0, x_start)
    src_y_end = min(image.shape[0], y_end)
    src_x_end = min(image.shape[1], x_end)

    dst_y_start = src_y_start - y_start
    dst_x_start = src_x_start - x_start
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    crop = np.full((target_size, target_size), fill_value, dtype=image.dtype)
    crop[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[src_y_start:src_y_end, src_x_start:src_x_end]
    return crop


def get_centered_crop_bounds(center_y, center_x, target_size):
    half_size = int(target_size) // 2
    y_start = int(round(center_y)) - half_size
    x_start = int(round(center_x)) - half_size
    y_end = y_start + int(target_size)
    x_end = x_start + int(target_size)
    return y_start, y_end, x_start, x_end


def _normalize_to_uint8(image):
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    elif max_value > 0:
        image = image / max_value

    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def _make_labeled_tile(image_array, label, tile_size=220, label_height=24):
    image = Image.fromarray(_normalize_to_uint8(image_array))
    image = ImageOps.contain(image, (tile_size, tile_size))

    canvas = Image.new("L", (tile_size, tile_size + label_height), color=255)
    x_offset = (tile_size - image.width) // 2
    canvas.paste(image, (x_offset, 0))

    draw = ImageDraw.Draw(canvas)
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    draw.text(((tile_size - text_width) // 2, tile_size + 4), label, fill=0)
    return canvas


def _make_operator_tile(symbol, tile_size=90, label_height=24):
    canvas = Image.new("RGB", (tile_size, tile_size + label_height), color="white")
    draw = ImageDraw.Draw(canvas)
    text_bbox = draw.textbbox((0, 0), symbol)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.text(
        ((tile_size - text_width) // 2, (tile_size - text_height) // 2 - 4),
        symbol,
        fill="black",
    )
    return canvas


def _make_crop_box_tile(image_array, crop_bounds, label, tile_size=220, label_height=24):
    image = Image.fromarray(_normalize_to_uint8(image_array)).convert("RGB")
    original_width, original_height = image.size
    image = ImageOps.contain(image, (tile_size, tile_size))

    scale_x = image.width / original_width
    scale_y = image.height / original_height
    y_start, y_end, x_start, x_end = crop_bounds

    draw = ImageDraw.Draw(image)
    left = max(0, x_start) * scale_x
    top = max(0, y_start) * scale_y
    right = min(original_width, x_end) * scale_x
    bottom = min(original_height, y_end) * scale_y
    draw.rectangle([left, top, right, bottom], outline=(255, 64, 64), width=3)

    canvas = Image.new("RGB", (tile_size, tile_size + label_height), color="white")
    x_offset = (tile_size - image.width) // 2
    canvas.paste(image, (x_offset, 0))

    draw = ImageDraw.Draw(canvas)
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    draw.text(((tile_size - text_width) // 2, tile_size + 4), label, fill="black")
    return canvas


def _save_process_schematic(
    cell_dir,
    source_img,
    soft_mask,
    weighted_image,
    crop_bounds,
    final_crop,
    sigma,
    padded,
):
    tiles = [
        _make_labeled_tile(soft_mask, f"blurred ROI (sigma={sigma:g})"),
        _make_operator_tile("x"),
        _make_labeled_tile(source_img, "processed Actin"),
        _make_operator_tile("="),
        _make_crop_box_tile(weighted_image, crop_bounds, "weighted image + 224 crop box"),
        _make_operator_tile("->"),
        _make_labeled_tile(final_crop, f"final 224x224 crop{' (padded)' if padded else ''}"),
    ]

    padding = 12
    tile_widths = [tile.width for tile in tiles]
    tile_heights = [tile.height for tile in tiles]
    canvas_width = padding + sum(tile_widths) + padding * len(tiles)
    canvas_height = max(tile_heights) + (2 * padding)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")

    x_offset = padding
    for tile in tiles:
        y_offset = padding + (max(tile_heights) - tile.height) // 2
        canvas.paste(tile, (x_offset, y_offset))
        x_offset += tile.width + padding

    sigma_label = str(int(sigma)) if float(sigma).is_integer() else str(sigma).replace(".", "p")
    canvas.save(cell_dir / f"sigma_{sigma_label}.png")


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_smooth_boundary(sample_folder, image_number, base_path, params=None, verbose=True):
    """
    Generate one smoothing schematic per cell for one image.

    Args:
        sample_folder: Sample folder name (e.g., "sample1")
        image_number: Image number (e.g., "1")
        base_path: Base directory path
        params: Optional dictionary of processing parameters
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean
            - 'error': Error message if failed
            - 'num_processed': Number of processed cells
            - 'results': Dict of output paths and settings
    """
    if params is None:
        params = PARAMS.copy()
    else:
        params = {**PARAMS, **params}

    source_image = params["source_image"]
    roi_dir_name = params["roi_dir_name"]
    output_dir_name = params["output_dir_name"]
    sigma_values = [float(sigma) for sigma in params["sigma_values"]]
    target_size = int(params["target_size"])

    base_dir = Path(base_path) / sample_folder / str(image_number)
    roi_dir = base_dir / roi_dir_name
    source_path = base_dir / source_image
    output_dir = base_dir / output_dir_name

    results = {
        "base_dir": str(base_dir),
        "roi_dir": str(roi_dir),
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "sigma_values": sigma_values,
        "target_size": target_size,
    }

    if not roi_dir.exists():
        return {
            "success": False,
            "error": f"ROI directory not found: {roi_dir}",
            "num_processed": 0,
            "results": results,
        }

    if not source_path.exists():
        return {
            "success": False,
            "error": f"Source image not found: {source_path}",
            "num_processed": 0,
            "results": results,
        }

    output_dir.mkdir(exist_ok=True)

    source_img = tifffile.imread(source_path)
    if source_img.ndim != 2:
        raise ValueError(f"Expected 2D source image, got shape {source_img.shape}")

    roi_files = sorted(roi_dir.glob("*.tif"))

    processed_count = 0
    for cell_index, roi_file in enumerate(roi_files, start=1):
        roi_mask = tifffile.imread(roi_file)
        if roi_mask.ndim != 2:
            raise ValueError(f"Expected 2D ROI mask for {roi_file.name}, got shape {roi_mask.shape}")
        binary_mask = roi_mask > 0
        coords = np.where(binary_mask)
        if len(coords[0]) == 0:
            continue

        center_y = (float(coords[0].min()) + float(coords[0].max())) / 2.0
        center_x = (float(coords[1].min()) + float(coords[1].max())) / 2.0

        cell_dir = output_dir / f"cell_{cell_index}"
        cell_dir.mkdir(exist_ok=True)

        for sigma in sigma_values:
            # Build the blurred ROI on the full-size mask, weight the full image,
            # then extract a fixed-size crop centered on the cell.
            soft_mask = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma, radius=7)
            weighted_image = source_img.astype(np.float32) * soft_mask
            crop_bounds = get_centered_crop_bounds(center_y, center_x, target_size)
            padded = (
                crop_bounds[0] < 0
                or crop_bounds[2] < 0
                or crop_bounds[1] > source_img.shape[0]
                or crop_bounds[3] > source_img.shape[1]
            )

            smoothed_crop = extract_centered_crop(
                weighted_image,
                center_y=center_y,
                center_x=center_x,
                target_size=target_size,
                fill_value=0.0,
            )

            _save_process_schematic(
                cell_dir=cell_dir,
                source_img=source_img,
                soft_mask=soft_mask,
                weighted_image=weighted_image,
                crop_bounds=crop_bounds,
                final_crop=smoothed_crop,
                sigma=sigma,
                padded=padded,
            )
        processed_count += 1


    return {
        "success": True,
        "error": None,
        "num_processed": processed_count,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create smoothing schematics for per-cell crops.")
    parser.add_argument("--base-path", default=BASE_PATH, help="Base directory path")
    parser.add_argument("--sample-folder", default=SAMPLE_FOLDER, help="Sample folder name")
    parser.add_argument("--image-number", default=IMAGE_NUMBER, help="Image number within sample")
    parser.add_argument("--source-image", default=PARAMS["source_image"], help="Source image filename")
    parser.add_argument("--roi-dir-name", default=PARAMS["roi_dir_name"], help="Input ROI directory name")
    parser.add_argument("--output-dir-name", default=PARAMS["output_dir_name"], help="Output visualization directory")
    parser.add_argument("--sigma-start", type=int, default=int(PARAMS["sigma_values"][0]), help="First sigma value")
    parser.add_argument("--sigma-end", type=int, default=int(PARAMS["sigma_values"][-1]), help="Last sigma value")
    parser.add_argument("--target-size", type=int, default=PARAMS["target_size"], help="Fixed centered crop size")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    args = parser.parse_args()

    params = {
        "source_image": args.source_image,
        "roi_dir_name": args.roi_dir_name,
        "output_dir_name": args.output_dir_name,
        "sigma_values": list(range(args.sigma_start, args.sigma_end + 1)),
        "target_size": args.target_size,
    }

    result = run_smooth_boundary(
        sample_folder=args.sample_folder,
        image_number=args.image_number,
        base_path=args.base_path,
        params=params,
        verbose=not args.quiet,
    )

    if result["success"]:
        print(f"Done: {result['num_processed']} cells")
    else:
        print(f"Error: {result['error']}")


# Backward-compatible alias after renaming the file/module.
run_smooth_segmentation = run_smooth_boundary
