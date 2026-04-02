#!/usr/bin/env python3
"""
Generate hard-mask vs Gaussian soft-mask comparisons for the first cells in an image.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageOps

try:
    from process_image.soften_edges import get_mask_crop_bounds, make_soft_mask
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from process_image.soften_edges import get_mask_crop_bounds, make_soft_mask


SIGMAS = (1, 2, 3)
TILE_SIZE = 220
LABEL_HEIGHT = 24
PADDING = 12


def _find_source_image(base_dir):
    for candidate in ("Actin-FITC.tif", "processed_Actin-FITC.tif", "original_image.tif"):
        path = base_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No source image found in {base_dir}")


def _normalize_for_display(image):
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


def _to_tile(image_array):
    image = Image.fromarray(_normalize_for_display(image_array))
    return ImageOps.contain(image, (TILE_SIZE, TILE_SIZE))


def _draw_labeled_tile(canvas, tile, label, x0, y0):
    tile_x = x0 + (TILE_SIZE - tile.width) // 2
    tile_y = y0
    canvas.paste(tile, (tile_x, tile_y))
    draw = ImageDraw.Draw(canvas)
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    label_x = x0 + max(0, (TILE_SIZE - text_width) // 2)
    label_y = y0 + TILE_SIZE + 4
    draw.text((label_x, label_y), label, fill=0)


def _crop_to_mask(image, roi_mask, sigma=None):
    binary_mask = (roi_mask > 0).astype(np.uint8)
    crop_bounds = get_mask_crop_bounds(binary_mask, sigma=sigma)
    if crop_bounds is None:
        return None

    y_min, y_max, x_min, x_max = crop_bounds

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = binary_mask[y_min:y_max, x_min:x_max]
    return cropped_image, cropped_mask, (y_min, y_max, x_min, x_max)


def _save_comparison_grid(binary_mask, hard_masked, soft_masks, soft_masked_images, output_path):
    labels_top = ["Binary Mask"] + [f"Soft Mask s={sigma}" for sigma in SIGMAS]
    labels_bottom = ["Hard-Masked"] + [f"Soft-Masked s={sigma}" for sigma in SIGMAS]

    tiles_top = [_to_tile(binary_mask)] + [_to_tile(soft_masks[sigma]) for sigma in SIGMAS]
    tiles_bottom = [_to_tile(hard_masked)] + [_to_tile(soft_masked_images[sigma]) for sigma in SIGMAS]

    rows = [tiles_top, tiles_bottom]
    labels = [labels_top, labels_bottom]
    width = PADDING + len(rows[0]) * (TILE_SIZE + PADDING)
    height = PADDING + len(rows) * (TILE_SIZE + LABEL_HEIGHT + PADDING)
    canvas = Image.new("L", (width, height), color=255)

    for row_idx, row_tiles in enumerate(rows):
        y0 = PADDING + row_idx * (TILE_SIZE + LABEL_HEIGHT + PADDING)
        for col_idx, tile in enumerate(row_tiles):
            x0 = PADDING + col_idx * (TILE_SIZE + PADDING)
            _draw_labeled_tile(canvas, tile, labels[row_idx][col_idx], x0, y0)

    canvas.save(output_path)


def generate_soft_mask_comparisons(base_path, sample_folder="sample1", image_number="1", num_cells=5):
    base_dir = Path(base_path) / sample_folder / str(image_number)
    roi_dir = base_dir / "cell_rois"
    output_dir = base_dir / "soft_mask_comparisons"
    output_dir.mkdir(exist_ok=True)

    source_path = _find_source_image(base_dir)
    source_image = tifffile.imread(source_path)
    if source_image.ndim > 2:
        source_image = np.squeeze(source_image)
    if source_image.ndim == 3:
        source_image = source_image[..., 0]

    roi_files = sorted(roi_dir.glob("*.tif"))[:num_cells]
    if not roi_files:
        raise FileNotFoundError(f"No ROI masks found in {roi_dir}")

    saved_paths = []
    for roi_file in roi_files:
        roi_mask = tifffile.imread(roi_file)
        if roi_mask.ndim > 2:
            roi_mask = np.squeeze(roi_mask)
        if roi_mask.ndim == 3:
            roi_mask = roi_mask[..., 0]

        cropped = _crop_to_mask(source_image, roi_mask, sigma=max(SIGMAS))
        if cropped is None:
            continue

        cropped_image, binary_mask, (y_min, y_max, x_min, x_max) = cropped
        cropped_image = cropped_image.astype(np.float32)
        hard_masked = cropped_image * binary_mask.astype(np.float32)
        full_binary_mask = (roi_mask > 0).astype(np.uint8)
        soft_masks = {
            sigma: make_soft_mask(full_binary_mask, sigma=sigma)[y_min:y_max, x_min:x_max]
            for sigma in SIGMAS
        }
        soft_masked_images = {
            sigma: cropped_image * soft_masks[sigma] for sigma in SIGMAS
        }

        output_path = output_dir / f"{roi_file.stem}_soft_mask_comparison.png"
        _save_comparison_grid(binary_mask, hard_masked, soft_masks, soft_masked_images, output_path)
        saved_paths.append(output_path)

    return {
        "base_dir": str(base_dir),
        "source_image": str(source_path),
        "output_dir": str(output_dir),
        "saved_paths": [str(path) for path in saved_paths],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create soft-mask comparisons for the first cells in an image.")
    parser.add_argument("--base-path", required=True, help="Base directory containing sample folders")
    parser.add_argument("--sample-folder", default="sample1", help="Sample folder name")
    parser.add_argument("--image-number", default="1", help="Image number within sample")
    parser.add_argument("--num-cells", type=int, default=5, help="Number of cells to compare")
    args = parser.parse_args()

    result = generate_soft_mask_comparisons(
        base_path=args.base_path,
        sample_folder=args.sample_folder,
        image_number=args.image_number,
        num_cells=args.num_cells,
    )

    print(f"Source image: {result['source_image']}")
    print(f"Saved {len(result['saved_paths'])} figures to {result['output_dir']}")
