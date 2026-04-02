#!/usr/bin/env python3
"""
Temporary helper to compare Cellpose segmentations across multiple
cellprob thresholds.
"""

from pathlib import Path
import sys

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageOps
from skimage import measure
from skimage.segmentation import find_boundaries

try:
    import segmentation.segment_cells_cellpose as segment_cells_cellpose
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import segmentation.segment_cells_cellpose as segment_cells_cellpose


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

SAMPLE_FOLDER = "sample1"
IMAGE_NUMBER = "1"
BASE_PATH = '/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241'

MODEL_TYPE = "cyto2"
DIAMETER = 250
CELLPROB_THRESHOLD = -2.0
USE_GPU = True
MIN_SIZE = 180
MAX_SIZE = 100000

FLOW_THRESHOLD = 0.6
CELLPROB_THRESHOLDS = [-2.0, -4.0, -6.0, -8.0, -10.0]
CHANNEL_CONFIG = {"actin": "Actin-FITC.tif"}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _normalize_to_uint8(image):
    image = np.asarray(image, dtype=np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    elif max_value > 0:
        image = image / max_value
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def _load_input_image(base_dir, channel_config):
    actin_filename = channel_config.get("actin", "Actin-FITC.tif")
    actin_tif = base_dir / actin_filename
    raw_files = list(base_dir.glob("*_raw.jpg"))

    if actin_tif.exists():
        image_path = actin_tif
    elif raw_files:
        image_path = raw_files[0]
    else:
        raise FileNotFoundError(
            f"No image found in {base_dir} (looked for {actin_filename} or *_raw.jpg)"
        )

    image = segment_cells_cellpose.load_image(str(image_path))
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")
    return image


def _load_optional_mask(base_dir):
    mask_files = list(base_dir.glob("*_mask.jpg"))
    if not mask_files:
        return None
    return segment_cells_cellpose.load_image(str(mask_files[0]))


def _make_overlay(raw_img_uint8, labeled_mask):
    boundaries = find_boundaries(labeled_mask, mode="inner")
    overlay = np.stack([raw_img_uint8] * 3, axis=-1)
    overlay[boundaries] = [255, 0, 0]
    return overlay


def _make_labeled_panel(overlay, cellprob_threshold, num_cells):
    panel = Image.fromarray(overlay)
    panel = ImageOps.contain(panel, (420, 420))

    canvas = Image.new("RGB", (440, 480), color="white")
    canvas.paste(panel, ((440 - panel.width) // 2, 10))

    draw = ImageDraw.Draw(canvas)
    title = f"cellprob={cellprob_threshold:g}"
    subtitle = f"{num_cells} cells"
    draw.text((20, 438), title, fill="black")
    draw.text((20, 458), subtitle, fill="black")
    return canvas


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def compare_cellprob_thresholds(
    sample_folder,
    image_number,
    base_path,
    cellprob_thresholds=None,
    model_type=MODEL_TYPE,
    diameter=DIAMETER,
    flow_threshold=FLOW_THRESHOLD,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    use_gpu=USE_GPU,
    channel_config=None,
    verbose=True,
):
    """
    Run Cellpose segmentation across multiple cellprob thresholds and save a single
    side-by-side comparison image.
    """
    if cellprob_thresholds is None:
        cellprob_thresholds = CELLPROB_THRESHOLDS
    if channel_config is None:
        channel_config = CHANNEL_CONFIG.copy()

    base_dir = Path(base_path) / sample_folder / str(image_number)
    output_dir = base_dir / "cellpose_flow_comparisons"
    output_dir.mkdir(exist_ok=True)

    image = _load_input_image(base_dir, channel_config)
    mask = _load_optional_mask(base_dir)
    image_uint8 = _normalize_to_uint8(image)

    if verbose:
        print(f"Comparing Cellpose cellprob thresholds for {sample_folder}/{image_number}")
        print(f"Cellprob thresholds: {cellprob_thresholds}")

    panels = []
    results = []
    for cellprob_threshold in cellprob_thresholds:
        if verbose:
            print(f"  Running cellprob_threshold={cellprob_threshold}")

        regions, _, labeled_mask, _ = segment_cells_cellpose.segment_cells_with_cellpose(
            image=image,
            mask=mask,
            model_type=model_type,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            max_size=max_size,
            use_gpu=use_gpu,
            verbose=False,
        )

        overlay = _make_overlay(image_uint8, labeled_mask)
        panels.append(_make_labeled_panel(overlay, cellprob_threshold, len(regions)))
        results.append({
            "cellprob_threshold": cellprob_threshold,
            "num_cells": len(regions),
        })

    panel_width, panel_height = panels[0].size
    padding = 16
    canvas_width = padding + len(panels) * (panel_width + padding)
    canvas_height = panel_height + (2 * padding)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")

    for idx, panel in enumerate(panels):
        x_offset = padding + idx * (panel_width + padding)
        canvas.paste(panel, (x_offset, padding))

    output_path = output_dir / "cellpose_cellprob_threshold_comparison.png"
    canvas.save(output_path)

    if verbose:
        print(f"Saved comparison: {output_path}")

    return {
        "success": True,
        "output_path": str(output_path),
        "output_dir": str(output_dir),
        "results": results,
    }


if __name__ == "__main__":
    result = compare_cellprob_thresholds(
        sample_folder=SAMPLE_FOLDER,
        image_number=IMAGE_NUMBER,
        base_path=BASE_PATH,
    )

    if result["success"]:
        print(f"Done: {result['output_path']}")
