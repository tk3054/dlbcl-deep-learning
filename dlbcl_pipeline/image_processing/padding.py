#!/usr/bin/env python3
"""Pad variable-size cell crops to fixed square images for classifier input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def _resize_nearest(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resize with nearest-neighbor sampling without adding another dependency."""
    height, width = image.shape[:2]
    if height == target_height and width == target_width:
        return image

    y_idx = np.linspace(0, height - 1, target_height).round().astype(int)
    x_idx = np.linspace(0, width - 1, target_width).round().astype(int)
    return image[np.ix_(y_idx, x_idx)]


def pad_to_square(image: np.ndarray, target_size: int = 224) -> np.ndarray | None:
    """Center-pad a crop to target_size x target_size, resizing down if needed."""
    if image.ndim < 2:
        return None

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return None

    if height > target_size or width > target_size:
        scale = min(target_size / height, target_size / width)
        resized_h = max(1, int(round(height * scale)))
        resized_w = max(1, int(round(width * scale)))
        image = _resize_nearest(image, resized_h, resized_w)
        height, width = image.shape[:2]

    pad_h = target_size - height
    pad_w = target_size - width
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    pad_spec = ((top, bottom), (left, right))
    if image.ndim > 2:
        pad_spec += tuple((0, 0) for _ in image.shape[2:])

    return np.pad(image, pad_spec, mode="constant", constant_values=0)


def _padded_name(crop_path: Path) -> str:
    stem = crop_path.stem
    if stem.endswith("_raw"):
        stem = stem[:-len("_raw")]
    return f"{stem}_padded{crop_path.suffix}"


def pad_crop_folder(
    sample_folder: str,
    image_number: str,
    base_path: str,
    target_size: int = 224,
    input_dir_name: str = "raw_actin",
    output_dir_name: str = "padded_cells",
    verbose: bool = True,
) -> dict[str, object]:
    """Create fixed-size padded cell images from variable-size crop files."""
    base_dir = Path(base_path) / sample_folder / str(image_number)
    input_dir = base_dir / input_dir_name
    output_dir = base_dir / output_dir_name

    if not input_dir.exists():
        return {
            "success": False,
            "error": f"Crop directory not found: {input_dir}",
            "num_padded": 0,
            "output_dir": str(output_dir),
        }

    crop_files = sorted(input_dir.glob("*.tif"))
    if not crop_files:
        return {
            "success": False,
            "error": f"No crop files found in: {input_dir}",
            "num_padded": 0,
            "output_dir": str(output_dir),
        }

    output_dir.mkdir(exist_ok=True)

    padded_count = 0
    for crop_file in crop_files:
        crop = tifffile.imread(crop_file)
        if crop.ndim > 3:
            crop = np.squeeze(crop)
        if crop.ndim == 3 and crop.shape[-1] == 1:
            crop = crop[..., 0]

        padded = pad_to_square(crop, target_size=target_size)
        if padded is None:
            if verbose:
                print(f"  Skipping invalid crop: {crop_file.name}")
            continue

        output_path = output_dir / _padded_name(crop_file)
        tifffile.imwrite(output_path, padded)
        padded_count += 1

    if verbose:
        print(f"  Padded {padded_count}/{len(crop_files)} crops to {target_size}x{target_size}")

    return {
        "success": True,
        "num_padded": padded_count,
        "output_dir": str(output_dir),
    }
