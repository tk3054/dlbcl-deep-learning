#!/usr/bin/env python3
"""Convert TIFF images to 8-bit JPEG previews."""

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def _to_uint8(image):
    """Linearly scale an image's finite value range to uint8."""
    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image

    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)

    image_min = image[finite].min()
    image_max = image[finite].max()

    if image_max == image_min:
        fill_value = 0 if image_min == 0 else 255
        return np.full(image.shape, fill_value, dtype=np.uint8)

    scaled = np.zeros(image.shape, dtype=np.float32)
    scaled[finite] = (
        (image[finite].astype(np.float32) - image_min)
        * (255.0 / (image_max - image_min))
    )
    return np.clip(scaled, 0, 255).astype(np.uint8)


def convert_tif_to_jpg(input_folder, output_folder, quality=100, verbose=True):
    """
    Convert all TIFF files in a folder to JPEG.

    Each image is linearly scaled from its own value range to 8-bit [0, 255].
    For multi-page TIFFs, only the first page is converted.

    Args:
        input_folder: Folder containing TIFF images.
        output_folder: Folder in which JPEG images will be created.
        quality: JPEG quality from 1 to 100.
        verbose: Print progress.

    Returns:
        dict containing success, output_path, num_processed, and num_skipped.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.is_dir():
        return {
            "success": False,
            "error": f"Input folder not found: {input_path}",
            "output_path": str(output_path),
            "num_processed": 0,
            "num_skipped": 0,
        }

    if not 1 <= quality <= 100:
        return {
            "success": False,
            "error": "JPEG quality must be between 1 and 100.",
            "output_path": str(output_path),
            "num_processed": 0,
            "num_skipped": 0,
        }

    tif_files = sorted(
        list(input_path.glob("*.tif"))
        + list(input_path.glob("*.tiff"))
        + list(input_path.glob("*.TIF"))
        + list(input_path.glob("*.TIFF"))
    )

    if not tif_files:
        return {
            "success": False,
            "error": f"No TIFF files found in {input_path}",
            "output_path": str(output_path),
            "num_processed": 0,
            "num_skipped": 0,
        }

    output_path.mkdir(parents=True, exist_ok=True)
    num_processed = 0
    num_skipped = 0

    if verbose:
        print(f"Converting {len(tif_files)} TIFF files in {input_path.name}...")

    for tif_path in tif_files:
        try:
            with tifffile.TiffFile(tif_path) as tif:
                image = tif.pages[0].asarray()

            image_8bit = _to_uint8(image)

            if image_8bit.ndim == 2:
                jpg_image = Image.fromarray(image_8bit, mode="L").convert("RGB")
            elif image_8bit.ndim == 3 and image_8bit.shape[-1] in (3, 4):
                jpg_image = Image.fromarray(image_8bit)
                if jpg_image.mode == "RGBA":
                    jpg_image = jpg_image.convert("RGB")
            else:
                raise ValueError(f"Unsupported TIFF shape: {image_8bit.shape}")

            jpg_image.save(
                output_path / f"{tif_path.stem}.jpg",
                "JPEG",
                quality=quality,
                optimize=False,
            )
            num_processed += 1

        except Exception as exc:
            if verbose:
                print(f"  Error processing {tif_path.name}: {exc}")
            num_skipped += 1

    if verbose:
        print(
            f"  ✓ Done: {num_processed} converted, "
            f"{num_skipped} skipped → {output_path}"
        )

    return {
        "success": num_processed > 0,
        "output_path": str(output_path),
        "num_processed": num_processed,
        "num_skipped": num_skipped,
    }

