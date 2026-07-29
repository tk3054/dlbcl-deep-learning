#!/usr/bin/env python3
"""
Robust per-image normalization of TIF files to float32 [0, 1].

Each image is normalized independently using percentiles computed from
its own non-zero pixels, so zero-background pixels stay zero.
"""

from pathlib import Path

import numpy as np
import tifffile


def normalize_tif_folder(
    input_folder,
    output_folder="normalized_tif",
    lower_pct=1,
    upper_pct=99,
    verbose=True,
):
    """
    Normalize all TIF files in a folder to float32 [0, 1].

    Args:
        input_folder: Path to folder containing TIF files.
        output_folder: Name of output subfolder created inside input_folder.
        lower_pct: Lower percentile for robust min (computed on non-zero pixels).
        upper_pct: Upper percentile for robust max (computed on non-zero pixels).
        verbose: Print progress.

    Returns:
        dict with keys:
            - 'success': bool
            - 'output_path': str path to output folder
            - 'num_processed': int
            - 'num_skipped': int
            - 'error': str (only on failure)
    """
    input_path = Path(input_folder)
    output_path = input_path / output_folder
    output_path.mkdir(exist_ok=True)

    tif_files = sorted(
        list(input_path.glob("*.tif"))
        + list(input_path.glob("*.tiff"))
        + list(input_path.glob("*.TIF"))
        + list(input_path.glob("*.TIFF"))
    )

    if not tif_files:
        return {
            "success": False,
            "error": f"No TIF files found in {input_folder}",
            "output_path": str(output_path),
            "num_processed": 0,
            "num_skipped": 0,
        }

    if verbose:
        print(f"Normalizing {len(tif_files)} TIF files in {input_path.name}...")

    num_processed = 0
    num_skipped = 0

    for tif_path in tif_files:
        try:
            img = tifffile.imread(tif_path).astype(np.float32)

            non_zero_mask = img > 0
            non_zero_pixels = img[non_zero_mask]

            if len(non_zero_pixels) == 0:
                output_array = np.zeros_like(img, dtype=np.float32)
            else:
                robust_min = np.percentile(non_zero_pixels, lower_pct)
                robust_max = np.percentile(non_zero_pixels, upper_pct)

                output_array = np.zeros_like(img, dtype=np.float32)
                if robust_max > robust_min:
                    normalized = (non_zero_pixels - robust_min) / (robust_max - robust_min)
                    output_array[non_zero_mask] = np.clip(normalized, 0, 1)
                else:
                    output_array[non_zero_mask] = 0.5

            out_file = output_path / f"{tif_path.stem}_normalized.tif"
            tifffile.imwrite(out_file, output_array)
            num_processed += 1

        except Exception as e:
            if verbose:
                print(f"  Error processing {tif_path.name}: {e}")
            num_skipped += 1

    if verbose:
        print(f"  ✓ Done: {num_processed} normalized, {num_skipped} skipped → {output_path}")

    return {
        "success": True,
        "output_path": str(output_path),
        "num_processed": num_processed,
        "num_skipped": num_skipped,
    }


def normalize_donor_channels(donor_dir, channels=("actin", "ccr7", "cd45ra"), verbose=True):
    """
    Normalize the formatted channel folders for a donor.

    Expects folders named formatted_{channel}/ inside donor_dir.
    Writes normalized TIFs into formatted_{channel}/normalized_tif/.

    Args:
        donor_dir: Path to the donor directory.
        channels: Channel names to normalize.
        verbose: Print progress.

    Returns:
        dict mapping channel name → normalize_tif_folder result.
    """
    donor_dir = Path(donor_dir)
    results = {}

    for channel in channels:
        channel_folder = donor_dir / f"formatted_{channel}"
        if not channel_folder.exists():
            if verbose:
                print(f"  ⚠️  Missing channel folder: {channel_folder.name}")
            results[channel] = {
                "success": False,
                "error": f"Folder not found: {channel_folder}",
            }
            continue

        if verbose:
            print(f"\nChannel: {channel}")

        results[channel] = normalize_tif_folder(
            input_folder=channel_folder,
            verbose=verbose,
        )

    return results
