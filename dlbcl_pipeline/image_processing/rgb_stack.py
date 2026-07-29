#!/usr/bin/env python3
"""
Stack three already-normalized single-channel TIFs into a float32 RGB TIF.

Channel mapping: R=CCR7, G=Actin, B=CD45RA
Reads from formatted_{channel}/normalized_tif/ inside the donor directory.
Channel values are preserved when stacking. Outputs float32 RGB TIFs and
8-bit JPG previews.
"""

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


_CHANNEL_MAP = {
    "r": "ccr7",
    "g": "actin",
    "b": "cd45ra",
}

_NORMALIZED_SUFFIXES = ("_normalized.tif",)


def _base_name(filename, channel):
    """Strip the _<channel>_normalized.tif suffix to get the cell base name."""
    stem = Path(filename).stem
    suffix = f"_{channel}_normalized"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def stack_rgb_channels(
    donor_dir,
    r_channel="ccr7",
    g_channel="actin",
    b_channel="cd45ra",
    output_folder="rgb_normalized",
    verbose=True,
):
    """
    Stack already-normalized single-channel TIFs into float32 RGB composites.

    This function does not normalize or clip channel values. Run channel
    normalization before calling it.

    Args:
        donor_dir: Path to the donor directory containing formatted_* folders.
        r_channel: Channel name to use for the red channel.
        g_channel: Channel name to use for the green channel.
        b_channel: Channel name to use for the blue channel.
        output_folder: Name of output folder created inside donor_dir.
        verbose: Print progress.

    Returns:
        dict with keys:
            - 'success': bool
            - 'tif_output': str path to TIF output folder
            - 'jpg_output': str path to JPG output folder
            - 'num_processed': int
            - 'num_skipped': int
            - 'error': str (only on failure)
    """
    donor_dir = Path(donor_dir)

    r_folder = donor_dir / f"formatted_{r_channel}" / "normalized_tif"
    g_folder = donor_dir / f"formatted_{g_channel}" / "normalized_tif"
    b_folder = donor_dir / f"formatted_{b_channel}" / "normalized_tif"

    for folder in (r_folder, g_folder, b_folder):
        if not folder.exists():
            return {
                "success": False,
                "error": f"Normalized channel folder not found: {folder}\nRun normalization first.",
                "num_processed": 0,
                "num_skipped": 0,
            }

    out_base = donor_dir / output_folder
    tif_out = out_base / "tif_files"
    jpg_out = out_base / "jpg_files"
    tif_out.mkdir(parents=True, exist_ok=True)
    jpg_out.mkdir(parents=True, exist_ok=True)

    r_files = sorted(r_folder.glob("*.tif"))
    if not r_files:
        return {
            "success": False,
            "error": f"No TIF files found in {r_folder}",
            "num_processed": 0,
            "num_skipped": 0,
        }

    if verbose:
        print(f"Stacking RGB for {len(r_files)} cells...")

    num_processed = 0
    num_skipped = 0

    for r_file in r_files:
        try:
            base = _base_name(r_file.name, r_channel)

            g_file = g_folder / f"{base}_{g_channel}_normalized.tif"
            b_file = b_folder / f"{base}_{b_channel}_normalized.tif"

            if not g_file.exists():
                if verbose:
                    print(f"  Missing G channel for {base}")
                num_skipped += 1
                continue
            if not b_file.exists():
                if verbose:
                    print(f"  Missing B channel for {base}")
                num_skipped += 1
                continue

            r_arr = tifffile.imread(r_file).astype(np.float32)
            g_arr = tifffile.imread(g_file).astype(np.float32)
            b_arr = tifffile.imread(b_file).astype(np.float32)

            # Squeeze single-channel dim if present
            if r_arr.ndim == 3 and r_arr.shape[2] == 1:
                r_arr = r_arr[:, :, 0]
            if g_arr.ndim == 3 and g_arr.shape[2] == 1:
                g_arr = g_arr[:, :, 0]
            if b_arr.ndim == 3 and b_arr.shape[2] == 1:
                b_arr = b_arr[:, :, 0]

            if not (r_arr.shape == g_arr.shape == b_arr.shape):
                if verbose:
                    print(f"  Shape mismatch for {base}: R={r_arr.shape} G={g_arr.shape} B={b_arr.shape}")
                num_skipped += 1
                continue

            rgb = np.stack([r_arr, g_arr, b_arr], axis=-1)

            # Save float32 TIF
            tif_path = tif_out / f"{base}_rgb.tif"
            tifffile.imwrite(
                tif_path,
                rgb,
                photometric="rgb",
                metadata={"axes": "YXC", "Description": f"R={r_channel} G={g_channel} B={b_channel}"},
            )

            # Save 8-bit JPG preview
            jpg_path = jpg_out / f"{base}_rgb.jpg"
            Image.fromarray((rgb * 255).astype(np.uint8)).save(jpg_path, quality=95)

            num_processed += 1

        except Exception as e:
            if verbose:
                print(f"  Error processing {r_file.name}: {e}")
            num_skipped += 1

    if verbose:
        print(f"  ✓ Done: {num_processed} stacked, {num_skipped} skipped")
        print(f"    TIF → {tif_out}")
        print(f"    JPG → {jpg_out}")

    return {
        "success": True,
        "tif_output": str(tif_out),
        "jpg_output": str(jpg_out),
        "num_processed": num_processed,
        "num_skipped": num_skipped,
    }
