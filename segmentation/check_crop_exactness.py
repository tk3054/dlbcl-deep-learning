#!/usr/bin/env python3
"""
Check whether saved raw channel crops exactly match the source TIFF ROI bbox.

Edit CONFIG below, then run:
  python segmentation/check_crop_exactness.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


# ============================================================================
# CONFIG - EDIT THESE
# ============================================================================
BASE_PATH = "/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241"
SAMPLE = "sample1"
IMAGE = "3"
CELL = 14

CCR7_FILENAME = "CCR7-AF594.tif"
CD45RA_FILENAME = "CD45RA-PacBlue.tif"
RAW_CCR7_DIR = "raw_ccr7"
RAW_CD45RA_DIR = "raw_cd45ra"


def _load_tiff(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    arr = np.squeeze(arr)
    return arr


def _roi_bbox(roi_path: Path) -> tuple[int, int, int, int]:
    roi = _load_tiff(roi_path)
    if roi.ndim > 2:
        roi = roi[..., 0]
    ys, xs = np.where(roi > 0)
    if ys.size == 0 or xs.size == 0:
        raise ValueError(f"ROI has no non-zero pixels: {roi_path}")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return y0, y1, x0, x1


def _stats(arr: np.ndarray) -> str:
    return (
        f"shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min()}, max={arr.max()}"
    )


def _compare_one(
    base_dir: Path,
    cell_id: int,
    source_filename: str,
    raw_dir_name: str,
) -> bool:
    roi_path = base_dir / "cell_rois" / f"cell_{cell_id:02d}.tif"
    source_path = base_dir / source_filename
    crop_path = base_dir / raw_dir_name / f"cell_{cell_id:02d}_raw.tif"

    if not roi_path.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if not crop_path.exists():
        raise FileNotFoundError(f"Crop file not found: {crop_path}")

    y0, y1, x0, x1 = _roi_bbox(roi_path)
    source = _load_tiff(source_path)
    crop = _load_tiff(crop_path)

    expected = source[y0:y1, x0:x1, ...] if source.ndim >= 3 else source[y0:y1, x0:x1]

    exact = (
        expected.shape == crop.shape
        and expected.dtype == crop.dtype
        and np.array_equal(expected, crop)
    )

    print(f"\n[{raw_dir_name}] cell_{cell_id:02d}")
    print(f"source:  {source_path.name}")
    print(f"crop:    {crop_path.name}")
    print(f"bbox:    y={y0}:{y1}, x={x0}:{x1}")
    print(f"expect:  {_stats(expected)}")
    print(f"actual:  {_stats(crop)}")
    print(f"exact:   {'YES' if exact else 'NO'}")

    if not exact and expected.shape == crop.shape:
        exp_f = expected.astype(np.float64, copy=False)
        got_f = crop.astype(np.float64, copy=False)
        diff = got_f - exp_f
        print(
            "diff:    "
            f"min={diff.min()}, max={diff.max()}, "
            f"mean={diff.mean():.6f}, abs_max={np.abs(diff).max()}"
        )

    return exact


def main() -> None:
    base_dir = Path(BASE_PATH) / SAMPLE / IMAGE
    if not base_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {base_dir}")

    ok_ccr7 = _compare_one(
        base_dir=base_dir,
        cell_id=CELL,
        source_filename=CCR7_FILENAME,
        raw_dir_name=RAW_CCR7_DIR,
    )
    ok_cd45ra = _compare_one(
        base_dir=base_dir,
        cell_id=CELL,
        source_filename=CD45RA_FILENAME,
        raw_dir_name=RAW_CD45RA_DIR,
    )

    print("\nSummary")
    print(f"- CCR7 exact match:   {'YES' if ok_ccr7 else 'NO'}")
    print(f"- CD45RA exact match: {'YES' if ok_cd45ra else 'NO'}")


if __name__ == "__main__":
    main()
