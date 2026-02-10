#!/usr/bin/env python3
"""
Create raw crops for additional channels using existing ROI masks.

Edit the CONFIG section below, then run:
    python segmentation/three_channel_raw_crops.py
"""

from pathlib import Path
import sys

# Allow running this script directly by adding repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from process_image.make_raw_crops import make_raw_crops
from utils.config_helpers import (
    extract_sample_number,
    filter_image_folders,
    normalize_image_filter_config,
)


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================
BASE_PATH = '/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241'
SAMPLES_TO_PROCESS = []  # Leave empty to process all samples

# Optional per-sample image filtering. Define image numbers as ints.
# Examples:
#   {1: [5, 13]}  → run images 5 & 13 for sample1 only
# Leave empty, if you don't want to use filtering.
IMAGES_TO_PROCESS = {}

# Channel filenames in the image folder
CCR7_FILENAME = "CCR7-AF594.tif"
CD45RA_FILENAME = "CD45RA-PacBlue.tif"

# ROI folder name
ROI_DIR_NAME = "cell_rois"

# Output folders (created inside <base>/<sample>/<image_number>/)
OUTPUT_DIR_CCR7 = "raw_ccr7"
OUTPUT_DIR_CD45RA = "raw_cd45ra"

# Background outside ROI: "white", "black", or "transparent"
BACKGROUND = "transparent"


def _pick_existing(base_dir: Path, candidates):
    for name in candidates:
        if (base_dir / name).exists():
            return name
    return None


def _discover_samples(base_path: Path):
    sample_folders = [
        item.name
        for item in base_path.iterdir()
        if item.is_dir() and item.name.lower().startswith("sample")
    ]
    return sorted(sample_folders, key=extract_sample_number)


def _discover_images(sample_path: Path):
    image_folders = [item.name for item in sample_path.iterdir() if item.is_dir()]
    return sorted(image_folders, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))


def main():
    base_path = Path(BASE_PATH)
    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {BASE_PATH}")

    samples = _discover_samples(base_path)
    if SAMPLES_TO_PROCESS:
        samples = [
            s for s in samples if extract_sample_number(s) in SAMPLES_TO_PROCESS
        ]

    if not samples:
        raise RuntimeError(f"No sample folders found in {BASE_PATH}")

    image_filters, image_filters_default = normalize_image_filter_config(IMAGES_TO_PROCESS)

    for sample_folder in samples:
        sample_path = base_path / sample_folder
        image_folders = _discover_images(sample_path)
        image_folders = filter_image_folders(
            sample_folder,
            image_folders,
            image_filters,
            image_filters_default,
            announce=True,
        )

        if not image_folders:
            print(f"⚠️  No image folders found in {sample_folder}, skipping...")
            continue

        print(f"\nProcessing {sample_folder}: {', '.join(image_folders)}")

        for image_number in image_folders:
            base_dir = base_path / sample_folder / image_number

            ccr7_source = _pick_existing(base_dir, [CCR7_FILENAME])
            cd45ra_source = _pick_existing(base_dir, [CD45RA_FILENAME])

            if not ccr7_source:
                print(f"  ⚠️  CCR7 image not found in {base_dir}, skipping.")
                continue
            if not cd45ra_source:
                print(f"  ⚠️  CD45RA image not found in {base_dir}, skipping.")
                continue

            result_ccr7 = make_raw_crops(
                sample_folder=sample_folder,
                image_number=image_number,
                base_path=BASE_PATH,
                source_image=ccr7_source,
                roi_dir_name=ROI_DIR_NAME,
                output_dir_name=OUTPUT_DIR_CCR7,
                background=BACKGROUND,
                verbose=True,
            )

            if not result_ccr7["success"]:
                raise RuntimeError(result_ccr7["error"])

            result_cd45ra = make_raw_crops(
                sample_folder=sample_folder,
                image_number=image_number,
                base_path=BASE_PATH,
                source_image=cd45ra_source,
                roi_dir_name=ROI_DIR_NAME,
                output_dir_name=OUTPUT_DIR_CD45RA,
                background=BACKGROUND,
                verbose=True,
            )

            if not result_cd45ra["success"]:
                raise RuntimeError(result_cd45ra["error"])

            print(
                f"  ✓ {sample_folder}/{image_number}: "
                f"CCR7 {result_ccr7['num_extracted']} -> {result_ccr7['output_dir']}; "
                f"CD45RA {result_cd45ra['num_extracted']} -> {result_cd45ra['output_dir']}"
            )


if __name__ == "__main__":
    main()
