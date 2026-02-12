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
SAMPLES_TO_PROCESS = [1]  # Leave empty to process all samples

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

# Crop mode: "original", "white", "black", or "transparent"
# "original" keeps pixel intensities exactly as in source TIFF within ROI bbox.
BACKGROUND = "original"


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


def _compact_error(message: str) -> str:
    # Convert messages like "X not found: /full/path" -> "X not found"
    if ": " in message:
        prefix, suffix = message.split(": ", 1)
        if "/" in suffix or "\\" in suffix:
            return prefix
    return message


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

    all_jobs = []
    for sample_folder in samples:
        sample_path = base_path / sample_folder
        image_folders = _discover_images(sample_path)
        image_folders = filter_image_folders(
            sample_folder,
            image_folders,
            image_filters,
            image_filters_default,
            announce=False,
        )
        for image_number in image_folders:
            all_jobs.append((sample_folder, image_number))

    total_jobs = len(all_jobs)
    if total_jobs == 0:
        raise RuntimeError("No image folders matched your filters.")

    print(f"Processing {total_jobs} images...")

    success_count = 0
    failures = []

    for idx, (sample_folder, image_number) in enumerate(all_jobs, start=1):
        job_label = f"{sample_folder}/{image_number}"
        base_dir = base_path / sample_folder / image_number

        ccr7_source = _pick_existing(base_dir, [CCR7_FILENAME])
        cd45ra_source = _pick_existing(base_dir, [CD45RA_FILENAME])

        if not ccr7_source:
            error = "CCR7 image not found"
            failures.append((job_label, error))
            print(f"[{idx}/{total_jobs}] FAIL {job_label} - {error}")
            continue

        if not cd45ra_source:
            error = "CD45RA image not found"
            failures.append((job_label, error))
            print(f"[{idx}/{total_jobs}] FAIL {job_label} - {error}")
            continue

        result_ccr7 = make_raw_crops(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=BASE_PATH,
            source_image=ccr7_source,
            roi_dir_name=ROI_DIR_NAME,
            output_dir_name=OUTPUT_DIR_CCR7,
            background=BACKGROUND,
            verbose=False,
        )

        result_cd45ra = make_raw_crops(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=BASE_PATH,
            source_image=cd45ra_source,
            roi_dir_name=ROI_DIR_NAME,
            output_dir_name=OUTPUT_DIR_CD45RA,
            background=BACKGROUND,
            verbose=False,
        )

        image_errors = []
        if not result_ccr7["success"]:
            image_errors.append(f"CCR7: {_compact_error(result_ccr7['error'])}")
        if not result_cd45ra["success"]:
            image_errors.append(f"CD45RA: {_compact_error(result_cd45ra['error'])}")

        if image_errors:
            error = " | ".join(image_errors)
            failures.append((job_label, error))
            print(f"[{idx}/{total_jobs}] FAIL {job_label} - {error}")
            continue

        success_count += 1
        print(f"[{idx}/{total_jobs}] OK {job_label}")

    print("\nDone.")
    print(f"Successful images: {success_count}/{total_jobs}")
    print(f"Failed images: {len(failures)}")
    if failures:
        print("\nFailed image list:")
        for job_label, error in failures:
            print(f"- {job_label}: {error}")


if __name__ == "__main__":
    main()
