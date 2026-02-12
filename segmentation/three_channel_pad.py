#!/usr/bin/env python3
"""
Create padded per-cell crops for additional channels using existing ROI masks.

Edit the CONFIG section below, then run:
    python segmentation/three_channel_padded.py
"""

from pathlib import Path
import sys

import cv2
from skimage import io

# Allow running this script directly by adding repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from process_image.pad_raw_crops import extract_masked_cell, pad_to_square
from utils.image_iterator import (
    collect_image_jobs,
    resolve_channel_filenames,
)


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================
BASE_PATH = '/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241'
SAMPLES_TO_PROCESS = []  # Leave empty to process all samples

# Optional per-sample image filtering. Define image numbers as ints.
# Examples:
#   {1: [5, 13]}  -> run images 5 & 13 for sample1 only
IMAGES_TO_PROCESS = {}

# Channel filenames in the image folder
ACTIN_FILENAME = "processed_Actin-FITC.tif"
CCR7_FILENAME = "processed_CCR7-AF594.tif"
CD45RA_FILENAME = "processed_CD45RA-PacBlue.tif"

# ROI folder name
ROI_DIR_NAME = "cell_rois"

# Output folders (created inside <base>/<sample>/<image_number>/)
# Keep actin output in padded_cells so downstream scripts can consume it directly.
OUTPUT_DIR_ACTIN = "padded_cells"
OUTPUT_DIR_CCR7 = "padded_ccr7"
OUTPUT_DIR_CD45RA = "padded_cd45ra"

# Output size
TARGET_SIZE = 224

# If True, prompt to choose a .tif when automatic matching fails.
INTERACTIVE_FILENAME_PROMPT = True


def _load_grayscale(path: Path):
    image = io.imread(path)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def _compact_error(message: str) -> str:
    # Convert messages like "X not found: /full/path" -> "X not found"
    if ": " in message:
        prefix, suffix = message.split(": ", 1)
        if "/" in suffix or "\\" in suffix:
            return prefix
    return message


def _process_channel(
    base_dir: Path,
    source_filename: str,
    roi_dir_name: str,
    output_dir_name: str,
    target_size: int,
):
    roi_dir = base_dir / roi_dir_name
    source_path = base_dir / source_filename
    output_dir = base_dir / output_dir_name

    if not roi_dir.exists():
        return {
            "success": False,
            "error": f"ROI directory not found: {roi_dir}",
            "num_padded": 0,
            "output_dir": str(output_dir),
        }

    if not source_path.exists():
        return {
            "success": False,
            "error": f"Source image not found: {source_path}",
            "num_padded": 0,
            "output_dir": str(output_dir),
        }

    roi_files = sorted(roi_dir.glob("*.tif"))
    if not roi_files:
        return {
            "success": False,
            "error": f"No ROI masks found in: {roi_dir}",
            "num_padded": 0,
            "output_dir": str(output_dir),
        }

    output_dir.mkdir(exist_ok=True)
    source_image = _load_grayscale(source_path)

    success_count = 0
    for roi_file in roi_files:
        roi_mask = _load_grayscale(roi_file)
        masked_cell, _ = extract_masked_cell(source_image, roi_mask)  # non-cell pixels -> 0
        if masked_cell is None:
            continue

        padded = pad_to_square(masked_cell, target_size)  # pad with 0
        if padded is None:
            continue

        output_name = f"{roi_file.stem}_padded{roi_file.suffix}"
        io.imsave(output_dir / output_name, padded, check_contrast=False)
        success_count += 1

    return {
        "success": True,
        "num_padded": success_count,
        "output_dir": str(output_dir),
    }


def main():
    base_path = Path(BASE_PATH)
    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {BASE_PATH}")

    all_jobs = collect_image_jobs(
        base_path=base_path,
        samples_to_process=SAMPLES_TO_PROCESS,
        images_to_process=IMAGES_TO_PROCESS,
        announce_filters=False,
    )

    total_jobs = len(all_jobs)
    if total_jobs == 0:
        raise RuntimeError("No image folders matched your filters.")

    print(f"Processing {total_jobs} images...")

    success_count = 0
    failures = []
    total_actin_padded = 0
    total_ccr7_padded = 0
    total_cd45ra_padded = 0

    for idx, job in enumerate(all_jobs, start=1):
        job_label = job.label
        base_dir = job.image_path

        resolved = resolve_channel_filenames(
            image_dir=job.image_path,
            configured_map={
                "actin": ACTIN_FILENAME,
                "ccr7": CCR7_FILENAME,
                "cd45ra": CD45RA_FILENAME,
            },
            interactive_prompt=INTERACTIVE_FILENAME_PROMPT,
            job_label=job.label,
        )

        actin_source = resolved.get("actin")
        ccr7_source = resolved.get("ccr7")
        cd45ra_source = resolved.get("cd45ra")

        if not actin_source:
            error = "Actin image not found"
            failures.append((job_label, error))
            print(f"[{idx}/{total_jobs}] FAIL {job_label} - {error}")
            continue

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

        result_actin = _process_channel(
            base_dir=base_dir,
            source_filename=actin_source,
            roi_dir_name=ROI_DIR_NAME,
            output_dir_name=OUTPUT_DIR_ACTIN,
            target_size=TARGET_SIZE,
        )

        result_ccr7 = _process_channel(
            base_dir=base_dir,
            source_filename=ccr7_source,
            roi_dir_name=ROI_DIR_NAME,
            output_dir_name=OUTPUT_DIR_CCR7,
            target_size=TARGET_SIZE,
        )

        result_cd45ra = _process_channel(
            base_dir=base_dir,
            source_filename=cd45ra_source,
            roi_dir_name=ROI_DIR_NAME,
            output_dir_name=OUTPUT_DIR_CD45RA,
            target_size=TARGET_SIZE,
        )

        image_errors = []
        if not result_actin["success"]:
            image_errors.append(f"Actin: {_compact_error(result_actin['error'])}")
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
        total_actin_padded += int(result_actin.get("num_padded", 0))
        total_ccr7_padded += int(result_ccr7.get("num_padded", 0))
        total_cd45ra_padded += int(result_cd45ra.get("num_padded", 0))
        print(f"[{idx}/{total_jobs}] OK {job_label}")

    print("\nDone.")
    print(f"Successful images: {success_count}/{total_jobs}")
    print(f"Failed images: {len(failures)}")
    print(f"Total padded segmentations (Actin): {total_actin_padded}")
    print(f"Total padded segmentations (CCR7): {total_ccr7_padded}")
    print(f"Total padded segmentations (CD45RA): {total_cd45ra_padded}")
    if failures:
        print("\nFailed image list:")
        for job_label, error in failures:
            print(f"- {job_label}: {error}")


if __name__ == "__main__":
    main()
