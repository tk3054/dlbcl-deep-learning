#!/usr/bin/env python3
"""
Remote Pipeline Runner
Copies a raw dataset to a processed tree, then iterates all patient folders.

Usage:
    python main_remote.py
    (Edit RAW_BASE_PATH below to point to the raw dataset root)
"""

import sys
import shutil
from pathlib import Path
from urllib.parse import urlparse, unquote
from process_single_image import run_pipeline, PARAMS
from csvOps.combine_images import combine_sample
from utils.channel_aliases import canonicalize_channel_config, canonicalize_channel_list
from utils.name_builder import NameBuilder
from utils.config_helpers import (
    filter_image_folders,
    normalize_image_filter_config,
)
from compare_edge_methods import compare_multiple_cells
from pipeline_helpers import (
    export_images_for_export,
    prepare_run,
    prompt_channel_filenames,
)


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Relative to the working directory when you run the script (e.g., repo root).
RAW_BASE_PATHS = [
    '../DLBCL/Non-responder',
    '../DLBCL/Responder',
]
PROCESSED_BASE_NAME = 'DLBCL_processed'
COPY_RAW_TO_PROCESSED = True

# Leave empty to process all samples found for each patient.
SAMPLES_TO_PROCESS = []

# Optional per-sample image filtering. Define image numbers as ints.
# Examples:
#   {1: [5, 13]}  → run images 5 & 13 for sample1 only
# 
# Leave empty, if you don't want to use filtering. 
IMAGES_TO_PROCESS = {}

# Exporting named cell images into the patient folder (copied from padded_cells)
EXPORT_IMAGES_FOR_EXPORT = True

# Export naming config
EXPORT_PDMS_STIFFNESS = "1to10"
EXPORT_CLASSIFIED_CSV = "all_samples_combined_classified.csv"
EXPORT_SOURCE_DIR = "padded_cells"
EXPORT_OUTPUT_DIR = "formatted_cells"
EXPORT_DILUTION = "1to10"
EXPORT_NAME_ORDER = [
    "response",
    "patient_id",
    "date",
    "stiffness",
    "sample",
    "image",
    "cell_label",
    "classification",
]


# Master CSV column config
# Required columns are always kept. Optional columns follow MASTER_COLUMN_TOGGLES.
MASTER_REQUIRED_COLUMNS = [
    "global_cell_id",
    "unique_id",
    "sample",
    "image",
    "cell_id",
    "cd4_median",
    "cd4_mean",
    "ccr7_median",
    "ccr7_mean",
    "cd45ra_af647_median",
    "cd45ra_af647_mean",
    "cd45ra_sparkviolet_median",
    "cd45ra_sparkviolet_mean",
    "cd19car_median",
    "cd19car_mean",
]

# True = keep, False = drop. Any column not listed falls back to _default.
MASTER_COLUMN_TOGGLES = {
    "_default": False,
    "area": False,
    "x": False,
    "y": False,
    "circ": False,
    "ar": False,
    "round": False,
    "solidity": False,
}

# Edge softening comparison
GENERATE_EDGE_COMPARISONS = False  # DISABLED: Generate edge softening comparison images
NUM_CELLS_TO_COMPARE = 5  # Number of cells per image to compare

# File names for images for each channel. These must match the names in the folder. 
CHANNEL_CONFIG = {
    'actin': 'Actin-FITC.tif',
    'cd4': 'CD4-PerCP.tif',
    'cd45ra_PacBlue': 'CD45RA-PacBlue.tif',
    # 'cd45ra_sparkviolet': 'CD45RA-SparkViolet.tif',
    'cd19car': 'CD19CAR-AF647.tif',
    'ccr7': 'CCR7-AF594.tif',
}


CHANNEL_CONFIG = canonicalize_channel_config(CHANNEL_CONFIG)
INTENSITY_COLUMNS = ["mean", "median", "std", "min", "max", "intden", "rawintden"]
for _channel in CHANNEL_CONFIG:
    for _stat in INTENSITY_COLUMNS:
        MASTER_COLUMN_TOGGLES.setdefault(f"{_channel}_{_stat}", True)
IMAGE_FILTERS, IMAGE_FILTERS_DEFAULT = normalize_image_filter_config(IMAGES_TO_PROCESS)

# ============================================================================
def _normalize_input_path(path_text: str) -> str:
    if not path_text:
        return ""
    if path_text.startswith("sftp://"):
        parsed = urlparse(path_text)
        return unquote(parsed.path or "")
    return unquote(path_text)


def _is_patient_dir(path: Path) -> bool:
    return any(
        child.is_dir() and child.name.lower().startswith("sample")
        for child in path.iterdir()
    )


def _find_patient_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []

    candidates = [
        item for item in root.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    ]

    direct_patients = [item for item in candidates if _is_patient_dir(item)]
    if direct_patients:
        return sorted(direct_patients, key=lambda p: p.name.lower())

    patient_dirs: list[Path] = []
    for response_dir in sorted(candidates, key=lambda p: p.name.lower()):
        for patient_dir in sorted(
            [p for p in response_dir.iterdir() if p.is_dir()],
            key=lambda p: p.name.lower()
        ):
            if _is_patient_dir(patient_dir):
                patient_dirs.append(patient_dir)

    return patient_dirs


def _prepare_processed_root(raw_path: Path) -> Path:
    processed_path = raw_path.parent / PROCESSED_BASE_NAME

    if processed_path.exists():
        print(f"✓ Using existing processed tree: {processed_path}")
        return processed_path

    if not COPY_RAW_TO_PROCESSED:
        print(f"✗ ERROR: Processed path not found: {processed_path}")
        sys.exit(1)

    print(f"Copying raw dataset to processed tree:\n  {raw_path}\n→ {processed_path}")
    shutil.copytree(raw_path, processed_path)
    print("✓ Copy complete")
    return processed_path


def run_patient_pipeline(base_path: str):
    base_path_obj, sample_folders, ij = prepare_run(base_path, SAMPLES_TO_PROCESS)

    total_processed = 0
    total_failed = 0
    all_results = []
    failed_images = []

    for sample_idx, sample_folder in enumerate(sample_folders, 1):
        sample_path = base_path_obj / sample_folder

        image_folders = [item.name for item in sample_path.iterdir() if item.is_dir()]
        # Sort numerically if all digits, otherwise alphabetically (numeric first)
        image_folders = sorted(image_folders, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
        image_folders = filter_image_folders(
            sample_folder,
            image_folders,
            IMAGE_FILTERS,
            IMAGE_FILTERS_DEFAULT,
            announce=True,
        )

        if not image_folders:
            print(f"⚠️  No image folders found in {sample_folder}, skipping...")
            continue

        print(f"\n{'='*40}")
        print(f"PROCESSING SAMPLE {sample_idx}/{len(sample_folders)}: {sample_folder}")
        print(f"Found {len(image_folders)} image folders: {', '.join(image_folders)}")
        print(f"{'='*40}\n")

        for img_idx, image_number in enumerate(image_folders, 1):
            print(f"\n{'-'*40}")
            print(f"Image {img_idx}: {sample_folder}/{image_number}")
            print(f"{'-'*40}\n")

            try:
                channel_config_for_image = prompt_channel_filenames(
                    base_path_obj,
                    sample_folder,
                    image_number,
                    CHANNEL_CONFIG,
                )

                result = run_pipeline(
                    sample_folder=sample_folder,
                    image_number=image_number,
                    base_path=base_path,
                    segmentation_method='cellpose',
                    params=PARAMS,
                    channel_config=channel_config_for_image,
                    combine_channels=None,
                    null_channels=None,
                    ij=ij,
                    verbose=False
                )

                all_results.append({
                    'sample': sample_folder,
                    'image_number': image_number,
                    'success': result['success'],
                    'result': result
                })

                if result['success']:
                    total_processed += 1
                    print(f"\n✓ Successfully processed {sample_folder}/{image_number}")

                    # Generate edge softening comparison images
                    if GENERATE_EDGE_COMPARISONS:
                        try:
                            print(f"\n  Creating edge softening comparisons...")
                            compare_multiple_cells(
                                sample_folder=sample_folder,
                                image_number=image_number,
                    base_path=base_path,
                                num_cells=NUM_CELLS_TO_COMPARE
                            )
                            print(f"  ↳ Generated edge comparisons for {sample_folder}/{image_number}")
                        except Exception as compare_err:
                            print(f"  ⚠️  Failed to generate edge comparisons: {compare_err}")
                else:
                    total_failed += 1
                    print(f"\n✗ Failed to process {sample_folder}/{image_number}")
                    error_msg = result.get('error', 'Unknown error')
                    print(f"   Error: {error_msg}")
                    failed_images.append({
                        'sample': sample_folder,
                        'image_number': image_number,
                        'error': error_msg,
                    })

            except Exception as e:
                total_failed += 1
                error_msg = str(e)
                print(f"\n✗ Failed to process {sample_folder}/{image_number}")
                print(f"   Error: {error_msg}")
                all_results.append({
                    'sample': sample_folder,
                    'image_number': image_number,
                    'success': False,
                    'error': error_msg
                })
                failed_images.append({
                    'sample': sample_folder,
                    'image_number': image_number,
                    'error': error_msg,
                })

        # After all images in this sample, combine them into sample-level CSV
        try:
            combine_sample(
                sample_name=sample_folder,
                base_path=base_path,
                verbose=True
            )
        except Exception as e:
            print(f"  ⚠️  Failed to combine sample {sample_folder}: {e}")

    print("\n" + "="*40)
    print("BATCH PROCESSING COMPLETE")
    print("="*40)
    print(f"Total samples: {len(sample_folders)}")
    print(f"Total images: {total_processed + total_failed}")
    print(f"✓ Processed successfully: {total_processed}")
    print(f"✗ Failed: {total_failed}")
    print("="*40 + "\n")

    if failed_images:
        log_path = base_path_obj / "failed_images.log"
        with log_path.open("w") as log_file:
            log_file.write("Failed images:\n")
            for item in failed_images:
                log_file.write(
                    f"- {item['sample']}/{item['image_number']}: {item['error']}\n"
                )
        print(f"✗ Wrote failure log: {log_path}")

    if total_failed > 0:
        return {
            "success": False,
            "total_failed": total_failed,
            "failed_images": failed_images,
        }

    print("\n" + "="*80)
    print("COMBINING ALL SAMPLES INTO MASTER CSV")
    print("="*80 + "\n")

    try:
        from csvOps import combine_samples
        combine_samples.BASE_PATH = base_path
        combine_samples.CHANNEL_CONFIG = CHANNEL_CONFIG
        combine_samples.CHANNELS_LIST = canonicalize_channel_list(CHANNEL_CONFIG.keys())
        combine_samples.SAMPLES_TO_PROCESS = SAMPLES_TO_PROCESS
        combine_samples.IMAGES_TO_PROCESS = IMAGES_TO_PROCESS
        combine_samples.main()

        from analysis.classify_tcells import classify_tcells
        classify_result = classify_tcells(
            base_path=base_path,
            verbose=True
        )

        if classify_result['success']:
            print(f"✓ Classified {classify_result['num_cd4_pos']} CD4+ T cells into subsets")

        if EXPORT_IMAGES_FOR_EXPORT:
            name_builder = NameBuilder(order=EXPORT_NAME_ORDER)
            export_images_for_export(
                base_path_obj=base_path_obj,
                name_builder=name_builder,
                output_dir_name=EXPORT_OUTPUT_DIR,
                source_dir_name=EXPORT_SOURCE_DIR,
                classified_csv=EXPORT_CLASSIFIED_CSV,
                dilution=EXPORT_DILUTION,
                pdms_stiffness=EXPORT_PDMS_STIFFNESS,
                verbose=True,
            )

    except Exception as e:
        print(f"\n✗ Failed to complete post-processing: {str(e)}")
        import traceback
        traceback.print_exc()

    return {"success": True, "total_failed": total_failed, "failed_images": failed_images}


def main():
    raw_path_texts = [
        _normalize_input_path(p) for p in RAW_BASE_PATHS if _normalize_input_path(p)
    ]
    if not raw_path_texts:
        print("✗ ERROR: RAW_BASE_PATHS is empty.")
        sys.exit(1)

    total_failed_patients = 0
    total_patients_seen = 0

    for raw_path_text in raw_path_texts:
        raw_path = Path(raw_path_text)
        if not raw_path.exists():
            print(f"✗ ERROR: Raw path not found: {raw_path}")
            total_failed_patients += 1
            continue

        processed_root = _prepare_processed_root(raw_path)
        patient_dirs = _find_patient_dirs(processed_root)
        if not patient_dirs:
            print(f"✗ ERROR: No patient folders found under {processed_root}")
            total_failed_patients += 1
            continue

        print("\n" + "=" * 80)
        print("REMOTE BATCH PIPELINE: PROCESSING ALL PATIENTS")
        print("=" * 80)
        print(f"Processed root: {processed_root}")
        print(f"Found {len(patient_dirs)} patient folders")
        print("=" * 80 + "\n")

        total_patients_seen += len(patient_dirs)
        for idx, patient_dir in enumerate(patient_dirs, start=1):
            print("\n" + "=" * 80)
            print(f"PATIENT {idx}/{len(patient_dirs)}: {patient_dir.name}")
            print(f"Path: {patient_dir}")
            print("=" * 80 + "\n")

            result = run_patient_pipeline(str(patient_dir))
            if not result.get("success"):
                total_failed_patients += 1

    if total_patients_seen == 0:
        print("\n✗ ERROR: No patient folders found in any RAW_BASE_PATHS entry.")
        sys.exit(1)

    if total_failed_patients:
        print(f"\n⚠️  Completed with {total_failed_patients} patient(s) reporting failures.")
        sys.exit(1)

if __name__ == "__main__":
    main()
