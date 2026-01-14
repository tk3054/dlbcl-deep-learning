#!/usr/bin/env python3
"""
Main Pipeline Runner
Automatically discovers and processes all samples and images in a directory

Usage:
    python main.py
    (Edit BASE_PATH below to change which directory to process)
"""

import imagej
import os
import sys
from contextlib import contextmanager
import sys
from pathlib import Path
from process_single_image import run_pipeline, PARAMS
from csvOps.combine_channel import combine_measurements
from utils.channel_aliases import canonicalize_channel_config
from utils.config_helpers import (
    extract_sample_number,
    filter_image_folders,
    normalize_image_filter_config,
)
from compare_edge_methods import compare_multiple_cells


@contextmanager
def _suppress_output():
    """Suppress ImageJ stdout/stderr noise during initialization."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

BASE_PATH = '/Users/taeeonkong/Desktop/DL Project/responder/01-06-2026 DLBCL 118830'
SAMPLES_TO_PROCESS = [1, 2]  # Process all available samples

# Optional per-sample image filtering. Define image numbers as ints.
# Examples:
#   {1: [5, 13]}  → run images 5 & 13 for sample1 only
# 
# Leave empty, if you don't want to use filtering. 
IMAGES_TO_PROCESS = {}

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
IMAGE_FILTERS, IMAGE_FILTERS_DEFAULT = normalize_image_filter_config(IMAGES_TO_PROCESS)

# ============================================================================
# HELPERS
# ============================================================================

def prepare_run():
    """Preflight: validate base path, gather samples, and initialize ImageJ."""
    base_path_obj = Path(BASE_PATH)

    if not base_path_obj.exists():
        print(f"✗ ERROR: Base path not found: {BASE_PATH}")
        sys.exit(1)

    sample_folders = [
        item.name
        for item in base_path_obj.iterdir()
        if item.is_dir() and item.name.lower().startswith('sample')
    ]
    sample_folders = sorted(sample_folders, key=extract_sample_number)
    sample_folders = [s for s in sample_folders if extract_sample_number(s) in SAMPLES_TO_PROCESS]

    if not sample_folders:
        print(f"✗ ERROR: No sample folders found in {BASE_PATH}")
        sys.exit(1)

    print("\n" + "="*40)
    print("BATCH PIPELINE: PROCESSING ALL SAMPLES")
    print("="*40)
    print(f"Base path: {BASE_PATH}")
    print(f"Found {len(sample_folders)} samples: {', '.join(sample_folders)}")
    print("="*40 + "\n")

    print("Initializing ImageJ (will be reused for all images)...")
    with _suppress_output():
        ij = imagej.init('sc.fiji:fiji')
    print(f"✓ ImageJ version: {ij.getVersion()}\n")
    print("="*40 + "\n")

    return base_path_obj, sample_folders, ij

def prompt_channel_filenames(base_path_obj, sample_folder, image_number, channel_config):
    """
    Check channel filenames for an image folder; if a configured name is missing,
    prompt the user to pick between the configured name and any .tif present.
    """
    image_dir = base_path_obj / sample_folder / image_number
    if not image_dir.exists():
        return channel_config

    available_tifs = sorted([p.name for p in image_dir.glob("*.tif")])
    resolved = dict(channel_config)

    for key, filename in channel_config.items():
        expected = image_dir / filename
        processed = image_dir / f"processed_{filename}"

        if expected.exists() or processed.exists():
            continue

        if not available_tifs:
            continue

        print(f"\nChannel '{key}' file not found for {sample_folder}/{image_number}.")
        print(f"Configured: {filename}")
        print("Available .tif files in this folder:")
        for idx, name in enumerate(available_tifs, start=1):
            print(f"  {idx}. {name}")
        print("  0. Keep configured name")

        choice = input("Select a file number to use for this channel (default 0): ").strip()
        if not choice:
            choice = "0"

        try:
            choice_num = int(choice)
        except ValueError:
            choice_num = 0

        if 1 <= choice_num <= len(available_tifs):
            resolved[key] = available_tifs[choice_num - 1]
            print(f"  → Using {resolved[key]} for channel '{key}'")
        else:
            print(f"  → Keeping configured filename for '{key}'")

    return resolved


# ============================================================================
def main():
    base_path_obj, sample_folders, ij = prepare_run()

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
                    base_path=BASE_PATH,
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
                    try:
                        combine_measurements(
                            sample_folder=sample_folder,
                            image_number=image_number,
                            base_path=BASE_PATH,
                            include_channels=None,
                            null_channels=None,
                            verbose=False,
                        )
                        print(f"  ↳ Combined channel CSVs for {sample_folder}/{image_number}")
                    except Exception as combine_err:
                        print(f"  ⚠️  Failed to combine channels for {sample_folder}/{image_number}: {combine_err}")

                    # Generate edge softening comparison images
                    if GENERATE_EDGE_COMPARISONS:
                        try:
                            print(f"\n  Creating edge softening comparisons...")
                            compare_multiple_cells(
                                sample_folder=sample_folder,
                                image_number=image_number,
                                base_path=BASE_PATH,
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
        import sys
        sys.exit(1)

    print("\n" + "="*80)
    print("COMBINING ALL SAMPLES INTO MASTER CSV")
    print("="*80 + "\n")

    try:
        from csvOps import combine_all
        combine_all.main()

        from analysis.classify_tcells import classify_tcells
        classify_result = classify_tcells(
            base_path=BASE_PATH,
            verbose=True
        )

        if classify_result['success']:
            print(f"✓ Classified {classify_result['num_cd4_pos']} CD4+ T cells into subsets")

    except Exception as e:
        print(f"\n✗ Failed to complete post-processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
