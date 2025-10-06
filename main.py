#!/usr/bin/env python3
"""
Main Pipeline Runner
Automatically discovers and processes all samples and images in a directory

Usage:
    python main.py
    (Edit BASE_PATH below to change which directory to process)
"""

import imagej
from pathlib import Path
from process_single_image import run_pipeline, PARAMS

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

BASE_PATH = "/Users/taeeonkong/Desktop/2025 Fall Images/09-26-2025 DLBCL"
SAMPLES_TO_PROCESS = [1, 2, 3, 4, 5, 6]  # List of sample numbers to process, e.g., [1, 2, 3] or [1]

# Channel filenames - Edit these if your channel files have different names
CHANNEL_CONFIG = {
    'actin': 'Actin-FITC.tif',
    'cd4': 'CD4-PerCP.tif',
    'cd45ra_af647': 'CD45RA-AF647.tif',
    'cd45ra_sparkviolet': 'CD45RA-SparkViolet.tif',
    'cd19car': 'CD19CAR-AF647.tif',
    'ccr7': 'CCR7-PE.tif',
}

# ============================================================================
# MAIN
# ============================================================================

def main():
    base_path_obj = Path(BASE_PATH)

    if not base_path_obj.exists():
        print(f"✗ ERROR: Base path not found: {BASE_PATH}")
        import sys
        sys.exit(1)

    # Find all sample folders (sample1, sample2, etc.)
    sample_folders = []
    for item in base_path_obj.iterdir():
        if item.is_dir() and item.name.lower().startswith('sample'):
            sample_folders.append(item.name)

    # Sort by sample number
    def extract_sample_number(name):
        import re
        # Extract digits after "sample" or "Sample"
        match = re.search(r'sample(\d+)', name, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    sample_folders = sorted(sample_folders, key=extract_sample_number)

    # Filter to only requested samples
    sample_folders = [s for s in sample_folders if extract_sample_number(s) in SAMPLES_TO_PROCESS]

    if not sample_folders:
        print(f"✗ ERROR: No sample folders found in {BASE_PATH}")
        import sys
        sys.exit(1)

    print("\n" + "="*40)
    print("BATCH PIPELINE: PROCESSING ALL SAMPLES")
    print("="*40)
    print(f"Base path: {BASE_PATH}")
    print(f"Found {len(sample_folders)} samples: {', '.join(sample_folders)}")
    print("="*40 + "\n")

    # Initialize ImageJ once (reuse for all images)
    print("Initializing ImageJ (will be reused for all images)...")
    ij = imagej.init('sc.fiji:fiji')
    print(f"✓ ImageJ version: {ij.getVersion()}\n")
    print("="*40 + "\n")

    # Process each sample
    total_processed = 0
    total_failed = 0
    all_results = []

    for sample_idx, sample_folder in enumerate(sample_folders, 1):
        sample_path = base_path_obj / sample_folder

        # Find all subdirectories (image folders) - automatically process all found
        image_folders = [item.name for item in sample_path.iterdir() if item.is_dir()]

        # Sort alphabetically/numerically
        image_folders = sorted(image_folders)

        if not image_folders:
            print(f"⚠️  No image folders found in {sample_folder}, skipping...")
            continue

        print(f"\n{'='*40}")
        print(f"PROCESSING SAMPLE {sample_idx}/{len(sample_folders)}: {sample_folder}")
        print(f"Found {len(image_folders)} image folders: {', '.join(image_folders)}")
        print(f"{'='*40}\n")

        # Process each image in this sample
        for img_idx, image_number in enumerate(image_folders, 1):
            print(f"\n{'-'*40}")
            print(f"Image {img_idx}/{len(image_folders)}: {sample_folder}/{image_number}")
            print(f"{'-'*40}\n")

            try:
                result = run_pipeline(
                    sample_folder=sample_folder,
                    image_number=image_number,
                    base_path=BASE_PATH,
                    segmentation_method='cellpose',
                    params=PARAMS,
                    channel_config=CHANNEL_CONFIG,
                    ij=ij,  # Reuse ImageJ instance
                    verbose=True
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
                else:
                    total_failed += 1
                    print(f"\n✗ Failed to process {sample_folder}/{image_number}")
                    print(f"   Error: {result.get('error', 'Unknown error')}")

            except Exception as e:
                total_failed += 1
                print(f"\n✗ Failed to process {sample_folder}/{image_number}")
                print(f"   Error: {str(e)}")
                all_results.append({
                    'sample': sample_folder,
                    'image_number': image_number,
                    'success': False,
                    'error': str(e)
                })

    # Summary
    print("\n" + "="*40)
    print("BATCH PROCESSING COMPLETE")
    print("="*40)
    print(f"Total samples: {len(sample_folders)}")
    print(f"Total images: {total_processed + total_failed}")
    print(f"✓ Processed successfully: {total_processed}")
    print(f"✗ Failed: {total_failed}")
    print("="*40 + "\n")

    if total_failed > 0:
        import sys
        sys.exit(1)

    # Auto-run CSV combination
    print("\n" + "="*80)
    print("AUTO-COMBINING CSVs")
    print("="*80 + "\n")

    try:
        # Step 1: Combine channels for each image
        print("Step 1: Combining channels within each image...")
        from csvOps.combine_channel import combine_measurements

        for sample_folder in sample_folders:
            sample_path = base_path_obj / sample_folder
            image_folders = [item.name for item in sample_path.iterdir() if item.is_dir()]

            for image_number in sorted(image_folders):
                combine_measurements(
                    sample_folder=sample_folder,
                    image_number=image_number,
                    base_path=BASE_PATH,
                    verbose=False  # Quiet mode
                )
        print("✓ Image-level combination complete\n")

        # Step 2: Combine all samples
        print("Step 2: Combining all samples into master CSV...")
        from csvOps import combine_all
        combine_all.main()

        # Step 3: Classify T cells
        print("\nStep 3: Classifying T cell subsets...")
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
