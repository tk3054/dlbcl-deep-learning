#!/usr/bin/env python3
"""
Combine All Measurements
Runs combine_measurements on all image folders in specified samples

Usage:
    python combine_all_measurements.py
    (Edit BASE_PATH and SAMPLES_TO_PROCESS below)
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from csvOps.combine_channel import combine_measurements

# Import BASE_PATH from main.py
try:
    from main import BASE_PATH
except ImportError:
    # Fallback if main.py is not available
    BASE_PATH = "/Users/taeeonkong/Desktop/2025 Fall Images/09-26-2025 DLBCL"

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

SAMPLES_TO_PROCESS = [1,2]  # List of sample numbers to process, e.g., [1, 2, 3] or [1]

# ============================================================================
# MAIN
# ============================================================================

def main():
    import re

    base_path_obj = Path(BASE_PATH)

    if not base_path_obj.exists():
        print(f"✗ ERROR: Base path not found: {BASE_PATH}")
        import sys
        sys.exit(1)

    # Find all sample folders
    sample_folders = []
    for item in base_path_obj.iterdir():
        if item.is_dir() and item.name.lower().startswith('sample'):
            sample_folders.append(item.name)

    # Sort by sample number
    def extract_sample_number(name):
        match = re.search(r'sample(\d+)', name, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    sample_folders = sorted(sample_folders, key=extract_sample_number)

    # Filter to only requested samples
    sample_folders = [s for s in sample_folders if extract_sample_number(s) in SAMPLES_TO_PROCESS]

    if not sample_folders:
        print(f"✗ ERROR: No sample folders found matching {SAMPLES_TO_PROCESS} in {BASE_PATH}")
        import sys
        sys.exit(1)

    print("\n" + "="*80)
    print("BATCH COMBINE MEASUREMENTS")
    print("="*80)
    print(f"Base path: {BASE_PATH}")
    print(f"Processing samples: {', '.join(sample_folders)}")
    print("="*80 + "\n")

    # Process each sample
    total_processed = 0
    total_skipped = 0
    total_failed = 0

    for sample_idx, sample_folder in enumerate(sample_folders, 1):
        sample_path = base_path_obj / sample_folder

        # Find all subdirectories (image folders)
        image_folders = [item.name for item in sample_path.iterdir() if item.is_dir()]
        image_folders = sorted(image_folders)

        if not image_folders:
            print(f"⚠️  No image folders found in {sample_folder}, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"SAMPLE {sample_idx}/{len(sample_folders)}: {sample_folder}")
        print(f"Found {len(image_folders)} image folders: {', '.join(image_folders)}")
        print(f"{'='*80}\n")

        # Process each image
        for img_idx, image_number in enumerate(image_folders, 1):
            print(f"\n{'-'*80}")
            print(f"Image {img_idx}/{len(image_folders)}: {sample_folder}/{image_number}")
            print(f"{'-'*80}\n")

            try:
                result = combine_measurements(
                    sample_folder=sample_folder,
                    image_number=image_number,
                    base_path=BASE_PATH,
                    verbose=True
                )

                if result['success']:
                    if result.get('skipped'):
                        total_skipped += 1
                        print(f"⊘ Skipped {sample_folder}/{image_number} (no CSVs found)")
                    else:
                        total_processed += 1
                        print(f"✓ Successfully processed {sample_folder}/{image_number}")
                else:
                    total_failed += 1
                    print(f"✗ Failed to process {sample_folder}/{image_number}")
                    print(f"   Error: {result.get('error', 'Unknown error')}")

            except Exception as e:
                total_failed += 1
                print(f"✗ Failed to process {sample_folder}/{image_number}")
                print(f"   Error: {str(e)}")

    # Summary
    print("\n" + "="*80)
    print("BATCH COMBINE COMPLETE")
    print("="*80)
    print(f"Total samples: {len(sample_folders)}")
    print(f"✓ Combined successfully: {total_processed}")
    print(f"⊘ Skipped (no CSVs): {total_skipped}")
    print(f"✗ Failed: {total_failed}")
    print("="*80 + "\n")

    if total_failed > 0:
        import sys
        sys.exit(1)

    # Auto-combine all samples into master CSV
    print("\n" + "="*80)
    print("AUTO-COMBINING ALL SAMPLES")
    print("="*80 + "\n")

    try:
        from csvOps import combine_all
        combine_all.main()
    except Exception as e:
        print(f"\n✗ Failed to create master combined CSV: {str(e)}")
        import sys
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
