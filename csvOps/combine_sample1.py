#!/usr/bin/env python3
"""
Combine Sample1 Measurements
Combines all image CSVs within sample1 into a single sample-level CSV

Usage:
    python combine_sample1.py
"""

import pandas as pd
from pathlib import Path
import sys

# Import BASE_PATH from main.py
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from main import BASE_PATH
except ImportError:
    # Fallback if main.py is not available
    BASE_PATH = "/Users/taeeonkong/Desktop/2025 Fall Images/09-26-2025 DLBCL"

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_NAME = "sample1"
OUTPUT_FILE = "combined_measurements.csv"

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def combine_sample1(base_path=BASE_PATH, sample_name=SAMPLE_NAME, verbose=True):
    """Combine all image CSVs in sample1 into one sample-level CSV"""

    if verbose:
        print("\n" + "="*80)
        print("COMBINE SAMPLE1 MEASUREMENTS")
        print("="*80)
        print(f"Base path: {base_path}")
        print(f"Sample: {sample_name}")
        print("="*80 + "\n")

    # Build sample path
    sample_folder = Path(base_path) / sample_name

    if not sample_folder.exists():
        print(f"✗ ERROR: Sample folder not found: {sample_folder}")
        sys.exit(1)

    # Delete old sample-level CSV if it exists
    old_csv = sample_folder / OUTPUT_FILE
    if old_csv.exists():
        if verbose:
            print(f"Deleting old {OUTPUT_FILE}...")
        old_csv.unlink()
        if verbose:
            print(f"  ✓ Deleted\n")

    # Find all image subdirectories
    image_folders = sorted([item for item in sample_folder.iterdir() if item.is_dir()])

    if not image_folders:
        print(f"✗ ERROR: No image folders found in {sample_name}")
        sys.exit(1)

    if verbose:
        print(f"Found {len(image_folders)} image folders:")
        for img in image_folders:
            print(f"  • {img.name}")
        print()

    # Collect all image CSVs
    all_image_data = []
    found_count = 0
    missing_count = 0

    for image_folder in image_folders:
        csv_path = image_folder / "combined_measurements.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)

            # Remove unique_id column if it exists (we'll regenerate it)
            if 'unique_id' in df.columns:
                df = df.drop(columns=['unique_id'])

            # Ensure sample and image columns exist
            if 'sample' not in df.columns:
                df.insert(0, 'sample', sample_name)
            if 'image' not in df.columns:
                df.insert(1, 'image', image_folder.name)

            all_image_data.append(df)
            found_count += 1
            if verbose:
                print(f"  ✓ {image_folder.name}: {len(df)} cells")
        else:
            missing_count += 1
            if verbose:
                print(f"  ⚠️  {image_folder.name}: No combined_measurements.csv found")

    if not all_image_data:
        print(f"\n✗ ERROR: No CSV files found in any image folder!")
        sys.exit(1)

    # Combine all images within this sample
    if verbose:
        print(f"\n{'='*80}")
        print(f"Combining {len(all_image_data)} image CSV(s)...")
        print(f"{'='*80}\n")

    sample_df = pd.concat(all_image_data, ignore_index=True)

    # Add unique_id column (format: sample_image_cellid)
    sample_df['unique_id'] = sample_df.apply(
        lambda row: f"{row['sample']}_{row['image']}_{row['cell_id']}", axis=1
    )
    # Move unique_id to first column
    cols = list(sample_df.columns)
    cols = [cols[-1]] + cols[:-1]
    sample_df = sample_df[cols]

    # Save sample-level CSV
    output_path = sample_folder / OUTPUT_FILE
    sample_df.to_csv(output_path, index=False)

    if verbose:
        print(f"{'='*80}")
        print("COMBINATION COMPLETE")
        print(f"{'='*80}")
        print(f"Output: {output_path}")
        print(f"Total images combined: {found_count}")
        print(f"Images skipped (no CSV): {missing_count}")
        print(f"Total cells: {len(sample_df)}")
        print(f"Columns: {len(sample_df.columns)}")
        print(f"{'='*80}\n")

    return {
        'success': True,
        'output_path': str(output_path),
        'num_cells': len(sample_df),
        'num_images': len(all_image_data)
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    result = combine_sample1(verbose=True)

    if not result['success']:
        sys.exit(1)


if __name__ == "__main__":
    main()
