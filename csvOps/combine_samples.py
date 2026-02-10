#!/usr/bin/env python3
"""
Master CSV Combination Script
Combines all measurements from all images and samples into one master CSV

Flow:
1. Deletes old sample-level combined_measurements.csv files
2. For each sample: combines all image CSVs into sample-level CSV
3. Deletes old all_samples_combined.csv
4. Combines all sample-level CSVs into master all_samples_combined.csv

Usage:
    python csvOps/combine_samples.py
"""

import pandas as pd
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from csvOps.combine_channels import combine_measurements
from utils.channel_aliases import canonicalize_channel_config, canonicalize_channel_list
from utils.config_helpers import normalize_image_filter_config, filter_image_folders
from utils.name_builder import extract_image_number, extract_patient_id_from_path

# Import BASE_PATH / channel config / samples from main.py (single source of truth)
try:
    from main import (
        BASE_PATH,
        CHANNEL_CONFIG,
        SAMPLES_TO_PROCESS,
        IMAGES_TO_PROCESS,
    )
except ImportError as exc:
    raise ImportError("combine_all.py must be run where main.py is importable so channel configuration is shared.") from exc

CHANNEL_CONFIG = canonicalize_channel_config(CHANNEL_CONFIG)
CHANNELS_LIST = canonicalize_channel_list(CHANNEL_CONFIG.keys())

# ============================================================================
# CONFIGURATION
# ============================================================================

# SAMPLES_TO_PROCESS is now imported from main.py above
OUTPUT_FILE = "all_samples_combined.csv"
PLOT_AREA_HISTOGRAM = True
AREA_HISTOGRAM_BINS = 50
AREA_HISTOGRAM_FILE = "cell_area_histogram.png"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_sample_number(name):
    """Extract sample number from folder name"""
    match = re.search(r'sample(\d+)', name, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def get_sample_folders(base_path, sample_numbers=None):
    """Get list of sample folders to process"""
    base_path_obj = Path(base_path)

    if not base_path_obj.exists():
        print(f"✗ ERROR: Base path not found: {base_path}")
        sys.exit(1)

    # Find all sample folders
    all_samples = [item for item in base_path_obj.iterdir()
                   if item.is_dir() and item.name.lower().startswith('sample')]

    # Filter by requested numbers if specified
    if sample_numbers:
        all_samples = [s for s in all_samples if extract_sample_number(s.name) in sample_numbers]

    # Sort by sample number
    all_samples = sorted(all_samples, key=lambda x: extract_sample_number(x.name))

    return all_samples

# ============================================================================
# STEP 1: DELETE OLD SAMPLE-LEVEL CSVs
# ============================================================================

def delete_old_sample_csvs(sample_folders, verbose=True):
    """Delete old sample-level combined_measurements.csv files"""
    if verbose:
        print("\n" + "="*80)
        print("STEP 1: Cleaning old sample-level CSVs")
        print("="*80 + "\n")

    deleted_count = 0
    for sample_folder in sample_folders:
        csv_path = sample_folder / "combined_measurements.csv"
        if csv_path.exists():
            csv_path.unlink()
            deleted_count += 1
            if verbose:
                print(f"  ✓ Deleted {sample_folder.name}/combined_measurements.csv")

    if verbose:
        if deleted_count > 0:
            print(f"\nDeleted {deleted_count} old sample-level CSV(s)")
        else:
            print("No old sample-level CSVs found")

# ============================================================================
# STEP 2: COMBINE IMAGES WITHIN EACH SAMPLE
# ============================================================================

def combine_images_per_sample(sample_folders, verbose=True):
    """For each sample, combine all image CSVs into sample-level CSV"""
    base_path_obj = Path(BASE_PATH)
    filters_map, filters_default = normalize_image_filter_config(IMAGES_TO_PROCESS)
    if verbose:
        print("\n" + "="*80)
        print("STEP 2: Combining images within each sample")
        print("="*80 + "\n")

    sample_results = []

    for sample_folder in sample_folders:
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing {sample_folder.name}")
            print(f"{'='*80}\n")

        # Find all image subdirectories
        image_folders = sorted([item for item in sample_folder.iterdir() if item.is_dir()])
        image_folder_names = [item.name for item in image_folders]
        image_folder_names = filter_image_folders(
            sample_folder.name,
            image_folder_names,
            filters_map,
            filters_default,
            announce=verbose,
        )
        image_folders = [sample_folder / name for name in image_folder_names]

        if not image_folders:
            if verbose:
                print(f"  ⚠️  No image folders found in {sample_folder.name}")
            continue

        if verbose:
            print(f"Found {len(image_folders)} image folders")

        # Collect all image CSVs
        all_image_data = []

        for image_folder in image_folders:
            csv_path = image_folder / "combined_measurements.csv"

            if not csv_path.exists():
                if verbose:
                    print(f"  • {image_folder.name}: generating combined_measurements.csv for configured channels...")
                combine_result = combine_measurements(
                    sample_folder=sample_folder.name,
                    image_number=image_folder.name,
                    base_path=BASE_PATH,
                    include_channels=CHANNELS_LIST,
                    channel_config=CHANNEL_CONFIG,
                    verbose=verbose
                )
                if not combine_result['success'] or combine_result.get('skipped'):
                    if verbose:
                        print(f"    ⚠️  Unable to combine channels for {image_folder.name}: {combine_result.get('error', 'No measurement CSVs found')}")
                    continue

            df = pd.read_csv(csv_path)

            # Remove unique_id column if it exists (we'll regenerate it)
            if 'unique_id' in df.columns:
                df = df.drop(columns=['unique_id'])

            # Ensure sample and image columns exist
            if 'sample' not in df.columns:
                df.insert(0, 'sample', sample_folder.name)
            if 'image' not in df.columns:
                df.insert(1, 'image', image_folder.name)

            all_image_data.append(df)
            if verbose:
                print(f"  ✓ {image_folder.name}: {len(df)} cells")

        if not all_image_data:
            if verbose:
                print(f"\n  ⚠️  No CSV files found in {sample_folder.name}, skipping...")
            continue

        # Combine all images within this sample
        if verbose:
            print(f"\nCombining {len(all_image_data)} image CSVs...")

        sample_df = pd.concat(all_image_data, ignore_index=True)

        # Normalize image column to zero-padded numbers
        def _image_number(text):
            value = extract_image_number(text)
            return f"{int(value):02d}" if value is not None else str(text)

        sample_df['image'] = sample_df['image'].apply(_image_number)

        # Add unique_id column (format: patientId_sampleNumber_image_cellid)
        patient_id = extract_patient_id_from_path(base_path_obj) or "UnknownPatient"
        sample_df['unique_id'] = sample_df.apply(
            lambda row: f"{patient_id}_{extract_sample_number(row['sample'])}_{row['image']}_{int(row['cell_id']):02d}",
            axis=1,
        )
        # Move unique_id to first column
        cols = list(sample_df.columns)
        cols = [cols[-1]] + cols[:-1]
        sample_df = sample_df[cols]

        # Save sample-level CSV
        output_path = sample_folder / "combined_measurements.csv"
        sample_df.to_csv(output_path, index=False)

        if verbose:
            print(f"\n✓ Saved {sample_folder.name}/combined_measurements.csv")
            print(f"  Total cells: {len(sample_df)}")
            print(f"  Columns: {len(sample_df.columns)}")

        sample_results.append({
            'sample': sample_folder.name,
            'num_cells': len(sample_df),
            'num_images': len(all_image_data)
        })

    return sample_results

# ============================================================================
# STEP 3: DELETE OLD MASTER CSV
# ============================================================================

def delete_old_master_csv(base_path, output_file, verbose=True):
    """Delete old all_samples_combined.csv"""
    if verbose:
        print("\n" + "="*80)
        print("STEP 3: Cleaning old master CSV")
        print("="*80 + "\n")

    master_csv = Path(base_path) / output_file
    if master_csv.exists():
        master_csv.unlink()
        if verbose:
            print(f"  ✓ Deleted {output_file}")
    else:
        if verbose:
            print(f"  No old {output_file} found")

# ============================================================================
# STEP 4: COMBINE ALL SAMPLES INTO MASTER CSV
# ============================================================================

def combine_all_samples(base_path, sample_folders, output_file, verbose=True):
    """Combine all sample-level CSVs into one master CSV"""
    if verbose:
        print("\n" + "="*80)
        print("STEP 4: Combining all samples into master CSV")
        print("="*80 + "\n")

    all_data = []

    for sample_folder in sample_folders:
        csv_file = sample_folder / "combined_measurements.csv"

        if csv_file.exists():
            if verbose:
                print(f"Reading {sample_folder.name}/combined_measurements.csv...")
            df = pd.read_csv(csv_file)
            all_data.append(df)
            if verbose:
                print(f"  ✓ Found {len(df)} cells")
        else:
            if verbose:
                print(f"  ⚠️  No combined_measurements.csv in {sample_folder.name}")

    if not all_data:
        print(f"\n✗ No sample-level CSV files found!")
        return None

    # Combine all dataframes
    if verbose:
        print(f"\nCombining {len(all_data)} sample CSV(s)...")

    combined_df = pd.concat(all_data, ignore_index=True)

    # Add global sequential cell ID
    if verbose:
        print(f"Adding global cell IDs...")
    combined_df.insert(0, 'global_cell_id', range(1, len(combined_df) + 1))


    # Save combined file
    output_path = Path(base_path) / output_file
    combined_df.to_csv(output_path, index=False)

    # Plot histogram of cell area (if available)
    if PLOT_AREA_HISTOGRAM:
        if 'area' in combined_df.columns:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Headless-safe
                import matplotlib.pyplot as plt

                area_data = combined_df['area'].dropna()
                plt.figure(figsize=(8, 5))
                plt.hist(area_data, bins=AREA_HISTOGRAM_BINS, color="#2a6f97", edgecolor="#1b3b5a")
                plt.title("Cell Area Histogram")
                plt.xlabel("Area (um^2)")
                plt.ylabel("Count")
                plt.tight_layout()

                hist_path = Path(base_path) / AREA_HISTOGRAM_FILE
                plt.savefig(hist_path, dpi=150)
                plt.close()

                if verbose:
                    print(f"✓ Saved area histogram: {hist_path}")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Failed to plot area histogram: {e}")
        elif verbose:
            print("⚠️  'area' column not found; skipping histogram.")

    if verbose:
        print(f"\n{'='*80}")
        print("MASTER CSV COMPLETE")
        print(f"{'='*80}")
        print(f"Output: {output_file}")
        print(f"Total samples: {len(all_data)}")
        print(f"Total cells: {len(combined_df)}")
        print(f"Columns: {len(combined_df.columns)}")
        print(f"{'='*80}\n")

    return {
        'output_path': str(output_path),
        'num_samples': len(all_data),
        'num_cells': len(combined_df)
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("MASTER CSV COMBINATION")
    print("="*80)
    print(f"Base path: {BASE_PATH}")
    print(f"Processing samples: {SAMPLES_TO_PROCESS if SAMPLES_TO_PROCESS else 'ALL'}")
    if CHANNELS_LIST:
        print(f"Channel set: {', '.join(CHANNELS_LIST)}")
    print("="*80)

    # Get sample folders
    sample_folders = get_sample_folders(BASE_PATH, SAMPLES_TO_PROCESS)

    if not sample_folders:
        print(f"\n✗ No sample folders found!")
        sys.exit(1)

    print(f"\nFound {len(sample_folders)} sample(s): {', '.join([s.name for s in sample_folders])}")

    # Step 1: Delete old sample-level CSVs
    delete_old_sample_csvs(sample_folders, verbose=True)

    # Step 2: Combine images within each sample
    sample_results = combine_images_per_sample(sample_folders, verbose=True)

    if not sample_results:
        print("\n✗ No samples were successfully processed!")
        sys.exit(1)

    # Step 3: Delete old master CSV
    delete_old_master_csv(BASE_PATH, OUTPUT_FILE, verbose=True)

    # Step 4: Combine all samples into master CSV
    result = combine_all_samples(BASE_PATH, sample_folders, OUTPUT_FILE, verbose=True)

    if not result:
        print("\n✗ Failed to create master CSV!")
        sys.exit(1)

    # Final summary
    print("\n" + "="*80)
    print("ALL COMBINATION COMPLETE!")
    print("="*80)
    print(f"✓ Processed {len(sample_folders)} samples")
    print(f"✓ Total cells: {result['num_cells']}")
    print(f"✓ Output: {result['output_path']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
