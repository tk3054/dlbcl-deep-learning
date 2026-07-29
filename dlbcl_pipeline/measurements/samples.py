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
    python -m dlbcl_pipeline.measurements.samples
"""

import pandas as pd
from pathlib import Path
import sys
import re

from dlbcl_pipeline.measurements.channels import combine_measurements
from dlbcl_pipeline.utils.config_helpers import (
    filter_image_folders,
    normalize_image_filter_config,
    selected_sample_numbers,
)
from dlbcl_pipeline.utils.name_builder import extract_image_number, extract_donor_id_from_path

# ============================================================================
# CONFIGURATION
# ============================================================================

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


def _ordered_unique(values):
    return list(dict.fromkeys(values))

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
    selected_samples = selected_sample_numbers(sample_numbers)
    if selected_samples:
        all_samples = [
            s for s in all_samples if extract_sample_number(s.name) in selected_samples
        ]

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

def combine_images_per_sample(
    base_path,
    sample_folders,
    channel_config=None,
    images_to_process=None,
    verbose=True,
):
    """For each sample, combine all image CSVs into sample-level CSV"""
    base_path_obj = Path(base_path)
    channel_config = dict(channel_config or {})
    channels_list = _ordered_unique(channel_config.keys()) if channel_config else None
    filters_map, filters_default = normalize_image_filter_config(images_to_process)
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
                    base_path=base_path,
                    include_channels=channels_list,
                    channel_config=channel_config,
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

        # Add unique_id column (format: donorId_sampleNumber_image_cellid)
        donor_id = extract_donor_id_from_path(base_path_obj) or "UnknownDonor"
        sample_df['unique_id'] = sample_df.apply(
            lambda row: f"{donor_id}_{extract_sample_number(row['sample'])}_{row['image']}_{int(row['cell_id']):02d}",
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
# MAIN ORCHESTRATION
# ============================================================================

def combine_donor_measurements(
    base_path,
    channel_config=None,
    samples_to_process=None,
    images_to_process=None,
    output_file=OUTPUT_FILE,
    verbose=True,
):
    channel_config = dict(channel_config or {})
    channels_list = _ordered_unique(channel_config.keys()) if channel_config else None

    if verbose:
        print("\n" + "="*80)
        print("MASTER CSV COMBINATION")
        print("="*80)
        print(f"Base path: {base_path}")
        print(f"Processing samples: {samples_to_process if samples_to_process else 'ALL'}")
        if channels_list:
            print(f"Channel set: {', '.join(channels_list)}")
        print("="*80)

    sample_folders = get_sample_folders(base_path, samples_to_process)

    if not sample_folders:
        if verbose:
            print(f"\n✗ No sample folders found!")
        return {'success': False, 'error': 'No sample folders found'}

    if verbose:
        print(f"\nFound {len(sample_folders)} sample(s): {', '.join([s.name for s in sample_folders])}")

    delete_old_sample_csvs(sample_folders, verbose=verbose)

    sample_results = combine_images_per_sample(
        base_path=base_path,
        sample_folders=sample_folders,
        channel_config=channel_config,
        images_to_process=images_to_process,
        verbose=verbose,
    )

    if not sample_results:
        if verbose:
            print("\n✗ No samples were successfully processed!")
        return {'success': False, 'error': 'No samples were successfully processed'}

    delete_old_master_csv(base_path, output_file, verbose=verbose)
    result = combine_all_samples(base_path, sample_folders, output_file, verbose=verbose)

    if not result:
        if verbose:
            print("\n✗ Failed to create master CSV!")
        return {'success': False, 'error': 'Failed to create master CSV'}

    if verbose:
        print("\n" + "="*80)
        print("ALL COMBINATION COMPLETE!")
        print("="*80)
        print(f"✓ Processed {len(sample_folders)} samples")
        print(f"✓ Total cells: {result['num_cells']}")
        print(f"✓ Output: {result['output_path']}")
        print("="*80 + "\n")

    return {'success': True, **result}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Combine image/sample CSVs into one donor-level CSV.")
    parser.add_argument("--base-path", required=True, help="Donor directory containing sample folders")
    parser.add_argument("--samples", nargs="*", type=int, help="Sample numbers to include")
    parser.add_argument("--images", nargs="*", type=int, help="Image numbers to include for all samples")
    parser.add_argument("--output-file", default=OUTPUT_FILE, help="Output CSV filename")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    args = parser.parse_args()

    result = combine_donor_measurements(
        base_path=args.base_path,
        samples_to_process=args.samples,
        images_to_process=set(args.images or []),
        output_file=args.output_file,
        verbose=not args.quiet,
    )

    if not result['success']:
        print(f"Error: {result.get('error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
