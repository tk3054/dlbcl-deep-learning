#!/usr/bin/env python3
"""
Combine Channel Measurements
Merges all channel measurement CSVs into a single wide-format table

Each channel's measurements are prefixed with the channel name.
Morphology columns (area, x, y, etc.) are kept from first channel only.

Usage:
    python combine_measurements.py
    (Edit SAMPLE_FOLDER and IMAGE_NUMBER below to change inputs)
"""

import pandas as pd
from pathlib import Path


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Change these values to process different samples
SAMPLE_FOLDER = "sample1"  # Options: "sample1", "sample2", "sample3"
IMAGE_NUMBER = "5"         # Options: "1", "2", "3", "4", etc.

# Auto-generated paths
BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"
BASE_DIR = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def combine_measurements(sample_folder, image_number, base_path, verbose=True):
    """
    Combine all channel measurement CSVs into a single table.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'output_csv': Path to combined CSV
            - 'num_cells': Number of cells in combined table
            - 'num_channels': Number of channels combined
    """
    if verbose:
        print("="*60)
        print("COMBINE CHANNEL MEASUREMENTS")
        print("="*60)
        print(f"Sample: {sample_folder}/{image_number}\n")

    # Build paths
    base_dir = Path(f"{base_path}/{sample_folder}/{image_number}")

    # Define channels and their CSV files
    channels = {
        'actin': base_dir / 'actin-fitc-measurements.csv',
        'cd4': base_dir / 'cd4-percp-measurements.csv',
        'cd45ra': base_dir / 'cd45ra-af647-measurements.csv',
        'ccr7': base_dir / 'ccr7-pe-measurements.csv'
    }

    # Check which CSVs exist
    available_channels = {}
    for channel_name, csv_path in channels.items():
        if csv_path.exists():
            available_channels[channel_name] = csv_path
        elif verbose:
            print(f"⚠️  Missing: {csv_path.name}")

    if not available_channels:
        return {
            'success': False,
            'error': 'No measurement CSVs found',
            'output_csv': None,
            'num_cells': 0,
            'num_channels': 0
        }

    if verbose:
        print(f"Found {len(available_channels)} channel CSVs:")
        for name in available_channels.keys():
            print(f"  • {name}")
        print()

    # Columns to keep from each channel (intensity-related only)
    intensity_cols = ['mean', 'std', 'min', 'max', 'intden', 'rawintden']

    # Morphology columns to keep (from first channel only)
    morphology_cols = ['area', 'x', 'y', 'circ', 'ar', 'round', 'solidity']

    # Load and merge CSVs
    combined_df = None

    for i, (channel_name, csv_path) in enumerate(available_channels.items()):
        if verbose:
            print(f"Loading {channel_name}...")

        df = pd.read_csv(csv_path)

        if i == 0:
            # First channel: keep cell_id + morphology + intensity columns
            cols_to_keep = ['cell_id'] + morphology_cols + intensity_cols
            combined_df = df[cols_to_keep].copy()

            # Rename intensity columns with channel prefix
            rename_map = {col: f"{channel_name}_{col}" for col in intensity_cols}
            combined_df = combined_df.rename(columns=rename_map)

            if verbose:
                print(f"  Base table: {len(combined_df)} cells, {len(combined_df.columns)} columns")
        else:
            # Subsequent channels: keep cell_id + intensity columns only
            cols_to_keep = ['cell_id'] + intensity_cols
            channel_df = df[cols_to_keep].copy()

            # Rename intensity columns with channel prefix
            rename_map = {col: f"{channel_name}_{col}" for col in intensity_cols}
            channel_df = channel_df.rename(columns=rename_map)

            # Merge with combined table
            combined_df = combined_df.merge(channel_df, on='cell_id', how='outer')

            if verbose:
                print(f"  Merged: {len(combined_df)} cells, {len(combined_df.columns)} columns")

    # Sort by cell_id
    combined_df = combined_df.sort_values('cell_id').reset_index(drop=True)

    # Add sample and image info columns
    combined_df.insert(0, 'sample', sample_folder)
    combined_df.insert(1, 'image', image_number)

    # Save combined CSV to sample folder (e.g., 1to10/sample1/combined_measurements.csv)
    output_dir = Path(base_path) / sample_folder
    output_csv = output_dir / 'combined_measurements.csv'

    # Check if file exists - if so, append without header
    if output_csv.exists():
        combined_df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        combined_df.to_csv(output_csv, index=False)

    if verbose:
        print(f"\n{'='*60}")
        print("COMBINATION COMPLETE")
        print(f"{'='*60}")
        print(f"Output: {output_csv.name}")
        print(f"Cells: {len(combined_df)}")
        print(f"Columns: {len(combined_df.columns)}")
        print(f"\nColumn groups:")
        print(f"  • cell_id: 1")
        print(f"  • Morphology: {len(morphology_cols)}")
        print(f"  • Channel intensities: {len(intensity_cols)} × {len(available_channels)} = {len(intensity_cols) * len(available_channels)}")
        print(f"  • Total: {len(combined_df.columns)}")

    return {
        'success': True,
        'output_csv': str(output_csv),
        'num_cells': len(combined_df),
        'num_channels': len(available_channels)
    }


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    result = combine_measurements(
        sample_folder=SAMPLE_FOLDER,
        image_number=IMAGE_NUMBER,
        base_path=BASE_PATH,
        verbose=True
    )

    if not result['success']:
        print(f"\n✗ Error: {result['error']}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
