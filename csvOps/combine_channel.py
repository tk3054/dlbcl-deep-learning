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
SAMPLE_FOLDER = "sample2"  # Options: "sample1", "sample2", "sample3"
IMAGE_NUMBER = "1"         # Options: "1", "2", "3", "4", etc.

# Auto-generated paths
BASE_PATH = "/Users/taeeonkong/Desktop/2025 Fall Images/09-26-2025 DLBCL"
BASE_DIR = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def combine_measurements(sample_folder, image_number, base_path, channel_config=None, verbose=True):
    """
    Combine all channel measurement CSVs into a single table.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        channel_config: Dictionary mapping channel keys to filenames (optional)
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

    # Define channel patterns to search for (flexible naming, including car- prefix)
    channel_patterns = {
        'actin': ['actin-fitc-measurements.csv', 'car-actin-fitc-measurements.csv', 'Actin-FITC-measurements.csv', 'actin_measurements.csv'],
        'cd4': ['cd4-percp-measurements.csv', 'car-cd4-percp-measurements.csv', 'CD4-PerCP-measurements.csv', 'cd4_measurements.csv'],
        'cd45ra_af647': ['cd45ra-af647-measurements.csv', 'car-cd45ra-af647-measurements.csv', 'CD45RA-AF647-measurements.csv', 'cd45ra_measurements.csv'],
        'cd45ra_sparkviolet': ['cd45ra-sparkviolet-measurements.csv', 'car-cd45ra-sparkviolet-measurements.csv', 'CD45RA-SparkViolet-measurements.csv'],
        'cd19car': ['cd19car-af647-measurements.csv', 'car-cd19car-af647-measurements.csv', 'CD19CAR-AF647-measurements.csv'],
        'ccr7': ['ccr7-pe-measurements.csv', 'car-ccr7-pe-measurements.csv', 'CCR7-PE-measurements.csv', 'ccr7_measurements.csv']
    }

    # Find CSVs that exist (try all patterns)
    available_channels = {}
    for channel_name, patterns in channel_patterns.items():
        found = False
        for pattern in patterns:
            csv_path = base_dir / pattern
            if csv_path.exists():
                available_channels[channel_name] = csv_path
                found = True
                break

        # If no pattern matched, try glob pattern as last resort
        if not found:
            # Try case-insensitive glob for any file containing the channel name
            matching_files = list(base_dir.glob(f"*{channel_name}*measurements.csv"))
            if matching_files:
                available_channels[channel_name] = matching_files[0]
                found = True

        if not found and verbose:
            print(f"⚠️  Missing: {channel_name}-measurements.csv")

    if not available_channels:
        if verbose:
            print("⚠️  No measurement CSVs found, skipping...\n")
        return {
            'success': True,  # Don't fail, just skip
            'skipped': True,
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
    intensity_cols = ['mean', 'median', 'std', 'min', 'max', 'intden', 'rawintden']

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
            if 'cell_id' not in df.columns:
                if verbose:
                    print(f"  ⚠️  Warning: 'cell_id' column not found in {channel_name}, cannot use as base")
                continue

            cols_to_keep = ['cell_id']
            # Add morphology columns that exist
            for col in morphology_cols:
                if col in df.columns:
                    cols_to_keep.append(col)
            # Add intensity columns that exist
            for col in intensity_cols:
                if col in df.columns:
                    cols_to_keep.append(col)

            combined_df = df[cols_to_keep].copy()

            # Rename intensity columns with channel prefix (don't rename cell_id or morphology)
            rename_map = {col: f"{channel_name}_{col}" for col in intensity_cols if col in cols_to_keep}
            combined_df = combined_df.rename(columns=rename_map)

            if verbose:
                print(f"  Base table: {len(combined_df)} cells, {len(combined_df.columns)} columns")
        else:
            # Subsequent channels: keep cell_id + intensity columns only
            # Make sure we always keep cell_id
            if 'cell_id' not in df.columns:
                if verbose:
                    print(f"  ⚠️  Warning: 'cell_id' column not found in {channel_name}, skipping this channel")
                continue

            cols_to_keep = ['cell_id']
            # Add intensity columns that exist
            for col in intensity_cols:
                if col in df.columns:
                    cols_to_keep.append(col)

            channel_df = df[cols_to_keep].copy()

            # Rename intensity columns with channel prefix (don't rename cell_id)
            rename_map = {col: f"{channel_name}_{col}" for col in intensity_cols if col in cols_to_keep}
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

    # Add unique_id column (format: sample_image_cellid)
    combined_df['unique_id'] = combined_df.apply(
        lambda row: f"{row['sample']}_{row['image']}_{row['cell_id']}", axis=1
    )
    # Move unique_id to first column
    cols = list(combined_df.columns)
    cols = [cols[-1]] + cols[:-1]  # Move last column to first
    combined_df = combined_df[cols]

    # Save combined CSV to image folder (e.g., sample1/1/combined_measurements.csv)
    output_csv = base_dir / 'combined_measurements.csv'

    # Always overwrite (don't append)
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
