#!/usr/bin/env python3
"""
Combine Sample Measurements
Combines all image CSVs within a sample into a single sample-level CSV

Usage:
    python combine_sample.py sample1
    python combine_sample.py sample2
"""

import pandas as pd
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.name_builder import extract_image_number, extract_patient_id_from_path

try:
    from main import BASE_PATH
except ImportError:
    BASE_PATH = None

OUTPUT_FILE = "combined_measurements.csv"


def combine_sample(sample_name, base_path, verbose=True):
    """
    Combine all image CSVs in a sample into one sample-level CSV.

    Args:
        sample_name: Name of the sample folder (e.g., "sample1")
        base_path: Base directory path containing sample folders
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'output_path': Path to output CSV
            - 'num_cells': Number of cells in combined table
            - 'num_images': Number of images combined
    """
    if verbose:
        print(f"\n  Combining images for {sample_name}...")

    sample_folder = Path(base_path) / sample_name

    if not sample_folder.exists():
        if verbose:
            print(f"  ⚠️  Sample folder not found: {sample_folder}")
        return {'success': False, 'error': f"Sample folder not found: {sample_folder}"}

    # Delete old sample-level CSV if it exists
    old_csv = sample_folder / OUTPUT_FILE
    if old_csv.exists():
        old_csv.unlink()

    # Find all image subdirectories
    image_folders = sorted([item for item in sample_folder.iterdir() if item.is_dir()])

    if not image_folders:
        if verbose:
            print(f"  ⚠️  No image folders found in {sample_name}")
        return {'success': False, 'error': "No image folders found"}

    # Collect all image CSVs
    all_image_data = []

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

    if not all_image_data:
        if verbose:
            print(f"  ⚠️  No image CSVs found in {sample_name}")
        return {'success': False, 'error': "No image CSVs found"}

    # Combine all images within this sample
    sample_df = pd.concat(all_image_data, ignore_index=True)

    # Normalize image column to zero-padded numbers
    def _image_number(text):
        value = extract_image_number(text)
        return f"{int(value):02d}" if value is not None else str(text)

    sample_df['image'] = sample_df['image'].apply(_image_number)

    # Add unique_id column (format: sampleNumber_image_cellid)
    def _sample_number(text):
        match = re.search(r"(\d+)", str(text))
        return match.group(1) if match else str(text)

    patient_id = extract_patient_id_from_path(Path(base_path)) or "UnknownPatient"
    sample_df['unique_id'] = sample_df.apply(
        lambda row: f"{patient_id}_{_sample_number(row['sample'])}_{_image_number(row['image'])}_{int(row['cell_id']):02d}",
        axis=1,
    )
    # Move unique_id to first column
    cols = list(sample_df.columns)
    cols = [cols[-1]] + cols[:-1]
    sample_df = sample_df[cols]

    # Save sample-level CSV
    output_path = sample_folder / OUTPUT_FILE
    sample_df.to_csv(output_path, index=False)

    if verbose:
        print(f"  ✓ {sample_name}: combined {len(all_image_data)} images, {len(sample_df)} cells → {OUTPUT_FILE}")

    return {
        'success': True,
        'output_path': str(output_path),
        'num_cells': len(sample_df),
        'num_images': len(all_image_data)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine image CSVs within a sample")
    parser.add_argument("sample_name", help="Sample folder name (e.g., sample1)")
    parser.add_argument("--base-path", default=BASE_PATH, help="Base directory path")

    args = parser.parse_args()

    if not args.base_path:
        print("Error: BASE_PATH not set. Provide --base-path or ensure main.py is importable.")
        sys.exit(1)

    result = combine_sample(args.sample_name, args.base_path, verbose=True)

    if not result['success']:
        print(f"Error: {result.get('error')}")
        sys.exit(1)
