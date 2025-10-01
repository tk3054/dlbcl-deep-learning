#!/usr/bin/env python3
"""
ImageJ macro wrapper functions
Provides parameterized functions for running ImageJ macros via PyImageJ
"""

import imagej
from pathlib import Path


def preprocess_actin(sample_folder, image_number, base_path, ij=None, create_mask=True, verbose=True):
    """
    Run preprocessing macro to generate raw and optionally mask images.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        ij: ImageJ instance (if None, will initialize)
        create_mask: Whether to create mask (True for watershed, False for Cellpose)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'raw_file': Path to generated raw image
            - 'mask_file': Path to generated mask image (or None if not created)
    """
    if verbose:
        print("\n" + "="*60)
        print("IMAGEJ: PREPROCESSING")
        print("="*60)
        print(f"Sample: {sample_folder}/{image_number}")

    # Initialize ImageJ if needed
    if ij is None:
        if verbose:
            print("Initializing ImageJ...")
        ij = imagej.init('sc.fiji:fiji')

    # Read macro
    macro_path = Path(__file__).parent / "headless_process_actin_fitc.ijm"
    with open(macro_path, 'r') as f:
        macro_code = f.read()

    # Update parameters
    macro_code_updated = macro_code.replace(
        'sampleFolder = "sample1";',
        f'sampleFolder = "{sample_folder}";'
    ).replace(
        'imageNumber = "4";',
        f'imageNumber = "{image_number}";'
    ).replace(
        'basePath = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10";',
        f'basePath = "{base_path}";'
    ).replace(
        'createMask = true;',
        f'createMask = {"true" if create_mask else "false"};'
    )

    if verbose:
        print("Running preprocessing macro...")
        if not create_mask:
            print("  (Mask creation disabled - using Cellpose)")

    try:
        ij.py.run_macro(macro_code_updated)

        # Check if files were created
        base_dir = f"{base_path}/{sample_folder}/{image_number}"
        raw_files = list(Path(base_dir).glob("*_raw.jpg"))
        mask_files = list(Path(base_dir).glob("*_mask.jpg"))

        # Validate raw file always exists
        if not raw_files:
            return {
                'success': False,
                'error': 'Raw file not created'
            }

        # Validate mask file only if createMask was True
        if create_mask and not mask_files:
            return {
                'success': False,
                'error': 'Mask file not created (was requested)'
            }

        if verbose:
            print("✓ Preprocessing complete")
            print(f"  Raw: {raw_files[0].name}")
            if create_mask and mask_files:
                print(f"  Mask: {mask_files[0].name}")

        return {
            'success': True,
            'raw_file': str(raw_files[0]),
            'mask_file': str(mask_files[0]) if mask_files else None
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def load_rois(sample_folder, image_number, base_path, ij=None, verbose=True):
    """
    Load cell ROIs onto original image in ImageJ.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        ij: ImageJ instance (if None, will initialize)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'num_rois': Number of ROIs loaded
    """
    if verbose:
        print("\n" + "="*60)
        print("IMAGEJ: LOAD ROIS")
        print("="*60)
        print(f"Sample: {sample_folder}/{image_number}")

    # Initialize ImageJ if needed
    if ij is None:
        if verbose:
            print("Initializing ImageJ...")
        ij = imagej.init('sc.fiji:fiji')

    # Read macro
    macro_path = Path(__file__).parent / "headless_load_rois.ijm"
    with open(macro_path, 'r') as f:
        macro_code = f.read()

    # Update parameters
    macro_code_updated = macro_code.replace(
        'sampleFolder = "sample1";',
        f'sampleFolder = "{sample_folder}";'
    ).replace(
        'imageNumber = "4";',
        f'imageNumber = "{image_number}";'
    ).replace(
        'basePath = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10";',
        f'basePath = "{base_path}";'
    )

    if verbose:
        print("Loading ROIs...")

    try:
        ij.py.run_macro(macro_code_updated)

        # Count ROI files
        base_dir = f"{base_path}/{sample_folder}/{image_number}"
        roi_dir = Path(base_dir) / "cell_rois"
        num_rois = len(list(roi_dir.glob("*.tif"))) if roi_dir.exists() else 0

        if verbose:
            print(f"✓ Loaded {num_rois} ROIs")

        return {
            'success': True,
            'num_rois': num_rois
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'num_rois': 0
        }


def extract_channel_measurements(sample_folder, image_number, base_path, ij=None, verbose=True):
    """
    Extract fluorescence measurements for all channels.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        ij: ImageJ instance (if None, will initialize)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'measurement_files': List of paths to CSV measurement files
    """
    if verbose:
        print("\n" + "="*60)
        print("IMAGEJ: EXTRACT CHANNEL MEASUREMENTS")
        print("="*60)
        print(f"Sample: {sample_folder}/{image_number}")

    # Initialize ImageJ if needed
    if ij is None:
        if verbose:
            print("Initializing ImageJ...")
        ij = imagej.init('sc.fiji:fiji')

    # Read macro (v3 - builds CSV manually)
    macro_path = Path(__file__).parent / "headless_extract_channels_v3.ijm"
    with open(macro_path, 'r') as f:
        macro_code = f.read()

    # Update parameters
    macro_code_updated = macro_code.replace(
        'sampleFolder = "sample1";',
        f'sampleFolder = "{sample_folder}";'
    ).replace(
        'imageNumber = "4";',
        f'imageNumber = "{image_number}";'
    ).replace(
        'basePath = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10";',
        f'basePath = "{base_path}";'
    )

    # Count ROIs first
    base_dir = Path(f"{base_path}/{sample_folder}/{image_number}")
    roi_dir = base_dir / "cell_rois"
    num_rois = len(list(roi_dir.glob("*.tif"))) if roi_dir.exists() else 0

    # Count channels
    channels = ['actin-fitc', 'cd4-percp', 'cd45ra-af647', 'ccr7-pe']
    channel_files = []
    for ch in channels:
        ch_path = base_dir / f"{ch}.tif"
        if ch_path.exists():
            channel_files.append(ch)

    if verbose:
        print(f"Found {num_rois} ROIs")
        print(f"Found {len(channel_files)} channel images: {', '.join(channel_files)}")
        print(f"Total measurements to extract: {num_rois * len(channel_files)}")
        print("\nExtracting measurements...")

    try:
        ij.py.run_macro(macro_code_updated)

        # Check for output files
        measurement_files = []

        for channel in channels:
            csv_path = base_dir / f"{channel}-measurements.csv"
            if csv_path.exists():
                measurement_files.append(str(csv_path))

                # Count measurements in CSV
                import pandas as pd
                try:
                    df = pd.read_csv(csv_path)
                    num_measurements = len(df)
                    if verbose:
                        print(f"✓ {channel}: {num_measurements} measurements")
                except:
                    if verbose:
                        print(f"✓ {channel}: measurements saved")

        if verbose:
            print(f"\n✓ Complete! Extracted {len(measurement_files)} channel measurement files")
            for file in measurement_files:
                print(f"  • {Path(file).name}")

        return {
            'success': True,
            'measurement_files': measurement_files
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'measurement_files': []
        }
