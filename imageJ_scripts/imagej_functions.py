#!/usr/bin/env python3
"""
ImageJ macro wrapper functions
Provides parameterized functions for running ImageJ macros via PyImageJ
"""

import imagej
import os
import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _suppress_output():
    """Suppress ImageJ stdout/stderr noise during init and macro execution."""
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
    # Initialize ImageJ if needed
    if ij is None:
        if verbose:
            print("Initializing ImageJ...")
        with _suppress_output():
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
        mask_msg = " (no mask)" if not create_mask else ""
        print(f"Preprocessing with ImageJ{mask_msg}...")

    try:
        with _suppress_output():
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
            print(f"  ✓ Created raw and mask files")

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


def load_rois(sample_folder, image_number, base_path, roi_dir_name='cell_rois', ij=None, verbose=True):
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
    # Initialize ImageJ if needed
    if ij is None:
        if verbose:
            print("Initializing ImageJ...")
        with _suppress_output():
            ij = imagej.init('sc.fiji:fiji')

    # Read macro
    macro_path = Path(__file__).parent / "load_ROIs.ijm"
    with open(macro_path, 'r') as f:
        macro_code = f.read()

    # Update parameters
    macro_code_updated = macro_code.replace(
        'sampleFolder = "SAMPLE_PLACEHOLDER";',
        f'sampleFolder = "{sample_folder}";'
    ).replace(
        'imageNumber = "IMAGE_PLACEHOLDER";',
        f'imageNumber = "{image_number}";'
    ).replace(
        'basePath = "BASE_PATH_PLACEHOLDER";',
        f'basePath = "{base_path}";'
    ).replace(
        'ROI_DIR_PLACEHOLDER',
        roi_dir_name
    )

    # Count ROI files
    base_dir = f"{base_path}/{sample_folder}/{image_number}"
    roi_dir = Path(base_dir) / roi_dir_name
    num_rois = len(list(roi_dir.glob("*.tif"))) if roi_dir.exists() else 0

    if verbose:
        print(f"Loading ROIs in ImageJ ({num_rois} cells)...")

    try:
        with _suppress_output():
            ij.py.run_macro(macro_code_updated)

        if verbose:
            print(f"  ✓ Loaded {num_rois} ROIs")

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


def preprocess_channels(sample_folder, image_number, base_path, channel_config=None, ij=None, verbose=True):
    """
    Preprocess channel TIF files and save processed versions.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        channel_config: Dictionary mapping channel keys to filenames
        ij: ImageJ instance (if None, will initialize)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'num_processed': Number of processed files created
    """
    # Initialize ImageJ if needed
    if ij is None:
        if verbose:
            print("Initializing ImageJ...")
        with _suppress_output():
            ij = imagej.init('sc.fiji:fiji')

    # Read macro
    macro_path = Path(__file__).parent / "preprocess_channels.ijm"
    with open(macro_path, 'r') as f:
        macro_code = f.read()

    # Update parameters
    macro_code_updated = macro_code.replace(
        'sampleFolder = "SAMPLE_PLACEHOLDER";',
        f'sampleFolder = "{sample_folder}";'
    ).replace(
        'imageNumber = "IMAGE_PLACEHOLDER";',
        f'imageNumber = "{image_number}";'
    ).replace(
        'basePath = "BASE_PATH_PLACEHOLDER";',
        f'basePath = "{base_path}";'
    )

    if verbose:
        print(f"Preprocessing channel images...")

    try:
        with _suppress_output():
            ij.py.run_macro(macro_code_updated)

        # Check for created processed files
        base_dir = Path(f"{base_path}/{sample_folder}/{image_number}")
        processed_files = list(base_dir.glob("processed_*.tif"))

        if verbose:
            print(f"  ✓ Created {len(processed_files)} processed channel files")

        return {
            'success': True,
            'num_processed': len(processed_files)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'num_processed': 0
        }


def make_duplicate_jpg(sample_folder, image_number, base_path, channel_config=None, ij=None, verbose=True):
    """
    Create JPG duplicates of all channel TIF files.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        channel_config: Dictionary mapping channel keys to filenames
        ij: ImageJ instance (if None, will initialize)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'num_jpgs': Number of JPG files created
    """
    # Initialize ImageJ if needed
    if ij is None:
        if verbose:
            print("Initializing ImageJ...")
        with _suppress_output():
            ij = imagej.init('sc.fiji:fiji')

    # Read macro
    macro_path = Path(__file__).parent / "make_duplicate_jpg.ijm"
    with open(macro_path, 'r') as f:
        macro_code = f.read()

    # Update parameters
    macro_code_updated = macro_code.replace(
        'sampleFolder = "SAMPLE_PLACEHOLDER";',
        f'sampleFolder = "{sample_folder}";'
    ).replace(
        'imageNumber = "IMAGE_PLACEHOLDER";',
        f'imageNumber = "{image_number}";'
    ).replace(
        'basePath = "BASE_PATH_PLACEHOLDER";',
        f'basePath = "{base_path}";'
    )

    if verbose:
        print(f"Creating JPG duplicates for all channels...")

    try:
        with _suppress_output():
            ij.py.run_macro(macro_code_updated)

        # Check for created JPG files
        base_dir = Path(f"{base_path}/{sample_folder}/{image_number}")
        jpg_files = list(base_dir.glob("*_raw.jpg"))

        if verbose:
            print(f"  ✓ Created {len(jpg_files)} JPG duplicate files")

        return {
            'success': True,
            'num_jpgs': len(jpg_files)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'num_jpgs': 0
        }


def extract_channel_measurements(sample_folder, image_number, base_path, channel_config=None, roi_dir_name='cell_rois', ij=None, verbose=True):
    """
    Extract fluorescence measurements for all channels.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        channel_config: Dictionary mapping channel keys to filenames
        base_path: Base directory path (e.g., "/path/to/data")
        ij: ImageJ instance (if None, will initialize)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'measurement_files': List of paths to CSV measurement files
    """
    # Use default channel config if not provided
    if channel_config is None:
        channel_config = {
            'actin': 'Actin-FITC.tif',
            'cd4': 'CD4-PerCP.tif',
            'cd45ra_sparkviolet': 'CD45RA-SparkViolet.tif',
            'cd45ra_PacBlue': 'CD45RA-PacBlue.tif',
            'cd19car': 'CD19CAR-AF647.tif',
            'ccr7': 'CCR7-PE.tif',
        }

    # Initialize ImageJ if needed
    if ij is None:
        if verbose:
            print("Initializing ImageJ...")
        with _suppress_output():
            ij = imagej.init('sc.fiji:fiji')

    # Read macro
    macro_path = Path(__file__).parent / "extract_channels.ijm"
    with open(macro_path, 'r') as f:
        macro_code = f.read()

    # Update parameters
    macro_code_updated = macro_code.replace(
        'sampleFolder = "SAMPLE_PLACEHOLDER";',
        f'sampleFolder = "{sample_folder}";'
    ).replace(
        'imageNumber = "IMAGE_PLACEHOLDER";',
        f'imageNumber = "{image_number}";'
    ).replace(
        'basePath = "BASE_PATH_PLACEHOLDER";',
        f'basePath = "{base_path}";'
    ).replace(
        'actinFile = "ACTIN_FILE_PLACEHOLDER";',
        f'actinFile = "{channel_config.get("actin", "Actin-FITC.tif")}";'
    ).replace(
        'cd4File = "CD4_FILE_PLACEHOLDER";',
        f'cd4File = "{channel_config.get("cd4", "CD4-PerCP.tif")}";'
    ).replace(
        'cd45raSparkVioletFile = "CD45RA_SPARKVIOLET_FILE_PLACEHOLDER";',
        f'cd45raSparkVioletFile = "{channel_config.get("cd45ra_sparkviolet", channel_config.get("cd45ra_PacBlue", "CD45RA-SparkViolet.tif"))}";'
    ).replace(
        'cd19carFile = "CD19CAR_FILE_PLACEHOLDER";',
        f'cd19carFile = "{channel_config.get("cd19car", "CD19CAR-AF647.tif")}";'
    ).replace(
        'ccr7File = "CCR7_FILE_PLACEHOLDER";',
        f'ccr7File = "{channel_config.get("ccr7", "CCR7-PE.tif")}";'
    ).replace(
        'ROI_DIR_PLACEHOLDER',
        roi_dir_name
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
        print(f"Extracting channel measurements ({num_rois} cells × {len(channel_files)} channels)...")

    try:
        with _suppress_output():
            ij.py.run_macro(macro_code_updated)

        # Check for output files
        measurement_files = []

        for channel in channels:
            csv_path = base_dir / f"{channel}-measurements.csv"
            if csv_path.exists():
                measurement_files.append(str(csv_path))

        if verbose:
            print(f"  ✓ Extracted {len(measurement_files)} channel measurement files")

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
