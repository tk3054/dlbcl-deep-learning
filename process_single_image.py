#!/usr/bin/env python3
"""
Complete Cell Analysis Pipeline
Runs the full pipeline for a single image (excluding manual review)

Usage:
    python run_pipeline.py
    (Edit SAMPLE_FOLDER and IMAGE_NUMBER below to change inputs)
"""

import imagej
import os
import sys
from contextlib import contextmanager
from pathlib import Path

# Import pipeline modules
import imageJ_scripts.imagej_functions as imagej_functions
import segmentation.segment_cells_cellpose as segment_cells_cellpose
import segmentation.shrink_roi as shrink_roi
import process_image.make_raw_crops as make_raw_crops
import process_image.pad_raw_crops as pad_raw_crops
import csvOps.combine_channels as combine_channel
from utils.channel_aliases import canonicalize_channel_config, canonicalize_channel_list


# Optional aliases for channel filenames when multiple naming conventions exist
CHANNEL_FILENAME_ALIASES = {
    'cd45ra_sparkviolet': ['CD45RA-PacBlue.tif'],
    'cd45ra_PacBlue': ['CD45RA-SparkViolet.tif'],
}


def resolve_channel_filenames(channel_config, base_path, sample_folder, image_number, verbose=False):
    """
    Resolve channel filenames to match what actually exists on disk.

    Falls back to known aliases (e.g., SparkViolet) when the configured name
    is not found for the current image.
    """
    resolved = channel_config.copy()
    base_dir_path = Path(base_path) / sample_folder / image_number

    for key, filename in channel_config.items():
        candidates = [filename]
        aliases = CHANNEL_FILENAME_ALIASES.get(key, [])

        for alias in aliases:
            if alias not in candidates:
                candidates.append(alias)

        selected = None
        for candidate in candidates:
            if (base_dir_path / candidate).exists() or (base_dir_path / f"processed_{candidate}").exists():
                selected = candidate
                break

        if selected:
            if selected != filename and verbose:
                print(f"  • Channel '{key}': using '{selected}' (found on disk)")
            resolved[key] = selected
        else:
            if verbose:
                print(f"  • Channel '{key}': no matching file found for candidates {candidates}")

    return resolved


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

SAMPLE_FOLDER = "sample2"
IMAGE_NUMBER = "1"
BASE_PATH = "'/Users/taeeonkong/Desktop/113614(FITC-500ms)'"

SEGMENTATION_METHOD = 'cellpose'  # Options: 'cellpose' or 'watershed'

PARAMS = {
    # Segmentation - Cellpose (deep learning)
    'cellpose_model': 'cyto2',
    'cellpose_diameter': 250,      # Increased for larger cells
    'cellpose_flow_threshold': 0.6,  # Higher = tighter fit (was 0.4)
    'cellpose_cellprob_threshold': -2.0,
    'cellpose_use_gpu': True,
    'min_size': 180,
    'max_size': 100000,

    # Filtering
    'consecutive_threshold': 20,

    # ROI Visualization
    'shrink_pixels': 8,           # Number of pixels to shrink ROIs for visualization

    # Soft Edges - Generate multiple versions for comparison
    # DISABLED: Only using hard cutoff (no soft edges) for now
    'soft_edge_methods': [
        {'name': 'hard', 'use_soft_edges': False},
    ]
}


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_pipeline(sample_folder, image_number, base_path,
                 segmentation_method='cellpose',
                 params=None, channel_config=None, combine_channels=None,
                 null_channels=None, ij=None, verbose=False):
    """
    Run complete cell analysis pipeline for one image.

    Args:
        sample_folder: Sample folder name (e.g., "sample1")
        image_number: Image number (e.g., "5")
        base_path: Base directory path
        segmentation_method: 'cellpose' or 'watershed'
        params: Dictionary of pipeline parameters
        channel_config: Dictionary mapping channel keys to filenames
        combine_channels: Optional list of channel keys to include in combined CSV
        null_channels: Optional list of channel keys to nullify after combining
        ij: ImageJ instance (if None, will initialize)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean
            - 'error': Error message if failed
            - 'results': Dict of all step results
    """
    if verbose:
        print("\n" + "="*40)
        print("CELL ANALYSIS PIPELINE")
        print("="*40)
        print(f"Sample: {sample_folder}/{image_number}")
        print(f"Segmentation: {segmentation_method}")
        print("="*40 + "\n")

    results = {}

    # Use default params if not provided
    if params is None:
        params = PARAMS

    # Require caller-provided channel config so it stays defined in main.py
    if channel_config is None:
        return {'success': False, 'error': "channel_config must be provided by caller", 'results': results}
    channel_config = channel_config.copy()

    # Normalize any aliased channel keys so the rest of the pipeline receives
    # the canonical names it expects (e.g., map "cd45ra_PacBlue" to
    # "cd45ra_sparkviolet").
    channel_config = canonicalize_channel_config(channel_config, verbose=verbose)
    combine_channels = canonicalize_channel_list(combine_channels, verbose=verbose)
    null_channels = canonicalize_channel_list(null_channels, verbose=verbose)

    base_dir_path = Path(base_path) / sample_folder / image_number

    channel_config = resolve_channel_filenames(
        channel_config=channel_config,
        base_path=base_path,
        sample_folder=sample_folder,
        image_number=image_number,
        verbose=verbose
    )

    roi_dir_name = "cell_rois"

    try:
        # ====================================================================
        # STEP 1: Initialize ImageJ
        # ====================================================================
        if ij is None:
            if verbose:
                print("STEP 1: Initializing ImageJ...")
            with _suppress_output():
                ij = imagej.init('sc.fiji:fiji')
            if verbose:
                print(f"✓ ImageJ version: {ij.getVersion()}\n")
        results['imagej'] = ij

        # ====================================================================
        # STEP 2: Cellpose Segmentation
        # ====================================================================
        if verbose:
            print("\nSTEP 2: Cellpose Segmentation")
            print("-" * 80)

        result = segment_cells_cellpose.segment_cells_cellpose(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            model_type=params['cellpose_model'],
            diameter=params['cellpose_diameter'],
            flow_threshold=params['cellpose_flow_threshold'],
            cellprob_threshold=params['cellpose_cellprob_threshold'],
            min_size=params['min_size'],
            max_size=params['max_size'],
            use_gpu=params['cellpose_use_gpu'],
            channel_config=channel_config,
            verbose=verbose
        )
        results['segmentation'] = result

        if not result['success']:
            return {'success': False, 'error': f"Segmentation failed: {result['error']}", 'results': results}

        # ====================================================================
        # STEP 3: Shrink ROIs (disabled)
        # ====================================================================
        # if verbose:
        #     print("\nSTEP 3: Shrink ROIs")
        #     print("-" * 80)
        #
        # base_dir = f"{base_path}/{sample_folder}/{image_number}"
        # result_shrink = shrink_roi.shrink_all_rois(
        #     base_dir=base_dir,
        #     shrink_pixels=params.get('shrink_pixels', 3),
        #     output_suffix="_shrunk",
        #     verbose=verbose
        # )
        # results['shrink_rois'] = result_shrink
        #
        # if not result_shrink['success']:
        #     return {'success': False, 'error': f"ROI shrinking failed: {result_shrink['error']}", 'results': results}

        # ====================================================================
        # STEP 4: Preprocess Channels
        # ====================================================================
        if verbose:
            print("\nSTEP 4: Preprocess Channels")
            print("-" * 80)

        result = imagej_functions.preprocess_channels(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            channel_config=channel_config,
            ij=ij,
            verbose=verbose
        )
        results['preprocess_channels'] = result

        if not result['success']:
            return {'success': False, 'error': f"Channel preprocessing failed: {result['error']}", 'results': results}

        # # ====================================================================
        # # STEP 5: Create JPG Duplicates (from processed files)
        # # ====================================================================
        # if verbose:
        #     print("\nSTEP 5: Create JPG Duplicates")
        #     print("-" * 80)

        # result = imagej_functions.make_duplicate_jpg(
        #     sample_folder=sample_folder,
        #     image_number=image_number,
        #     base_path=base_path,
        #     channel_config=channel_config,
        #     ij=ij,
        #     verbose=verbose
        # )
        # results['make_jpg'] = result

        # if not result['success']:
        #     return {'success': False, 'error': f"JPG creation failed: {result['error']}", 'results': results}

        # ====================================================================
        # STEP 6: Create ROI-Masked Raw Crops
        # ====================================================================
        if verbose:
            print("\nSTEP 6: Create ROI-Masked Raw Crops")
            print("-" * 80)

        result_raw_crops = make_raw_crops.make_raw_crops(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            source_image=channel_config.get('actin', 'Actin-FITC.tif'),
            roi_dir_name=roi_dir_name,
            output_dir_name="raw_crops",
            background="transparent",
            verbose=False
        )
        results['raw_crops'] = result_raw_crops

        result_padded = pad_raw_crops.pad_masked_cells(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            target_size=224,
            verbose=False
        )
        results['padded_cells'] = result_padded

        if verbose:
            print("\n  ✓ Generated ROI-masked raw crops and padded cells")

        # ====================================================================
        # STEP 8: Load ROIs in ImageJ
        # ====================================================================
        if verbose:
            print("\nSTEP 8: Load ROIs in ImageJ")
            print("-" * 80)

        result = imagej_functions.load_rois(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            roi_dir_name=roi_dir_name,
            ij=ij,
            verbose=verbose
        )
        results['load_rois'] = result

        if not result['success']:
            return {'success': False, 'error': f"Load ROIs failed: {result['error']}", 'results': results}

        # ====================================================================
        # STEP 9: Extract Channel Measurements
        # ====================================================================
        if verbose:
            print("\nSTEP 9: Extract Channel Measurements")
            print("-" * 80)

        result = imagej_functions.extract_channel_measurements(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            channel_config=channel_config,
            roi_dir_name=roi_dir_name,
            ij=ij,
            verbose=False
        )
        results['measurements'] = result

        if not result['success']:
            return {'success': False, 'error': f"Extract measurements failed: {result['error']}", 'results': results}

        # ====================================================================
        # STEP 10: Filter Bad Cells
        # ====================================================================
        # if verbose:
        #     print("\nSTEP 10: Filter Bad Cells")
        #     print("-" * 80)

        # result = filter_bad_cells.filter_bad_cells(
        #     sample_folder=sample_folder,
        #     image_number=image_number,
        #     base_path=base_path,
        #     consecutive_threshold=params['consecutive_threshold'],
        #     verbose=verbose
        # )
        # results['filter'] = result

        # if not result['success']:
        #     return {'success': False, 'error': f"Filter bad cells failed: {result['error']}", 'results': results}

        # ====================================================================
        # STEP 11: Combine Measurements
        # ====================================================================
        if verbose:
            print("\nSTEP 11: Combine Measurements")
            print("-" * 80)

        result = combine_channel.combine_measurements(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            channel_config=channel_config,
            include_channels=combine_channels,
            null_channels=null_channels,
            verbose=verbose
        )
        results['combine'] = result

        if not result['success']:
            return {'success': False, 'error': f"Combine measurements failed: {result['error']}", 'results': results}

        # ====================================================================
        # COMPLETE
        # ====================================================================
        if verbose:
            print("\n" + "="*40)
            print("PIPELINE COMPLETE!")
            print("="*40)
            print(f"✓ Processed {results['segmentation']['num_cells']} cells")
            # print(f"✓ Quality: {results['filter']['whole_count']} WHOLE, {results['filter']['cut_count']} CUT")
            print(f"✓ Combined CSV: {results['combine']['output_csv']}")
            print("="*40 + "\n")

        return {
            'success': True,
            'results': results
        }

    except Exception as e:
        import traceback
        error_msg = f"Pipeline failed: {str(e)}\n{traceback.format_exc()}"
        return {
            'success': False,
            'error': error_msg,
            'results': results
        }


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    import sys

    # Accept command-line arguments (optional)
    if len(sys.argv) >= 3:
        sample_folder = sys.argv[1]
        image_number = sys.argv[2]
        base_path = sys.argv[3] if len(sys.argv) >= 4 else BASE_PATH
        print(f"Using command-line arguments: {sample_folder}/{image_number}")
    else:
        sample_folder = SAMPLE_FOLDER
        image_number = IMAGE_NUMBER
        base_path = BASE_PATH
        print(f"Using default configuration: {sample_folder}/{image_number}")

    result = run_pipeline(
        sample_folder=sample_folder,
        image_number=image_number,
        base_path=base_path,
        segmentation_method=SEGMENTATION_METHOD,
        params=PARAMS,
        ij=None,
        verbose=True
    )

    if not result['success']:
        print(f"\n✗ ERROR: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
# Suppress ImageJ stdout/stderr during init when needed.
@contextmanager
def _suppress_output():
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
