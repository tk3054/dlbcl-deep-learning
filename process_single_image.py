#!/usr/bin/env python3
"""
Complete Cell Analysis Pipeline
Runs the full pipeline for a single image (excluding manual review)

Usage:
    python run_pipeline.py
    (Edit SAMPLE_FOLDER and IMAGE_NUMBER below to change inputs)
"""

import imagej

# Import pipeline modules
import imageJ_scripts.imagej_functions as imagej_functions
import segmentation.segment_cells_cellpose as segment_cells_cellpose
import segmentation.shrink_roi as shrink_roi
import process_image.extract_roi_crops as extract_roi_crops
import process_image.visualize_rois_on_jpg as visualize_rois_on_jpg
import csvOps.combine_channel as combine_channel


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

SAMPLE_FOLDER = "sample1"
IMAGE_NUMBER = "1"
BASE_PATH = "/Users/taeeonkong/Desktop/2025 Fall Images/09-26-2025 DLBCL"

SEGMENTATION_METHOD = 'cellpose'  # Options: 'cellpose' or 'watershed'

PARAMS = {
    # Segmentation - Cellpose (deep learning)
    'cellpose_model': 'cyto2',
    'cellpose_diameter': 50,      # Increased for larger cells
    'cellpose_flow_threshold': 0.6,  # Higher = tighter fit (was 0.4)
    'cellpose_cellprob_threshold': 0.0,  # Higher = tighter fit, less background (was -2.0)
    'cellpose_use_gpu': True,
    'min_size': 500,
    'max_size': 20000,            # Increased for larger cells

    # Filtering
    'consecutive_threshold': 20,

    # ROI Visualization
    'shrink_pixels': 8,           # Number of pixels to shrink ROIs for visualization
}


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_pipeline(sample_folder, image_number, base_path,
                 segmentation_method='cellpose',
                 params=None, channel_config=None, ij=None, verbose=False):
    """
    Run complete cell analysis pipeline for one image.

    Args:
        sample_folder: Sample folder name (e.g., "sample1")
        image_number: Image number (e.g., "5")
        base_path: Base directory path
        segmentation_method: 'cellpose' or 'watershed'
        params: Dictionary of pipeline parameters
        channel_config: Dictionary mapping channel keys to filenames
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

    # Use default channel config if not provided
    if channel_config is None:
        channel_config = {
            'actin': 'Actin-FITC.tif',
            'cd4': 'CD4-PerCP.tif',
            'cd45ra_af647': 'CD45RA-AF647.tif',
            'cd45ra_sparkviolet': 'CD45RA-SparkViolet.tif',
            'cd19car': 'CD19CAR-AF647.tif',
            'ccr7': 'CCR7-PE.tif',
        }

    try:
        # ====================================================================
        # STEP 1: Initialize ImageJ
        # ====================================================================
        if ij is None:
            if verbose:
                print("STEP 1: Initializing ImageJ...")
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
        # STEP 3: Shrink ROIs
        # ====================================================================
        if verbose:
            print("\nSTEP 3: Shrink ROIs")
            print("-" * 80)

        base_dir = f"{base_path}/{sample_folder}/{image_number}"
        result_shrink = shrink_roi.shrink_all_rois(
            base_dir=base_dir,
            shrink_pixels=params.get('shrink_pixels', 3),
            output_suffix="_shrunk",
            verbose=verbose
        )
        results['shrink_rois'] = result_shrink

        if not result_shrink['success']:
            return {'success': False, 'error': f"ROI shrinking failed: {result_shrink['error']}", 'results': results}

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

        # ====================================================================
        # STEP 5: Create JPG Duplicates (from processed files)
        # ====================================================================
        if verbose:
            print("\nSTEP 5: Create JPG Duplicates")
            print("-" * 80)

        result = imagej_functions.make_duplicate_jpg(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            channel_config=channel_config,
            ij=ij,
            verbose=verbose
        )
        results['make_jpg'] = result

        if not result['success']:
            return {'success': False, 'error': f"JPG creation failed: {result['error']}", 'results': results}

        # ====================================================================
        # STEP 6: Extract ROI Crops (3 versions)
        # ====================================================================
        # if verbose:
        #     print("\nSTEP 5: Extract ROI Crops")
        #     print("-" * 80)

        # 5a: TIF version (full dynamic range) - use original file + shrunk ROIs
        result_tif = extract_roi_crops.extract_roi_crops(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            source_image=channel_config.get('actin', 'Actin-FITC.tif'),
            output_dir_name="roi_crops_tif",
            use_transparency=False,
            roi_dir_name="cell_rois_shrunk",
            verbose=verbose
        )
        results['roi_crops_tif'] = result_tif

        # 5b: PNG with white background - use original Actin + shrunk ROIs
        from pathlib import Path
        base_dir_path = Path(f"{base_path}/{sample_folder}/{image_number}")

        # Use original Actin file for PNG crops
        original_actin = channel_config.get('actin', 'Actin-FITC.tif')
        if (base_dir_path / original_actin).exists():
            result_whitebg = extract_roi_crops.extract_roi_crops(
                sample_folder=sample_folder,
                image_number=image_number,
                base_path=base_path,
                source_image=original_actin,
                output_dir_name="roi_crops_whiteBg",
                use_transparency=True,
                background_color=255,
                roi_dir_name="cell_rois_shrunk",
                verbose=verbose
            )
            results['roi_crops_whiteBg'] = result_whitebg

            # 5c: PNG with black background - use original Actin + shrunk ROIs
            result_blackbg = extract_roi_crops.extract_roi_crops(
                sample_folder=sample_folder,
                image_number=image_number,
                base_path=base_path,
                source_image=original_actin,
                output_dir_name="roi_crops_blackBg",
                use_transparency=True,
                background_color=0,
                roi_dir_name="cell_rois_shrunk",
                verbose=verbose
            )
            results['roi_crops_blackBg'] = result_blackbg

        # ====================================================================
        # STEP 7: Visualize ROIs on JPG Images
        # ====================================================================
        if verbose:
            print("\nSTEP 7: Visualize ROIs on JPG Images")
            print("-" * 80)

        result = visualize_rois_on_jpg.create_roi_visualization(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            shrink_pixels=params.get('shrink_pixels', 3),
            verbose=verbose
        )
        results['roi_visualization'] = result

        if not result['success']:
            # Non-fatal: continue even if visualization fails
            if verbose:
                print(f"  ⚠️  Visualization failed: {result['error']}")

        # Create multi-channel comparison
        result_comparison = visualize_rois_on_jpg.visualize_roi_all_channels(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            shrink_pixels=params.get('shrink_pixels', 3),
            verbose=verbose
        )
        results['roi_comparison'] = result_comparison

        if not result_comparison['success']:
            # Non-fatal: continue even if comparison fails
            if verbose:
                print(f"  ⚠️  Multi-channel comparison failed: {result_comparison['error']}")

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
