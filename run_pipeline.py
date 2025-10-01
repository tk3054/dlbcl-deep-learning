#!/usr/bin/env python3
"""
Complete Cell Analysis Pipeline
Runs the full pipeline for a single image (excluding manual review)

Usage:
    python run_pipeline.py
    (Edit SAMPLE_FOLDER and IMAGE_NUMBER below to change inputs)
"""

import imagej
import numpy as np
import pandas as pd
from pathlib import Path

# Import pipeline modules
import imagej_functions
import segment_cells_cellpose
import extract_roi_crops
import filter_bad_cells
import combine_measurements


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

SAMPLE_FOLDER = "sample1"
IMAGE_NUMBER = "5"
BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"

SEGMENTATION_METHOD = 'cellpose'  # Options: 'cellpose' or 'watershed'

PARAMS = {
    # Segmentation - Cellpose (deep learning)
    'cellpose_model': 'cyto2',
    'cellpose_diameter': None,
    'cellpose_flow_threshold': 0.4,
    'cellpose_cellprob_threshold': 0.0,
    'cellpose_use_gpu': True,
    'cellpose_enhance_contrast': True,  # Apply CLAHE to enhance dim cells
    'min_size': 500,
    'max_size': 9000,

    # Filtering
    'consecutive_threshold': 20,
}


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_pipeline(sample_folder, image_number, base_path,
                 segmentation_method='cellpose',
                 params=None, ij=None, verbose=True):
    """
    Run complete cell analysis pipeline for one image.

    Args:
        sample_folder: Sample folder name (e.g., "sample1")
        image_number: Image number (e.g., "5")
        base_path: Base directory path
        segmentation_method: 'cellpose' or 'watershed'
        params: Dictionary of pipeline parameters
        ij: ImageJ instance (if None, will initialize)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean
            - 'error': Error message if failed
            - 'results': Dict of all step results
    """
    if verbose:
        print("\n" + "="*80)
        print("CELL ANALYSIS PIPELINE")
        print("="*80)
        print(f"Sample: {sample_folder}/{image_number}")
        print(f"Segmentation: {segmentation_method}")
        print("="*80 + "\n")

    results = {}

    # Use default params if not provided
    if params is None:
        params = PARAMS

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
        # STEP 2: Preprocessing
        # ====================================================================
        if verbose:
            print("STEP 2: Preprocessing")
            print("-" * 80)

        result = imagej_functions.preprocess_actin(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            ij=ij,
            create_mask=(segmentation_method == 'watershed'),
            verbose=verbose
        )
        results['preprocessing'] = result

        if not result['success']:
            return {'success': False, 'error': f"Preprocessing failed: {result['error']}", 'results': results}

        # ====================================================================
        # STEP 3: Cellpose Segmentation
        # ====================================================================
        if verbose:
            print("\nSTEP 3: Cellpose Segmentation")
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
            enhance_contrast=params['cellpose_enhance_contrast'],
            verbose=verbose
        )
        results['segmentation'] = result

        if not result['success']:
            return {'success': False, 'error': f"Segmentation failed: {result['error']}", 'results': results}

        # ====================================================================
        # STEP 4: Extract ROI Crops (3 versions)
        # ====================================================================
        if verbose:
            print("\nSTEP 4: Extract ROI Crops")
            print("-" * 80)

        # 4a: TIF version (full dynamic range)
        result_tif = extract_roi_crops.extract_roi_crops(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            source_image="Actin-FITC.tif",
            output_dir_name="roi_crops_tif",
            use_transparency=False,
            verbose=verbose
        )
        results['roi_crops_tif'] = result_tif

        # 4b: PNG with transparency
        import glob
        base_dir_path = f"{base_path}/{sample_folder}/{image_number}"
        raw_jpg_files = glob.glob(f"{base_dir_path}/*_raw.jpg")

        if raw_jpg_files:
            raw_jpg = raw_jpg_files[0].split("/")[-1]

            result_png = extract_roi_crops.extract_roi_crops(
                sample_folder=sample_folder,
                image_number=image_number,
                base_path=base_path,
                source_image=raw_jpg,
                output_dir_name="roi_crops_whiteBg",
                use_transparency=True,
                verbose=verbose
            )
            results['roi_crops_png'] = result_png

            # 4c: JPG version
            result_jpg = extract_roi_crops.extract_roi_crops(
                sample_folder=sample_folder,
                image_number=image_number,
                base_path=base_path,
                source_image=raw_jpg,
                output_dir_name="roi_crops_jpg",
                use_transparency=False,
                verbose=verbose
            )
            results['roi_crops_jpg'] = result_jpg

        # ====================================================================
        # STEP 5: Load ROIs in ImageJ
        # ====================================================================
        if verbose:
            print("\nSTEP 5: Load ROIs in ImageJ")
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
        # STEP 6: Extract Channel Measurements
        # ====================================================================
        if verbose:
            print("\nSTEP 6: Extract Channel Measurements")
            print("-" * 80)

        result = imagej_functions.extract_channel_measurements(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            ij=ij,
            verbose=verbose
        )
        results['measurements'] = result

        if not result['success']:
            return {'success': False, 'error': f"Extract measurements failed: {result['error']}", 'results': results}

        # ====================================================================
        # STEP 7: Filter Bad Cells
        # ====================================================================
        if verbose:
            print("\nSTEP 7: Filter Bad Cells")
            print("-" * 80)

        result = filter_bad_cells.filter_bad_cells(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            consecutive_threshold=params['consecutive_threshold'],
            verbose=verbose
        )
        results['filter'] = result

        if not result['success']:
            return {'success': False, 'error': f"Filter bad cells failed: {result['error']}", 'results': results}

        # ====================================================================
        # STEP 8: Combine Measurements
        # ====================================================================
        if verbose:
            print("\nSTEP 8: Combine Measurements")
            print("-" * 80)

        result = combine_measurements.combine_measurements(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            verbose=verbose
        )
        results['combine'] = result

        if not result['success']:
            return {'success': False, 'error': f"Combine measurements failed: {result['error']}", 'results': results}

        # ====================================================================
        # COMPLETE
        # ====================================================================
        if verbose:
            print("\n" + "="*80)
            print("PIPELINE COMPLETE!")
            print("="*80)
            print(f"✓ Processed {results['segmentation']['num_cells']} cells")
            print(f"✓ Quality: {results['filter']['whole_count']} WHOLE, {results['filter']['cut_count']} CUT")
            print(f"✓ Combined CSV: {results['combine']['output_csv']}")
            print("="*80 + "\n")

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
    result = run_pipeline(
        sample_folder=SAMPLE_FOLDER,
        image_number=IMAGE_NUMBER,
        base_path=BASE_PATH,
        segmentation_method=SEGMENTATION_METHOD,
        params=PARAMS,
        ij=None,
        verbose=True
    )

    if not result['success']:
        print(f"\n✗ ERROR: {result['error']}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
