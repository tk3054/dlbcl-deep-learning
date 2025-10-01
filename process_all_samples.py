#!/usr/bin/env python3
"""
Process All Samples in Batch
Loops through all images in a sample folder and runs the complete pipeline

Usage:
    python process_all_samples.py
    (Edit SAMPLE_FOLDER below to change which sample to process)
"""

import os
import imagej
from pathlib import Path
from run_pipeline import run_pipeline


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

SAMPLE_FOLDER = "sample2"  # Options: "sample1", "sample2", "sample3"
BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"

SEGMENTATION_METHOD = 'cellpose'  # Options: 'cellpose' or 'watershed'

PARAMS = {
    # Segmentation - Cellpose (deep learning)
    # 'cellpose_model': 'cyto2',
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
# MAIN BATCH PROCESSING FUNCTION
# ============================================================================

def process_all_samples(sample_folder, base_path, segmentation_method='cellpose',
                       params=None, verbose=True):
    """
    Process all images in a sample folder.

    Args:
        sample_folder: Sample folder name (e.g., "sample1")
        base_path: Base directory path
        segmentation_method: 'cellpose' or 'watershed'
        params: Dictionary of pipeline parameters
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean
            - 'num_processed': Number of images processed
            - 'num_failed': Number of images failed
            - 'results': List of results for each image
    """
    if verbose:
        print("\n" + "="*80)
        print("BATCH PROCESSING: PROCESS ALL SAMPLES")
        print("="*80)
        print(f"Sample folder: {sample_folder}")
        print(f"Base path: {base_path}")
        print("="*80 + "\n")

    # Use default params if not provided
    if params is None:
        params = PARAMS

    # Find all image folders in sample folder
    sample_path = Path(base_path) / sample_folder

    if not sample_path.exists():
        return {
            'success': False,
            'error': f'Sample folder not found: {sample_path}',
            'num_processed': 0,
            'num_failed': 0,
            'results': []
        }

    # Get all subdirectories (image folders are numbered: 1, 2, 3, etc.)
    image_folders = []
    for item in sample_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            image_folders.append(item.name)

    # Sort numerically
    image_folders = sorted(image_folders, key=lambda x: int(x))

    if not image_folders:
        return {
            'success': False,
            'error': f'No image folders found in {sample_path}',
            'num_processed': 0,
            'num_failed': 0,
            'results': []
        }

    if verbose:
        print(f"Found {len(image_folders)} images to process: {', '.join(image_folders)}\n")

    # Initialize ImageJ once (reuse for all images)
    if verbose:
        print("Initializing ImageJ (will be reused for all images)...")
    ij = imagej.init('sc.fiji:fiji')
    if verbose:
        print(f"✓ ImageJ version: {ij.getVersion()}\n")
        print("="*80 + "\n")

    # Process each image
    results = []
    num_processed = 0
    num_failed = 0

    for i, image_number in enumerate(image_folders, 1):
        if verbose:
            print(f"\n{'='*80}")
            print(f"PROCESSING IMAGE {i}/{len(image_folders)}: {sample_folder}/{image_number}")
            print(f"{'='*80}\n")

        try:
            result = run_pipeline(
                sample_folder=sample_folder,
                image_number=image_number,
                base_path=base_path,
                segmentation_method=segmentation_method,
                params=params,
                ij=ij,  # Reuse ImageJ instance
                verbose=verbose
            )

            results.append({
                'image_number': image_number,
                'success': result['success'],
                'result': result
            })

            if result['success']:
                num_processed += 1
                if verbose:
                    print(f"\n✓ Successfully processed {sample_folder}/{image_number}")
            else:
                num_failed += 1
                if verbose:
                    print(f"\n✗ Failed to process {sample_folder}/{image_number}")
                    print(f"   Error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            num_failed += 1
            if verbose:
                print(f"\n✗ Failed to process {sample_folder}/{image_number}")
                print(f"   Error: {str(e)}")
            results.append({
                'image_number': image_number,
                'success': False,
                'error': str(e)
            })

    # Summary
    if verbose:
        print("\n" + "="*80)
        print("BATCH PROCESSING COMPLETE")
        print("="*80)
        print(f"Total images: {len(image_folders)}")
        print(f"✓ Processed successfully: {num_processed}")
        print(f"✗ Failed: {num_failed}")
        print("="*80 + "\n")

    return {
        'success': num_failed == 0,
        'num_processed': num_processed,
        'num_failed': num_failed,
        'results': results
    }


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    result = process_all_samples(
        sample_folder=SAMPLE_FOLDER,
        base_path=BASE_PATH,
        segmentation_method=SEGMENTATION_METHOD,
        params=PARAMS,
        verbose=True
    )

    if not result['success']:
        print(f"\n⚠️  Some images failed to process: {result['num_failed']}/{result['num_processed'] + result['num_failed']}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
