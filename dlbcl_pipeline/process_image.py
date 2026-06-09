#!/usr/bin/env python3
"""
Single Image Processing
Runs the full cell analysis workflow for one image (excluding manual review)
"""

import os
import sys
from contextlib import contextmanager
from typing import Any, Mapping

from dlbcl_pipeline.config import PipelineConfig
from dlbcl_pipeline.data_types import ImageFolder, ImageProcessingResult, SampleFolder
from dlbcl_pipeline.utils.name_builder import processed_channel_filename


# ============================================================================
# MAIN IMAGE PROCESSING FUNCTION
# ============================================================================

def process_single_image(
    sample: SampleFolder,
    image: ImageFolder,
    config: PipelineConfig,
    ij=None,
    params: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> ImageProcessingResult:
    """
    Process one image through segmentation, ImageJ measurements, crops, and table merge.

    Args:
        sample: Modeled sample folder containing the image.
        image: Modeled image folder to process.
        config: Pipeline configuration for the donor run.
        ij: ImageJ instance (if None, will initialize)
        params: Optional override for Cellpose/image-processing parameters.
        verbose: Print progress messages

    Returns:
        ImageProcessingResult with success/error metadata for this image.
    """
    sample_folder = sample.name
    image_number = image.name
    base_path = str(config.donor_dir)
    params = dict(params or config.cellpose.to_pipeline_params())
    channel_config = {channel: path.name for channel, path in image.channels.items()}
    combine_channels = None
    null_channels = None

    if verbose:
        print("\n" + "="*40)
        print("SINGLE IMAGE PROCESSING")
        print("="*40)
        print(f"Sample: {sample_folder}/{image_number}")
        print("Segmentation: cellpose")
        print("="*40 + "\n")

    results = {}

    try:
        # ====================================================================
        # STEP 1: Initialize ImageJ
        # ====================================================================
        if ij is None:
            if verbose:
                print("STEP 1: Initializing ImageJ...")
            import imagej
            with _suppress_output():
                ij = imagej.init(
                    "sc.fiji:fiji",
                    mode=config.runtime.imagej_mode,
                    add_legacy=True,
                )
            if verbose:
                print(f"✓ ImageJ version: {ij.getVersion()}\n")
        results['imagej'] = ij

        # ====================================================================
        # STEP 2: Cellpose Segmentation
        # ====================================================================
        if verbose:
            print("\nSTEP 2: Cellpose Segmentation")
            print("-" * 80)

        import dlbcl_pipeline.segmentation.cellpose as segment_cells_cellpose

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
            return ImageProcessingResult(
                success=False,
                error=f"Segmentation failed: {result['error']}",
                sample_name=sample_folder,
                image_name=image_number,
                results=results,
            )

        # ====================================================================
        # STEP 3: Preprocess Channels
        # ====================================================================
        if verbose:
            print("\nSTEP 3: Preprocess Channels")
            print("-" * 80)

        import dlbcl_pipeline.imagej.functions as imagej_functions

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
            return ImageProcessingResult(
                success=False,
                error=f"Channel preprocessing failed: {result['error']}",
                sample_name=sample_folder,
                image_name=image_number,
                results=results,
            )

        # ====================================================================
        # STEP 4: Create ROI-Masked Crops from Step 3 Preprocessed Images
        # ====================================================================
        if verbose:
            print("\nSTEP 4: Create ROI-Masked Crops from Preprocessed Images")
            print("-" * 80)

        import dlbcl_pipeline.image_processing.crops as make_raw_crops
        from dlbcl_pipeline.image_processing.padding import pad_crop_folder

        preprocessed_actin_source = processed_channel_filename(
            channel_config.get('actin', 'Actin-FITC.tif')
        )
        preprocessed_ccr7_source = processed_channel_filename(
            channel_config.get('ccr7', 'CCR7-PE.tif')
        )
        preprocessed_cd45ra_source = processed_channel_filename(
            channel_config.get('cd45ra_sparkviolet')
            or channel_config.get('cd45ra_PacBlue')
            or 'CD45RA-PacBlue.tif'
        )

        result_raw_crops = make_raw_crops.make_raw_crops(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            source_image=preprocessed_actin_source,
            output_dir_name="raw_actin",
            background="transparent",
            verbose=False
        )
        results['raw_actin'] = result_raw_crops

        if not result_raw_crops['success']:
            return ImageProcessingResult(
                success=False,
                error=f"Raw actin crops failed: {result_raw_crops['error']}",
                sample_name=sample_folder,
                image_name=image_number,
                results=results,
            )

        result_padded_cells = pad_crop_folder(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            input_dir_name="raw_actin",
            output_dir_name="padded_cells",
            target_size=224,
            verbose=False,
        )
        results['padded_cells'] = result_padded_cells

        if not result_padded_cells['success']:
            return ImageProcessingResult(
                success=False,
                error=f"Pad cells failed: {result_padded_cells['error']}",
                sample_name=sample_folder,
                image_name=image_number,
                results=results,
            )

        # STEP 4b: Create ROI-masked crops from preprocessed CCR7
        result_raw_crops_ccr7 = make_raw_crops.make_raw_crops(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            source_image=preprocessed_ccr7_source,
            output_dir_name="raw_ccr7",
            background="transparent",
            verbose=False
        )
        results['raw_ccr7'] = result_raw_crops_ccr7

        if not result_raw_crops_ccr7['success']:
            return ImageProcessingResult(
                success=False,
                error=f"Raw CCR7 crops failed: {result_raw_crops_ccr7['error']}",
                sample_name=sample_folder,
                image_name=image_number,
                results=results,
            )

        result_padded_ccr7 = pad_crop_folder(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            input_dir_name="raw_ccr7",
            output_dir_name="padded_ccr7",
            target_size=224,
            verbose=False,
        )
        results['padded_ccr7'] = result_padded_ccr7

        if not result_padded_ccr7['success']:
            return ImageProcessingResult(
                success=False,
                error=f"Pad CCR7 cells failed: {result_padded_ccr7['error']}",
                sample_name=sample_folder,
                image_name=image_number,
                results=results,
            )

        # STEP 4c: Create ROI-masked crops from preprocessed CD45RA
        result_raw_crops_cd45ra = make_raw_crops.make_raw_crops(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            source_image=preprocessed_cd45ra_source,
            output_dir_name="raw_cd45ra",
            background="transparent",
            verbose=False
        )
        results['raw_cd45ra'] = result_raw_crops_cd45ra

        if not result_raw_crops_cd45ra['success']:
            return ImageProcessingResult(
                success=False,
                error=f"Raw CD45RA crops failed: {result_raw_crops_cd45ra['error']}",
                sample_name=sample_folder,
                image_name=image_number,
                results=results,
            )

        result_padded_cd45ra = pad_crop_folder(
            sample_folder=sample_folder,
            image_number=image_number,
            base_path=base_path,
            input_dir_name="raw_cd45ra",
            output_dir_name="padded_cd45ra",
            target_size=224,
            verbose=False,
        )
        results['padded_cd45ra'] = result_padded_cd45ra

        if not result_padded_cd45ra['success']:
            return ImageProcessingResult(
                success=False,
                error=f"Pad CD45RA cells failed: {result_padded_cd45ra['error']}",
                sample_name=sample_folder,
                image_name=image_number,
                results=results,
            )

        if verbose:
            print("\n  ✓ Generated ROI-masked crops and padded 224x224 channel images")

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
            return ImageProcessingResult(
                success=False,
                error=f"Load ROIs failed: {result['error']}",
                sample_name=sample_folder,
                image_name=image_number,
                results=results,
            )

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
            channel_config=channel_config,
            ij=ij,
            verbose=False
        )
        results['measurements'] = result

        if not result['success']:
            return ImageProcessingResult(
                success=False,
                error=f"Extract measurements failed: {result['error']}",
                sample_name=sample_folder,
                image_name=image_number,
                results=results,
            )

        # ====================================================================
        # STEP 7: Combine Measurements
        # ====================================================================
        if verbose:
            print("\nSTEP 7: Combine Measurements")
            print("-" * 80)

        import dlbcl_pipeline.measurements.channels as combine_channel

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
            return ImageProcessingResult(
                success=False,
                error=f"Combine measurements failed: {result['error']}",
                sample_name=sample_folder,
                image_name=image_number,
                results=results,
            )

        # ====================================================================
        # COMPLETE
        # ====================================================================
        if verbose:
            print("\n" + "="*40)
            print("IMAGE PROCESSING COMPLETE!")
            print("="*40)
            print(f"✓ Processed {results['segmentation']['num_cells']} cells")
            print(f"✓ Combined CSV: {results['combine']['output_csv']}")
            print("="*40 + "\n")

        return ImageProcessingResult(
            success=True,
            sample_name=sample_folder,
            image_name=image_number,
            results=results,
        )

    except Exception as e:
        import traceback
        error_msg = f"Image processing failed: {str(e)}\n{traceback.format_exc()}"
        return ImageProcessingResult(
            success=False,
            error=error_msg,
            sample_name=sample_folder,
            image_name=image_number,
            results=results,
        )


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
