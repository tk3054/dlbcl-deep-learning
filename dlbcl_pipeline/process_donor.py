"""Donor and image orchestration functions.

This module coordinates discovery, image processing, and table building.
"""

from __future__ import annotations

from typing import Any, Mapping

from dlbcl_pipeline import build_paths as paths
from dlbcl_pipeline.config import PipelineConfig
from dlbcl_pipeline.process_image import process_single_image
from dlbcl_pipeline.measurements.aggregation import build_donor_table, build_sample_table
from dlbcl_pipeline.data_types import (
    DonorFolderStructure,
    DonorProcessingResult,
)


def process_donor(
    config: PipelineConfig,
    donor_folder_structure: DonorFolderStructure,
    params: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> DonorProcessingResult:
    """Process images and build donor-level tables for one donor."""
    from dlbcl_pipeline.segmentation.imagej_config import initialize_imagej

    if donor_folder_structure is None:
        return DonorProcessingResult(
            success=False,
            error="donor_folder_structure is required",
            donor_dir=config.donor_dir,
        )

    base_path_obj = donor_folder_structure.path
    sample_folders = [sample.name for sample in donor_folder_structure.samples]

    if verbose:
        print("\n" + "=" * 40)
        print("BATCH PIPELINE: PROCESSING ALL SAMPLES")
        print("=" * 40)
        print(f"Base path: {base_path_obj}")
        print(f"Found {len(sample_folders)} samples: {', '.join(sample_folders)}")
        print("=" * 40 + "\n")

    ij = initialize_imagej(mode=config.runtime.imagej_mode, verbose=verbose)

    total_processed = 0
    total_failed = 0
    failed_images = []
    image_results = []

    for sample_idx, sample in enumerate(donor_folder_structure.samples, 1):
        sample_folder = sample.name
        if not sample.images:
            if verbose:
                print(f"⚠️  No image folders found in {sample_folder}, skipping...")
            continue

        if verbose:
            image_labels = [image.name for image in sample.images]
            print(f"\n{'='*40}")
            print(f"PROCESSING SAMPLE {sample_idx}/{len(sample_folders)}: {sample_folder}")
            print(f"Found {len(sample.images)} image folders: {', '.join(image_labels)}")
            print(f"{'='*40}\n")

        for img_idx, image in enumerate(sample.images, 1):
            if verbose:
                print(f"\n{'-'*40}")
                print(f"Image {img_idx}: {sample.name}/{image.name}")
                print(f"{'-'*40}\n")

            result = process_single_image(
                sample=sample,
                image=image,
                config=config,
                ij=ij,
                params=params,
                verbose=False,
            )
            image_results.append(result)

            if result.success:
                total_processed += 1
                if verbose:
                    print(f"\n✓ Successfully processed {sample.name}/{image.name}")
            else:
                total_failed += 1
                error_msg = result.error or "Unknown error"
                if verbose:
                    print(f"\n✗ Failed to process {sample.name}/{image.name}")
                    print(f"   Error: {error_msg}")
                failed_images.append({
                    "sample": sample.name,
                    "image_number": image.name,
                    "error": error_msg,
                })

        sample_result = build_sample_table(
            base_path_obj / sample_folder,
            images_to_process=config.images_to_process,
            verbose=verbose,
        )
        if not sample_result.success and verbose:
            print(f"  ⚠️  Failed to combine sample {sample_folder}: {sample_result.error}")

    total_images = total_processed + total_failed
    if verbose:
        print("\n" + "="*40)
        print("BATCH PROCESSING COMPLETE")
        print("="*40)
        print(f"Total samples: {len(sample_folders)}")
        print(f"Total images: {total_images}")
        print(f"✓ Processed successfully: {total_processed}")
        print(f"✗ Failed: {total_failed}")
        print("="*40 + "\n")

    if failed_images:
        log_path = paths.failed_images_log(base_path_obj)
        with log_path.open("w") as log_file:
            log_file.write("Failed images:\n")
            for item in failed_images:
                log_file.write(
                    f"- {item['sample']}/{item['image_number']}: {item['error']}\n"
                )
        if verbose:
            print(f"✗ Wrote failure log: {log_path}")

    if total_failed > 0:
        return DonorProcessingResult(
            success=False,
            error=f"{total_failed} image(s) failed",
            donor_dir=config.donor_dir,
            total_images=total_images,
            total_processed=total_processed,
            total_failed=total_failed,
            failed_images=tuple(failed_images),
            image_results=tuple(image_results),
        )

    donor_table = None
    if verbose:
        print("\n" + "="*80)
        print("COMBINING ALL SAMPLES INTO MASTER CSV")
        print("="*80 + "\n")
    donor_table = build_donor_table(config, verbose=verbose)
    if not donor_table.success:
        return DonorProcessingResult(
            success=False,
            error=donor_table.error or "Failed to combine samples",
            donor_dir=config.donor_dir,
            total_images=total_images,
            total_processed=total_processed,
            total_failed=total_failed,
            failed_images=tuple(failed_images),
            image_results=tuple(image_results),
            donor_table=donor_table,
        )

    return DonorProcessingResult(
        success=True,
        donor_dir=config.donor_dir,
        total_images=total_images,
        total_processed=total_processed,
        total_failed=total_failed,
        failed_images=tuple(failed_images),
        image_results=tuple(image_results),
        donor_table=donor_table,
    )
