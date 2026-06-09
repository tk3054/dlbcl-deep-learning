#!/usr/bin/env python3
"""
Remote Pipeline Runner
Copies a raw dataset to a processed tree, then iterates all donor folders.

Usage:
    python main_remote.py
    (Edit RAW_BASE_PATHS and CHANNEL_CONFIG below to point to your dataset)
"""

import sys
import shutil
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse, unquote

from dlbcl_pipeline.config import ExportConfig, PipelineConfig
from dlbcl_pipeline.model_folder_structure import (
    build_donor_folder_structure,
    find_donor_dirs,
)
from dlbcl_pipeline.process_donor import process_donor

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

RAW_BASE_PATHS = [
    # '/mnt/HDD16TB/LanceKam_Lab/Daizong/Project/DLBCL/DLBCL/Non-responder',
    '/mnt/HDD16TB/LanceKam_Lab/Daizong/Project/DLBCL/DLBCL/Responder',
]
PROCESSED_BASE_NAME = 'DLBCL-edge'
COPY_RAW_TO_PROCESSED = True

FIJI_PATH = '/mnt/HDD16TB/LanceKam_Lab/Daizong/Project/DLBCL/Fiji.app'

# Leave empty to process all samples / images found for each donor.
SAMPLES_TO_PROCESS = []
IMAGES_TO_PROCESS = {}

# File names for each channel. Must match the names in the raw image folders.
CHANNEL_CONFIG = {
    'actin': 'Actin-FITC.tif',
    'cd4': 'CD4-PerCP.tif',
    'cd45ra_PacBlue': 'CD45RA-PacBlue.tif',
    # 'cd45ra_sparkviolet': 'CD45RA-SparkViolet.tif',
    'cd19car': 'CD19CAR-AF647.tif',
    'ccr7': 'CCR7-AF594.tif',
}

# Export named cell images into a formatted_cells folder.
EXPORT_IMAGES = True
EXPORT_PDMS_STIFFNESS = "1to10"
EXPORT_DILUTION = "1to10"
EXPORT_NAME_ORDER = [
    "response",
    "donor_id",
    "date",
    "stiffness",
    "sample",
    "image",
    "cell_label",
    "classification",
]

# ============================================================================


def _normalize_input_path(path_text: str) -> str:
    if not path_text:
        return ""
    if path_text.startswith("sftp://"):
        parsed = urlparse(path_text)
        return unquote(parsed.path or "")
    return unquote(path_text)


def _prepare_processed_root(raw_path: Path) -> Path:
    processed_path = raw_path.parent / PROCESSED_BASE_NAME

    if processed_path.exists():
        print(f"✓ Using existing processed tree: {processed_path}")
        return processed_path

    if not COPY_RAW_TO_PROCESSED:
        print(f"✗ ERROR: Processed path not found: {processed_path}")
        sys.exit(1)

    print(f"Copying raw dataset to processed tree:\n  {raw_path}\n→ {processed_path}")
    shutil.copytree(raw_path, processed_path)
    print("✓ Copy complete")
    return processed_path


def process_remote_donor(donor_dir: Path) -> dict:
    config = PipelineConfig.server(
        donor_dir=donor_dir,
        channels=CHANNEL_CONFIG,
        samples_to_process=SAMPLES_TO_PROCESS or None,
        images_to_process=IMAGES_TO_PROCESS or None,
        fiji_path=FIJI_PATH,
        processed_base_name=PROCESSED_BASE_NAME,
    )
    config = replace(config, export=ExportConfig(
        enabled=EXPORT_IMAGES,
        pdms_stiffness=EXPORT_PDMS_STIFFNESS,
        dilution=EXPORT_DILUTION,
        name_order=tuple(EXPORT_NAME_ORDER),
    ))

    donor_folder_structure = build_donor_folder_structure(
        config,
        announce_filters=True,
    )
    result = process_donor(
        config,
        donor_folder_structure=donor_folder_structure,
        verbose=True,
    )
    if not result.success:
        return {"success": False, "error": result.error}

    return {"success": True}


def main():
    raw_path_texts = [
        _normalize_input_path(p) for p in RAW_BASE_PATHS if _normalize_input_path(p)
    ]
    if not raw_path_texts:
        print("✗ ERROR: RAW_BASE_PATHS is empty.")
        sys.exit(1)

    total_failed_donors = 0
    total_donors_seen = 0

    for raw_path_text in raw_path_texts:
        raw_path = Path(raw_path_text)
        if not raw_path.exists():
            print(f"✗ ERROR: Raw path not found: {raw_path}")
            total_failed_donors += 1
            continue

        processed_root = _prepare_processed_root(raw_path)
        donor_dirs = find_donor_dirs(processed_root)
        if not donor_dirs:
            print(f"✗ ERROR: No donor folders found under {processed_root}")
            total_failed_donors += 1
            continue

        print("\n" + "=" * 80)
        print("REMOTE BATCH PIPELINE: PROCESSING ALL DONORS")
        print("=" * 80)
        print(f"Processed root: {processed_root}")
        print(f"Found {len(donor_dirs)} donor folders")
        print("=" * 80 + "\n")

        total_donors_seen += len(donor_dirs)
        for idx, donor_dir in enumerate(donor_dirs, start=1):
            print("\n" + "=" * 80)
            print(f"DONOR {idx}/{len(donor_dirs)}: {donor_dir.name}")
            print(f"Path: {donor_dir}")
            print("=" * 80 + "\n")

            result = process_remote_donor(donor_dir)
            if not result.get("success"):
                total_failed_donors += 1

    if total_donors_seen == 0:
        print("\n✗ ERROR: No donor folders found in any RAW_BASE_PATHS entry.")
        sys.exit(1)

    if total_failed_donors:
        print(f"\n⚠️  Completed with {total_failed_donors} donor(s) reporting failures.")
        sys.exit(1)


if __name__ == "__main__":
    main()
