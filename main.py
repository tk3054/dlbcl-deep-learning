#!/usr/bin/env python3
"""
Main Pipeline Runner
Automatically discovers and processes all samples and images in a directory

Usage:
    python main.py
    (Edit BASE_PATH below to change which directory to process)
"""

import sys
from dlbcl_pipeline.config import build_local_pipeline_config
from dlbcl_pipeline.model_folder_structure import build_donor_folder_structure
from dlbcl_pipeline.process_donor import process_donor


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

BASE_PATH = '/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241'
SAMPLES_TO_PROCESS = {1:[1,2,3]}  # sample number -> image numbers

# SAMPLES_TO_PROCESS = {1: [1, 2, 3]}  # sample number -> image numbers

# ============================================================================
def main():
    config = build_local_pipeline_config(
        donor_dir=BASE_PATH,
        samples_to_process=SAMPLES_TO_PROCESS,
    )
    donor_folder_structure = build_donor_folder_structure(
        config,
        announce_filters=True,
    )
    
    donor_result = process_donor(
        config,
        donor_folder_structure=donor_folder_structure,
        verbose=True,
    )
    return donor_result


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.success else 1)
