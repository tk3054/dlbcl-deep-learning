#!/usr/bin/env python3
"""
T Cell Classifier
Run after inspecting pixel intensities from all_samples_combined.csv to set thresholds.

Usage:
    python classify_cells.py
    (Edit BASE_PATH and CLASSIFICATION below to match your data)
"""

import sys
from dlbcl_pipeline.classification.classify_tcells import classify_tcells
from dlbcl_pipeline.config import ClassificationConfig

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

BASE_PATH = '/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241'

CLASSIFICATION = ClassificationConfig(
    thresholds={
        'cd4':     260.0,   # CD4-PerCP
        'cd45ra':  120.0,   # CD45RA (exactly one of AF647 or SparkViolet/PacBlue must be present)
        'ccr7':    114.0,   # CCR7-AF594
        'cd19car': 43.0,   # CD19CAR-AF647
    },
)

# ============================================================================
def main():
    result = classify_tcells(
        base_path=BASE_PATH,
        thresholds=dict(CLASSIFICATION.thresholds),
        verbose=True,
    )

    if not result['success']:
        print(f"\n✗ Classification failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
