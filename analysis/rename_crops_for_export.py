#!/usr/bin/env python3
"""
Rename Cell ROI Crops for Export
Renames cell crop files with standardized naming convention including experiment info,
sample/image numbers, and cell classification.

Naming format: {experiment_prefix}_sample{N}_image{NN}_cell{NN}_{classification}.tif

Examples:
- DLBCL114357_20250926_1to10_40min_sample1_image01_cell01_CD4_Naive.tif
- DLBCL114357_20250926_1to10_40min_sample1_image02_cell03_CAR-T_TEM.tif
- DLBCL114357_20250926_1to10_40min_sample2_image05_cell10_CD4-negative.tif

Usage:
    python rename_crops_for_export.py
"""

import pandas as pd
from pathlib import Path
import sys
import re

# Import BASE_PATH from main.py
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from main import BASE_PATH
except ImportError:
    BASE_PATH = "/Users/taeeonkong/Desktop/2025 Fall Images/09-26-2025 DLBCL"


# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# Experiment prefix (extracted from base path or set manually)
EXPERIMENT_PREFIX = "DLBCL114357_20250926_1to10_40min"

# Input CSV with classifications
INPUT_CSV = "all_samples_combined_classified.csv"

# Target directory name to rename files in
TARGET_DIR = "tif_roi_crops_blackbg"

# Dry run mode (if True, only show what would be renamed without actually renaming)
DRY_RUN = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_abbreviation(text):
    """
    Extract abbreviation from parentheses in text.

    Examples:
        "Central Memory (TCM)" -> "TCM"
        "Effector Memory (TEM)" -> "TEM"
        "Terminal Effector (TEMRA)" -> "TEMRA"
        "Naive" -> "Naive"

    Args:
        text: Input text string

    Returns:
        Abbreviation if found, otherwise original text
    """
    match = re.search(r'\(([^)]+)\)', text)
    if match:
        return match.group(1)
    return text


def format_cell_classification(cell_type, subset):
    """
    Format cell classification for filename.

    Rules:
    - CD4+ cells: CD4_{subset}
    - CAR-T cells: CAR-T_{subset} (remove CD4+/CD4- prefix)
    - CD4- cells: CD4-negative

    Args:
        cell_type: Cell type (e.g., "CD4+", "CD4-", "CAR-T")
        subset: Cell subset (e.g., "Naive", "Central Memory (TCM)", "N/A")

    Returns:
        Formatted classification string
    """
    # Handle CD4- cells
    if cell_type == "CD4-":
        return "CD4-negative"

    # Handle CAR-T cells
    if cell_type == "CAR-T":
        # Remove "CAR-T CD4+ " or "CAR-T CD4- " prefix from subset
        if subset.startswith("CAR-T CD4+ "):
            subset = subset.replace("CAR-T CD4+ ", "")
        elif subset.startswith("CAR-T CD4- "):
            subset = subset.replace("CAR-T CD4- ", "")

        # Extract abbreviation
        abbrev = extract_abbreviation(subset)
        return f"CAR-T_{abbrev}"

    # Handle CD4+ cells
    if cell_type == "CD4+":
        abbrev = extract_abbreviation(subset)
        return f"CD4_{abbrev}"

    # Fallback
    return f"{cell_type}_{subset}".replace(" ", "_")


def rename_crops_for_export(base_path, experiment_prefix=None, input_csv=None,
                            target_dir=None, dry_run=True, verbose=True):
    """
    Rename cell crop files with standardized naming convention.

    Args:
        base_path: Base directory path
        experiment_prefix: Experiment prefix string (default: auto-detect from base_path)
        input_csv: Input CSV filename (default: all_samples_combined_classified.csv)
        target_dir: Target directory name (default: tif_roi_crops_blackbg)
        dry_run: If True, only show what would be renamed (default: True)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean
            - 'num_renamed': Number of files renamed
            - 'renames': List of (old_name, new_name) tuples
    """
    if input_csv is None:
        input_csv = INPUT_CSV
    if target_dir is None:
        target_dir = TARGET_DIR
    if experiment_prefix is None:
        experiment_prefix = EXPERIMENT_PREFIX

    if verbose:
        print("\n" + "="*80)
        print("RENAME CELL CROPS FOR EXPORT")
        print("="*80)
        if dry_run:
            print("  [DRY RUN MODE - No files will be renamed]")
        print(f"\nExperiment prefix: {experiment_prefix}")

    # Load classified CSV
    csv_path = Path(base_path) / input_csv
    if not csv_path.exists():
        return {
            'success': False,
            'error': f"CSV not found: {csv_path}",
            'num_renamed': 0,
            'renames': []
        }

    if verbose:
        print(f"Loading: {input_csv}")

    df = pd.read_csv(csv_path)

    if verbose:
        print(f"  Total cells in CSV: {len(df)}")

    # Track renames
    renames = []
    num_renamed = 0
    num_skipped = 0

    # Process each row in CSV
    for idx, row in df.iterrows():
        sample_folder = row['sample']
        image_number = row['image']
        cell_number = row['cell_id']
        cell_type = row['cell_type']
        subset = row['tcell_subset']

        # Format components
        sample_num = sample_folder.replace('sample', '')
        image_num = str(image_number).zfill(2)
        cell_num = str(cell_number).zfill(2)
        classification = format_cell_classification(cell_type, subset)

        # Build paths
        base_dir = Path(base_path) / sample_folder / image_number
        crop_dir = base_dir / target_dir

        if not crop_dir.exists():
            if verbose and idx == 0:
                print(f"  ⚠️  Warning: {target_dir}/ not found in {sample_folder}/{image_number}")
            num_skipped += 1
            continue

        # Find the original file (should be named like "cell_001_crop.tif")
        original_filename = f"cell_{str(cell_number).zfill(3)}_crop.tif"
        original_path = crop_dir / original_filename

        if not original_path.exists():
            if verbose:
                print(f"  ⚠️  File not found: {original_path}")
            num_skipped += 1
            continue

        # Build new filename
        new_filename = f"{experiment_prefix}_sample{sample_num}_image{image_num}_cell{cell_num}_{classification}.tif"
        new_path = crop_dir / new_filename

        # Record rename
        renames.append((str(original_path.name), str(new_filename)))

        # Perform rename (unless dry run)
        if not dry_run:
            original_path.rename(new_path)

        num_renamed += 1

        if verbose and num_renamed <= 5:
            print(f"\n  {original_path.name}")
            print(f"    → {new_filename}")

    if verbose:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Files renamed: {num_renamed}")
        print(f"Files skipped: {num_skipped}")

        if dry_run:
            print("\n⚠️  DRY RUN MODE - No files were actually renamed")
            print("Set DRY_RUN = False to apply changes")
        else:
            print("\n✓ Files renamed successfully!")

        print("="*80 + "\n")

    return {
        'success': True,
        'num_renamed': num_renamed,
        'num_skipped': num_skipped,
        'renames': renames
    }


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    result = rename_crops_for_export(
        base_path=BASE_PATH,
        experiment_prefix=EXPERIMENT_PREFIX,
        input_csv=INPUT_CSV,
        target_dir=TARGET_DIR,
        dry_run=DRY_RUN,
        verbose=True
    )

    if not result['success']:
        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
