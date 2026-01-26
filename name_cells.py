#!/usr/bin/env python3
"""
Name Cell ROI Crops for Export
Renames cell crop files in the target ROI directory with standardized naming
based on patient folder, sample/image numbers, and cell classification.

Naming format: {experiment_prefix}_sample{N}_image{NN}_cell{NN}_{classification}.tif

Example:
- nonresponder_109241_01-03-2026_1to10_sample1_image01_cell01_native_CD4_Naive.tif
- nonresponder_109241_01-03-2026_1to10_sample1_image02_cell03_CART_CD8_TEM.tif

Usage:
    python name_cells.py
"""

import pandas as pd
from pathlib import Path
import sys
import re

from utils.name_builder import extract_image_number


# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# Base path fallback for standalone usage (prefer passing via CLI or main.py)
BASE_PATH = ""

# Shared naming/export configs
EXPORT_PDMS_STIFFNESS = "1to10"
EXPORT_CLASSIFIED_CSV = "all_samples_combined_classified.csv"
EXPORT_SOURCE_DIR = "padded_cells"
EXPORT_OUTPUT_DIR = "formatted_cells"
EXPORT_DILUTION = "1to10"
EXPORT_NAME_ORDER = [
    "response",
    "patient_id",
    "date",
    "stiffness",
    "sample",
    "image",
    "cell_label",
    "classification",
]

# Local rename configs
NAME_CELLS_INPUT_CSV = "all_samples_combined_classified.csv"
NAME_CELLS_TARGET_DIR = "padded_cells"


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


def _normalize_subset(subset):
    if subset is None:
        return None
    subset = str(subset).strip()
    if subset in {"", "N/A", "NA"}:
        return None
    return subset


def _cd_tag_from_text(text):
    if not text:
        return None
    if "CD4+" in text:
        return "CD4"
    if "CD4-" in text:
        return "CD8"
    if "CD8" in text:
        return "CD8"
    return None


def detect_experiment_prefix(base_path):
    """
    Build experiment prefix from the patient folder path.

    Example:
      /.../non-responder/01-03-2026 DLBCL 109241
      -> nonresponder_109241
    """
    base_path = Path(base_path)
    patient_folder = base_path.name
    group_folder = base_path.parent.name

    group = re.sub(r"[^A-Za-z0-9]+", "", group_folder).lower()
    if not group:
        group = "unknown"

    match = re.search(r"\bDLBCL\s+(\d+)\b", patient_folder, flags=re.IGNORECASE)
    patient_id = match.group(1) if match else None
    if not patient_id:
        match = re.search(r"(\d+)\s*$", patient_folder)
        patient_id = match.group(1) if match else "unknown"

    return f"{group}_{patient_id}"


def format_cell_classification(cell_type, subset):
    """
    Format cell classification for filename.

    Rules:
    - Native CD4+ cells: native_CD4_{subset}
    - Native CD4- cells: native_CD8_{subset}
    - CAR-T cells: CART_CD4_{subset} or CART_CD8_{subset}

    Args:
        cell_type: Cell type (e.g., "CD4+", "CD4-", "CAR-T")
        subset: Cell subset (e.g., "Naive", "Central Memory (TCM)", "N/A")

    Returns:
        Formatted classification string
    """
    subset = _normalize_subset(subset)
    abbrev = extract_abbreviation(subset) if subset else None

    # Handle CAR-T cells
    if cell_type == "CAR-T":
        cd_tag = _cd_tag_from_text(subset)
        if subset:
            subset = re.sub(r"^CAR-T\s+(CD4\+|CD4-|CD8)\s+", "", subset)
        abbrev = extract_abbreviation(subset) if subset else None
        if not cd_tag:
            cd_tag = "CD4"
        return f"CART_{cd_tag}_{abbrev}" if abbrev else f"CART_{cd_tag}"

    # Handle native CD4/CD8 cells
    if cell_type in {"CD4+", "CD4-"}:
        cd_tag = "CD4" if cell_type == "CD4+" else "CD8"
        return f"native_{cd_tag}_{abbrev}" if abbrev else f"native_{cd_tag}"

    # Fallback
    return f"{cell_type}_{abbrev}".replace(" ", "_") if abbrev else str(cell_type).replace(" ", "_")


def name_cells(base_path, verbose=True):
    """
    Rename cell crop files with standardized naming convention.

    Args:
        base_path: Base directory path
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean
            - 'num_renamed': Number of files renamed
            - 'renames': List of (old_name, new_name) tuples
    """
    experiment_prefix = detect_experiment_prefix(base_path)
    input_csv = NAME_CELLS_INPUT_CSV
    target_dir = NAME_CELLS_TARGET_DIR

    if verbose:
        print("\n" + "="*80)
        print("RENAME CELL CROPS FOR EXPORT")
        print("="*80)
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
        sample_raw = sample_folder.replace('sample', '')
        sample_num = sample_raw.zfill(2) if sample_raw.isdigit() else sample_raw
        image_num_value = extract_image_number(image_number)
        image_num = (
            str(image_num_value).zfill(2)
            if image_num_value is not None
            else str(image_number)
        )
        cell_num = str(cell_number).zfill(2)
        classification = format_cell_classification(cell_type, subset)

        # Build paths (support folders like "14[large cell]" when CSV uses "14")
        sample_dir = Path(base_path) / str(sample_folder)
        image_folder = None
        image_str = str(image_number)
        if sample_dir.exists():
            candidate = sample_dir / image_str
            if candidate.exists():
                image_folder = image_str
            else:
                for entry in sorted(sample_dir.iterdir()):
                    if not entry.is_dir():
                        continue
                    name = entry.name
                    if image_str.isdigit() and name.startswith(image_str):
                        next_char = name[len(image_str):len(image_str)+1]
                        if next_char == "" or not next_char.isdigit():
                            image_folder = name
                            break

        if not image_folder:
            if verbose:
                print(f"  ⚠️  Image folder not found for {sample_folder}/{image_number}")
            num_skipped += 1
            continue

        base_dir = sample_dir / image_folder
        crop_dir = base_dir / target_dir

        if not crop_dir.exists():
            if verbose and idx == 0:
                print(f"  ⚠️  Warning: {target_dir}/ not found in {sample_folder}/{image_number}")
            num_skipped += 1
            continue

        # Find the original file (strict naming only)
        original_filename = f"cell_{str(cell_number).zfill(2)}_padded.tif"
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

        # Perform rename
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
    base_path = BASE_PATH
    if len(sys.argv) > 1:
        base_path = sys.argv[1]

    if not base_path:
        raise RuntimeError("✗ ERROR: Base path not provided. Pass a patient folder path or run via main.py.")

    result = name_cells(
        base_path=base_path,
        verbose=True,
    )

    if not result['success']:
        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
