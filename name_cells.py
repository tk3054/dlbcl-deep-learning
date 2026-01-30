#!/usr/bin/env python3
"""
Name Cell ROI Crops for Export
Renames cell crop files in the target ROI directory with standardized naming
based on patient folder, sample/image numbers, and cell classification.

Naming format: {response}_{experiment_prefix}_sample{N}_image{NN}_cell{NN}_{lineage}_{cell_type}_{subset}.tif

Example:
- nonresponder_12-16-2025_DLBCL_108859_sample01_image01_cell01_native_CD4_Naive.tif
- nonresponder_12-16-2025_DLBCL_108859_sample01_image02_cell03_CART_CD8_TEM.tif

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
        "Central Memory (CM)" -> "CM"
        "Effector Memory (EM)" -> "EM"
        "Effector" -> "Effector"
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


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return False


def _normalize_component(text):
    cleaned = re.sub(r"\s+", "_", str(text).strip())
    cleaned = re.sub(r"[^\w\-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _normalize_response_label(text):
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(text).strip()).lower()
    return cleaned


def detect_experiment_prefix(base_path):
    """
    Build experiment prefix from the patient folder path.

    Example:
      /.../non-responder/12-16-2025 DLBCL 108859
      -> 12-16-2025_DLBCL_108859
    """
    base_path = Path(base_path)
    patient_folder = base_path.name
    return _normalize_component(patient_folder) or "unknown_experiment"


def detect_response_label(base_path):
    base_path = Path(base_path)
    response_folder = base_path.parent.name
    response = _normalize_response_label(response_folder)
    return response or "unknownresponse"


def format_lineage_label(car_positive, cell_type, subset):
    """
    Format lineage/type/subset label for filename.

    Rules:
    - Lineage is native or CART based on car_positive
    - CD4 vs CD8 is determined from cell_type
    - Subtype is derived from tcell_subset (abbreviation if present)
    - No QC classification suffix

    Args:
        car_positive: Bool or value indicating CAR positivity
        cell_type: CD4/CD8 label (e.g., "CD4+", "CD8+")
        subset: Cell subset (e.g., "CD4+ Central Memory (TCM)")

    Returns:
        Formatted label string
    """
    cell_type = (cell_type or "").strip()
    subset = _normalize_subset(subset)

    lineage = "CART" if _to_bool(car_positive) else "native"

    cd_tag = _cd_tag_from_text(cell_type)
    if not cd_tag and subset:
        cd_tag = _cd_tag_from_text(subset)
    cd_tag = cd_tag or "Unknown"

    if subset:
        subset = re.sub(r"^CAR-T\s+", "", subset)
        subset = re.sub(r"^(CD4\+|CD4-|CD8\+|CD8-)\s+", "", subset)
    abbrev = extract_abbreviation(subset) if subset else None
    subtype = _normalize_component(abbrev or "Unknown")

    return f"{lineage}_{cd_tag}_{subtype}"


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
    response = detect_response_label(base_path)
    experiment_prefix = detect_experiment_prefix(base_path)
    input_csv = NAME_CELLS_INPUT_CSV
    target_dir = NAME_CELLS_TARGET_DIR

    # Load classified CSV
    csv_path = Path(base_path) / input_csv
    if not csv_path.exists():
        return {
            'success': False,
            'error': f"CSV not found: {csv_path}",
            'num_renamed': 0,
            'renames': []
        }

    df = pd.read_csv(csv_path)

    # Track renames
    renames = []
    num_renamed = 0
    num_skipped = 0

    # Process each row in CSV
    for idx, row in df.iterrows():
        sample_folder = row['sample']
        image_number = row['image']
        cell_number = row['cell_id']
        car_positive = row.get('car_positive')
        cell_type = row.get('cell_type')
        subset = row.get('tcell_subset')

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
        lineage_label = format_lineage_label(car_positive, cell_type, subset)

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
            num_skipped += 1
            continue

        base_dir = sample_dir / image_folder
        crop_dir = base_dir / target_dir

        if not crop_dir.exists():
            num_skipped += 1
            continue

        # Find the original file (strict naming only)
        original_filename = f"cell_{str(cell_number).zfill(2)}_padded.tif"
        original_path = crop_dir / original_filename

        if not original_path.exists():
            num_skipped += 1
            continue

        # Build new filename
        new_filename = f"{response}_{experiment_prefix}_sample{sample_num}_image{image_num}_cell{cell_num}_{lineage_label}.tif"
        new_path = crop_dir / new_filename

        # Record rename
        renames.append((str(original_path.name), str(new_filename)))

        # Perform rename
        original_path.rename(new_path)

        num_renamed += 1

    if verbose:
        status = "✓ Files renamed successfully!" if num_renamed > 0 else "✗ No files renamed."
        print(status)

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
        print(f"✗ Failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
