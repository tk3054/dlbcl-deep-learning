# need a script to go through a csv, and name the cells based on the thresholds. 
# need to specify the thresholds for each patients 
# (so we would have the four below for all 6 patients seprately)
#   - CD4, CD45RA, CCR7, and CD19CAR

# need to iterate through csv, and extract patient number. 
    # read each line, and choose the threshold set.
    # extract the "Responder_dlbclprocessed1_113056_12-18-2025_1to10_sample1_image01_cell01_CD8_CD8_Effector_Memory_TEM.jpg"
        # make a response type column "Responder," or "Non-responder" 
        # make a date column, and put "12-18-2025"
        # make a new column for all the channels, indicating positive or negative
        # make a new column for T cell type, car vs native, and subtype
            # CD4(+): CD4, CD4(-): CD8
            # CD19CAR(+): CART, CD19CAR(-): native
            # Use the subtype matrix CCR7 x CD45RA
        # {responderType}_{date}_{DLBCL}_{1to10}_{patient number}_{sampleNumber}_{imageNumber}_{cellNumber}_{native/CART}_{CD4/CD8}_{subtype}.tif
        # nonresponder_12-16-2025_DLBCL_108859_sample01_image02_cell03_CART_CD8_TEM.tif

    # subtype logic: 
        #CCR7(+) CD45RA(+): naive
        #CCR7(+) CD45RA(-): CM
        #CCR7(-) CD45RA(+): effector
        #CCR7(-) CD45RA(-): CM

#!/usr/bin/env python3
"""
Name cells from a CSV using per-patient thresholds and create standardized filenames.

Input CSV is expected to include at least:
  - sample, image, cell_id
  - cd4_median, cd45ra_median, ccr7_median, cd19car_median
  - matched_photo (used to extract response, patient id, date)

Output adds:
  - response, date, patient_id
  - cd4_positive, cd45ra_positive, ccr7_positive, car_positive
  - cell_type (CD4/CD8), lineage (CART/native), tcell_subset
  - new_filename (standardized .tif name)

Usage:
  python analysis/name_cells_csv.py
  python analysis/name_cells_csv.py --input analysis/all_patients_combined_clean.csv --output analysis/named_cells.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default I/O
DEFAULT_INPUT_CSV = "analysis/all_patients_combined_clean.csv"
DEFAULT_OUTPUT_CSV = "analysis/named_cells.csv"

# These constants are fixed for naming based on your examples
EXPERIMENT_LABEL = "DLBCL"
PDMS_STIFFNESS = "1to10"

# Per-patient thresholds as JSON (fill in for each patient id)
# Example patient ids: "113056", "108859", ...
THRESHOLDS_JSON = """
{
  "118830": {"cd4": 175, "cd45ra": 110, "ccr7": 75, "cd19car": 60},
  "118867": {"cd4": 170, "cd45ra": 110, "ccr7": 75, "cd19car": 60},
  "108859": {"cd4": 200, "cd45ra": 120, "ccr7": 65, "cd19car": 60},
  "109241": {"cd4": 220, "cd45ra": 120, "ccr7": 65, "cd19car": 60},
  "109317": {"cd4": 200, "cd45ra": 110, "ccr7": 65, "cd19car": 60},
  "113056": {"cd4": 200, "cd45ra": 120, "ccr7": 60, "cd19car": 60}
}
"""

THRESHOLDS_BY_PATIENT: Dict[str, Dict[str, float]] = json.loads(THRESHOLDS_JSON)


# ============================================================================
# HELPERS
# ============================================================================

def _normalize_response(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(text).strip()).lower()
    if cleaned in {"responder", "r"}:
        return "Responder"
    if cleaned in {"nonresponder", "non-responder", "nr", "nonresponder"}:
        return "Non-responder"
    return cleaned or "unknown"


def _extract_from_matched_photo(name: str) -> Dict[str, Optional[str]]:
    """
    Expected pattern (examples):
      Responder_dlbclprocessed1_113056_12-18-2025_1to10_sample1_image01_cell01_...jpg
      Non-responder_dlbclprocessed1_108859_12-16-2025_1to10_sample1_image02_cell03_...jpg
    """
    if not name:
        return {"response": None, "patient_id": None, "date": None}

    # response: first token before underscore
    parts = str(name).split("_")
    response_raw = parts[0] if parts else ""
    response = _normalize_response(response_raw)

    # patient id: last long number before date or as 6-digit group
    patient_id = None
    id_match = re.search(r"_(\d{5,})_", name)
    if id_match:
        patient_id = id_match.group(1)
    else:
        numbers = re.findall(r"\d{5,}", name)
        if numbers:
            patient_id = numbers[0]

    # date: mm-dd-yyyy
    date = None
    date_match = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", name)
    if date_match:
        month, day, year = date_match.groups()
        date = f"{int(month):02d}-{int(day):02d}-{year}"

    return {"response": response, "patient_id": patient_id, "date": date}


def _get_thresholds(patient_id: Optional[str]) -> Optional[Dict[str, float]]:
    if not patient_id:
        return None
    return THRESHOLDS_BY_PATIENT.get(patient_id)


def _to_bool(value: bool) -> bool:
    return bool(value)


def classify_row(row: pd.Series, thresholds: Dict[str, float]) -> Dict[str, object]:
    cd4 = row["cd4_median"]
    cd45ra = row["cd45ra_median"]
    ccr7 = row["ccr7_median"]
    cd19car = row["cd19car_median"]

    is_cd4_pos = cd4 > thresholds["cd4"]
    is_cd45ra_pos = cd45ra > thresholds["cd45ra"]
    is_ccr7_pos = ccr7 > thresholds["ccr7"]
    is_car_pos = cd19car > thresholds["cd19car"]

    if is_cd45ra_pos and is_ccr7_pos:
        subset = "Naive"
    elif not is_cd45ra_pos and is_ccr7_pos:
        subset = "CM"
    elif not is_cd45ra_pos and not is_ccr7_pos:
        subset = "EM"
    else:
        subset = "Effector"

    cell_type = "CD4" if is_cd4_pos else "CD8"
    lineage = "CART" if is_car_pos else "native"

    return {
        "cd4_positive": _to_bool(is_cd4_pos),
        "cd45ra_positive": _to_bool(is_cd45ra_pos),
        "ccr7_positive": _to_bool(is_ccr7_pos),
        "car_positive": _to_bool(is_car_pos),
        "cell_type": cell_type,
        "lineage": lineage,
        "tcell_subset": subset,
    }


def build_new_filename(row: pd.Series, response: str, date: str, patient_id: str) -> str:
    sample_num = str(row["sample"]).replace("sample", "")
    if sample_num.isdigit():
        sample_num = sample_num.zfill(2)
    image_num = str(row["image"]).zfill(2)
    cell_num = str(row["cell_id"]).zfill(2)

    lineage = row["lineage"]
    cell_type = row["cell_type"]
    subset = row["tcell_subset"]

    return (
        f"{response}_{date}_{EXPERIMENT_LABEL}_{PDMS_STIFFNESS}_{patient_id}_"
        f"sample{sample_num}_image{image_num}_cell{cell_num}_"
        f"{lineage}_{cell_type}_{subset}.jpg"
    )


# ============================================================================
# MAIN
# ============================================================================

def name_cells_csv(input_csv: str, output_csv: str, verbose: bool = True) -> Dict[str, object]:
    input_path = Path(input_csv)
    if not input_path.exists():
        return {"success": False, "error": f"CSV not found: {input_path}"}

    df = pd.read_csv(input_path)
    required = {"sample", "image", "cell_id", "matched_photo", "cd4_median", "cd45ra_median", "ccr7_median", "cd19car_median"}
    missing = required - set(df.columns)
    if missing:
        return {"success": False, "error": f"Missing required columns: {', '.join(sorted(missing))}"}

    # Extract response/patient/date
    extracted = df["matched_photo"].apply(_extract_from_matched_photo)
    df["response"] = extracted.apply(lambda x: x.get("response"))
    df["patient_id"] = extracted.apply(lambda x: x.get("patient_id"))
    df["date"] = extracted.apply(lambda x: x.get("date"))

    # Classify using thresholds per patient
    results = []
    skipped = 0
    for _, row in df.iterrows():
        thresholds = _get_thresholds(row.get("patient_id"))
        if not thresholds:
            results.append(
                {
                    "cd4_positive": None,
                    "cd45ra_positive": None,
                    "ccr7_positive": None,
                    "car_positive": None,
                    "cell_type": None,
                    "lineage": None,
                    "tcell_subset": None,
                }
            )
            skipped += 1
            continue
        results.append(classify_row(row, thresholds))

    result_df = pd.DataFrame(results)
    for col in result_df.columns:
        df[col] = result_df[col]

    # Build filenames
    df["new_filename"] = df.apply(
        lambda row: build_new_filename(
            row=row,
            response=row["response"] or "unknown",
            date=row["date"] or "unknown-date",
            patient_id=row["patient_id"] or "unknown",
        ),
        axis=1,
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"✓ Saved: {output_path}")
        if skipped:
            print(f"⚠️  Skipped {skipped} rows (missing thresholds for patient_id)")

    return {"success": True, "output_csv": str(output_path), "skipped": skipped, "total": len(df)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Name cells using per-patient thresholds.")
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    result = name_cells_csv(args.input, args.output, verbose=True)
    if not result.get("success"):
        raise SystemExit(f"✗ Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
