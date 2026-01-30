#!/usr/bin/env python3
"""
T Cell Classification
Classifies T cells into subsets based on CD4, CD45RA, CCR7, and CD19CAR expression

Classification:
- CAR-T cells: CD19CAR positive (takes priority)
- CD4+ T cells: CD4 positive
  - Naive: CD45RA+ CCR7+
  - Central Memory (TCM): CD45RA- CCR7+
  - Effector Memory (TEM): CD45RA- CCR7-
  - Terminal Effector (TEMRA): CD45RA+ CCR7-
- CD4- cells: CD4 negative

Usage:
    python classify_tcells.py
"""

import pandas as pd
from pathlib import Path
import sys

# Import BASE_PATH from main.py
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from main import BASE_PATH
except ImportError:
    BASE_PATH = "'/Users/taeeonkong/Desktop/10-16-2025/new objective"


# ============================================================================
# CONFIGURATION - EDIT THESE THRESHOLDS
# ============================================================================

# Manual thresholds - Edit these values based on your data
THRESHOLDS = {
    'cd4': 260,      # CD4-PerCP
    'cd45ra': 120,   # CD45RA (uses SparkViolet channel; plots label it as PacBlue)
    'ccr7': 114,     # CCR7-PE
    'cd19car': 430,  # CD19CAR-AF647
}

# Which CD45RA channel to use? Options: 'af647', 'sparkviolet', 'average', 'max'
CD45RA_CHANNEL_MODE = 'average'

# CSV filename
INPUT_CSV = "all_samples_combined.csv"
OUTPUT_CSV = "all_samples_combined_classified.csv"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_cd45ra_value(row, mode='average'):
    """Get CD45RA value based on mode (average, max, af647, or sparkviolet)"""

    # Check which CD45RA channels exist
    has_af647_mean = 'cd45ra_af647_mean' in row.index
    has_sparkviolet_mean = 'cd45ra_sparkviolet_mean' in row.index

    # Get available values (mean only)
    af647_val = None
    sparkviolet_val = None

    if has_af647_mean:
        af647_val = row['cd45ra_af647_mean']

    if has_sparkviolet_mean:
        sparkviolet_val = row['cd45ra_sparkviolet_mean']

    if not has_af647_mean and not has_sparkviolet_mean:
        raise KeyError("Missing CD45RA mean columns: expected cd45ra_af647_mean or cd45ra_sparkviolet_mean")

    # Handle different modes
    if mode == 'af647':
        if af647_val is not None:
            return af647_val
        elif sparkviolet_val is not None:
            return sparkviolet_val  # Fallback
        else:
            return 0
    elif mode == 'sparkviolet':
        if sparkviolet_val is not None:
            return sparkviolet_val
        elif af647_val is not None:
            return af647_val  # Fallback
        else:
            return 0
    elif mode == 'average':
        if af647_val is not None and sparkviolet_val is not None:
            return (af647_val + sparkviolet_val) / 2
        elif sparkviolet_val is not None:
            return sparkviolet_val
        elif af647_val is not None:
            return af647_val
        else:
            return 0
    elif mode == 'max':
        if af647_val is not None and sparkviolet_val is not None:
            return max(af647_val, sparkviolet_val)
        elif sparkviolet_val is not None:
            return sparkviolet_val
        elif af647_val is not None:
            return af647_val
        else:
            return 0
    else:
        # Default to average mode
        if af647_val is not None and sparkviolet_val is not None:
            return (af647_val + sparkviolet_val) / 2
        elif sparkviolet_val is not None:
            return sparkviolet_val
        elif af647_val is not None:
            return af647_val
        else:
            return 0


def classify_cell(row, thresholds):
    """
    Classify a single cell based on marker expression.

    Args:
        row: DataFrame row with cell measurements
        thresholds: Dict of thresholds for each marker

    Returns:
        Dict with classification results
    """
    # Get marker values (mean only)
    if 'cd4_mean' not in row.index:
        raise KeyError("Missing cd4_mean column")
    if 'ccr7_mean' not in row.index:
        raise KeyError("Missing ccr7_mean column")
    cd4 = row['cd4_mean']
    cd45ra = get_cd45ra_value(row, CD45RA_CHANNEL_MODE)
    ccr7 = row['ccr7_mean']
    if 'cd19car_mean' not in row.index:
        raise KeyError("Missing cd19car_mean column")
    cd19car = row['cd19car_mean']

    # Determine positivity
    is_cd4_pos = cd4 > thresholds['cd4']
    is_cd45ra_pos = cd45ra > thresholds['cd45ra']
    is_ccr7_pos = ccr7 > thresholds['ccr7']
    car_threshold = thresholds.get('cd19car')
    is_car_pos = bool(
        cd19car is not None and car_threshold is not None and cd19car > car_threshold
    )

    # Determine memory subset based on CD45RA and CCR7
    if is_cd45ra_pos and is_ccr7_pos:
        memory_subset = "Naive"
    elif not is_cd45ra_pos and is_ccr7_pos:
        memory_subset = "CM"
    elif not is_cd45ra_pos and not is_ccr7_pos:
        memory_subset = "EM"
    elif is_cd45ra_pos and not is_ccr7_pos:
        memory_subset = "Effector"
    else:
        memory_subset = "Unclassified"

    cell_type = "CD4+" if is_cd4_pos else "CD8+"
    subset = memory_subset

    return {
        'cd4_positive': is_cd4_pos,
        'cd45ra_positive': is_cd45ra_pos,
        'ccr7_positive': is_ccr7_pos,
        'car_positive': is_car_pos,
        'cell_type': cell_type,
        'tcell_subset': subset
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def classify_tcells(base_path, input_csv=None, output_csv=None, verbose=True):
    """
    Classify T cells from combined measurements CSV.

    Args:
        base_path: Base directory path
        input_csv: Input CSV filename (default: all_samples_combined.csv)
        output_csv: Output CSV filename (default: all_samples_combined_classified.csv)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean
            - 'output_csv': Path to output file
            - 'num_cells': Total cells
            - 'num_car_pos': Number of CAR-T cells
            - 'num_cd4_pos': Number of CD4+ cells
            - 'num_cd8_pos': Number of CD8+ cells
            - 'subset_counts': Dict of subset counts
    """
    if input_csv is None:
        input_csv = INPUT_CSV
    if output_csv is None:
        output_csv = OUTPUT_CSV

    if verbose:
        print("\n" + "="*60)
        print("T CELL CLASSIFICATION")
        print("="*60)

    # Load data
    csv_path = Path(base_path) / input_csv
    if not csv_path.exists():
        return {
            'success': False,
            'error': f"CSV not found: {csv_path}",
            'output_csv': None
        }

    if verbose:
        print(f"\nLoading: {input_csv}")

    df = pd.read_csv(csv_path)

    if verbose:
        print(f"  Total cells: {len(df)}")

    # Use manual thresholds
    if verbose:
        print("\nUsing manual thresholds:")
        for marker, value in THRESHOLDS.items():
            print(f"  {marker.upper()}: {value:.2f}")

    thresholds = THRESHOLDS

    # Classify each cell
    if verbose:
        print("\nClassifying cells...")

    classifications = df.apply(lambda row: classify_cell(row, thresholds), axis=1)

    # Add classification columns
    df['cd4_positive'] = [c['cd4_positive'] for c in classifications]
    df['cd45ra_positive'] = [c['cd45ra_positive'] for c in classifications]
    df['ccr7_positive'] = [c['ccr7_positive'] for c in classifications]
    df['car_positive'] = [c['car_positive'] for c in classifications]
    df['cell_type'] = [c['cell_type'] for c in classifications]
    df['tcell_subset'] = [c['tcell_subset'] for c in classifications]

    # Calculate statistics
    cell_type_counts = df['cell_type'].value_counts()
    lineage_counts = cell_type_counts.to_dict()
    subset_counts = df['tcell_subset'].value_counts().to_dict()

    if verbose:
        print("\n" + "="*60)
        print("CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Total cells: {len(df)}")
        print("\nLineage distribution:")
        for lineage in ["CAR-T", "CD4+", "CD8+"]:
            count = int(lineage_counts.get(lineage, 0))
            pct = count / len(df) * 100 if len(df) else 0
            print(f"  {lineage}: {count} ({pct:.1f}%)")

        print("\nChannel positivity:")
        channel_positive_series = [
            ("CD4", df['cd4_positive']),
            ("CD45RA", df['cd45ra_positive']),
            ("CCR7", df['ccr7_positive']),
        ]
        if thresholds.get('cd19car') is not None:
            channel_positive_series.append(("CD19CAR", df['car_positive']))

        for label, series in channel_positive_series:
            count = int(series.sum())
            pct = count / len(df) * 100 if len(df) else 0
            print(f"  {label}: {count} ({pct:.1f}%)")

        print("\nSubset distribution:")
        for subset, count in sorted(subset_counts.items()):
            pct = count / len(df) * 100
            print(f"  {subset}: {count} ({pct:.1f}%)")

    # Save output
    output_path = Path(base_path) / output_csv
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\n✓ Saved classified data to: {output_csv}")
        print("="*60 + "\n")

    num_car_pos = int(lineage_counts.get("CAR-T", 0))
    num_cd4_pos = int(lineage_counts.get("CD4+", 0))
    num_cd8_pos = int(lineage_counts.get("CD8+", 0))

    return {
        'success': True,
        'output_csv': str(output_path),
        'num_cells': len(df),
        'num_car_pos': num_car_pos,
        'num_cd4_pos': num_cd4_pos,
        'num_cd8_pos': num_cd8_pos,
        'subset_counts': subset_counts,
        'thresholds': thresholds
    }


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    result = classify_tcells(
        base_path=BASE_PATH,
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        verbose=True
    )

    if not result['success']:
        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
