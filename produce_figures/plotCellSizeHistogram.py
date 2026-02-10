#!/usr/bin/env python3
"""
Plot Cell Size Histogram
Creates a histogram of cell area from all_samples_combined.csv

Usage:
    python plotCellSizeHistogram.py
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI, saves to file only)
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import BASE_PATH from main.py

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
   from main import BASE_PATH
except ImportError:
    #Fallback if main.py is not available 
    BASE_PATH = Path("/mnt/HDD16TB/LanceKam_Lab/Daizong/Project/DLBCL/DLBCL/DLBCL_processed1/01-16-2026 DLBCL 109317")


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

CSV_FILE = "all_samples_combined.csv"
OUTPUT_FILE = "cell_area_histogram.png"
BINS = 50


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def plot_cell_area_histogram(base_path, csv_file=None, output_file=None, bins=None, verbose=True):
    """
    Plot a histogram of cell areas (pixels) from the combined CSV.

    Args:
        base_path: Base directory path containing the combined CSV
        csv_file: Name of combined CSV file (default: all_samples_combined.csv)
        output_file: Output filename (default: cell_area_histogram.png)
        bins: Histogram bin count (default: 50)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'figure_path': Path to saved figure
    """
    if csv_file is None:
        csv_file = "all_samples_combined.csv"
    if output_file is None:
        output_file = "cell_area_histogram.png"
    if bins is None:
        bins = 50

    if verbose:
        print(f"\n{'='*40}")
        print("PLOT CELL AREA HISTOGRAM")
        print(f"{'='*40}")

    csv_path = Path(csv_file) if csv_file else (Path(base_path) / "all_samples_combined.csv")
    output_path = Path(output_file) if output_file else (Path(base_path) / "cell_area_histogram.png")

    if not csv_path.exists():
        return {
            'success': False,
            'error': f'Combined measurements CSV not found: {csv_path}',
            'figure_path': None
        }

    if verbose:
        print(f"Loading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return {
            'success': False,
            'error': 'No data found in CSV',
            'figure_path': None
        }

    if 'area' not in df.columns:
        return {
            'success': False,
            'error': "Column 'area' not found in CSV",
            'figure_path': None
        }

    data = df['area'].dropna()
    if len(data) == 0:
        return {
            'success': False,
            'error': "No non-null values found in 'area' column",
            'figure_path': None
        }

    if verbose:
        print(f"  Total cells: {len(df)}")
        print(f"  Area values: {len(data)}")

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color="#2a6f97", edgecolor="#1b3b5a")
    plt.title("Cell Area Histogram")
    plt.xlabel("Area (um^2)")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\n✓ Figure saved: {output_path}")
        print(f"{'='*40}")
        print("PLOTTING COMPLETE")
        print(f"{'='*40}\n")

    return {
        'success': True,
        'figure_path': str(output_path),
        'num_cells': len(df)
    }


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cell area histogram from a CSV file.")
    parser.add_argument("--csv", dest="csv_path", default=None,
                        help="Path to CSV file (defaults to base_path/all_samples_combined.csv)")
    parser.add_argument("--output", dest="output_path", default=None,
                        help="Output image path (defaults to base_path/cell_area_histogram.png)")
    parser.add_argument("--bins", dest="bins", type=int, default=BINS,
                        help=f"Number of histogram bins (default: {BINS})")
    parser.add_argument("--base-path", dest="base_path", default=BASE_PATH,
                        help="Base directory used when --csv/--output are not provided")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")

    args = parser.parse_args()

    result = plot_cell_area_histogram(
        base_path=args.base_path,
        csv_file=args.csv_path,
        output_file=args.output_path,
        bins=args.bins,
        verbose=not args.quiet
    )

    if not result['success']:
        print(f"\n✗ Error: {result['error']}")
        import sys
        sys.exit(1)
