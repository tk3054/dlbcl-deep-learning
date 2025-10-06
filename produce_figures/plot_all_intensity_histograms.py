#!/usr/bin/env python3
"""
Plot Intensity Histograms for All Samples Combined
Creates histograms showing median intensity distribution for each channel across all samples

Usage:
    python plot_all_intensity_histograms.py
"""

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
    # Fallback if main.py is not available
    BASE_PATH = "/Users/taeeonkong/Desktop/2025 Fall Images/09-26-2025 DLBCL"

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

CSV_FILE = "all_samples_combined.csv"
OUTPUT_FILE = "all_samples_intensity_histograms.png"


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def plot_all_intensity_histograms(base_path, csv_file=None, output_file=None, verbose=True):
    """
    Plot histograms of median intensity distributions for all channels across all samples.

    Args:
        base_path: Base directory path containing the combined CSV
        csv_file: Name of combined CSV file (default: all_samples_combined.csv)
        output_file: Output filename (default: all_samples_intensity_histograms.png)
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
        output_file = "all_samples_intensity_histograms.png"

    if verbose:
        print(f"\n{'='*40}")
        print("PLOT INTENSITY HISTOGRAMS (ALL SAMPLES)")
        print(f"{'='*40}")

    # Build paths
    csv_path = Path(base_path) / csv_file
    output_path = Path(base_path) / output_file

    # Check if CSV exists
    if not csv_path.exists():
        return {
            'success': False,
            'error': f'Combined measurements CSV not found: {csv_path}',
            'figure_path': None
        }

    # Load data
    if verbose:
        print(f"Loading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return {
            'success': False,
            'error': 'No data found in CSV',
            'figure_path': None
        }

    if verbose:
        print(f"  Total cells: {len(df)}")
        print(f"  Samples: {df['sample'].unique().tolist()}")

    # Define channels to plot
    channels = [
        ('actin_median', 'Actin-FITC', 'green'),
        ('cd4_median', 'CD4-PerCP', 'blue'),
        ('cd45ra_sparkviolet_median', 'CD45RA-SparkViolet', 'purple'),
        ('cd19car_median', 'CD19CAR-AF647', 'red'),
        ('ccr7_median', 'CCR7-PE', 'orange')
    ]

    # Create figure with subplots (3x2 grid for 5 channels)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Plot histogram for each channel
    for i, (col_name, channel_name, color) in enumerate(channels):
        ax = axes[i]

        # Check if column exists
        if col_name not in df.columns:
            if verbose:
                print(f"  ⚠️  Column '{col_name}' not found, skipping {channel_name}")
            ax.text(0.5, 0.5, f'{channel_name}\nNot Available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Get data
        data = df[col_name].dropna()
        n_cells = len(data)

        if n_cells == 0:
            if verbose:
                print(f"  ⚠️  No data for {channel_name}")
            ax.text(0.5, 0.5, f'{channel_name}\nNo Data',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot histogram with more bins for better resolution
        ax.hist(data, bins=100, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Styling
        ax.set_title(f'{channel_name} Intensity Distribution', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Median Intensity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add statistics text
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        ax.text(0.98, 0.98,
                f'n = {n_cells}\nmean = {mean_val:.1f}\nstd = {std_val:.1f}\nmedian = {median_val:.1f}',
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)

        if verbose:
            print(f"  {channel_name}: n={n_cells}, mean={mean_val:.1f}, std={std_val:.1f}, median={median_val:.1f}")

    # Hide the extra subplot (we have 5 channels in a 2x3 grid)
    axes[5].set_visible(False)

    plt.suptitle(f'All Samples Combined - Intensity Distributions (n={len(df)} cells)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save figure with higher DPI for better resolution
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    if verbose:
        print(f"\n✓ Figure saved: {output_path}")

    # Close figure to avoid memory issues
    plt.close(fig)

    if verbose:
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
    result = plot_all_intensity_histograms(
        base_path=BASE_PATH,
        csv_file=CSV_FILE,
        output_file=OUTPUT_FILE,
        verbose=True
    )

    if not result['success']:
        print(f"\n✗ Error: {result['error']}")
        import sys
        sys.exit(1)
