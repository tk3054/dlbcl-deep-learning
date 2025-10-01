#!/usr/bin/env python3
"""
Plot Intensity Histograms
Creates histograms showing mean intensity distribution for each channel

Usage:
    python plot_intensity_histograms.py
    (Edit SAMPLE_FOLDER and IMAGE_NUMBER below to change inputs)
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI, saves to file only)
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Change these values to process different samples
SAMPLE_FOLDER = "sample1"  # Options: "sample1", "sample2", "sample3"
IMAGE_NUMBER = "5"         # Options: "1", "2", "3", "4", etc.

# Auto-generated paths
BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"
BASE_DIR = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def plot_intensity_histograms(sample_folder, image_number, base_path, save_figure=True, verbose=True):
    """
    Plot histograms of mean intensity distributions for all channels.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        save_figure: Save figure as PNG (default: True)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'figure_path': Path to saved figure (if save_figure=True)
    """
    if verbose:
        print("="*60)
        print("PLOT INTENSITY HISTOGRAMS")
        print("="*60)
        print(f"Sample: {sample_folder}/{image_number}\n")

    # Build paths
    csv_path = Path(base_path) / 'combined_measurements.csv'

    # Check if CSV exists
    if not csv_path.exists():
        return {
            'success': False,
            'error': f'Combined measurements CSV not found: {csv_path}',
            'figure_path': None
        }

    # Load data and filter for this sample/image
    if verbose:
        print(f"Loading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Filter for specific sample and image (convert image_number to int for comparison)
    df = df[(df['sample'] == sample_folder) & (df['image'] == int(image_number))]

    if len(df) == 0:
        return {
            'success': False,
            'error': f'No data found for {sample_folder}/{image_number}',
            'figure_path': None
        }

    # Define channels to plot
    channels = [
        ('actin_mean', 'Actin-FITC', 'green'),
        ('cd4_mean', 'CD4-PerCP', 'blue'),
        ('cd45ra_mean', 'CD45RA-AF647', 'red'),
        ('ccr7_mean', 'CCR7-PE', 'orange')
    ]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot histogram for each channel
    for i, (col_name, channel_name, color) in enumerate(channels):
        ax = axes[i]

        # Get data
        data = df[col_name].dropna()
        n_cells = len(data)

        # Plot histogram
        ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Styling
        ax.set_title(f'{channel_name} intensity distribution', fontsize=14, pad=10)
        ax.set_xlabel('Mean Intensity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add statistics text
        mean_val = data.mean()
        std_val = data.std()
        ax.text(0.98, 0.98, f'n = {n_cells}\nmean = {mean_val:.1f}\nstd = {std_val:.1f}',
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)

        if verbose:
            print(f"  {channel_name}: n={n_cells}, mean={mean_val:.1f}, std={std_val:.1f}")

    plt.tight_layout()

    # Save figure to 1to10 directory
    if save_figure:
        output_dir = Path(base_path)
        output_path = output_dir / f'{sample_folder}_{image_number}_intensity_histograms.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"\n✓ Figure saved: {output_path.name}")
        figure_path = str(output_path)
    else:
        figure_path = None

    # Close figure to avoid memory issues and blocking
    plt.close(fig)

    if verbose:
        print("="*60)
        print("PLOTTING COMPLETE")
        print("="*60)

    return {
        'success': True,
        'figure_path': figure_path
    }


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    result = plot_intensity_histograms(
        sample_folder=SAMPLE_FOLDER,
        image_number=IMAGE_NUMBER,
        base_path=BASE_PATH,
        save_figure=True,
        verbose=True
    )

    if not result['success']:
        print(f"\n✗ Error: {result['error']}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
