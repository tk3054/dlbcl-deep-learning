#!/usr/bin/env python3
"""
Plot Intensity Histograms for All Patients Combined
Creates histograms showing intensity distribution for each channel across all patients
in a responder or non-responder category.

Usage:
    python plotAllPatients.py
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path("/Users/taeeonkong/Desktop/DL Project")
CATEGORY = "responder"  # "responder" or "non-responder"
# =============================================================================


def plot_all_patients_histograms(base_dir, category, verbose=True):
    """
    Plot histograms of intensity distributions for all patients in a category.

    Args:
        base_dir: Base directory containing responder/non-responder folders
        category: "responder" or "non-responder"
        verbose: Print progress messages

    Returns:
        dict with success status and output path
    """
    csv_path = Path(base_dir) / category / "all_patients_combined.csv"
    output_path = Path(base_dir) / category / f"{category}_intensity_histograms.png"

    if verbose:
        print("=" * 60)
        print(f"PLOT ALL PATIENTS - {category.upper()}")
        print("=" * 60)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return {'success': False, 'error': f'CSV not found: {csv_path}'}

    # Load data
    if verbose:
        print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return {'success': False, 'error': 'No data found in CSV'}

    patients = df['patient_id'].unique()
    if verbose:
        print(f"  Total cells: {len(df)}")
        print(f"  Patients: {list(patients)}")

    # Define channels
    channels = [
        (['actin_mean'], 'Actin-FITC', 'green'),
        (['cd4_mean'], 'CD4-PerCP', 'blue'),
        (['cd45ra_PacBlue_mean', 'cd45ra_sparkviolet_mean'], 'CD45RA-PacBlue', 'purple'),
        (['cd19car_mean'], 'CD19CAR-AF647', 'red'),
        (['ccr7_mean'], 'CCR7-PE', 'orange'),
    ]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Color palette for different patients
    colors = plt.cm.tab10(np.linspace(0, 1, len(patients)))

    for i, (col_names, channel_name, _) in enumerate(channels):
        ax = axes[i]

        col_name = next((c for c in col_names if c in df.columns), None)

        if not col_name:
            if verbose:
                print(f"  Warning: No columns found for {channel_name}")
            ax.text(0.5, 0.5, f'{channel_name}\nNot Available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot histogram for each patient (stacked/overlaid)
        for j, patient in enumerate(patients):
            patient_data = df[df['patient_id'] == patient][col_name].dropna()
            if len(patient_data) > 0:
                # Use shorter label for legend
                short_label = patient.split()[-1] if len(patient.split()) > 2 else patient
                ax.hist(patient_data, bins=50, alpha=0.5, label=short_label,
                        color=colors[j], edgecolor='black', linewidth=0.3)

        # Styling
        ax.set_title(f'{channel_name}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Mean Intensity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8, loc='upper right')

        # Add overall stats
        all_data = df[col_name].dropna()
        if len(all_data) > 0:
            ax.text(0.02, 0.98,
                    f'n={len(all_data)}\nmean={all_data.mean():.1f}\nmedian={all_data.median():.1f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)

        if verbose:
            print(f"  {channel_name}: n={len(all_data)}")

    # Hide extra subplot
    axes[5].set_visible(False)

    plt.suptitle(f'{category.upper()} - All Patients Combined (n={len(df)} cells, {len(patients)} patients)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"\nSaved: {output_path}")
        print("=" * 60)

    return {'success': True, 'figure_path': str(output_path)}


def plot_patients_comparison(base_dir, category, verbose=True):
    """
    Create a grid showing each patient's distribution separately for comparison.
    """
    csv_path = Path(base_dir) / category / "all_patients_combined.csv"
    output_path = Path(base_dir) / category / f"{category}_patients_comparison.png"

    if not csv_path.exists():
        return {'success': False, 'error': f'CSV not found: {csv_path}'}

    df = pd.read_csv(csv_path)
    patients = df['patient_id'].unique()

    if verbose:
        print(f"\nCreating patient comparison grid...")

    # Define channels
    channels = [
        (['actin_mean'], 'Actin', 'green'),
        (['cd4_mean'], 'CD4', 'blue'),
        (['cd45ra_PacBlue_mean', 'cd45ra_sparkviolet_mean'], 'CD45RA', 'purple'),
        (['cd19car_mean'], 'CAR', 'red'),
        (['ccr7_mean'], 'CCR7', 'orange'),
    ]

    # Create grid: rows = patients, cols = channels
    n_patients = len(patients)
    n_channels = len(channels)

    fig, axes = plt.subplots(n_patients, n_channels, figsize=(3*n_channels, 2.5*n_patients))

    if n_patients == 1:
        axes = axes.reshape(1, -1)

    for row, patient in enumerate(patients):
        patient_df = df[df['patient_id'] == patient]
        # Short patient label
        short_label = patient.split()[-1] if len(patient.split()) > 2 else patient

        for col, (col_names, channel_name, color) in enumerate(channels):
            ax = axes[row, col]

            col_name = next((c for c in col_names if c in patient_df.columns), None)

            if col_name and col_name in patient_df.columns:
                data = patient_df[col_name].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.3)
                    ax.text(0.95, 0.95, f'n={len(data)}', transform=ax.transAxes,
                            ha='right', va='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Labels
            if row == 0:
                ax.set_title(channel_name, fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(short_label, fontsize=9, fontweight='bold')

            ax.tick_params(labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.suptitle(f'{category.upper()} - Patient Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"Saved: {output_path}")

    return {'success': True, 'figure_path': str(output_path)}


def plot_combined_simple(base_dir, category, verbose=True):
    """
    Plot simple combined histograms treating all data as one cohort (no patient separation).
    Similar to plotAvgIntensity but for the all_patients_combined.csv file.
    """
    csv_path = Path(base_dir) / category / "all_patients_combined.csv"
    output_path = Path(base_dir) / category / f"{category}_combined_histograms.png"

    if verbose:
        print("=" * 60)
        print(f"PLOT COMBINED SIMPLE - {category.upper()}")
        print("=" * 60)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return {'success': False, 'error': f'CSV not found: {csv_path}'}

    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return {'success': False, 'error': 'No data found in CSV'}

    n_patients = df['patient_id'].nunique()
    if verbose:
        print(f"  Total cells: {len(df)}")
        print(f"  Number of patients: {n_patients}")

    # Define channels
    channels = [
        (['actin_mean'], 'Actin-FITC', 'green'),
        (['cd4_mean'], 'CD4-PerCP', 'blue'),
        (['cd45ra_PacBlue_mean', 'cd45ra_sparkviolet_mean'], 'CD45RA-PacBlue', 'purple'),
        (['cd19car_mean'], 'CD19CAR-AF647', 'red'),
        (['ccr7_mean'], 'CCR7-PE', 'orange'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (col_names, channel_name, color) in enumerate(channels):
        ax = axes[i]

        col_name = next((c for c in col_names if c in df.columns), None)

        if not col_name:
            ax.text(0.5, 0.5, f'{channel_name}\nNot Available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        data = df[col_name].dropna()
        n_cells = len(data)

        if n_cells == 0:
            ax.text(0.5, 0.5, f'{channel_name}\nNo Data',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Single histogram for all data
        ax.hist(data, bins=100, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_title(f'{channel_name} Intensity Distribution', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Mean Intensity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Stats
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        ax.text(0.98, 0.98,
                f'n = {n_cells}\nmean = {mean_val:.1f}\nstd = {std_val:.1f}\nmedian = {median_val:.1f}',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

        if verbose:
            print(f"  {channel_name}: n={n_cells}, mean={mean_val:.1f}, median={median_val:.1f}")

    axes[5].set_visible(False)

    plt.suptitle(f'{category.upper()} - Combined Intensity Distributions (n={len(df)} cells)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"\nSaved: {output_path}")
        print("=" * 60)

    return {'success': True, 'figure_path': str(output_path)}


if __name__ == "__main__":
    # Plot simple combined (all data as one)
    result0 = plot_combined_simple(BASE_DIR, CATEGORY, verbose=True)

    # Plot combined histograms (patients overlaid)
    result1 = plot_all_patients_histograms(BASE_DIR, CATEGORY, verbose=True)

    # Plot patient comparison grid
    result2 = plot_patients_comparison(BASE_DIR, CATEGORY, verbose=True)

    if not result0['success']:
        print(f"Error: {result0.get('error')}")
    if not result1['success']:
        print(f"Error: {result1.get('error')}")
    if not result2['success']:
        print(f"Error: {result2.get('error')}")
