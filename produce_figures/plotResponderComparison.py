#!/usr/bin/env python3
"""
Plot Responder vs Non-Responder Comparison
Creates a grid comparing all patients with color-coded backgrounds.

Usage:
    python plotResponderComparison.py
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path("/Users/taeeonkong/Desktop/DL Project")
# =============================================================================


def plot_all_comparison(base_dir, verbose=True):
    """
    Create comparison grid for all patients (responders + non-responders).
    Responders have light green background band, non-responders have orangish-red band.
    """
    base_dir = Path(base_dir)
    from matplotlib.patches import Rectangle, Patch

    # Load both CSVs
    responder_csv = base_dir / "responder" / "all_patients_combined.csv"
    non_responder_csv = base_dir / "non-responder" / "all_patients_combined.csv"

    if verbose:
        print("=" * 60)
        print("RESPONDER VS NON-RESPONDER COMPARISON")
        print("=" * 60)

    dfs = []

    if responder_csv.exists():
        df_r = pd.read_csv(responder_csv)
        df_r['response'] = 'responder'
        dfs.append(df_r)
        if verbose:
            print(f"Responders: {df_r['patient_id'].nunique()} patients, {len(df_r)} cells")

    if non_responder_csv.exists():
        df_nr = pd.read_csv(non_responder_csv)
        df_nr['response'] = 'non-responder'
        dfs.append(df_nr)
        if verbose:
            print(f"Non-responders: {df_nr['patient_id'].nunique()} patients, {len(df_nr)} cells")

    if not dfs:
        print("Error: No CSV files found")
        return {'success': False}

    df = pd.concat(dfs, ignore_index=True)

    # Get patients in order: responders first, then non-responders
    responder_patients = df[df['response'] == 'responder']['patient_id'].unique().tolist()
    non_responder_patients = df[df['response'] == 'non-responder']['patient_id'].unique().tolist()
    all_patients = responder_patients + non_responder_patients

    if verbose:
        print(f"\nTotal: {len(all_patients)} patients, {len(df)} cells")

    # Define channels
    channels = [
        (['actin_mean'], 'Actin', 'green'),
        (['cd4_mean'], 'CD4', 'blue'),
        (['cd45ra_PacBlue_mean', 'cd45ra_sparkviolet_mean'], 'CD45RA', 'purple'),
        (['cd19car_mean'], 'CAR', 'red'),
        (['ccr7_mean'], 'CCR7', 'orange'),
    ]

    n_patients = len(all_patients)
    n_channels = len(channels)

    # Background colors
    bg_responder = '#d4edda'      # Light green
    bg_non_responder = '#f8d7da'  # Light red/salmon

    fig, axes = plt.subplots(n_patients, n_channels, figsize=(3*n_channels, 2.5*n_patients))

    if n_patients == 1:
        axes = axes.reshape(1, -1)

    # Store row info for background bands
    row_info = []

    for row, patient in enumerate(all_patients):
        patient_df = df[df['patient_id'] == patient]
        response = patient_df['response'].iloc[0]
        bg_color = bg_responder if response == 'responder' else bg_non_responder
        row_info.append((row, bg_color))

        # Short patient label (just the ID number)
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

            # Column titles (top row only)
            if row == 0:
                ax.set_title(channel_name, fontsize=11, fontweight='bold')

            # Row labels (first column only)
            if col == 0:
                ax.set_ylabel(short_label, fontsize=10, fontweight='bold')

            ax.tick_params(labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Add legend for background colors
    legend_elements = [
        Patch(facecolor=bg_responder, edgecolor='black', label='Responder'),
        Patch(facecolor=bg_non_responder, edgecolor='black', label='Non-Responder')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=10)

    plt.suptitle(f'Responder vs Non-Responder Comparison\n({len(responder_patients)} responders, {len(non_responder_patients)} non-responders)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Add row-spanning background highlights
    fig.canvas.draw()
    for row, bg_color in row_info:
        bbox_first = axes[row, 0].get_position()
        bbox_last = axes[row, -1].get_position()

        rect = Rectangle(
            (bbox_first.x0 - 0.02, bbox_first.y0 - 0.01),
            (bbox_last.x1 - bbox_first.x0) + 0.04,
            bbox_first.height + 0.02,
            transform=fig.transFigure,
            facecolor=bg_color,
            edgecolor='gray',
            linewidth=0.5,
            zorder=-1
        )
        fig.patches.append(rect)

    output_path = base_dir / "responder_vs_nonresponder_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"\nSaved: {output_path}")
        print("=" * 60)

    return {'success': True, 'figure_path': str(output_path)}


def plot_combined_comparison(base_dir, verbose=True):
    """
    Compare responders vs non-responders as two combined groups.
    All responder data combined as one, all non-responder data combined as one.
    """
    base_dir = Path(base_dir)
    from matplotlib.patches import Rectangle

    responder_csv = base_dir / "responder" / "all_patients_combined.csv"
    non_responder_csv = base_dir / "non-responder" / "all_patients_combined.csv"

    if verbose:
        print("=" * 60)
        print("RESPONDER VS NON-RESPONDER (COMBINED)")
        print("=" * 60)

    groups = []

    if responder_csv.exists():
        df_r = pd.read_csv(responder_csv)
        groups.append(('Responder', df_r, '#d4edda'))  # light green
        if verbose:
            print(f"Responders: {df_r['patient_id'].nunique()} patients, {len(df_r)} cells")

    if non_responder_csv.exists():
        df_nr = pd.read_csv(non_responder_csv)
        groups.append(('Non-Responder', df_nr, '#f8d7da'))  # light red
        if verbose:
            print(f"Non-responders: {df_nr['patient_id'].nunique()} patients, {len(df_nr)} cells")

    if not groups:
        print("Error: No CSV files found")
        return {'success': False}

    # Define channels
    channels = [
        (['actin_mean'], 'Actin', 'green'),
        (['cd4_mean'], 'CD4', 'blue'),
        (['cd45ra_PacBlue_mean', 'cd45ra_sparkviolet_mean'], 'CD45RA', 'purple'),
        (['cd19car_mean'], 'CAR', 'red'),
        (['ccr7_mean'], 'CCR7', 'orange'),
    ]

    n_groups = len(groups)
    n_channels = len(channels)

    fig, axes = plt.subplots(n_groups, n_channels, figsize=(3.5*n_channels, 3*n_groups))

    if n_groups == 1:
        axes = axes.reshape(1, -1)

    for row, (group_name, group_df, bg_color) in enumerate(groups):
        for col, (col_names, channel_name, color) in enumerate(channels):
            ax = axes[row, col]

            col_name = next((c for c in col_names if c in group_df.columns), None)

            if col_name and col_name in group_df.columns:
                data = group_df[col_name].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.3)

                    # Stats box
                    ax.text(0.95, 0.95,
                            f'n={len(data)}\nmean={data.mean():.1f}\nmed={data.median():.1f}',
                            transform=ax.transAxes, ha='right', va='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            if row == 0:
                ax.set_title(channel_name, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(group_name, fontsize=11, fontweight='bold')

            ax.tick_params(labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.suptitle('Responder vs Non-Responder (All Patients Combined)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Add row-spanning background highlights
    fig.canvas.draw()
    for row, (group_name, group_df, bg_color) in enumerate(groups):
        # Get bounding box of first and last subplot in this row
        bbox_first = axes[row, 0].get_position()
        bbox_last = axes[row, -1].get_position()

        # Create rectangle spanning the entire row with padding
        rect = Rectangle(
            (bbox_first.x0 - 0.02, bbox_first.y0 - 0.01),
            (bbox_last.x1 - bbox_first.x0) + 0.04,
            bbox_first.height + 0.02,
            transform=fig.transFigure,
            facecolor=bg_color,
            edgecolor='gray',
            linewidth=1,
            zorder=-1
        )
        fig.patches.append(rect)

    output_path = base_dir / "responder_vs_nonresponder_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"\nSaved: {output_path}")
        print("=" * 60)

    return {'success': True, 'figure_path': str(output_path)}


if __name__ == "__main__":
    # Individual patient comparison
    plot_all_comparison(BASE_DIR, verbose=True)

    print()

    # Combined group comparison
    plot_combined_comparison(BASE_DIR, verbose=True)
