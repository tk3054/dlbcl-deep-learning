#!/usr/bin/env python3
"""
CD45RA vs CCR7 Quadrant Analysis with GMM Clustering
Uses Gaussian Mixture Model (GMM) on asinh-transformed, z-scored markers
to identify T cell subsets:
- Naïve: CD45RA+ CCR7+
- TCM (Central Memory): CD45RA- CCR7+
- TEM (Effector Memory): CD45RA- CCR7-
- TEMRA (Effector Memory RA): CD45RA+ CCR7-

Usage:
    python plot_cd45ra_ccr7_quadrants.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib

# Import BASE_PATH from main.py
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from main import BASE_PATH
except ImportError:
    # Fallback if main.py is not available
    BASE_PATH = '/Users/taeeonkong/Desktop/10-16-2025/new objective'

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_FILE = "all_samples_combined.csv"
OUTPUT_2D_PLOT = "cd45ra_ccr7_gmm_clusters.png"
OUTPUT_DISTRIBUTIONS = "cd45ra_ccr7_distributions.png"
OUTPUT_STATS = "cd45ra_ccr7_statistics.txt"
OUTPUT_MODEL = "gmm_cluster_model.pkl"

# Cofactors for asinh transformation (channel-specific)
COFACTORS = {
    'cd45ra_sparkviolet_mean': 300,  # PacBlue
    'ccr7_mean': 150,  # PE
    'cd4_mean': 300,  # PerCP
    'actin_mean': 150  # FITC
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def asinh_transform(data, cofactor=150):
    """
    Apply asinh (inverse hyperbolic sine) transformation.
    Commonly used in flow cytometry for scaling.

    Args:
        data: Raw intensity values
        cofactor: Scaling factor (typically 150 for flow cytometry)

    Returns:
        Transformed data
    """
    return np.arcsinh(data / cofactor)


def annotate_cluster_to_subset(df, cluster_col='cluster'):
    """
    Annotate GMM clusters to biological T cell subsets based on median marker levels.

    Args:
        df: DataFrame with cluster assignments and marker columns
        cluster_col: Name of cluster column

    Returns:
        Dictionary mapping cluster ID to subset name
    """
    # Use the correct column names for asinh-transformed data
    cd45ra_asinh_col = 'cd45ra_sparkviolet_asinh'
    ccr7_asinh_col = 'ccr7_asinh'

    # Compute median marker levels per cluster
    cluster_medians = df.groupby(cluster_col)[[cd45ra_asinh_col, ccr7_asinh_col]].median()

    # Determine thresholds (median of all data)
    cd45ra_thresh = df[cd45ra_asinh_col].median()
    ccr7_thresh = df[ccr7_asinh_col].median()

    # Annotate each cluster
    cluster_mapping = {}
    for cluster_id in cluster_medians.index:
        cd45ra_med = cluster_medians.loc[cluster_id, cd45ra_asinh_col]
        ccr7_med = cluster_medians.loc[cluster_id, ccr7_asinh_col]

        if cd45ra_med >= cd45ra_thresh and ccr7_med >= ccr7_thresh:
            subset = 'Naïve'
        elif cd45ra_med < cd45ra_thresh and ccr7_med >= ccr7_thresh:
            subset = 'TCM'
        elif cd45ra_med < cd45ra_thresh and ccr7_med < ccr7_thresh:
            subset = 'TEM'
        else:  # cd45ra_med >= cd45ra_thresh and ccr7_med < ccr7_thresh
            subset = 'TEMRA'

        cluster_mapping[cluster_id] = subset

    return cluster_mapping


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def plot_cd45ra_ccr7_analysis(base_path, csv_file=None, n_clusters=4, verbose=True):
    """
    Create comprehensive CD45RA vs CCR7 analysis using GMM clustering.

    Args:
        base_path: Base directory path containing the combined CSV
        csv_file: Name of combined CSV file
        n_clusters: Number of clusters for GMM (default: 4 for Naive/TCM/TEM/TEMRA)
        verbose: Print progress messages

    Returns:
        dict with analysis results
    """
    if csv_file is None:
        csv_file = CSV_FILE

    if verbose:
        print(f"\n{'='*60}")
        print("GMM CLUSTERING ANALYSIS - T CELL SUBSETS")
        print(f"{'='*60}")

    # Build paths
    csv_path = Path(base_path) / csv_file
    output_2d_path = Path(base_path) / OUTPUT_2D_PLOT
    output_dist_path = Path(base_path) / OUTPUT_DISTRIBUTIONS
    output_stats_path = Path(base_path) / OUTPUT_STATS
    output_model_path = Path(base_path) / OUTPUT_MODEL

    # Check if CSV exists
    if not csv_path.exists():
        return {
            'success': False,
            'error': f'Combined measurements CSV not found: {csv_path}'
        }

    # Load data
    if verbose:
        print(f"Loading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return {'success': False, 'error': 'No data found in CSV'}

    # Define required columns
    required_cols = ['cd45ra_sparkviolet_mean', 'ccr7_mean', 'cd4_mean', 'actin_mean']

    # Check for required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {
            'success': False,
            'error': f'Required columns not found: {missing_cols}'
        }

    # Remove NaN values
    df_clean = df[required_cols].dropna().copy()

    if verbose:
        print(f"  Total cells: {len(df_clean)}")

    # ========================================================================
    # Step 1: Pre-processing - asinh transformation with channel-specific cofactors
    # ========================================================================
    if verbose:
        print(f"\nStep 1: Applying asinh transformation...")

    for col in required_cols:
        cofactor = COFACTORS[col]
        transformed_col = col.replace('_mean', '_asinh')
        df_clean[transformed_col] = asinh_transform(df_clean[col], cofactor=cofactor)
        if verbose:
            print(f"  {col}: cofactor={cofactor}")

    # ========================================================================
    # Step 2: Z-score standardization
    # ========================================================================
    if verbose:
        print(f"\nStep 2: Applying z-score standardization...")

    asinh_cols = [col.replace('_mean', '_asinh') for col in required_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[asinh_cols])

    if verbose:
        print(f"  Shape: {X_scaled.shape}")

    # ========================================================================
    # Step 3: GMM Clustering
    # ========================================================================
    if verbose:
        print(f"\nStep 3: Running Gaussian Mixture Model (n_clusters={n_clusters})...")

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full',
                          random_state=42, n_init=10)
    cluster_labels = gmm.fit_predict(X_scaled)
    df_clean['cluster'] = cluster_labels

    # Compute BIC and AIC for model evaluation
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)

    if verbose:
        print(f"  BIC: {bic:.2f}")
        print(f"  AIC: {aic:.2f}")
        print(f"  Converged: {gmm.converged_}")

    # ========================================================================
    # Step 4: Biological annotation
    # ========================================================================
    if verbose:
        print(f"\nStep 4: Annotating clusters to T cell subsets...")

    cluster_to_subset = annotate_cluster_to_subset(df_clean)
    df_clean['subset'] = df_clean['cluster'].map(cluster_to_subset)

    # Compute subset percentages
    subset_counts = df_clean['subset'].value_counts()
    subset_pcts = (subset_counts / len(df_clean) * 100).to_dict()

    if verbose:
        print(f"\nT Cell Subset Distribution:")
        for subset in ['Naïve', 'TCM', 'TEM', 'TEMRA']:
            if subset in subset_counts:
                count = subset_counts[subset]
                pct = subset_pcts[subset]
                print(f"  {subset}: {pct:.1f}% (n={count})")
            else:
                print(f"  {subset}: 0.0% (n=0)")

    # ========================================================================
    # Step 5: Save model for future use
    # ========================================================================
    if verbose:
        print(f"\nStep 5: Saving model and scaler...")

    model_data = {
        'gmm': gmm,
        'scaler': scaler,
        'cluster_to_subset': cluster_to_subset,
        'cofactors': COFACTORS,
        'features': required_cols
    }
    joblib.dump(model_data, output_model_path)

    if verbose:
        print(f"  ✓ Model saved: {output_model_path}")

    # ========================================================================
    # FIGURE 1: 2D Scatter Plots colored by GMM clusters and subsets
    # ========================================================================
    if verbose:
        print(f"\nGenerating 2D cluster plots...")

    fig1, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color maps for subsets
    subset_colors = {
        'Naïve': '#2E86AB',
        'TCM': '#A23B72',
        'TEM': '#F18F01',
        'TEMRA': '#C73E1D'
    }

    # --- Plot 1: Colored by GMM cluster ID ---
    ax1 = axes[0]

    for cluster_id in sorted(df_clean['cluster'].unique()):
        cluster_data = df_clean[df_clean['cluster'] == cluster_id]
        subset_name = cluster_to_subset[cluster_id]
        color = subset_colors.get(subset_name, 'gray')

        ax1.scatter(cluster_data['cd45ra_sparkviolet_asinh'], cluster_data['ccr7_asinh'],
                   c=color, alpha=0.4, s=10, edgecolors='none',
                   label=f'Cluster {cluster_id} ({subset_name})')

    ax1.set_xlabel('CD45RA asinh-transformed', fontsize=12)
    ax1.set_ylabel('CCR7 asinh-transformed', fontsize=12)
    ax1.set_title('GMM Clusters (asinh-transformed)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best', framealpha=0.9)
    ax1.grid(alpha=0.3)

    # --- Plot 2: Colored by T cell subset ---
    ax2 = axes[1]

    for subset in ['Naïve', 'TCM', 'TEM', 'TEMRA']:
        subset_data = df_clean[df_clean['subset'] == subset]
        if len(subset_data) > 0:
            color = subset_colors[subset]
            ax2.scatter(subset_data['cd45ra_sparkviolet_asinh'], subset_data['ccr7_asinh'],
                       c=color, alpha=0.4, s=10, edgecolors='none',
                       label=f'{subset} ({len(subset_data)} cells)')

    ax2.set_xlabel('CD45RA asinh-transformed', fontsize=12)
    ax2.set_ylabel('CCR7 asinh-transformed', fontsize=12)
    ax2.set_title('T Cell Subsets (GMM-based)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(alpha=0.3)

    plt.suptitle(f'GMM Clustering Results (n={len(df_clean)} cells, k={n_clusters})',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_2d_path, dpi=600, bbox_inches='tight')
    plt.close(fig1)

    if verbose:
        print(f"  ✓ 2D plot saved: {output_2d_path}")

    # ========================================================================
    # FIGURE 2: Distribution Analysis with Skewness
    # ========================================================================
    if verbose:
        print(f"Generating distribution plots...")

    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Compute skewness for all markers
    from scipy.stats import gaussian_kde

    markers_info = [
        ('cd45ra_sparkviolet_mean', 'CD45RA-PacBlue', 'purple'),
        ('ccr7_mean', 'CCR7-PE', 'orange'),
        ('cd4_mean', 'CD4-PerCP', 'blue'),
        ('actin_mean', 'Actin-FITC', 'green')
    ]

    for idx, (col, label, color) in enumerate(markers_info):
        ax = axes[idx // 2, idx % 2]
        data = df_clean[col]

        # Histogram with KDE
        ax.hist(data, bins=100, alpha=0.6, color=color, edgecolor='black',
                linewidth=0.5, density=True, label='Histogram')

        # Add KDE
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), 'k-', linewidth=2, label='KDE')

        # Compute skewness
        skewness = stats.skew(data)

        # Add mean and median lines
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label='Median')

        ax.set_xlabel(f'{label} Intensity', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{label} Distribution (Skewness: {skewness:.3f})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Marker Intensity Distributions with Skewness',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dist_path, dpi=600, bbox_inches='tight')
    plt.close(fig2)

    if verbose:
        print(f"  ✓ Distribution plot saved: {output_dist_path}")

    # ========================================================================
    # Save Statistics to Text File
    # ========================================================================
    if verbose:
        print(f"Saving statistics...")

    with open(output_stats_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GMM CLUSTERING ANALYSIS - T CELL SUBSETS\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total cells analyzed: {len(df_clean)}\n")
        f.write(f"Number of clusters: {n_clusters}\n")
        f.write(f"GMM Converged: {gmm.converged_}\n")
        f.write(f"BIC: {bic:.2f}\n")
        f.write(f"AIC: {aic:.2f}\n\n")

        f.write("="*70 + "\n")
        f.write("T CELL SUBSET DISTRIBUTION\n")
        f.write("="*70 + "\n\n")

        for subset in ['Naïve', 'TCM', 'TEM', 'TEMRA']:
            if subset in subset_counts:
                count = subset_counts[subset]
                pct = subset_pcts[subset]
                f.write(f"  {subset}: {pct:.2f}% (n={count})\n")
            else:
                f.write(f"  {subset}: 0.00% (n=0)\n")

        f.write("\n")
        f.write("="*70 + "\n")
        f.write("CLUSTER TO SUBSET MAPPING\n")
        f.write("="*70 + "\n\n")

        for cluster_id, subset_name in sorted(cluster_to_subset.items()):
            cluster_size = len(df_clean[df_clean['cluster'] == cluster_id])
            f.write(f"  Cluster {cluster_id} → {subset_name} (n={cluster_size})\n")

        f.write("\n")
        f.write("="*70 + "\n")
        f.write("MARKER STATISTICS (Raw Intensities)\n")
        f.write("="*70 + "\n\n")

        for col, label, _ in markers_info:
            data = df_clean[col]
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)

            f.write(f"{label}:\n")
            f.write(f"  Mean: {data.mean():.2f}\n")
            f.write(f"  Median: {data.median():.2f}\n")
            f.write(f"  Std Dev: {data.std():.2f}\n")
            f.write(f"  Skewness: {skewness:.4f}\n")
            f.write(f"  Kurtosis: {kurtosis:.4f}\n")
            f.write(f"  Range: {data.min():.2f} - {data.max():.2f}\n\n")

        f.write("="*70 + "\n")
        f.write("TRANSFORMATION PARAMETERS\n")
        f.write("="*70 + "\n\n")

        f.write("Cofactors for asinh transformation:\n")
        for col, cofactor in COFACTORS.items():
            f.write(f"  {col}: {cofactor}\n")

        f.write("\n")
        f.write("="*70 + "\n")

    if verbose:
        print(f"  ✓ Statistics saved: {output_stats_path}")
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}\n")

    return {
        'success': True,
        'subsets': subset_pcts,
        'cluster_mapping': cluster_to_subset,
        'bic': bic,
        'aic': aic,
        'figure_2d': str(output_2d_path),
        'figure_dist': str(output_dist_path),
        'stats_file': str(output_stats_path),
        'model_file': str(output_model_path)
    }


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    result = plot_cd45ra_ccr7_analysis(
        base_path=BASE_PATH,
        csv_file=CSV_FILE,
        verbose=True
    )

    if not result['success']:
        print(f"\n✗ Error: {result['error']}")
        sys.exit(1)
