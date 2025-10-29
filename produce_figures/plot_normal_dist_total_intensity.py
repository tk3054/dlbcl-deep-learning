#!/usr/bin/env python3
"""
Normal Distribution Fit to First Peak - Total Intensity Version

Fits a normal (Gaussian) distribution to the first peak of total intensity distributions
(mean intensity × cell area) for all fluorescent channels.
Uses peak detection to identify the first mode and fits only to data near that peak.

Usage examples:
    python plot_normal_dist_total_intensity.py
    python plot_normal_dist_total_intensity.py --peak-window 0.3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from main import BASE_PATH
except ImportError:
    BASE_PATH = "/Users/taeeonkong/Desktop/10-16-2025/new objective"

CSV_FILE = "all_samples_combined.csv"
OUTPUT_FILE = "all_samples_normal_dist_total_intensity.png"


@dataclass
class ChannelConfig:
    mean_column: str
    area_column: str
    label: str
    color: str


CHANNELS = {
    'actin': ChannelConfig("actin_mean", "area", "Actin-FITC", "green"),
    'cd4': ChannelConfig("cd4_mean", "area", "CD4-PerCP", "blue"),
    'pacblue': ChannelConfig("cd45ra_sparkviolet_mean", "area", "CD45RA-PacBlue", "purple"),
    'pe': ChannelConfig("ccr7_mean", "area", "CCR7-PE", "orange"),
}


def find_first_peak(data, bins=100, prominence_factor=0.1):
    """
    Find the first (leftmost) peak in the data histogram.

    Args:
        data: 1D array of intensity values
        bins: Number of histogram bins
        prominence_factor: Minimum prominence as fraction of max height

    Returns:
        Tuple of (peak_center, peak_height, bin_edges)
    """
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks with minimum prominence
    min_prominence = np.max(counts) * prominence_factor
    peaks, _ = find_peaks(counts, prominence=min_prominence)

    if len(peaks) == 0:
        # No clear peaks found, use the maximum
        peak_idx = np.argmax(counts)
    else:
        # Get the first (leftmost) peak
        peak_idx = peaks[0]

    peak_center = bin_centers[peak_idx]
    peak_height = counts[peak_idx]

    return peak_center, peak_height, bin_edges


def extract_peak_region(data, peak_center, peak_window=0.3):
    """
    Extract data around the peak for fitting.

    Args:
        data: 1D array of intensity values
        peak_center: Center of the peak
        peak_window: Fraction of data range to use as window around peak

    Returns:
        Filtered data array
    """
    data_range = data.max() - data.min()
    window_size = data_range * peak_window

    lower_bound = peak_center - window_size / 2
    upper_bound = peak_center + window_size / 2

    peak_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return peak_data


def fit_normal_distribution(data):
    """
    Fit a normal distribution to the data.

    Args:
        data: 1D array of intensity values

    Returns:
        Tuple of (mean, std, fit_info_dict)
    """
    mean = np.mean(data)
    std = np.std(data)

    # Compute goodness of fit metrics
    from scipy.stats import kstest, anderson
    ks_stat, ks_pval = kstest(data, 'norm', args=(mean, std))

    anderson_result = anderson(data, dist='norm')

    return mean, std, {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'anderson_statistic': anderson_result.statistic,
        'n_samples': len(data)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit normal distributions to first peaks in total intensity distributions."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(BASE_PATH),
        help="Directory containing the combined CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=CSV_FILE,
        help="Combined CSV filename (default: %(default)s)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=OUTPUT_FILE,
        help="Output figure filename (default: %(default)s)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of histogram bins (default: %(default)s)",
    )
    parser.add_argument(
        "--channels",
        nargs='+',
        choices=['actin', 'cd4', 'pacblue', 'pe', 'all'],
        default=['cd4', 'pacblue', 'pe'],
        help="Which channels to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--peak-window",
        type=float,
        default=0.3,
        help="Fraction of data range to use for peak region (default: %(default)s)",
    )
    parser.add_argument(
        "--prominence-factor",
        type=float,
        default=0.1,
        help="Minimum peak prominence as fraction of max height (default: %(default)s)",
    )
    return parser.parse_args()


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Combined CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Combined CSV contains no rows.")
    return df


def plot_normal_dist_fits(
    df: pd.DataFrame,
    channels_to_plot: list[str],
    bins: int,
    peak_window: float,
    prominence_factor: float,
    output_path: Path,
):
    """
    Create plots with normal distribution fits to first peaks of total intensity.
    Uses frequency (counts) on y-axis instead of density.
    """
    n_channels = len(channels_to_plot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    summaries = []

    for idx, channel_key in enumerate(channels_to_plot):
        if idx >= len(axes):
            break

        ax = axes[idx]
        channel = CHANNELS[channel_key]

        # Check if required columns exist
        if channel.mean_column not in df.columns or channel.area_column not in df.columns:
            ax.text(
                0.5, 0.5,
                f"{channel.label}\nColumn Missing",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=14
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Calculate total intensity
        data = (df[channel.mean_column] * df[channel.area_column]).dropna()
        data = data[data > 0]

        if data.empty:
            ax.text(
                0.5, 0.5,
                f"{channel.label}\nNo Positive Values",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=14
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Find first peak
        peak_center, peak_height, bin_edges = find_first_peak(
            data.to_numpy(), bins=bins, prominence_factor=prominence_factor
        )

        # Extract peak region
        peak_data = extract_peak_region(data.to_numpy(), peak_center, peak_window)

        # Fit normal distribution to peak region
        mean, std, fit_info = fit_normal_distribution(peak_data)

        # Calculate where distribution touches x-axis (±3σ from mean)
        x_axis_lower = mean - 3 * std
        x_axis_upper = mean + 3 * std

        # Calculate threshold as mean + 2*std (positive threshold)
        threshold = mean + 2 * std
        n_positive = np.sum(data >= threshold)
        pct_positive = (n_positive / len(data)) * 100

        # Plot histogram with frequency (counts) instead of density
        counts, bin_edges, patches = ax.hist(
            data,
            bins=bins,
            color=channel.color,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.4,
            label="Histogram (all data)"
        )

        # Plot the fitted normal distribution scaled to match histogram
        x_range = np.linspace(data.min(), data.max(), 500)
        bin_width = (data.max() - data.min()) / bins
        fitted_dist = norm.pdf(x_range, mean, std) * len(data) * bin_width
        ax.plot(
            x_range, fitted_dist,
            color="red", linewidth=2.5,
            label=f"Normal Fit\n(μ={mean:.1f}, σ={std:.1f})"
        )

        # Highlight the peak region used for fitting (removed red highlight per user request)

        # Mark the peak center
        ax.axvline(peak_center, color="darkred", linestyle="--", linewidth=1.5, alpha=0.7)

        # Mark where distribution touches x-axis
        ax.axvline(x_axis_lower, color="darkblue", linestyle=":", linewidth=2, alpha=0.6, label=f"±3σ bounds")
        ax.axvline(x_axis_upper, color="darkblue", linestyle=":", linewidth=2, alpha=0.6)

        # Draw threshold line and arrow (use dark green instead of red)
        ax.axvline(threshold, color="darkgreen", linestyle="-", linewidth=2.5, alpha=0.8, label=f"Threshold (μ+2σ)")

        # Get y-axis limits to draw arrow
        y_max = ax.get_ylim()[1]
        arrow_y = y_max * 0.8
        ax.annotate(
            "",
            xy=(threshold, arrow_y),
            xytext=(threshold, arrow_y * 0.5),
            arrowprops=dict(arrowstyle="-|>", color="darkgreen", lw=2.5)
        )

        ax.set_title(
            f"{channel.label}",
            fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Total Intensity (mean × area)")
        ax.set_ylabel("Frequency (Count)")

        # Scale y-axis to zoom in on the peak for all channels
        # Use 110% of the maximum count in the histogram
        ax.set_ylim(0, max(counts) * 1.1)

        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, loc="upper right")

        # Annotation
        annotation_lines = [
            f"n = {len(data)} (peak: {len(peak_data)})",
            f"μ = {mean:.2f}",
            f"σ = {std:.2f}",
            f"Threshold = {threshold:.2f}",
            f"KS p-val = {fit_info['ks_pvalue']:.4f}"
        ]

        ax.text(
            0.02, 0.98,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        # Display positive rate prominently
        marker_name = channel_key.upper()
        if channel_key == 'pacblue':
            marker_name = 'CD45RA'
        elif channel_key == 'pe':
            marker_name = 'CCR7'
        elif channel_key == 'cd4':
            marker_name = 'CD4'
        elif channel_key == 'actin':
            marker_name = 'Actin'

        ax.text(
            0.98, 0.25,
            f"{marker_name}+: {pct_positive:.1f}%",
            transform=ax.transAxes,
            ha="right", va="center",
            fontsize=13,
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        summaries.append({
            "channel": channel.label,
            "n_total": len(data),
            "n_peak_region": len(peak_data),
            "peak_center": peak_center,
            "fitted_mean": mean,
            "fitted_std": std,
            "threshold": threshold,
            "n_positive": n_positive,
            "pct_positive": pct_positive,
            "x_axis_lower": x_axis_lower,
            "x_axis_upper": x_axis_upper,
            "ks_statistic": fit_info['ks_statistic'],
            "ks_pvalue": fit_info['ks_pvalue'],
        })

    # Hide unused subplots
    for j in range(len(channels_to_plot), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Normal Distribution Fits to First Peak - Total Intensity (n={len(df)} cells)",
        fontsize=16, fontweight="bold", y=0.98
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return summaries


def main() -> None:
    args = parse_args()

    base_path = args.base_path.expanduser()
    csv_path = base_path / args.csv_file
    output_path = base_path / args.output_file

    # Determine which channels to plot
    if 'all' in args.channels:
        channels_to_plot = list(CHANNELS.keys())
    else:
        channels_to_plot = args.channels

    print("=" * 60)
    print("NORMAL DISTRIBUTION FITS TO FIRST PEAK - TOTAL INTENSITY")
    print("=" * 60)
    print(f"Base path: {base_path}")
    print(f"CSV file: {csv_path.name}")
    print(f"Output file: {output_path.name}")
    print(f"Channels: {', '.join(channels_to_plot)}")
    print(f"Bins: {args.bins}")
    print(f"Peak window: {args.peak_window}")
    print(f"Prominence factor: {args.prominence_factor}")
    print("=" * 60)

    df = load_dataframe(csv_path)
    print(f"Loaded {len(df)} rows.")

    summaries = plot_normal_dist_fits(
        df=df,
        channels_to_plot=channels_to_plot,
        bins=args.bins,
        peak_window=args.peak_window,
        prominence_factor=args.prominence_factor,
        output_path=output_path,
    )

    summary_df = pd.DataFrame(summaries)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    summary_path = output_path.with_suffix(".summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Figure saved to {output_path}")
    print(f"✓ Summary saved to {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n✗ Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
