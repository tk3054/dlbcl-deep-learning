#!/usr/bin/env python3
"""
Log-Transform + Robust Quantile Threshold Plots

This script creates log10 mean-intensity histograms for each channel and overlays
robust thresholds based on median + 2 × MAD (Median Absolute Deviation). An
optional negative-control cohort can be provided to draw the 95th percentile line.

Usage:
    python plot_log_mad_thresholds.py
    python plot_log_mad_thresholds.py --csv-file custom.csv --output-file custom.png
    python plot_log_mad_thresholds.py --negative-sample sample1 --negative-sample sample3
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # ensure we always render to file
import matplotlib.pyplot as plt

# Allow importing BASE_PATH from produce_figures/main.py parent
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from main import BASE_PATH
except ImportError:
    BASE_PATH = "/Users/taeeonkong/Desktop/10-16-2025/new objective"


CSV_FILE = "all_samples_combined.csv"
OUTPUT_FILE = "all_samples_log_mad_thresholds.png"


@dataclass
class ChannelConfig:
    column: str
    label: str
    color: str


CHANNELS = [
    ChannelConfig("actin_mean", "Actin-FITC", "green"),
    ChannelConfig("cd4_mean", "CD4-PerCP", "blue"),
    ChannelConfig("cd45ra_sparkviolet_mean", "CD45RA-PacBlue", "purple"),
    ChannelConfig("ccr7_mean", "CCR7-PE", "orange"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot log10 mean intensity histograms with median+2×MAD thresholds."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(BASE_PATH),
        help="Base directory containing the combined CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=CSV_FILE,
        help="Name of the combined CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=OUTPUT_FILE,
        help="Output filename for the figure (default: %(default)s)",
    )
    parser.add_argument(
        "--negative-sample",
        action="append",
        dest="negative_samples",
        default=None,
        help=(
            "Sample label to use as negative control (can be repeated). "
            "Draws the 95th percentile of the specified sample subset."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins (default: %(default)s)",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable per-axis text annotations.",
    )
    return parser.parse_args()


def robust_threshold(log_values: np.ndarray) -> tuple[float, float, float]:
    """
    Compute the log-domain median, MAD, and the median+2×MAD threshold.
    Returns (median_log10, mad_log10, threshold_log10).
    """
    median_log = float(np.median(log_values))
    mad_log = float(np.median(np.abs(log_values - median_log)))
    threshold_log = median_log + 2.0 * mad_log
    return median_log, mad_log, threshold_log


def percentile_cutoff(log_values: np.ndarray, percentile: float = 95.0) -> float:
    """
    Compute percentile in the log domain.
    """
    return float(np.percentile(log_values, percentile))


def format_threshold_text(threshold_raw: float) -> str:
    if threshold_raw >= 1000:
        return f"{threshold_raw:,.0f}"
    if threshold_raw >= 1:
        return f"{threshold_raw:,.2f}"
    return f"{threshold_raw:.2e}"


def plot_log_histograms(
    df: pd.DataFrame,
    negative_df: pd.DataFrame | None,
    bins: int,
    annotate: bool,
    output_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    summary_rows = []

    for idx, channel in enumerate(CHANNELS):
        ax = axes[idx]

        if channel.column not in df.columns:
            ax.text(
                0.5,
                0.5,
                f"{channel.label}\nColumn Missing",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        data = df[channel.column].dropna()
        positive_mask = data > 0
        valid_data = data[positive_mask]

        if valid_data.empty:
            ax.text(
                0.5,
                0.5,
                f"{channel.label}\nNo Positive Values",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        log_values = np.log10(valid_data.to_numpy())
        median_log, mad_log, thr_log = robust_threshold(log_values)
        thr_raw = 10 ** thr_log
        positives_fraction = float(np.mean(log_values > thr_log))

        neg_thr_log = None
        neg_thr_raw = None
        if negative_df is not None and channel.column in negative_df.columns:
            neg_vals = negative_df[channel.column].dropna()
            neg_vals = neg_vals[neg_vals > 0]
            if not neg_vals.empty:
                log_neg = np.log10(neg_vals.to_numpy())
                neg_thr_log = percentile_cutoff(log_neg, 95.0)
                neg_thr_raw = 10 ** neg_thr_log

        ax.hist(
            log_values,
            bins=bins,
            color=channel.color,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.axvline(median_log, color="black", linestyle="--", linewidth=1.0, label="Median")
        ax.axvline(
            thr_log,
            color="red",
            linestyle="-",
            linewidth=1.5,
            label="Median + 2×MAD",
        )
        if neg_thr_log is not None:
            ax.axvline(
                neg_thr_log,
                color="teal",
                linestyle=":",
                linewidth=1.5,
                label="Neg. ctrl 95th pct",
            )

        ax.set_title(f"{channel.label} (log10 mean intensity)", fontsize=14, fontweight="bold")
        ax.set_xlabel("log10(Mean Intensity)")
        ax.set_ylabel("Cell Count")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if annotate:
            annotation_lines = [
                f"n = {len(log_values)}",
                f"median = {median_log:.2f}",
                f"MAD = {mad_log:.2f}",
                f"thr (log10) = {thr_log:.2f}",
                f"thr (raw) ≈ {format_threshold_text(thr_raw)}",
                f"frac > thr = {positives_fraction*100:.1f}%",
            ]
            if neg_thr_log is not None:
                annotation_lines.append(f"neg 95th ≈ {format_threshold_text(neg_thr_raw)}")

            ax.text(
                0.98,
                0.98,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

        summary_rows.append(
            {
                "channel": channel.label,
                "median_log10": median_log,
                "mad_log10": mad_log,
                "threshold_log10": thr_log,
                "threshold_raw": thr_raw,
                "positive_fraction": positives_fraction,
                "neg95_log10": neg_thr_log,
                "neg95_raw": neg_thr_raw,
                "n_cells": len(log_values),
                "dropped_nonpositive": len(data) - len(valid_data),
            }
        )

    # Hide unused subplot if fewer than allocated axes
    for j in range(len(CHANNELS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Log10 Mean Intensity Distributions with Robust Thresholds (n={len(df)} cells)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return summary_rows


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Combined CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Combined CSV is empty.")
    return df


def main() -> None:
    args = parse_args()

    base_path = Path(args.base_path).expanduser()
    csv_path = base_path / args.csv_file
    output_path = base_path / args.output_file

    print("=" * 60)
    print("LOG-TRANSFORM + ROBUST QUANTILE THRESHOLDS")
    print("=" * 60)
    print(f"Base path: {base_path}")
    print(f"CSV: {csv_path.name}")
    print(f"Output: {output_path.name}")
    if args.negative_samples:
        neg_list = ", ".join(args.negative_samples)
        print(f"Negative control samples: {neg_list}")
    print("=" * 60)

    df = load_dataframe(csv_path)
    print(f"Loaded {len(df)} rows with columns: {', '.join(df.columns[:8])}...")

    negative_df = None
    if args.negative_samples:
        negative_df = df[df["sample"].isin(args.negative_samples)]
        print(f"Negative control subset size: {len(negative_df)}")
        if negative_df.empty:
            print("⚠️  Negative control sample filter returned no rows; skipping 95th percentile line.")
            negative_df = None

    summary_rows = plot_log_histograms(
        df=df,
        negative_df=negative_df,
        bins=args.bins,
        annotate=not args.no_annotate,
        output_path=output_path,
    )

    summary_df = pd.DataFrame(summary_rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    summary_csv_path = output_path.with_suffix(".summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n✓ Figure saved to {output_path}")
    print(f"✓ Summary saved to {summary_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"\n✗ Error: {exc}")
        sys.exit(1)

