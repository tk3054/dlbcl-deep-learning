#!/usr/bin/env python3
"""
KDE-Valley Threshold Plots for Mean Intensities

Uses a Gaussian KDE on log10-transformed mean intensities to find the deepest
valley between modes (if present). The resulting threshold is overlaid on the
histogram along with the KDE curve for visual confirmation.

Usage examples:
    python plot_kde_valley_thresholds.py
    python plot_kde_valley_thresholds.py --bw 0.25 --bins 120
    python plot_kde_valley_thresholds.py --no-log10  # operate in raw space
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from main import BASE_PATH
except ImportError:
    BASE_PATH = "/Users/taeeonkong/Desktop/10-16-2025/new objective"

CSV_FILE = "all_samples_combined.csv"
OUTPUT_FILE = "all_samples_kde_valley_thresholds.png"


def format_threshold_text(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:,.2f}"
    return f"{value:.2e}"


def kde_valley_threshold(
    x,
    log10=True,
    n_grid=2048,
    bw=None,
    order=10,
    edge_margin=0.02,
):
    """
    Estimate a threshold using the deepest valley in a KDE-smoothed density.
    Returns (threshold_raw, grid, density) if a valley is found, otherwise (None, grid, density).
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        return None, None, None

    if log10:
        x = np.log10(x)

    kde = gaussian_kde(x, bw_method=bw)
    grid = np.linspace(x.min(), x.max(), n_grid)
    dens = kde(grid)

    minima = argrelextrema(dens, np.less_equal, order=order)[0]
    if minima.size == 0:
        return None, grid, dens

    if edge_margin > 0:
        span = grid.max() - grid.min()
        margin = span * edge_margin
        mask = (grid[minima] > grid.min() + margin) & (grid[minima] < grid.max() - margin)
        minima = minima[mask]
        if minima.size == 0:
            return None, grid, dens

    idx = minima[np.argmin(dens[minima])]
    t_log = grid[idx]
    t = 10**t_log if log10 else t_log
    return t, grid, dens


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
        description="Plot KDE valley thresholds for mean intensity distributions."
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
        "--bw",
        type=float,
        default=None,
        help="Bandwidth passed to gaussian_kde (bw_method). Default lets SciPy choose.",
    )
    parser.add_argument(
        "--no-log10",
        action="store_true",
        help="Disable log10 transform (operate in raw intensity space).",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=10,
        help="Neighborhood order for local minima detection (default: %(default)s)",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.02,
        help=(
            "Fractional margin (0-0.5) of the KDE grid to ignore near the edges "
            "when selecting valleys (default: %(default)s). Increase if the red line hugs the edge."
        ),
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable per-panel text annotations.",
    )
    return parser.parse_args()


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Combined CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Combined CSV contains no rows.")
    return df


def plot_kde_thresholds(
    df: pd.DataFrame,
    log10: bool,
    bins: int,
    bw: float | None,
    order: int,
    edge_margin: float,
    annotate: bool,
    output_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    summaries = []

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
        data = data[data > 0]
        if data.empty:
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

        threshold, grid, density = kde_valley_threshold(
            data.to_numpy(),
            log10=log10,
            n_grid=2048,
            bw=bw,
            order=order,
            edge_margin=edge_margin,
        )

        if log10:
            plot_values = np.log10(data)
            hist_label = "log10(mean intensity)"
        else:
            plot_values = data
            hist_label = "Mean intensity"

        ax.hist(
            plot_values,
            bins=bins,
            color=channel.color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.4,
            density=True,
        )

        if grid is not None and density is not None:
            ax.plot(grid, density, color="black", linewidth=1.5, label="KDE")

        threshold_display = np.log10(threshold) if (threshold is not None and log10) else threshold

        ax.set_title(
            f"{channel.label} ({'log10 ' if log10 else ''}mean intensity)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(hist_label)
        ax.set_ylabel("Density")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        positive_fraction = float(np.mean(data > threshold)) if threshold is not None else None

        if annotate:
            annotation = [f"n = {len(data)}"]
            if threshold is not None:
                annotation.extend(
                    [
                        f"threshold ≈ {format_threshold_text(threshold)}",
                        f"frac > thr = {positive_fraction*100:.1f}%",
                    ]
                )
                if log10:
                    annotation.append(f"log10(thr) = {threshold_display:.2f}")
            else:
                annotation.append("no valley found")

            ax.text(
                0.98,
                0.98,
                "\n".join(annotation),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

        summaries.append(
            {
                "channel": channel.label,
                "n_cells": len(data),
                "threshold": threshold,
                "log10_threshold": np.log10(threshold) if threshold is not None else None,
                "positive_fraction": positive_fraction,
                "edge_margin": edge_margin,
                "status": "ok" if threshold is not None else "no valley",
            }
        )

    for j in range(len(CHANNELS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"KDE Valley Thresholds ({'log10' if log10 else 'raw'} mean intensities, n={len(df)} cells)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
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

    log10 = not args.no_log10

    print("=" * 60)
    print("KDE VALLEY THRESHOLDS")
    print("=" * 60)
    print(f"Base path: {base_path}")
    print(f"CSV file: {csv_path.name}")
    print(f"Output file: {output_path.name}")
    print(f"log10 transform: {log10}")
    print(f"Bins: {args.bins}, KDE bandwidth: {args.bw}")
    print(f"Order: {args.order}, edge margin: {args.edge_margin}")
    print("=" * 60)

    df = load_dataframe(csv_path)
    print(f"Loaded {len(df)} rows.")

    summaries = plot_kde_thresholds(
        df=df,
        log10=log10,
        bins=args.bins,
        bw=args.bw,
        order=args.order,
        edge_margin=args.edge_margin,
        annotate=not args.no_annotate,
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
        sys.exit(1)
