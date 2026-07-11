"""Plot donor-level channel intensity distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dlbcl_pipeline import constants


DEFAULT_CHANNELS = (
    (("actin_mean",), "Actin-FITC", "green"),
    (("cd4_mean",), "CD4-PerCP", "blue"),
    (("cd45ra_mean",), "CD45RA-PacBlue", "purple"),
    (("cd19car_mean",), "CD19CAR-AF647", "red"),
    (("ccr7_mean",), "CCR7-AF594", "orange"),
)


def plot_donor_intensity_distributions(
    donor_dir: str | Path,
    csv_file: str = constants.DONOR_COMBINED_CSV,
    output_file: str = "all_samples_intensity_histograms.png",
    donor_label: str | None = None,
    bins: int = 100,
    verbose: bool = True,
) -> dict[str, object]:
    """Plot mean-intensity histograms for one donor CSV."""
    donor_dir = Path(donor_dir)
    csv_path = donor_dir / csv_file
    output_path = donor_dir / output_file

    if not csv_path.exists():
        return {
            "success": False,
            "error": f"Combined measurements CSV not found: {csv_path}",
            "figure_path": None,
        }

    df = pd.read_csv(csv_path)
    if df.empty:
        return {
            "success": False,
            "error": f"No rows found in CSV: {csv_path}",
            "figure_path": None,
        }

    if verbose:
        print("\n" + "=" * 40)
        print("PLOT INTENSITY HISTOGRAMS")
        print("=" * 40)
        print(f"Loading: {csv_path}")
        print(f"Total cells: {len(df)}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    stats: dict[str, dict[str, float | int | str | None]] = {}

    for idx, (column_candidates, channel_name, color) in enumerate(DEFAULT_CHANNELS):
        ax = axes[idx]
        column = next((candidate for candidate in column_candidates if candidate in df.columns), None)

        if column is None:
            ax.text(
                0.5,
                0.5,
                f"{channel_name}\nNot Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            stats[channel_name] = {"column": None, "n": 0}
            if verbose:
                print(f"  Missing: {channel_name} ({', '.join(column_candidates)})")
            continue

        data = df[column].dropna()
        if data.empty:
            ax.text(
                0.5,
                0.5,
                f"{channel_name}\nNo Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            stats[channel_name] = {"column": column, "n": 0}
            if verbose:
                print(f"  No data: {channel_name} ({column})")
            continue

        ax.hist(data, bins=bins, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{channel_name} Intensity Distribution", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Mean Intensity", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        mean_val = float(data.mean())
        std_val = float(data.std())
        median_val = float(data.median())
        ax.text(
            0.98,
            0.98,
            f"n = {len(data)}\nmean = {mean_val:.1f}\nstd = {std_val:.1f}\nmedian = {median_val:.1f}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
        )

        stats[channel_name] = {
            "column": column,
            "n": int(len(data)),
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
        }
        if verbose:
            print(f"  {channel_name}: n={len(data)}, mean={mean_val:.1f}, median={median_val:.1f}")

    axes[5].set_visible(False)
    title = donor_label or donor_dir.name
    plt.suptitle(
        f"{title} - Intensity Distributions (n={len(df)} cells)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"Saved: {output_path}")
        print("=" * 40 + "\n")

    return {
        "success": True,
        "figure_path": str(output_path),
        "num_cells": int(len(df)),
        "stats": stats,
    }
