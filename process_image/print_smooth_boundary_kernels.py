#!/usr/bin/env python3
"""
Generate visualizations of the Gaussian kernels used for smooth_boundary.

Outputs two figures per sigma:
1. A heatmap with a colorbar showing the kernel magnitude scale.
2. A numbered grid with the kernel value written in each square.

This script uses SciPy's default Gaussian support directly:
radius = round(truncate * sigma), with default truncate=4.0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


DEFAULT_SIGMAS = (0.5, 1.0, 3.0, 5.0, 7.0)
DEFAULT_TRUNCATE = 4.0
DEFAULT_OUTPUT_DIR = Path("/Users/taeeonkong/Desktop/DL Project/smooth_boundary_gaussian_kernels")


def format_sigma_label(sigma: float) -> str:
    sigma_value = float(sigma)
    if sigma_value.is_integer():
        return str(int(sigma_value))
    return str(sigma_value).replace(".", "p")


def scipy_gaussian_kernel_2d(sigma: float, truncate: float) -> tuple[np.ndarray, int]:
    radius = int(round(float(truncate) * float(sigma)))
    size = 2 * radius + 1
    impulse = np.zeros((size, size), dtype=np.float64)
    impulse[radius, radius] = 1.0
    kernel = gaussian_filter(impulse, sigma=float(sigma), truncate=float(truncate))
    kernel /= kernel.sum()
    return kernel, radius


def save_heatmap(
    kernel: np.ndarray,
    sigma: float,
    radius: int,
    truncate: float,
    output_dir: Path,
) -> Path:
    sigma_label = format_sigma_label(sigma)
    size = kernel.shape[0]

    fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    im = ax.imshow(kernel, cmap="magma", interpolation="nearest")
    ax.set_title(f"smooth_boundary Gaussian kernel (sigma={sigma:g})")
    ax.set_xlabel(
        f"size={size}x{size}, radius={radius}, truncate={truncate:g}, normalized sum={kernel.sum():.6f}"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Kernel weight")

    output_path = output_dir / f"sigma_{sigma_label}_heatmap.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_numbered_grid(
    kernel: np.ndarray,
    sigma: float,
    radius: int,
    truncate: float,
    output_dir: Path,
) -> Path:
    sigma_label = format_sigma_label(sigma)
    size = kernel.shape[0]

    fig, ax = plt.subplots(figsize=(max(7.2, size * 0.72), max(6.2, size * 0.72)), constrained_layout=True)
    im = ax.imshow(kernel, cmap="magma", interpolation="nearest")
    ax.set_title(f"smooth_boundary Gaussian kernel values (sigma={sigma:g})")
    ax.set_xlabel(
        f"size={size}x{size}, radius={radius}, truncate={truncate:g}, normalized sum={kernel.sum():.6f}"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    for row in range(size):
        for col in range(size):
            value = kernel[row, col]
            ax.text(
                col,
                row,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "black",
                    "edgecolor": "black",
                    "linewidth": 0.0,
                    "alpha": 0.88,
                },
            )

    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Kernel weight")

    output_path = output_dir / f"sigma_{sigma_label}_numbered.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_kernel(kernel: np.ndarray, sigma: float, radius: int) -> None:
    print("=" * 80)
    print(f"sigma={sigma:g}, radius={radius}, size={kernel.shape[0]}x{kernel.shape[1]}")
    print(f"sum={kernel.sum():.8f}")
    print(np.array2string(kernel, precision=6, suppress_small=False))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print and visualize Gaussian kernels for smooth_boundary.")
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=list(DEFAULT_SIGMAS),
        help="Sigma values to visualize",
    )
    parser.add_argument(
        "--truncate",
        type=float,
        default=DEFAULT_TRUNCATE,
        help="SciPy truncate value used by gaussian_filter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures will be saved",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving kernel visualizations to: {output_dir}")
    for sigma in args.sigmas:
        kernel, radius = scipy_gaussian_kernel_2d(sigma=float(sigma), truncate=float(args.truncate))
        print_kernel(kernel=kernel, sigma=float(sigma), radius=radius)
        heatmap_path = save_heatmap(
            kernel=kernel,
            sigma=float(sigma),
            radius=radius,
            truncate=float(args.truncate),
            output_dir=output_dir,
        )
        numbered_path = save_numbered_grid(
            kernel=kernel,
            sigma=float(sigma),
            radius=radius,
            truncate=float(args.truncate),
            output_dir=output_dir,
        )
        print(f"Saved: {heatmap_path.name}")
        print(f"Saved: {numbered_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
