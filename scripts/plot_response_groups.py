#!/usr/bin/env python3
"""Thin CLI for response-group table building and comparison plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlbcl_pipeline.measurements.aggregation import build_group_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build response-group tables and comparison plots.")
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Project directory containing responder/ and non-responder/ folders",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["responder", "non-responder"],
        help="Response group folder names to combine",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Only build group CSVs")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir)

    failed = False
    for group in args.groups:
        result = build_group_table(base_dir / group, verbose=not args.quiet)
        if not result.success:
            failed = True
            print(f"Error building {group}: {result.error}")

    if failed:
        return 1

    if not args.skip_plots:
        from dlbcl_pipeline.plotting.response_groups import (
            plot_all_comparison,
            plot_combined_comparison,
        )

        plot_all_comparison(base_dir, verbose=not args.quiet)
        plot_combined_comparison(base_dir, verbose=not args.quiet)

    return 0


if __name__ == "__main__":
    sys.exit(main())
