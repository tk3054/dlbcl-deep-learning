#!/usr/bin/env python3
"""Thin CLI for rebuilding one donor-level measurement table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlbcl_pipeline.config import PipelineConfig
from dlbcl_pipeline.measurements.aggregation import build_donor_table
from scripts.defaults import parse_channel_overrides, parse_image_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine sample/image CSVs for one donor folder.")
    parser.add_argument("--base-path", required=True, help="Donor directory containing sample folders")
    parser.add_argument("--samples", nargs="*", type=int, help="Sample numbers to include")
    parser.add_argument("--images", nargs="*", type=int, help="Image numbers to include for all samples")
    parser.add_argument("--output-file", default="all_samples_combined.csv", help="Output CSV filename")
    parser.add_argument(
        "--channel",
        action="append",
        help="Override a channel filename as key=filename. Can be repeated.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig.local(
        donor_dir=Path(args.base_path),
        channels=parse_channel_overrides(args.channel),
        samples_to_process=args.samples,
        images_to_process=parse_image_filter(args.images),
    )
    result = build_donor_table(config, output_file=args.output_file, verbose=not args.quiet)
    if not result.success:
        print(f"Error: {result.error}")
        return 1
    print(f"Wrote {result.output_path} ({result.rows} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
