#!/usr/bin/env python3
"""Thin CLI for running one donor pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlbcl_pipeline.config import PipelineConfig
from dlbcl_pipeline.model_folder_structure import build_donor_folder_structure
from dlbcl_pipeline.process_donor import process_donor
from scripts.defaults import parse_channel_overrides, parse_image_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DLBCL pipeline for one donor folder.")
    parser.add_argument("--base-path", required=True, help="Donor directory containing sample folders")
    parser.add_argument("--samples", nargs="*", type=int, help="Sample numbers to process")
    parser.add_argument("--images", nargs="*", type=int, help="Image numbers to process for all samples")
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
    donor_folder_structure = build_donor_folder_structure(
        config,
        announce_filters=True,
    )
    result = process_donor(
        config,
        donor_folder_structure=donor_folder_structure,
        verbose=not args.quiet,
    )
    if not result.success:
        print(f"Error: {result.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
