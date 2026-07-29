#!/usr/bin/env python3
"""Import smoke checks for the current local pipeline entry points."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path


MODULES = (
    "dlbcl_pipeline.config",
    "dlbcl_pipeline.model_folder_structure",
    "dlbcl_pipeline.measurements.aggregation",
    "dlbcl_pipeline.build_paths",
    "dlbcl_pipeline.process_donor",
    "dlbcl_pipeline.data_types",
    "dlbcl_pipeline.process_image",
    "dlbcl_pipeline.segmentation.imagej_config",
    "main_remote",
    "dlbcl_pipeline.imagej.functions",
    "dlbcl_pipeline.image_processing.crops",
    "dlbcl_pipeline.segmentation.cellpose",
    "dlbcl_pipeline.measurements.channels",
    "dlbcl_pipeline.measurements.images",
    "dlbcl_pipeline.measurements.samples",
    "dlbcl_pipeline.classification.classify_tcells",
    "dlbcl_pipeline.export.formatted_channels",
    "dlbcl_pipeline.plotting.intensity_distributions",
    "dlbcl_pipeline.plotting.response_groups",
    "main",
    "scripts.run_donor",
    "scripts.combine_donor",
    "scripts.plot_response_groups",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    with tempfile.TemporaryDirectory(prefix="dlbcl-matplotlib-") as cache_dir:
        os.environ.setdefault("MPLCONFIGDIR", cache_dir)

        failures: list[tuple[str, Exception]] = []
        for module_name in MODULES:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - CLI diagnostic
                failures.append((module_name, exc))
                print(f"FAIL {module_name}: {exc}")
            else:
                print(f"OK   {module_name}")

    if failures:
        print(f"\n{len(failures)} import check(s) failed.")
        return 1

    print("\nAll import checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
