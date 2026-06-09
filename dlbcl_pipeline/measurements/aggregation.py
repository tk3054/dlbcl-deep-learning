"""Clear table-building API for image, sample, donor, and group measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from dlbcl_pipeline.measurements.channels import combine_measurements
from dlbcl_pipeline.measurements.images import combine_sample
from dlbcl_pipeline.measurements.samples import combine_donor_measurements

from dlbcl_pipeline import constants
from dlbcl_pipeline.config import PipelineConfig
from dlbcl_pipeline.data_types import ImageFolder, SampleFolder, TableResult


def _columns_from_csv(path: Path | None) -> tuple[str, ...]:
    if path is None or not path.exists():
        return ()
    return tuple(pd.read_csv(path, nrows=0).columns)


def _table_result_from_dict(
    result: Mapping[str, object],
    output_key: str,
    rows_key: str,
) -> TableResult:
    if not result.get("success"):
        return TableResult(
            success=False,
            error=str(result.get("error") or "Table build failed"),
        )

    if result.get("skipped"):
        return TableResult(
            success=False,
            error=str(result.get("error") or "No input CSVs found"),
        )

    output_value = result.get(output_key)
    output_path = Path(output_value) if output_value else None
    return TableResult(
        success=True,
        output_path=output_path,
        rows=int(result.get(rows_key) or 0),
        columns=_columns_from_csv(output_path),
    )


def _apply_table_export_config(config: PipelineConfig, output_path: Path, verbose: bool) -> tuple[int, tuple[str, ...]]:
    df = pd.read_csv(output_path)
    selected_columns = config.table_export.selected_columns(df.columns, config.channels)
    if selected_columns:
        df = df.loc[:, list(selected_columns)]
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"Applied table export config: kept {len(selected_columns)} columns")
    return len(df), tuple(df.columns)


def build_image_table(
    sample: SampleFolder,
    image: ImageFolder,
    config: PipelineConfig,
    channel_config: Mapping[str, str] | None = None,
    include_channels: Sequence[str] | None = None,
    null_channels: Sequence[str] | None = None,
    verbose: bool = True,
) -> TableResult:
    """Build one image-level `combined_measurements.csv` table."""
    channels = (
        dict(channel_config)
        if channel_config is not None
        else {channel: path.name for channel, path in image.channels.items()}
    )
    result = combine_measurements(
        sample_folder=sample.name,
        image_number=image.name,
        base_path=str(config.donor_dir),
        channel_config=channels,
        include_channels=include_channels,
        null_channels=null_channels,
        verbose=verbose,
    )
    return _table_result_from_dict(result, output_key="output_csv", rows_key="num_cells")


def build_sample_table(
    sample_dir: str | Path,
    images_to_process=None,
    verbose: bool = True,
) -> TableResult:
    """Build one sample-level `combined_measurements.csv` table."""
    sample_dir = Path(sample_dir)
    result = combine_sample(
        sample_name=sample_dir.name,
        base_path=str(sample_dir.parent),
        images_to_process=images_to_process,
        verbose=verbose,
    )
    return _table_result_from_dict(result, output_key="output_path", rows_key="num_cells")


def build_donor_table(
    config: PipelineConfig,
    output_file: str = constants.DONOR_COMBINED_CSV,
    verbose: bool = True,
) -> TableResult:
    """Build one donor-level `all_samples_combined.csv` table."""
    result = combine_donor_measurements(
        base_path=str(config.donor_dir),
        channel_config=config.channels,
        samples_to_process=config.samples_to_process,
        images_to_process=config.images_to_process,
        output_file=output_file,
        verbose=verbose,
    )
    table_result = _table_result_from_dict(result, output_key="output_path", rows_key="num_cells")
    if not table_result.success or table_result.output_path is None:
        return table_result

    rows, columns = _apply_table_export_config(
        config=config,
        output_path=table_result.output_path,
        verbose=verbose,
    )
    return TableResult(
        success=True,
        output_path=table_result.output_path,
        rows=rows,
        columns=columns,
    )


def build_group_table(
    group_dir: str | Path,
    output_file: str = constants.GROUP_COMBINED_CSV,
    prefer_classified: bool = True,
    verbose: bool = True,
) -> TableResult:
    """Build one response-group table by combining donor-level CSVs."""
    group_dir = Path(group_dir)
    if not group_dir.exists():
        return TableResult(success=False, error=f"Group directory not found: {group_dir}")

    donor_tables = []
    for donor_dir in sorted(group_dir.iterdir(), key=lambda path: path.name.lower()):
        if not donor_dir.is_dir() or donor_dir.name.startswith("."):
            continue

        candidates = []
        if prefer_classified:
            candidates.append(donor_dir / constants.DONOR_CLASSIFIED_CSV)
        candidates.append(donor_dir / constants.DONOR_COMBINED_CSV)

        csv_path = next((path for path in candidates if path.exists()), None)
        if csv_path is None:
            if verbose:
                print(f"Warning: No donor CSV found in {donor_dir}")
            continue

        if verbose:
            print(f"Reading: {csv_path}")
        df = pd.read_csv(csv_path)
        df.insert(0, "donor_id", donor_dir.name)
        donor_tables.append(df)

    if not donor_tables:
        return TableResult(success=False, error="No donor CSV files found")

    combined_df = pd.concat(donor_tables, ignore_index=True)
    output_path = group_dir / output_file
    combined_df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Total rows: {len(combined_df)}")

    return TableResult(
        success=True,
        output_path=output_path,
        rows=len(combined_df),
        columns=tuple(combined_df.columns),
    )
