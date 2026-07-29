"""Configuration dataclasses for pipeline runs.

These classes describe the shape of configuration shared by scripts, notebooks,
and future package-level pipeline functions. They intentionally avoid importing
ImageJ, Cellpose, or other heavy runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from . import constants

ImageFilter = set[int] | Mapping[int, Sequence[int]]
SampleFilter = Sequence[int] | Mapping[int, Sequence[int]] | None

DEFAULT_CHANNELS = {
    "actin": "Actin-FITC.tif",
    "cd4": "CD4-PerCP.tif",
    "cd45ra_PacBlue": "CD45RA-PacBlue.tif",
    "cd19car": "CD19CAR-AF647.tif",
    "ccr7": "CCR7-AF594.tif",
}

DEFAULT_INTENSITY_COLUMNS = ("mean", "median", "std", "min", "max", "intden", "rawintden")

DEFAULT_MASTER_REQUIRED_COLUMNS = (
    "global_cell_id",
    "unique_id",
    "sample",
    "image",
    "cell_id",
    "cd4_median",
    "cd4_mean",
    "ccr7_median",
    "ccr7_mean",
    "cd45ra_median",
    "cd45ra_mean",
    "cd19car_median",
    "cd19car_mean",
)

DEFAULT_MASTER_COLUMN_TOGGLES = {
    "area": False,
    "x": False,
    "y": False,
    "circ": False,
    "ar": False,
    "round": False,
    "solidity": False,
}

@dataclass(frozen=True)
class ClassificationConfig:
    """Thresholds and channel settings for T-cell subset classification.

    Set `enabled=False` to skip classification after donor table build.
    Thresholds are median fluorescence intensity cut-offs per marker.
    """

    enabled: bool = True
    thresholds: Mapping[str, float] = field(
        default_factory=lambda: {
            "cd4": 260.0,
            "cd45ra": 120.0,
            "ccr7": 114.0,
            "cd19car": 430.0,
        }
    )


_DYE_TOKENS = {
    "af647",
    "sparkviolet",
    "pacblue",
    "percp",
    "af594",
    "fitc",
}


def _measurement_prefix(channel_name: str) -> str:
    parts = str(channel_name).split("_")
    cleaned = [part for part in parts if part.lower() not in _DYE_TOKENS]
    return "_".join(cleaned) if cleaned else str(channel_name)


def _image_filter_from_sample_filter(samples_to_process: SampleFilter) -> dict[int, tuple[int, ...]]:
    if not isinstance(samples_to_process, Mapping):
        return {}

    filters: dict[int, tuple[int, ...]] = {}
    for sample, images in samples_to_process.items():
        try:
            sample_number = int(sample)
        except (TypeError, ValueError):
            continue

        if images is None:
            filters[sample_number] = ()
            continue

        if isinstance(images, (str, bytes)):
            image_values = [images]
        else:
            image_values = images

        normalized_images = []
        for image in image_values:
            try:
                normalized_images.append(int(image))
            except (TypeError, ValueError):
                continue
        filters[sample_number] = tuple(normalized_images)
    return filters


def _normalize_run_filters(
    samples_to_process: SampleFilter,
    images_to_process: ImageFilter | None,
) -> tuple[SampleFilter, ImageFilter]:
    image_filter_from_samples = _image_filter_from_sample_filter(samples_to_process)
    if image_filter_from_samples and not images_to_process:
        return samples_to_process, image_filter_from_samples
    return samples_to_process, images_to_process or set()


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime environment settings.

    `run_local` is a preset signal. Pipeline code should generally use the
    concrete settings below instead of scattering `if run_local` checks.
    """

    run_local: bool = True
    imagej_mode: str = "headless"
    fiji_path: Path | None = None
    copy_raw_to_processed: bool = False
    processed_base_name: str | None = None


@dataclass(frozen=True)
class CellposeConfig:
    model: str = "cyto2"
    diameter: int = 250
    flow_threshold: float = 0.6
    cellprob_threshold: float = -6.0
    use_gpu: bool = True
    min_size: int = 180
    max_size: int = 100000
    consecutive_threshold: int = 20

    def to_pipeline_params(self) -> dict[str, object]:
        """Return the dict shape expected by the image-level pipeline."""
        return {
            "cellpose_model": self.model,
            "cellpose_diameter": self.diameter,
            "cellpose_flow_threshold": self.flow_threshold,
            "cellpose_cellprob_threshold": self.cellprob_threshold,
            "cellpose_use_gpu": self.use_gpu,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "consecutive_threshold": self.consecutive_threshold,
        }


@dataclass(frozen=True)
class ExportConfig:
    enabled: bool = True
    pdms_stiffness: str = "1to10"
    classified_csv: str = constants.DONOR_CLASSIFIED_CSV
    source_dir: str = constants.RAW_CROPS_DIR
    output_dir: str = constants.FORMATTED_CELLS_DIR
    dilution: str = "1to10"
    name_order: tuple[str, ...] = (
        "response",
        "donor_id",
        "date",
        "stiffness",
        "sample",
        "image",
        "cell_label",
        "classification",
    )


@dataclass(frozen=True)
class TableExportConfig:
    """Column policy for donor-level measurement exports."""

    required_columns: tuple[str, ...] = DEFAULT_MASTER_REQUIRED_COLUMNS
    column_toggles: Mapping[str, bool] = field(
        default_factory=lambda: dict(DEFAULT_MASTER_COLUMN_TOGGLES)
    )
    intensity_columns: tuple[str, ...] = DEFAULT_INTENSITY_COLUMNS
    include_channel_intensities: bool = True
    default_keep: bool = False

    def selected_columns(
        self,
        available_columns: Sequence[str],
        channels: Mapping[str, str],
    ) -> tuple[str, ...]:
        """Return available columns in export order based on this policy."""
        available = tuple(available_columns)
        available_set = set(available)
        selected: list[str] = []

        def add(column: str) -> None:
            if column in available_set and column not in selected:
                selected.append(column)

        for column in self.required_columns:
            add(column)

        if self.include_channel_intensities:
            prefixes = []
            for channel in channels:
                for prefix in (str(channel), _measurement_prefix(str(channel))):
                    if prefix not in prefixes:
                        prefixes.append(prefix)
            for prefix in prefixes:
                for stat in self.intensity_columns:
                    add(f"{prefix}_{stat}")

        for column in available:
            if column in selected:
                continue
            keep = bool(self.column_toggles.get(column, self.default_keep))
            if keep:
                selected.append(column)

        return tuple(selected)


@dataclass(frozen=True)
class PipelineConfig:
    donor_dir: Path
    channels: Mapping[str, str]
    samples_to_process: SampleFilter = None
    images_to_process: ImageFilter = field(default_factory=set)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    cellpose: CellposeConfig = field(default_factory=CellposeConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    table_export: TableExportConfig = field(default_factory=TableExportConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "donor_dir", Path(self.donor_dir))
        object.__setattr__(self, "channels", dict(self.channels))
        samples, images = _normalize_run_filters(
            self.samples_to_process,
            self.images_to_process,
        )
        object.__setattr__(self, "samples_to_process", samples)
        object.__setattr__(self, "images_to_process", images)

    @classmethod
    def local(
        cls,
        donor_dir: str | Path,
        channels: Mapping[str, str],
        samples_to_process: SampleFilter = None,
        images_to_process: ImageFilter | None = None,
    ) -> "PipelineConfig":
        return cls(
            donor_dir=Path(donor_dir),
            channels=channels,
            samples_to_process=samples_to_process,
            images_to_process=images_to_process or set(),
            runtime=RuntimeConfig(run_local=True, imagej_mode="headless"),
        )

    @classmethod
    def server(
        cls,
        donor_dir: str | Path,
        channels: Mapping[str, str],
        samples_to_process: SampleFilter = None,
        images_to_process: ImageFilter | None = None,
        fiji_path: str | Path | None = None,
        processed_base_name: str = "DLBCL-edge",
    ) -> "PipelineConfig":
        return cls(
            donor_dir=Path(donor_dir),
            channels=channels,
            samples_to_process=samples_to_process,
            images_to_process=images_to_process or set(),
            runtime=RuntimeConfig(
                run_local=False,
                imagej_mode="headless",
                fiji_path=Path(fiji_path) if fiji_path else None,
                copy_raw_to_processed=True,
                processed_base_name=processed_base_name,
            ),
            export=ExportConfig(source_dir=constants.PADDED_CELLS_DIR),
        )


def build_local_pipeline_config(
    donor_dir: str | Path,
    samples_to_process: SampleFilter = None,
    images_to_process: ImageFilter | None = None,
    export_images: bool = True,
) -> PipelineConfig:
    """Build the standard local run config from the small set of run knobs."""
    config = PipelineConfig.local(
        donor_dir=donor_dir,
        channels=DEFAULT_CHANNELS,
        samples_to_process=samples_to_process,
        images_to_process=images_to_process,
    )
    return replace(config, export=ExportConfig(enabled=export_images))
