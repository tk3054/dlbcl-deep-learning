"""Shared dataclasses returned by pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ImageFolder:
    name: str
    path: Path
    image_number: int | None = None
    channels: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "channels",
            {channel: Path(path) for channel, path in dict(self.channels).items()},
        )


@dataclass(frozen=True)
class SampleFolder:
    name: str
    path: Path
    sample_number: int | None = None
    images: tuple[ImageFolder, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "images", tuple(self.images))


@dataclass(frozen=True)
class DonorFolderStructure:
    path: Path
    full_name: str
    samples: tuple[SampleFolder, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "samples", tuple(self.samples))


@dataclass(frozen=True)
class StepResult:
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class SegmentationResult(StepResult):
    num_cells: int = 0
    roi_dir: Path | None = None
    visualization_path: Path | None = None


@dataclass(frozen=True)
class MeasurementResult(StepResult):
    measurement_csvs: Mapping[str, Path] = field(default_factory=dict)
    combined_csv: Path | None = None
    rows: int = 0


@dataclass(frozen=True)
class TableResult(StepResult):
    output_path: Path | None = None
    rows: int = 0
    columns: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ClassificationResult(StepResult):
    num_cells: int = 0
    num_cd4_pos: int = 0
    num_cd8_pos: int = 0
    num_car_pos: int = 0
    output_path: Path | None = None


@dataclass(frozen=True)
class ImageProcessingResult(StepResult):
    sample_name: str | None = None
    image_name: str | None = None
    results: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DonorProcessingResult(StepResult):
    donor_dir: Path | None = None
    total_images: int = 0
    total_processed: int = 0
    total_failed: int = 0
    failed_images: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    image_results: Sequence[ImageProcessingResult] = field(default_factory=tuple)
    donor_table: TableResult | None = None
    classification: ClassificationResult | None = None
