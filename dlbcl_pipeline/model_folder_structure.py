"""Discover and model a donor folder's raw image structure."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from dlbcl_pipeline.config import PipelineConfig
from dlbcl_pipeline.data_types import (
    ImageFolder,
    DonorFolderStructure,
    SampleFolder,
)
from dlbcl_pipeline.utils.config_helpers import (
    extract_sample_number,
    filter_image_folders,
    normalize_image_filter_config,
    selected_sample_numbers,
)
from dlbcl_pipeline.utils.folder_resolution import resolve_channel_filenames
from dlbcl_pipeline.utils.name_builder import extract_image_number

def image_sort_key(name: str) -> tuple[int, int | str]:
    if str(name).isdigit():
        return (0, int(name))
    return (1, str(name))


def is_donor_dir(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return False
    return any(
        child.is_dir() and child.name.lower().startswith("sample")
        for child in path.iterdir()
    )


def find_donor_dirs(root: str | Path) -> list[Path]:
    """Find donor folders below a root or response-group directory."""
    root = Path(root)
    if not root.exists():
        return []

    candidates = [
        item for item in root.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    ]

    direct_donors = [item for item in candidates if is_donor_dir(item)]
    if direct_donors:
        return sorted(direct_donors, key=lambda p: p.name.lower())

    donor_dirs: list[Path] = []
    for response_dir in sorted(candidates, key=lambda p: p.name.lower()):
        if not response_dir.is_dir():
            continue
        for donor in sorted(response_dir.iterdir(), key=lambda p: p.name.lower()):
            if is_donor_dir(donor):
                donor_dirs.append(donor)
    return donor_dirs


def find_sample_dirs(donor_dir: str | Path, samples_to_process=None) -> list[Path]:
    """Find sample folders under a donor folder, optionally filtered by number."""
    donor_dir = Path(donor_dir)
    if not donor_dir.exists():
        raise ValueError(
            f"Donor directory not found: {donor_dir}\n"
            f"  Check BASE_PATH — see model_folder_structure.py"
        )

    all_sample_dirs = sorted(
        [item for item in donor_dir.iterdir()
         if item.is_dir() and item.name.lower().startswith("sample")],
        key=lambda p: extract_sample_number(p.name),
    )

    if not all_sample_dirs:
        raise ValueError(
            f"No sample folders found in: {donor_dir}\n"
            f"  Looked for subfolders whose name starts with 'sample'.\n"
            f"  Found subfolders: {[d.name for d in donor_dir.iterdir() if d.is_dir()] or 'none'}\n"
            f"  Check folder naming — see model_folder_structure.py"
        )

    selected = selected_sample_numbers(samples_to_process)
    if selected is None:
        return all_sample_dirs

    filtered = [s for s in all_sample_dirs if extract_sample_number(s.name) in selected]
    if not filtered:
        raise ValueError(
            f"SAMPLES_TO_PROCESS filter matched no sample folders in: {donor_dir}\n"
            f"  Filter requested sample numbers: {sorted(selected)}\n"
            f"  Available sample folders: {[d.name for d in all_sample_dirs]}\n"
            f"  Adjust SAMPLES_TO_PROCESS — see model_folder_structure.py"
        )
    return filtered


def find_image_dirs(
    sample_dir: str | Path,
    images_to_process=None,
    announce_filters: bool = False,
) -> list[Path]:
    """Find image folders under a sample folder, optionally filtered by image number."""
    sample_dir = Path(sample_dir)
    if not sample_dir.exists():
        raise ValueError(
            f"Sample directory not found: {sample_dir}\n"
            f"  See model_folder_structure.py"
        )

    all_image_names = sorted(
        [item.name for item in sample_dir.iterdir() if item.is_dir()],
        key=image_sort_key,
    )

    if not all_image_names:
        raise ValueError(
            f"No image subfolders found in: {sample_dir}\n"
            f"  Check the sample folder contains numbered image directories\n"
            f"  — see model_folder_structure.py"
        )

    filters_map, filters_default = normalize_image_filter_config(images_to_process)
    filtered_names = filter_image_folders(
        sample_dir.name,
        all_image_names,
        filters_map,
        filters_default,
        announce=announce_filters,
    )

    if not filtered_names:
        raise ValueError(
            f"IMAGES_TO_PROCESS filter matched no image folders in: {sample_dir}\n"
            f"  Available image folders: {all_image_names}\n"
            f"  Configured filter: {images_to_process}\n"
            f"  Check that the image numbers match actual folder names\n"
            f"  — see model_folder_structure.py"
        )

    return [sample_dir / name for name in filtered_names]


def resolve_channel_paths(
    image_dir: str | Path,
    channel_config: Mapping[str, str],
    job_label: str | None = None,
) -> dict[str, Path]:
    """Require configured channel filenames to exist for one image."""
    image_dir = Path(image_dir)
    resolved_names = resolve_channel_filenames(
        image_dir=image_dir,
        configured_map=dict(channel_config),
        job_label=job_label,
    )
    return {channel: image_dir / filename for channel, filename in resolved_names.items()}


def build_donor_folder_structure(
    config: PipelineConfig,
    resolve_channels: bool = True,
    announce_filters: bool = False,
) -> DonorFolderStructure:
    """Model the configured donor folder once, including samples, images, and channels."""
    donor_dir = Path(config.donor_dir)
    samples: list[SampleFolder] = []

    for sample_dir in find_sample_dirs(donor_dir, config.samples_to_process):
        images: list[ImageFolder] = []
        for image_dir in find_image_dirs(
            sample_dir,
            config.images_to_process,
            announce_filters=announce_filters,
        ):
            job_label = f"{sample_dir.name}/{image_dir.name}"
            channels = (
                resolve_channel_paths(
                    image_dir,
                    config.channels,
                    job_label=job_label,
                )
                if resolve_channels
                else {
                    channel: image_dir / filename
                    for channel, filename in config.channels.items()
                }
            )
            images.append(
                ImageFolder(
                    name=image_dir.name,
                    path=image_dir,
                    image_number=extract_image_number(image_dir.name),
                    channels=channels,
                )
            )

        samples.append(
            SampleFolder(
                name=sample_dir.name,
                path=sample_dir,
                sample_number=extract_sample_number(sample_dir.name),
                images=tuple(images),
            )
        )

    return DonorFolderStructure(
        path=donor_dir,
        full_name=donor_dir.name,
        samples=tuple(samples),
    )
