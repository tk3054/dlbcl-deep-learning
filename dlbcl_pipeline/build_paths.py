"""Path construction helpers for pipeline inputs and outputs."""

from __future__ import annotations

from pathlib import Path

from . import constants


def donor_dir(base_dir: str | Path, response_group: str, donor_name: str) -> Path:
    return Path(base_dir) / response_group / donor_name


def sample_dir(donor_dir: str | Path, sample_name: str) -> Path:
    return Path(donor_dir) / sample_name


def image_dir(donor_dir: str | Path, sample_name: str, image_name: str) -> Path:
    return sample_dir(donor_dir, sample_name) / str(image_name)


def channel_file(image_dir: str | Path, filename: str) -> Path:
    return Path(image_dir) / filename


def image_combined_csv(image_dir: str | Path) -> Path:
    return Path(image_dir) / constants.IMAGE_COMBINED_CSV


def sample_combined_csv(sample_dir: str | Path) -> Path:
    return Path(sample_dir) / constants.IMAGE_COMBINED_CSV


def donor_combined_csv(donor_dir: str | Path) -> Path:
    return Path(donor_dir) / constants.DONOR_COMBINED_CSV


def donor_classified_csv(donor_dir: str | Path) -> Path:
    return Path(donor_dir) / constants.DONOR_CLASSIFIED_CSV


def group_combined_csv(group_dir: str | Path) -> Path:
    return Path(group_dir) / constants.GROUP_COMBINED_CSV


def cell_rois_dir(image_dir: str | Path) -> Path:
    return Path(image_dir) / constants.CELL_ROIS_DIR


def raw_actin_dir(image_dir: str | Path) -> Path:
    return Path(image_dir) / constants.RAW_CROPS_DIR


def raw_crops_dir(image_dir: str | Path) -> Path:
    return raw_actin_dir(image_dir)


def padded_cells_dir(image_dir: str | Path) -> Path:
    return Path(image_dir) / constants.PADDED_CELLS_DIR


def formatted_cells_dir(donor_dir: str | Path) -> Path:
    return Path(donor_dir) / constants.FORMATTED_CELLS_DIR


def segmentation_visualization(image_dir: str | Path) -> Path:
    return Path(image_dir) / constants.SEGMENTATION_VISUALIZATION


def cell_area_histogram(donor_dir: str | Path) -> Path:
    return Path(donor_dir) / constants.CELL_AREA_HISTOGRAM


def failed_images_log(donor_dir: str | Path) -> Path:
    return Path(donor_dir) / constants.FAILED_IMAGES_LOG
