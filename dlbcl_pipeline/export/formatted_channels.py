#!/usr/bin/env python3
"""Build formatted per-channel image folders from a clean cell list."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dlbcl_pipeline import constants
from dlbcl_pipeline.model_folder_structure import find_donor_dirs, find_image_dirs, find_sample_dirs
from dlbcl_pipeline.utils.config_helpers import (
    extract_sample_number,
    normalize_image_filter_config,
    selected_sample_numbers,
)
from dlbcl_pipeline.utils.name_builder import extract_donor_id_from_path, extract_image_number

UID_PATTERN = re.compile(r"^(\d+)_([0-9]+)_([0-9]+)_([0-9]+)$")
CELL_FILE_PATTERN = re.compile(r"^cell_(\d+)_padded\.[^.]+$")

CHANNEL_PRESETS = {
    "actin": {
        "source_dirname": "padded_cells",
        "output_dirname": "formatted_actin",
        "suffix": "actin",
    },
    "ccr7": {
        "source_dirname": "padded_ccr7",
        "output_dirname": "formatted_ccr7",
        "suffix": "ccr7",
    },
    "cd45ra": {
        "source_dirname": "padded_cd45ra",
        "output_dirname": "formatted_cd45ra",
        "suffix": "cd45ra",
    },
}


@dataclass(frozen=True)
class CellKey:
    sample: int
    image: int
    cell: int


@dataclass(frozen=True)
class CleanEntry:
    key: CellKey
    target_name: str


@dataclass(frozen=True)
class ExcludedEntry:
    unique_id: str
    key: CellKey


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_unique_id(uid: str) -> tuple[str, CellKey] | None:
    """Parse donor_sample_image_cell IDs such as 118867_1_1_02."""
    match = UID_PATTERN.match(uid)
    if not match:
        return None
    donor_id, sample_s, image_s, cell_s = match.groups()
    return donor_id, CellKey(sample=int(sample_s), image=int(image_s), cell=int(cell_s))


def load_entries(
    clean_csv: Path,
    clean_col: str,
    clean_flag_col: str,
    name_col: str,
    donor_ids: Iterable[str],
    samples_to_process=None,
    images_to_process=None,
) -> tuple[dict[str, list[CleanEntry]], dict[str, list[ExcludedEntry]]]:
    """Load clean and excluded cell rows keyed by donor ID."""
    donor_set = set(donor_ids)
    selected_samples = selected_sample_numbers(samples_to_process)
    filters_map, filters_default = normalize_image_filter_config(images_to_process)
    clean_entries: dict[str, list[CleanEntry]] = defaultdict(list)
    excluded_entries: dict[str, list[ExcludedEntry]] = defaultdict(list)

    with clean_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {clean_col, clean_flag_col, name_col}
        if not required.issubset(fields):
            raise ValueError(
                f"{clean_csv} missing required columns {sorted(required)}; found: {sorted(fields)}"
            )

        for row in reader:
            uid = (row.get(clean_col) or "").strip()
            if not uid:
                continue

            parsed = parse_unique_id(uid)
            if not parsed:
                continue
            donor_id, key = parsed
            if donor_id not in donor_set:
                continue
            if not _cell_key_selected(key, selected_samples, filters_map, filters_default):
                continue

            if _is_truthy(row.get(clean_flag_col) or ""):
                name = (row.get(name_col) or "").strip()
                if name:
                    clean_entries[donor_id].append(CleanEntry(key=key, target_name=name))
            else:
                excluded_entries[donor_id].append(ExcludedEntry(unique_id=uid, key=key))

    return clean_entries, excluded_entries


def _cell_key_selected(
    key: CellKey,
    selected_samples: set[int] | None,
    filters_map: dict[int, list[str]],
    filters_default: list[str] | None,
) -> bool:
    if selected_samples and key.sample not in selected_samples:
        return False

    allowed = filters_map.get(key.sample)
    if allowed is None:
        allowed = filters_default
    if allowed is None or len(allowed) == 0:
        return True

    return str(key.image) in {str(value) for value in allowed}


def discover_donor_roots(root: Path, donor_folder_path: Path, single_donor_mode: bool) -> dict[str, list[Path]]:
    """Discover donor roots and group them by donor ID."""
    out: dict[str, list[Path]] = defaultdict(list)
    candidate = donor_folder_path if donor_folder_path.is_absolute() else root / donor_folder_path
    candidate = candidate.resolve()

    if not candidate.exists():
        return out

    donor_dirs = [candidate] if single_donor_mode else find_donor_dirs(candidate)
    for donor_dir in donor_dirs:
        donor_id = extract_donor_id_from_path(donor_dir)
        if donor_id:
            out[donor_id].append(donor_dir)

    return out


def build_source_index(
    donor_root: Path,
    source_dirname: str,
    samples_to_process=None,
    images_to_process=None,
) -> dict[CellKey, Path]:
    """Index padded channel files under one donor folder by sample/image/cell."""
    index: dict[CellKey, Path] = {}
    selected_samples = selected_sample_numbers(samples_to_process)
    filters_map, filters_default = normalize_image_filter_config(images_to_process)

    for sample_dir in find_sample_dirs(donor_root):
        sample_num = extract_sample_number(sample_dir.name)
        if sample_num <= 0:
            continue
        if selected_samples and sample_num not in selected_samples:
            continue

        for image_dir in find_image_dirs(sample_dir):
            image_num = extract_image_number(image_dir.name)
            if image_num is None:
                continue
            if not _cell_key_selected(
                CellKey(sample=sample_num, image=int(image_num), cell=0),
                selected_samples,
                filters_map,
                filters_default,
            ):
                continue

            source_dir = image_dir / source_dirname
            if not source_dir.exists():
                continue

            for path in source_dir.iterdir():
                if not path.is_file():
                    continue
                match = CELL_FILE_PATTERN.match(path.name)
                if not match:
                    continue
                key = CellKey(sample=sample_num, image=int(image_num), cell=int(match.group(1)))
                index[key] = path

    return index


def _clean_output_dir(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def _write_excluded_report(path: Path, rows: list[ExcludedEntry]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unique_id", "sample", "image", "cell"])
        for item in rows:
            writer.writerow([item.unique_id, item.key.sample, item.key.image, item.key.cell])


def build_formatted_channels(
    root: Path,
    donor_folder_path: Path,
    clean_csv: Path,
    single_donor_mode: bool = True,
    clean_col: str = "unique_id",
    clean_flag_col: str = "clean",
    name_col: str = "name",
    channels: Iterable[str] = ("actin", "ccr7", "cd45ra"),
    samples_to_process=None,
    images_to_process=None,
    dry_run: bool = False,
    skip_existing: bool = False,
    clean_output: bool = False,
    classified_csv: str = constants.DONOR_CLASSIFIED_CSV,
    excluded_report_name: str = "excluded_cells.csv",
    verbose: bool = True,
) -> dict[str, object]:
    """Copy selected padded channel images into formatted channel folders."""
    root = root.resolve()
    clean_csv = clean_csv if clean_csv.is_absolute() else root / clean_csv
    clean_csv = clean_csv.resolve()

    if not clean_csv.exists():
        raise FileNotFoundError(f"Clean CSV not found: {clean_csv}")
    if skip_existing and clean_output:
        raise ValueError("Use only one of skip_existing or clean_output")

    requested_channels = tuple(channels)
    unknown = sorted(set(requested_channels) - set(CHANNEL_PRESETS))
    if unknown:
        raise ValueError(f"Unknown channels: {unknown}; expected one of {sorted(CHANNEL_PRESETS)}")

    donor_roots = discover_donor_roots(root, donor_folder_path, single_donor_mode)
    donor_ids = sorted(donor_roots)
    if not donor_ids:
        return {
            "success": False,
            "error": f"No donor folders found from {donor_folder_path}",
            "donor_count": 0,
            "channel_results": [],
        }

    clean_entries, excluded_entries = load_entries(
        clean_csv=clean_csv,
        clean_col=clean_col,
        clean_flag_col=clean_flag_col,
        name_col=name_col,
        donor_ids=donor_ids,
        samples_to_process=samples_to_process,
        images_to_process=images_to_process,
    )

    channel_results = []
    for donor_id in donor_ids:
        for donor_root in donor_roots[donor_id]:
            classified_path = donor_root / classified_csv
            if not classified_path.exists():
                raise FileNotFoundError(
                    f"Classified donor CSV not found: {classified_path}\n"
                    "Run T-cell classification before formatted channel export."
                )

        entries = clean_entries.get(donor_id, [])
        if not entries:
            if verbose:
                print(f"[{donor_id}] no clean entries in {clean_csv.name}, skipped")
            continue

        for donor_root in donor_roots[donor_id]:
            excluded_report = donor_root / excluded_report_name

            for channel in requested_channels:
                preset = CHANNEL_PRESETS[channel]
                source_dirname = preset["source_dirname"]
                output_dirname = preset["output_dirname"]
                suffix = preset["suffix"]

                source_index = build_source_index(
                    donor_root,
                    source_dirname,
                    samples_to_process=samples_to_process,
                    images_to_process=images_to_process,
                )
                output_dir = donor_root / output_dirname

                if not dry_run:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    if clean_output:
                        _clean_output_dir(output_dir)

                copied = 0
                missing = 0
                missing_entries = []
                duplicate_name_overwrites = 0
                skipped_existing = 0

                for entry in entries:
                    source = source_index.get(entry.key)
                    if source is None:
                        missing += 1
                        missing_entries.append(
                            {
                                "sample": entry.key.sample,
                                "image": entry.key.image,
                                "cell": entry.key.cell,
                                "target_name": entry.target_name,
                                "expected_source_dir": str(
                                    donor_root
                                    / f"sample{entry.key.sample}"
                                    / str(entry.key.image)
                                    / source_dirname
                                ),
                            }
                        )
                        continue

                    output_stem = f"{Path(entry.target_name).stem}_{suffix}"
                    output_path = output_dir / f"{output_stem}{source.suffix}"

                    if output_path.exists():
                        duplicate_name_overwrites += 1
                        if skip_existing:
                            skipped_existing += 1
                            continue

                    if not dry_run:
                        shutil.copy2(source, output_path)
                    copied += 1

                result = {
                    "donor_id": donor_id,
                    "donor_root": str(donor_root),
                    "channel": channel,
                    "source_dirname": source_dirname,
                    "output_dir": str(output_dir),
                    "copied": copied,
                    "missing": missing,
                    "missing_entries": missing_entries,
                    "indexed_sources": len(source_index),
                    "duplicate_name_overwrites": duplicate_name_overwrites,
                    "skipped_existing": skipped_existing,
                    "dry_run": dry_run,
                }
                channel_results.append(result)

                if verbose:
                    mode = "DRY-RUN" if dry_run else "DONE"
                    print(
                        f"[{donor_id}] [{channel}] {mode} {donor_root} -> {output_dir}: "
                        f"copied={copied}, missing={missing}, indexed_sources={len(source_index)}, "
                        f"duplicate_name_overwrites={duplicate_name_overwrites}, "
                        f"skipped_existing={skipped_existing}"
                    )

            excluded_for_donor = excluded_entries.get(donor_id, [])
            if not dry_run:
                _write_excluded_report(excluded_report, excluded_for_donor)
                if verbose:
                    print(f"[{donor_id}] wrote excluded report: {excluded_report} rows={len(excluded_for_donor)}")
            elif verbose:
                print(f"[{donor_id}] DRY-RUN excluded report: {excluded_report} rows={len(excluded_for_donor)}")

    return {
        "success": True,
        "donor_count": len(donor_ids),
        "channel_results": channel_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect clean cells from padded channel folders into formatted channel folders."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root containing responder/ and non-responder/")
    parser.add_argument(
        "--donor-folder-path",
        type=Path,
        required=True,
        help="Path to one donor folder, or a parent folder when --multi-donor is used",
    )
    parser.add_argument(
        "--multi-donor",
        action="store_true",
        help="Treat --donor-folder-path as a parent folder and discover donor folders below it",
    )
    parser.add_argument("--clean-csv", type=Path, default=Path("clean_cell_list.csv"))
    parser.add_argument("--clean-col", default="unique_id")
    parser.add_argument("--clean-flag-col", default="clean")
    parser.add_argument("--name-col", default="name")
    parser.add_argument("--channels", nargs="+", default=["actin", "ccr7", "cd45ra"], choices=sorted(CHANNEL_PRESETS))
    parser.add_argument("--samples", nargs="*", type=int, help="Sample numbers to export")
    parser.add_argument("--images", nargs="*", type=int, help="Image numbers to export for all selected samples")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--classified-csv", default=constants.DONOR_CLASSIFIED_CSV)
    parser.add_argument("--excluded-report-name", default="excluded_cells.csv")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_formatted_channels(
        root=args.root,
        donor_folder_path=args.donor_folder_path,
        clean_csv=args.clean_csv,
        single_donor_mode=not args.multi_donor,
        clean_col=args.clean_col,
        clean_flag_col=args.clean_flag_col,
        name_col=args.name_col,
        channels=args.channels,
        samples_to_process=args.samples,
        images_to_process=args.images,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        clean_output=args.clean_output,
        classified_csv=args.classified_csv,
        excluded_report_name=args.excluded_report_name,
        verbose=not args.quiet,
    )
    if not result["success"]:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
