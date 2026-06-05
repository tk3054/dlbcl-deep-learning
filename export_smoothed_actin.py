#!/usr/bin/env python3
"""
Export selected smoothed Actin crops into patient-level folders.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.image_iterator import resolve_image_folder
from utils.name_builder import (
    build_channel_filename,
    format_cell_classification,
    format_sigma_label,
)


DEFAULT_ROOT = Path("/mnt/HDD16TB/LanceKam_Lab/Daizong/Project/DLBCL/DLBCL/DLBCL_processed")
DEFAULT_CLEAN_CSV = "clean_cell_list.csv"
DEFAULT_CLASSIFIED_CSV = "all_samples_combined_classified.csv"
DEFAULT_SMOOTHING_DIR = "visualize smoothing"
DEFAULT_SIGMAS = (0.7, 7.0)


@dataclass(frozen=True)
class CellKey:
    patient_id: str
    sample: int
    image: int
    cell: int


def is_truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_unique_id(unique_id: str) -> CellKey | None:
    parts = str(unique_id or "").strip().split("_")
    if len(parts) != 4:
        return None
    patient_id, sample_s, image_s, cell_s = parts
    if not (patient_id and sample_s.isdigit() and image_s.isdigit() and cell_s.isdigit()):
        return None
    return CellKey(
        patient_id=patient_id,
        sample=int(sample_s),
        image=int(image_s),
        cell=int(cell_s),
    )


def discover_patient_dirs(root_dir: Path) -> list[Path]:
    return sorted(
        [
            patient_dir
            for patient_dir in root_dir.iterdir()
            if patient_dir.is_dir() and any(
                child.is_dir() and child.name.lower().startswith("sample")
                for child in patient_dir.iterdir()
            )
        ],
        key=lambda path: path.name.lower(),
    )


def detect_clean_flag_column(fieldnames: list[str]) -> str | None:
    for candidate in ("clean", "clean_cell", "selected", "keep"):
        if candidate in fieldnames:
            return candidate
    return None


def load_clean_cells(clean_csv_path: Path) -> dict[str, set[CellKey]]:
    if not clean_csv_path.exists():
        raise FileNotFoundError(f"Clean CSV not found: {clean_csv_path}")

    selected: dict[str, set[CellKey]] = defaultdict(set)
    with clean_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if "unique_id" not in fieldnames:
            raise ValueError(f"{clean_csv_path} missing 'unique_id' column")
        clean_flag_col = detect_clean_flag_column(fieldnames)

        for row in reader:
            key = parse_unique_id(row.get("unique_id", ""))
            if key is None:
                continue
            if clean_flag_col and not is_truthy(row.get(clean_flag_col, "")):
                continue
            selected[key.patient_id].add(key)
    return selected


def resolve_clean_csv_path(root_dir: Path, clean_csv_arg: Path) -> Path:
    if clean_csv_arg.is_absolute():
        return clean_csv_arg

    candidates = [
        root_dir / clean_csv_arg,
        REPO_ROOT / clean_csv_arg,
        root_dir.parent / clean_csv_arg,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return root_dir / clean_csv_arg


def load_classified_rows(classified_csv_path: Path) -> dict[CellKey, dict[str, str]]:
    if not classified_csv_path.exists():
        raise FileNotFoundError(f"Classified CSV not found: {classified_csv_path}")

    rows: dict[CellKey, dict[str, str]] = {}
    with classified_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"unique_id", "sample", "image", "cell_id", "cell_type", "tcell_subset"}
        if not required.issubset(fieldnames):
            missing = sorted(required - fieldnames)
            raise ValueError(f"{classified_csv_path} missing required columns: {missing}")

        for row in reader:
            key = parse_unique_id(row.get("unique_id", ""))
            if key is None:
                continue
            rows[key] = row
    return rows


def resolve_smoothed_source(patient_dir: Path, key: CellKey, smoothing_dir_name: str, sigma: float) -> Path | None:
    sample_dir = patient_dir / f"sample{key.sample}"
    image_folder = resolve_image_folder(sample_dir, str(key.image))
    if image_folder is None:
        return None
    sigma_label = format_sigma_label(sigma).removeprefix("sigma")
    return (
        sample_dir
        / image_folder
        / smoothing_dir_name
        / f"cell_{key.cell}"
        / f"sigma_{sigma_label}.tif"
    )


def build_destination_name(patient_dir: Path, row: dict[str, str], sigma: float) -> str:
    classification = format_cell_classification(row.get("cell_type", ""), row.get("tcell_subset", ""))
    patient_id = row.get("patient_id") or parse_unique_id(row.get("unique_id", "")).patient_id
    sigma_suffix = format_sigma_label(sigma).replace("sigma", "smoothS", 1)
    base_name = build_channel_filename(
        response=row.get("response", patient_dir.parent.name),
        patient_folder_name=patient_dir.name,
        patient_id=patient_id,
        sample=row.get("sample", ""),
        image=row.get("image", ""),
        cell_id=row.get("cell_id", ""),
        classification=classification,
        channel_suffix=f"actin_{sigma_suffix}",
    )
    return f"{base_name}.tif"


def export_patient(patient_dir: Path, clean_keys: set[CellKey], args: argparse.Namespace) -> tuple[int, int]:
    if not clean_keys:
        print(f"[{patient_dir.name}] no selected cells, skipped")
        return 0, 0

    classified_rows = load_classified_rows(patient_dir / args.classified_csv)
    copied = 0
    missing = 0

    for sigma in args.sigmas:
        sigma_dir = format_sigma_label(sigma).replace("sigma", "smoothed_actin_s", 1)
        (patient_dir / sigma_dir).mkdir(exist_ok=True)

    for key in sorted(clean_keys, key=lambda item: (item.sample, item.image, item.cell)):
        row = classified_rows.get(key)
        if row is None:
            missing += len(args.sigmas)
            print(f"[{patient_dir.name}] missing classified row for {key.patient_id}_{key.sample}_{key.image:02d}_{key.cell:02d}")
            continue

        for sigma in args.sigmas:
            source_path = resolve_smoothed_source(
                patient_dir=patient_dir,
                key=key,
                smoothing_dir_name=args.smoothing_dir,
                sigma=sigma,
            )
            if source_path is None or not source_path.exists():
                missing += 1
                sigma_label = format_sigma_label(sigma).removeprefix("sigma")
                print(
                    f"[{patient_dir.name}] missing sigma_{sigma_label}.tif for "
                    f"sample{key.sample}/image{key.image:02d}/cell{key.cell:02d}"
                )
                continue

            dest_dir = patient_dir / format_sigma_label(sigma).replace("sigma", "smoothed_actin_s", 1)
            dest_name = build_destination_name(patient_dir, row, sigma)
            shutil.copy2(source_path, dest_dir / dest_name)
            copied += 1

    return copied, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected smoothed Actin crops into patient-level folders.")
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT, help="DLBCL_processed root directory")
    parser.add_argument("--clean-csv", type=Path, default=Path(DEFAULT_CLEAN_CSV), help="CSV with selected unique_id rows")
    parser.add_argument("--classified-csv", default=DEFAULT_CLASSIFIED_CSV, help="Per-patient classified CSV filename")
    parser.add_argument("--smoothing-dir", default=DEFAULT_SMOOTHING_DIR, help="Per-image smoothing output directory")
    parser.add_argument("--sigmas", nargs="+", type=float, default=DEFAULT_SIGMAS, help="Sigma values to export")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    clean_csv_path = resolve_clean_csv_path(root_dir, args.clean_csv)

    patient_dirs = discover_patient_dirs(root_dir)
    if not patient_dirs:
        raise FileNotFoundError(f"No patient folders found in {root_dir}")

    selected_by_patient = load_clean_cells(clean_csv_path)

    total_copied = 0
    total_missing = 0
    print("Patients discovered:")
    for patient_dir in patient_dirs:
        print(f"  - {patient_dir.name}")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name.split()[-1]
        copied, missing = export_patient(
            patient_dir=patient_dir,
            clean_keys=selected_by_patient.get(patient_id, set()),
            args=args,
        )
        total_copied += copied
        total_missing += missing
        print(f"[{patient_dir.name}] copied={copied}, missing={missing}")

    print(f"Done. copied={total_copied}, missing={total_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
