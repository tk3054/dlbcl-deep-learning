#!/usr/bin/env python3
"""Build per-patient formatted CD45RA image folders from clean cell list.

Workflow:
1. Read clean IDs + names from a CSV (default: clean_cell_list.csv).
2. For each patient root (e.g., */DLBCL 109241), index all source images under
   sample*/<image>/<source_dirname> exactly once.
3. Copy only clean cells into <patient_root>/<output_dirname>, renaming each file
   to the provided name while preserving source extension.

This avoids expensive repeated scans by using a dictionary lookup for
(sample, image, cell) -> source file.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

UID_PATTERN = re.compile(r"^(\d+)_([0-9]+)_([0-9]+)_([0-9]+)$")
CELL_FILE_PATTERN = re.compile(r"^cell_(\d+)_padded\.[^.]+$")

# Set this to either:
# 1) A single patient folder, e.g. "non-responder/01-03-2026 DLBCL 109241"
# 2) A parent folder containing many patient folders, e.g. "non-responder"
# Leave "." to use the whole project root behavior.
PATIENT_FOLDER_PATH = '/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241'


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect clean cells from padded CD45RA folders into per-patient formatted folders."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root containing responder/ and non-responder/ (default: .)",
    )
    parser.add_argument(
        "--patient-folder-path",
        type=Path,
        default=Path(PATIENT_FOLDER_PATH),
        help=(
            "Path to a single patient folder or a parent directory of patient folders "
            "(default: PATIENT_FOLDER_PATH variable at top of file)"
        ),
    )
    parser.add_argument(
        "--clean-csv",
        type=Path,
        default=Path("clean_cell_list.csv"),
        help="CSV with clean cell IDs and output names (default: clean_cell_list.csv)",
    )
    parser.add_argument(
        "--clean-col",
        default="clean",
        help="Column name containing clean unique_id values (default: clean)",
    )
    parser.add_argument(
        "--name-col",
        default="name",
        help="Column name containing output file names (default: name)",
    )
    parser.add_argument(
        "--bad-col",
        default="bad",
        help="Column name containing excluded/bad unique_id values (default: bad)",
    )
    parser.add_argument(
        "--patient-ids",
        nargs="+",
        default=["108859", "109241", "109317", "113056", "118830", "118867"],
        help="Patient IDs to process (default: 108859 109241 109317 113056 118830 118867)",
    )
    parser.add_argument(
        "--source-dirname",
        default="padded_cd45ra",
        help="Source folder name under each image directory (default: padded_cd45ra)",
    )
    parser.add_argument(
        "--output-dirname",
        default="formattd_cd45ra",
        help="Output folder created in each patient root (default: formattd_cd45ra)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without copying files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite files that already exist in output folder",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing files in output folder before copying",
    )
    parser.add_argument(
        "--excluded-report-name",
        default="excluded_cells.csv",
        help="CSV filename written in each patient root listing excluded cells (default: excluded_cells.csv)",
    )
    return parser.parse_args()


def parse_unique_id(uid: str) -> tuple[str, CellKey] | None:
    match = UID_PATTERN.match(uid)
    if not match:
        return None
    patient_id, sample_s, image_s, cell_s = match.groups()
    return patient_id, CellKey(sample=int(sample_s), image=int(image_s), cell=int(cell_s))


def load_clean_entries(clean_csv: Path, clean_col: str, name_col: str, patient_ids: Iterable[str]) -> Dict[str, List[CleanEntry]]:
    patient_set = set(patient_ids)
    entries: Dict[str, List[CleanEntry]] = defaultdict(list)

    with clean_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {clean_col, name_col}
        if not required.issubset(fields):
            raise ValueError(
                f"{clean_csv} missing required columns {sorted(required)}; found: {sorted(fields)}"
            )

        for row in reader:
            uid = (row.get(clean_col) or "").strip()
            name = (row.get(name_col) or "").strip()
            if not uid or not name:
                continue

            parsed = parse_unique_id(uid)
            if not parsed:
                continue

            patient_id, key = parsed
            if patient_id not in patient_set:
                continue

            entries[patient_id].append(CleanEntry(key=key, target_name=name))

    return entries


def load_excluded_entries(clean_csv: Path, bad_col: str, patient_ids: Iterable[str]) -> Dict[str, List[ExcludedEntry]]:
    patient_set = set(patient_ids)
    entries: Dict[str, List[ExcludedEntry]] = defaultdict(list)

    with clean_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if bad_col not in fields:
            return entries

        for row in reader:
            uid = (row.get(bad_col) or "").strip()
            if not uid:
                continue

            parsed = parse_unique_id(uid)
            if not parsed:
                continue

            patient_id, key = parsed
            if patient_id not in patient_set:
                continue

            entries[patient_id].append(ExcludedEntry(unique_id=uid, key=key))

    return entries


def _extract_patient_id_from_name(name: str) -> str | None:
    for token in name.replace("-", " ").split():
        if token.isdigit() and len(token) == 6:
            return token
    return None


def discover_patient_roots(root: Path, patient_ids: Iterable[str], patient_folder_path: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = defaultdict(list)
    patient_set = set(patient_ids)
    candidate = patient_folder_path if patient_folder_path.is_absolute() else (root / patient_folder_path)
    candidate = candidate.resolve()

    if not candidate.exists():
        return out

    # Case 0: project root provided; keep legacy scan behavior.
    if candidate == root:
        for base_name in ("responder", "non-responder"):
            base = root / base_name
            if not base.exists():
                continue
            for d in base.iterdir():
                if not d.is_dir() or "DLBCL" not in d.name:
                    continue
                pid = _extract_patient_id_from_name(d.name)
                if pid and pid in patient_set:
                    out[pid].append(d)
        return out

    # Case 1: direct patient root path provided
    if candidate.is_dir() and "DLBCL" in candidate.name:
        pid = _extract_patient_id_from_name(candidate.name)
        if pid and pid in patient_set:
            out[pid].append(candidate)
        return out

    # Case 2: parent folder provided; iterate child patient folders
    if candidate.is_dir():
        for d in candidate.iterdir():
            if not d.is_dir() or "DLBCL" not in d.name:
                continue
            pid = _extract_patient_id_from_name(d.name)
            if pid and pid in patient_set:
                out[pid].append(d)
    return out


def parse_sample_number(sample_dir_name: str) -> int | None:
    if not sample_dir_name.startswith("sample"):
        return None
    suffix = sample_dir_name.replace("sample", "", 1).strip()
    return int(suffix) if suffix.isdigit() else None


def build_source_index(patient_root: Path, source_dirname: str) -> Dict[CellKey, Path]:
    index: Dict[CellKey, Path] = {}
    for source_dir in patient_root.rglob(source_dirname):
        image_dir = source_dir.parent
        sample_dir = image_dir.parent

        sample_num = parse_sample_number(sample_dir.name)
        if sample_num is None:
            continue
        if not image_dir.name.isdigit():
            continue
        image_num = int(image_dir.name)

        for fp in source_dir.iterdir():
            if not fp.is_file():
                continue
            m = CELL_FILE_PATTERN.match(fp.name)
            if not m:
                continue
            key = CellKey(sample=sample_num, image=image_num, cell=int(m.group(1)))
            index[key] = fp
    return index


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    clean_csv = (root / args.clean_csv).resolve() if not args.clean_csv.is_absolute() else args.clean_csv

    if not clean_csv.exists():
        raise FileNotFoundError(f"Clean CSV not found: {clean_csv}")

    patient_roots = discover_patient_roots(root, args.patient_ids, args.patient_folder_path)
    target_patient_ids = sorted(patient_roots.keys())

    if not target_patient_ids:
        print("No patient folders found from --patient-folder-path. Nothing to do.")
        return 1

    clean_entries = load_clean_entries(
        clean_csv=clean_csv,
        clean_col=args.clean_col,
        name_col=args.name_col,
        patient_ids=target_patient_ids,
    )
    excluded_entries = load_excluded_entries(
        clean_csv=clean_csv,
        bad_col=args.bad_col,
        patient_ids=target_patient_ids,
    )

    print("Patient roots discovered (processing only these):")
    for pid in target_patient_ids:
        roots = patient_roots.get(pid, [])
        print(f"  {pid}: {len(roots)} root(s)")
        for r in roots:
            print(f"    - {r}")

    print("\nProcessing clean cells...")
    for pid in target_patient_ids:
        entries = clean_entries.get(pid, [])
        roots = patient_roots.get(pid, [])

        if not entries:
            print(f"[{pid}] no clean entries in {clean_csv.name}, skipped")
            continue
        if not roots:
            print(f"[{pid}] no patient root found, skipped ({len(entries)} clean entries)")
            continue

        for patient_root in roots:
            src_index = build_source_index(patient_root, args.source_dirname)
            out_dir = patient_root / args.output_dirname
            excluded_report = patient_root / args.excluded_report_name

            if args.skip_existing and args.clean_output:
                raise ValueError("Use only one of --skip-existing or --clean-output")

            if not args.dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                if args.clean_output:
                    for p in out_dir.iterdir():
                        if p.is_file():
                            p.unlink()

            copied = 0
            missing = 0
            duplicate_name_overwrites = 0
            skipped_existing = 0

            for entry in entries:
                src = src_index.get(entry.key)
                if src is None:
                    missing += 1
                    continue

                dst_stem = f"{Path(entry.target_name).stem}_cd45ra"
                dst = out_dir / f"{dst_stem}{src.suffix}"

                if dst.exists():
                    duplicate_name_overwrites += 1
                    if args.skip_existing:
                        skipped_existing += 1
                        continue

                if not args.dry_run:
                    shutil.copy2(src, dst)
                copied += 1

            mode = "DRY-RUN" if args.dry_run else "DONE"
            print(
                f"[{pid}] {mode} {patient_root} -> {out_dir}: "
                f"copied={copied}, missing={missing}, indexed_sources={len(src_index)}, "
                f"duplicate_name_overwrites={duplicate_name_overwrites}, skipped_existing={skipped_existing}"
            )

            excluded_for_patient = excluded_entries.get(pid, [])
            if args.dry_run:
                print(f"[{pid}] DRY-RUN excluded report: {excluded_report} rows={len(excluded_for_patient)}")
            else:
                with excluded_report.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["unique_id", "sample", "image", "cell"])
                    for item in excluded_for_patient:
                        writer.writerow([item.unique_id, item.key.sample, item.key.image, item.key.cell])
                print(f"[{pid}] wrote excluded report: {excluded_report} rows={len(excluded_for_patient)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
