#!/usr/bin/env python3
"""
Build per-patient formatted channel image folders from clean cell list.

Uses shared iterator helpers and a single clean list schema:
    unique_id, clean_cell, name
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from utils.config_helpers import extract_sample_number
from utils.image_iterator import collect_image_jobs
from utils.name_builder import extract_image_number

UID_PATTERN = re.compile(r"^(\d+)_([0-9]+)_([0-9]+)_([0-9]+)$")
CELL_FILE_PATTERN = re.compile(r"^cell_(\d+)_padded\.[^.]+$")

# Configure these for default execution.
PATIENT_FOLDER_PATH = '/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241'
CLEAN_CSV_PATH = "/Users/taeeonkong/Desktop/DL Project/clean_cell_list.csv"
SINGLE_PATIENT_MODE = True

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect clean cells from padded channel folders into per-patient formatted folders."
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
            "Path to a single patient folder (when --single-patient-mode=true) "
            "or a parent folder containing multiple patient folders "
            "(default: PATIENT_FOLDER_PATH variable at top of file)"
        ),
    )
    parser.add_argument(
        "--single-patient-mode",
        type=lambda s: str(s).strip().lower() in {"1", "true", "t", "yes", "y"},
        default=SINGLE_PATIENT_MODE,
        help="True: patient-folder-path is a single patient folder. False: iterate all patient folders under patient-folder-path.",
    )
    parser.add_argument(
        "--clean-csv",
        type=Path,
        default=Path(CLEAN_CSV_PATH),
        help="CSV with clean cell IDs and output names (default: CLEAN_CSV_PATH variable at top of file)",
    )
    parser.add_argument(
        "--clean-col",
        default="unique_id",
        help="Column name containing cell unique_id values (default: unique_id)",
    )
    parser.add_argument(
        "--clean-flag-col",
        default="clean",
        help="Column name indicating clean rows (True/False) (default: clean)",
    )
    parser.add_argument(
        "--name-col",
        default="name",
        help="Column name containing output file names (default: name)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["actin", "ccr7", "cd45ra"],
        choices=sorted(CHANNEL_PRESETS.keys()),
        help="Which channels to build (default: actin ccr7 cd45ra)",
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


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_unique_id(uid: str) -> tuple[str, CellKey] | None:
    match = UID_PATTERN.match(uid)
    if not match:
        return None
    patient_id, sample_s, image_s, cell_s = match.groups()
    return patient_id, CellKey(sample=int(sample_s), image=int(image_s), cell=int(cell_s))


def load_entries(
    clean_csv: Path,
    clean_col: str,
    clean_flag_col: str,
    name_col: str,
    patient_ids: Iterable[str],
) -> tuple[Dict[str, List[CleanEntry]], Dict[str, List[ExcludedEntry]]]:
    patient_set = set(patient_ids)
    clean_entries: Dict[str, List[CleanEntry]] = defaultdict(list)
    excluded_entries: Dict[str, List[ExcludedEntry]] = defaultdict(list)

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
            patient_id, key = parsed
            if patient_id not in patient_set:
                continue

            is_clean = _is_truthy(row.get(clean_flag_col) or "")
            if is_clean:
                name = (row.get(name_col) or "").strip()
                if not name:
                    continue
                clean_entries[patient_id].append(CleanEntry(key=key, target_name=name))
            else:
                excluded_entries[patient_id].append(ExcludedEntry(unique_id=uid, key=key))

    return clean_entries, excluded_entries


def _extract_patient_id_from_name(name: str) -> str | None:
    for token in name.replace("-", " ").split():
        if token.isdigit() and len(token) == 6:
            return token
    return None


def discover_patient_roots(root: Path, patient_folder_path: Path, single_patient_mode: bool) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = defaultdict(list)
    candidate = patient_folder_path if patient_folder_path.is_absolute() else (root / patient_folder_path)
    candidate = candidate.resolve()

    if not candidate.exists():
        return out

    if single_patient_mode:
        if candidate.is_dir() and "DLBCL" in candidate.name:
            pid = _extract_patient_id_from_name(candidate.name)
            if pid:
                out[pid].append(candidate)
        return out

    if candidate == root:
        for base_name in ("responder", "non-responder"):
            base = root / base_name
            if not base.exists():
                continue
            for d in base.iterdir():
                if not d.is_dir() or "DLBCL" not in d.name:
                    continue
                pid = _extract_patient_id_from_name(d.name)
                if pid:
                    out[pid].append(d)
        return out

    if candidate.is_dir():
        for d in candidate.iterdir():
            if not d.is_dir() or "DLBCL" not in d.name:
                continue
            pid = _extract_patient_id_from_name(d.name)
            if pid:
                out[pid].append(d)
    return out


def build_source_index(patient_root: Path, source_dirname: str) -> Dict[CellKey, Path]:
    index: Dict[CellKey, Path] = {}
    jobs = collect_image_jobs(
        base_path=patient_root,
        samples_to_process=None,
        images_to_process=None,
        announce_filters=False,
    )

    for job in jobs:
        source_dir = job.image_path / source_dirname
        if not source_dir.exists():
            continue

        sample_num = extract_sample_number(job.sample_folder)
        image_num = extract_image_number(job.image_folder)
        if sample_num <= 0 or image_num is None:
            continue

        for fp in source_dir.iterdir():
            if not fp.is_file():
                continue
            m = CELL_FILE_PATTERN.match(fp.name)
            if not m:
                continue
            key = CellKey(sample=sample_num, image=int(image_num), cell=int(m.group(1)))
            index[key] = fp

    return index


def _clean_output_dir(path: Path) -> None:
    if not path.exists():
        return
    for p in path.iterdir():
        if p.is_file():
            p.unlink()


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    clean_csv = (root / args.clean_csv).resolve() if not args.clean_csv.is_absolute() else args.clean_csv

    if not clean_csv.exists():
        raise FileNotFoundError(f"Clean CSV not found: {clean_csv}")
    if args.skip_existing and args.clean_output:
        raise ValueError("Use only one of --skip-existing or --clean-output")

    patient_roots = discover_patient_roots(root, args.patient_folder_path, args.single_patient_mode)
    target_patient_ids = sorted(patient_roots.keys())
    if not target_patient_ids:
        print("No patient folders found from --patient-folder-path. Nothing to do.")
        return 1

    clean_entries, excluded_entries = load_entries(
        clean_csv=clean_csv,
        clean_col=args.clean_col,
        clean_flag_col=args.clean_flag_col,
        name_col=args.name_col,
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

        for patient_root in roots:
            for channel in args.channels:
                preset = CHANNEL_PRESETS[channel]
                source_dirname = preset["source_dirname"]
                output_dirname = preset["output_dirname"]
                suffix = preset["suffix"]

                src_index = build_source_index(patient_root, source_dirname)
                out_dir = patient_root / output_dirname
                excluded_report = patient_root / args.excluded_report_name

                if not args.dry_run:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    if args.clean_output:
                        _clean_output_dir(out_dir)

                copied = 0
                missing = 0
                duplicate_name_overwrites = 0
                skipped_existing = 0

                for entry in entries:
                    src = src_index.get(entry.key)
                    if src is None:
                        missing += 1
                        continue

                    dst_stem = f"{Path(entry.target_name).stem}_{suffix}"
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
                    f"[{pid}] [{channel}] {mode} {patient_root} -> {out_dir}: "
                    f"copied={copied}, missing={missing}, indexed_sources={len(src_index)}, "
                    f"duplicate_name_overwrites={duplicate_name_overwrites}, skipped_existing={skipped_existing}"
                )

                excluded_for_patient = excluded_entries.get(pid, [])
                if args.dry_run:
                    print(
                        f"[{pid}] [{channel}] DRY-RUN excluded report: "
                        f"{excluded_report} rows={len(excluded_for_patient)}"
                    )
                else:
                    with excluded_report.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["unique_id", "sample", "image", "cell"])
                        for item in excluded_for_patient:
                            writer.writerow([item.unique_id, item.key.sample, item.key.image, item.key.cell])
                    print(
                        f"[{pid}] [{channel}] wrote excluded report: "
                        f"{excluded_report} rows={len(excluded_for_patient)}"
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
