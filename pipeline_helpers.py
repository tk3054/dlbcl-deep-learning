import os
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import imagej
import pandas as pd

from utils.config_helpers import extract_sample_number
from utils.name_builder import (
    NameBuilder,
    build_patient_context,
    extract_image_number,
    format_cell_classification,
)


@contextmanager
def suppress_output():
    """Suppress ImageJ stdout/stderr noise during initialization."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def prepare_run(base_path: str, samples_to_process: list[int]):
    """Preflight: validate base path, gather samples, and initialize ImageJ."""
    base_path_obj = Path(base_path)

    if not base_path_obj.exists():
        print(f"✗ ERROR: Base path not found: {base_path}")
        sys.exit(1)

    sample_folders = [
        item.name
        for item in base_path_obj.iterdir()
        if item.is_dir() and item.name.lower().startswith("sample")
    ]
    sample_folders = sorted(sample_folders, key=extract_sample_number)
    sample_folders = [
        s for s in sample_folders if extract_sample_number(s) in samples_to_process
    ]

    if not sample_folders:
        print(f"✗ ERROR: No sample folders found in {base_path}")
        sys.exit(1)

    print("\n" + "=" * 40)
    print("BATCH PIPELINE: PROCESSING ALL SAMPLES")
    print("=" * 40)
    print(f"Base path: {base_path}")
    print(f"Found {len(sample_folders)} samples: {', '.join(sample_folders)}")
    print("=" * 40 + "\n")

    print("Initializing ImageJ (will be reused for all images)...")
    with suppress_output():
        ij = imagej.init("sc.fiji:fiji")
    print(f"✓ ImageJ version: {ij.getVersion()}\n")
    print("=" * 40 + "\n")

    return base_path_obj, sample_folders, ij


def prompt_channel_filenames(
    base_path_obj: Path,
    sample_folder: str,
    image_number: str,
    channel_config: dict[str, str],
):
    """
    Check channel filenames for an image folder; if a configured name is missing,
    prompt the user to pick between the configured name and any .tif present.
    """
    image_dir = base_path_obj / sample_folder / image_number
    if not image_dir.exists():
        return channel_config

    available_tifs = sorted([p.name for p in image_dir.glob("*.tif")])
    resolved = dict(channel_config)

    for key, filename in channel_config.items():
        expected = image_dir / filename
        processed = image_dir / f"processed_{filename}"

        if expected.exists() or processed.exists():
            continue

        if not available_tifs:
            continue

        print(f"\nChannel '{key}' file not found for {sample_folder}/{image_number}.")
        print(f"Configured: {filename}")
        print("Available .tif files in this folder:")
        for idx, name in enumerate(available_tifs, start=1):
            print(f"  {idx}. {name}")
        print("  0. Keep configured name")

        choice = input("Select a file number to use for this channel (default 0): ").strip()
        if not choice:
            choice = "0"

        try:
            choice_num = int(choice)
        except ValueError:
            choice_num = 0

        if 1 <= choice_num <= len(available_tifs):
            resolved[key] = available_tifs[choice_num - 1]
            print(f"  → Using {resolved[key]} for channel '{key}'")
        else:
            print(f"  → Keeping configured filename for '{key}'")

    return resolved


def _find_cell_image_path(source_dir: Path, cell_id: int) -> Optional[Path]:
    patterns = [
        f"cell_{cell_id:02d}_padded.tif",
        f"cell_{cell_id:03d}_padded.tif",
        f"cell_{cell_id}_padded.tif",
        f"cell_{cell_id:02d}_raw.tif",
        f"cell_{cell_id:03d}_raw.tif",
        f"cell_{cell_id}_raw.tif",
        f"cell_{cell_id:02d}.tif",
        f"cell_{cell_id:03d}.tif",
        f"cell_{cell_id}.tif",
        f"cell_{cell_id:03d}_crop.tif",
        f"cell_{cell_id:02d}_crop.tif",
    ]

    for pattern in patterns:
        candidate = source_dir / pattern
        if candidate.exists():
            return candidate

    matches = sorted(source_dir.glob(f"*cell_{cell_id}*"))
    return matches[0] if matches else None


def _resolve_image_folder(sample_dir: Path, image_number: str) -> Optional[str]:
    """Resolve image folder name, supporting prefix matches like '14[large cell]'."""
    if not sample_dir.exists():
        return None
    candidate = sample_dir / image_number
    if candidate.exists():
        return image_number
    if image_number.isdigit():
        for entry in sorted(sample_dir.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith(image_number):
                next_char = name[len(image_number):len(image_number) + 1]
                if next_char == "" or not next_char.isdigit():
                    return name
    return None


def _build_destination(dest_dir: Path, filename: str) -> Path:
    """Return deterministic destination path (overwrites if exists)."""
    return dest_dir / filename


def export_images_for_export(
    base_path_obj: Path,
    name_builder: NameBuilder,
    output_dir_name: str,
    source_dir_name: str,
    classified_csv: str,
    dilution: str,
    pdms_stiffness: str,
    verbose: bool = True,
):
    csv_path = base_path_obj / classified_csv
    if not csv_path.exists():
        print(f"⚠️  Export skipped: CSV not found at {csv_path}")
        return {"success": False, "error": "classified_csv_missing"}

    df = pd.read_csv(csv_path)
    required_columns = {"sample", "image", "cell_id", "cell_type", "tcell_subset"}
    missing = required_columns - set(df.columns)
    if missing:
        print(f"⚠️  Export skipped: CSV missing columns: {', '.join(sorted(missing))}")
        return {"success": False, "error": "missing_columns"}

    context = build_patient_context(base_path_obj)
    response = context.get("response") or "unknownresponse"
    patient_id = context.get("patient_id") or "UnknownPatient"
    date_match = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", base_path_obj.name)
    date_code = date_match.group(1) if date_match else "UnknownDate"

    output_dir = base_path_obj / output_dir_name
    output_dir.mkdir(exist_ok=True)

    exported = 0
    skipped = 0
    missing_sources = 0
    skipped_duplicates = 0
    seen_filenames = set()

    for _, row in df.iterrows():
        sample_folder = str(row["sample"])
        image_number = str(row["image"])
        try:
            cell_id = int(row["cell_id"])
        except (TypeError, ValueError):
            skipped += 1
            continue

        sample_dir = base_path_obj / sample_folder
        image_folder = _resolve_image_folder(sample_dir, image_number)
        if not image_folder:
            missing_sources += 1
            continue
        source_dir = sample_dir / image_folder / source_dir_name
        if not source_dir.exists():
            missing_sources += 1
            continue

        source_path = _find_cell_image_path(source_dir, cell_id)
        if source_path is None:
            skipped += 1
            continue

        classification = format_cell_classification(row["cell_type"], row["tcell_subset"])
        image_num = extract_image_number(image_number)
        image_label = f"image{int(image_num):02d}" if image_num is not None else image_number
        sample_label = sample_folder
        if sample_label.lower().startswith("sample"):
            sample_suffix = sample_label[len("sample") :]
            sample_label = f"sample{sample_suffix}"
        parts = {
            "response": response,
            "patient_id": patient_id,
            "date": date_code,
            "stiffness": pdms_stiffness,
            "sample": sample_label,
            "image": image_label,
            "cell_label": f"cell{int(cell_id):02d}",
            "classification": classification,
        }
        base_name = name_builder.build(parts)
        dest_filename = f"{base_name}{source_path.suffix}"
        if dest_filename in seen_filenames:
            skipped_duplicates += 1
            continue
        seen_filenames.add(dest_filename)
        dest_path = _build_destination(output_dir, dest_filename)

        shutil.copy2(source_path, dest_path)
        exported += 1

    if verbose:
        print("\n" + "=" * 80)
        print("EXPORT IMAGES SUMMARY")
        print("=" * 80)
        print(f"Output dir: {output_dir}")
        print(f"Exported: {exported}")
        print(f"Skipped (missing files): {skipped}")
        print(f"Skipped (missing source dirs): {missing_sources}")
        print(f"Skipped (duplicate rows): {skipped_duplicates}")
        print("=" * 80 + "\n")

    return {
        "success": True,
        "exported": exported,
        "skipped": skipped,
        "missing_sources": missing_sources,
        "output_dir": str(output_dir),
    }
