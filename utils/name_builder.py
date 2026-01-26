#!/usr/bin/env python3
"""
Name builder utilities for export filenames.
"""

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, Iterable, Optional


def _normalize_component(text: str) -> str:
    cleaned = re.sub(r"\s+", "_", text.strip())
    cleaned = re.sub(r"[^\w\-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _normalize_response_label(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", text.strip()).lower()
    return cleaned


def _extract_date_code(patient_folder_name: str) -> Optional[str]:
    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", patient_folder_name)
    if not match:
        return None

    part1, part2, year_raw = match.groups()
    month = int(part1)
    day = int(part2)
    year = int(year_raw)

    if year < 100:
        year += 2000

    if month > 12 and day <= 12:
        month, day = day, month

    return f"{year:04d}{month:02d}{day:02d}"


def _extract_patient_id(patient_folder_name: str) -> Optional[str]:
    numbers = re.findall(r"\d+", patient_folder_name)
    return numbers[-1] if numbers else None


def build_patient_context(base_path: Path) -> Dict[str, Optional[str]]:
    patient_folder = base_path.name
    response_folder = base_path.parent.name

    return {
        "response": _normalize_response_label(response_folder),
        "patient_id": _extract_patient_id(patient_folder),
        "date": _extract_date_code(patient_folder),
    }


def extract_abbreviation(text: str) -> str:
    match = re.search(r"\(([^)]+)\)", text)
    return match.group(1) if match else text


def format_cell_classification(cell_type: str, subset: str) -> str:
    cell_type = (cell_type or "").strip()
    subset = (subset or "").strip()
    if subset in {"", "N/A", "NA"}:
        subset = ""

    def _cd_tag_from_text(text: str) -> Optional[str]:
        if not text:
            return None
        if "CD4+" in text:
            return "CD4"
        if "CD4-" in text:
            return "CD8"
        if "CD8" in text:
            return "CD8"
        return None

    abbrev = extract_abbreviation(subset) if subset else ""

    if cell_type == "CAR-T":
        cd_tag = _cd_tag_from_text(subset) or "CD4"
        if subset:
            subset = re.sub(r"^CAR-T\s+(CD4\+|CD4-|CD8)\s+", "", subset)
            abbrev = extract_abbreviation(subset) if subset else ""
        return _normalize_component(f"CART_{cd_tag}_{abbrev}".strip("_"))

    if cell_type in {"CD4+", "CD4-"}:
        cd_tag = "CD4" if cell_type == "CD4+" else "CD8"
        return _normalize_component(f"native_{cd_tag}_{abbrev}".strip("_"))

    if cell_type and subset:
        return _normalize_component(f"{cell_type}_{subset}")

    return _normalize_component(cell_type or subset or "Unknown")


def extract_image_number(image_label: str) -> Optional[int]:
    if image_label is None:
        return None
    text = str(image_label).strip()
    if not text:
        return None

    core = text.split("[", 1)[0].strip()
    # Strict: only accept a leading number, ignore the rest.
    match = re.match(r"^\s*(\d+)", core)
    return int(match.group(1)) if match else None


def extract_patient_id_from_path(base_path: Path) -> Optional[str]:
    return _extract_patient_id(base_path.name)


@dataclass
class NameBuilder:
    order: Iterable[str]
    separator: str = "_"

    def build(self, parts: Dict[str, Optional[str]]) -> str:
        tokens = []
        for key in self.order:
            value = parts.get(key)
            if value:
                tokens.append(str(value))
        return self.separator.join(tokens)
