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

    def _lineage_from_text(text: str) -> Optional[str]:
        if not text:
            return None
        if "CAR-T" in text or "CART" in text:
            return "CART"
        if text.lower() in {"native", "naive"}:
            return "native"
        return None

    lineage = _lineage_from_text(cell_type) or _lineage_from_text(subset)
    if not lineage and cell_type in {"CD4+", "CD4-", "CD8+", "CD8-"}:
        lineage = "native"

    cd_tag = _cd_tag_from_text(subset) or _cd_tag_from_text(cell_type)

    if subset:
        subset = re.sub(r"^CAR-T\s+", "", subset)
        subset = re.sub(r"^(CD4\+|CD4-|CD8\+|CD8-)\s+", "", subset)

    abbrev = extract_abbreviation(subset) if subset else ""

    if lineage and cd_tag:
        return _normalize_component(f"{lineage}_{cd_tag}_{abbrev}".strip("_"))
    if lineage:
        return _normalize_component(f"{lineage}_{abbrev}".strip("_"))

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


def format_patient_response_label(response: str) -> str:
    response = (response or "").strip().lower()
    if response in {"nonresponder", "non-responder", "non_responder"}:
        return "Non-responder"
    if response == "responder":
        return "Responder"
    return _normalize_component(response or "Unknown")


def format_patient_date_label(patient_folder_name: str) -> Optional[str]:
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

    return f"{month:02d}-{day:02d}-{year:04d}"


def format_sample_label(sample: str | int) -> str:
    sample_text = str(sample).strip()
    match = re.search(r"(\d+)", sample_text)
    if not match:
        return _normalize_component(sample_text)
    return f"sample{int(match.group(1)):02d}"


def format_image_label(image: str | int) -> str:
    image_num = extract_image_number(str(image))
    if image_num is None:
        return _normalize_component(str(image))
    return f"image{image_num:02d}"


def format_cell_label(cell_id: str | int) -> str:
    return f"cell{int(cell_id):02d}"


def format_sigma_label(sigma: float) -> str:
    sigma_value = float(sigma)
    if sigma_value.is_integer():
        return f"sigma{int(sigma_value)}"
    return f"sigma{str(sigma_value).replace('.', 'p')}"


def build_channel_filename(
    response: str,
    patient_folder_name: str,
    patient_id: str | int,
    sample: str | int,
    image: str | int,
    cell_id: str | int,
    stiffness: str = "1to10",
    classification: str = "",
    channel_suffix: str = "",
) -> str:
    parts = [
        format_patient_response_label(response),
        format_patient_date_label(patient_folder_name) or "UnknownDate",
        "DLBCL",
        _normalize_component(stiffness),
        str(patient_id),
        format_sample_label(sample),
        format_image_label(image),
        format_cell_label(cell_id),
        _normalize_component(classification),
        _normalize_component(channel_suffix),
    ]
    return "_".join(part for part in parts if part)


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
