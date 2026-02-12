"""
Shared image iteration and channel filename resolution helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from utils.config_helpers import (
    extract_sample_number,
    filter_image_folders,
    normalize_image_filter_config,
)


@dataclass(frozen=True)
class ImageJob:
    sample_folder: str
    image_folder: str
    sample_path: Path
    image_path: Path

    @property
    def label(self) -> str:
        return f"{self.sample_folder}/{self.image_folder}"


def discover_samples(base_path: Path, samples_to_process: list[int] | None = None) -> list[str]:
    samples = [
        item.name
        for item in base_path.iterdir()
        if item.is_dir() and item.name.lower().startswith("sample")
    ]
    samples = sorted(samples, key=extract_sample_number)
    if samples_to_process:
        samples = [s for s in samples if extract_sample_number(s) in samples_to_process]
    return samples


def discover_images(sample_path: Path) -> list[str]:
    image_folders = [item.name for item in sample_path.iterdir() if item.is_dir()]
    return sorted(
        image_folders,
        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
    )


def collect_image_jobs(
    base_path: Path,
    samples_to_process: list[int] | None = None,
    images_to_process=None,
    announce_filters: bool = False,
) -> list[ImageJob]:
    jobs: list[ImageJob] = []

    samples = discover_samples(base_path, samples_to_process)
    image_filters, image_filters_default = normalize_image_filter_config(images_to_process)

    for sample_folder in samples:
        sample_path = base_path / sample_folder
        image_folders = discover_images(sample_path)
        image_folders = filter_image_folders(
            sample_folder,
            image_folders,
            image_filters,
            image_filters_default,
            announce=announce_filters,
        )
        for image_folder in image_folders:
            jobs.append(
                ImageJob(
                    sample_folder=sample_folder,
                    image_folder=image_folder,
                    sample_path=sample_path,
                    image_path=sample_path / image_folder,
                )
            )
    return jobs


def resolve_image_folder(sample_dir: Path, image_number: str) -> str | None:
    """
    Resolve image folder name, supporting prefix matches like '14[large cell]'.
    """
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
                next_char = name[len(image_number) : len(image_number) + 1]
                if next_char == "" or not next_char.isdigit():
                    return name
    return None


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def channel_candidates(configured_filename: str) -> list[str]:
    """
    Build candidate names with processed/raw variants.
    """
    candidates = [configured_filename]
    if configured_filename.startswith("processed_"):
        candidates.append(configured_filename[len("processed_") :])
    else:
        candidates.append(f"processed_{configured_filename}")
    return _unique_preserve_order(candidates)


def _normalize_stem(stem: str) -> str:
    # Allow matching through modifiers like "(1)", "_v2", spaces, dashes.
    normalized = re.sub(r"[^a-z0-9]+", "", stem.lower())
    if normalized.startswith("processed"):
        normalized = normalized[len("processed") :]
    return normalized


def _match_candidate(available: list[str], candidate: str) -> str | None:
    """
    Return best matching filename from available list for one candidate.
    Matching levels:
    1) Exact
    2) Case-insensitive exact
    3) Normalized stem exact (ignores separators/modifier punctuation)
    4) Normalized stem containment (handles appended modifiers)
    """
    available_set = set(available)
    if candidate in available_set:
        return candidate

    lower_map = {name.lower(): name for name in available}
    if candidate.lower() in lower_map:
        return lower_map[candidate.lower()]

    cand_path = Path(candidate)
    cand_suffix = cand_path.suffix.lower()
    cand_norm = _normalize_stem(cand_path.stem)

    exact_norm = []
    contain_norm = []
    for name in available:
        p = Path(name)
        if p.suffix.lower() != cand_suffix:
            continue
        norm = _normalize_stem(p.stem)
        if norm == cand_norm:
            exact_norm.append(name)
        elif cand_norm and (cand_norm in norm or norm in cand_norm):
            contain_norm.append(name)

    if exact_norm:
        exact_norm.sort()
        return exact_norm[0]
    if contain_norm:
        contain_norm.sort()
        return contain_norm[0]
    return None


def resolve_channel_filename(
    image_dir: Path,
    configured_filename: str,
    interactive_prompt: bool = True,
    channel_key: str | None = None,
    job_label: str | None = None,
) -> str | None:
    """
    Resolve channel filename for one image folder.
    Tries automatic matching first, then optional interactive fallback.
    """
    if not image_dir.exists():
        return None

    available_tifs = sorted([p.name for p in image_dir.glob("*.tif")])
    if not available_tifs:
        return None

    for candidate in channel_candidates(configured_filename):
        matched = _match_candidate(available_tifs, candidate)
        if matched:
            return matched

    if not interactive_prompt:
        return None

    label = channel_key or "channel"
    where = f" for {job_label}" if job_label else ""
    print(f"\nChannel '{label}' file not found{where}.")
    print(f"Configured: {configured_filename}")
    print("Available .tif files in this folder:")
    for idx, name in enumerate(available_tifs, start=1):
        print(f"  {idx}. {name}")
    print("  0. Skip this channel")

    choice = input("Select a file number to use for this channel (default 0): ").strip()
    if not choice:
        choice = "0"
    try:
        choice_num = int(choice)
    except ValueError:
        choice_num = 0

    if 1 <= choice_num <= len(available_tifs):
        selected = available_tifs[choice_num - 1]
        print(f"  -> Using {selected} for channel '{label}'")
        return selected

    print(f"  -> Skipping channel '{label}'")
    return None


def resolve_channel_filenames(
    image_dir: Path,
    configured_map: dict[str, str],
    interactive_prompt: bool = True,
    job_label: str | None = None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, configured_filename in configured_map.items():
        match = resolve_channel_filename(
            image_dir=image_dir,
            configured_filename=configured_filename,
            interactive_prompt=interactive_prompt,
            channel_key=key,
            job_label=job_label,
        )
        if match:
            resolved[key] = match
    return resolved
