"""Image folder and channel filename resolution helpers."""

from __future__ import annotations

from pathlib import Path


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


def _channel_error(
    image_dir: Path,
    configured_map: dict[str, str],
    missing: list[tuple[str, str]],
    available_tifs: list[str],
    job_label: str | None,
) -> ValueError:
    where = job_label or image_dir.name
    lines = [
        f"Channel filename mismatch in image {where}.",
        f"Image folder: {image_dir}",
        "",
        "Expected channel files:",
    ]
    lines.extend(f"  - {channel}: {filename}" for channel, filename in configured_map.items())
    lines.extend(["", "Missing files:"])
    lines.extend(f"  - {channel}: {filename}" for channel, filename in missing)
    lines.extend(["", "Available .tif files:"])
    if available_tifs:
        lines.extend(f"  - {filename}" for filename in available_tifs)
    else:
        lines.append("  - none")
    lines.extend(
        [
            "",
            "Fix the filenames in this image folder, or update config.channels/DEFAULT_CHANNELS.",
        ]
    )
    return ValueError("\n".join(lines))


def resolve_channel_filenames(
    image_dir: Path,
    configured_map: dict[str, str],
    job_label: str | None = None,
) -> dict[str, str]:
    """Resolve configured channel filenames by exact filename match only."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image folder not found for channel check: {image_dir}")

    available_tifs = sorted([p.name for p in image_dir.glob("*.tif")])
    available = set(available_tifs)
    resolved: dict[str, str] = {}
    missing: list[tuple[str, str]] = []

    for key, configured_filename in configured_map.items():
        if configured_filename in available:
            resolved[key] = configured_filename
        else:
            missing.append((key, configured_filename))

    if missing:
        raise _channel_error(
            image_dir=image_dir,
            configured_map=configured_map,
            missing=missing,
            available_tifs=available_tifs,
            job_label=job_label,
        )

    return resolved
