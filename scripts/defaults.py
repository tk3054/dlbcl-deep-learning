"""Shared CLI parsing helpers for script entry points."""

from __future__ import annotations


def parse_channel_overrides(overrides: list[str] | None) -> dict[str, str] | None:
    if not overrides:
        return None

    channels: dict[str, str] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --channel value {item!r}; expected key=filename")
        key, filename = item.split("=", 1)
        key = key.strip()
        filename = filename.strip()
        if not key or not filename:
            raise ValueError(f"Invalid --channel value {item!r}; expected key=filename")
        channels[key] = filename
    return channels


def parse_image_filter(images: list[int] | None) -> list[int] | None:
    return images or None
