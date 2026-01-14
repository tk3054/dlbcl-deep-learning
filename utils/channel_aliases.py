#!/usr/bin/env python3
"""
Channel alias utilities.

These helpers normalize user-specified channel keys so the rest of the
pipeline can rely on a consistent set of canonical names. This prevents
errors when someone renames the config key (e.g., using "cd45ra_PacBlue"
instead of the expected "cd45ra_sparkviolet").
"""

import re
from typing import Dict, Iterable, List, Optional

def canonicalize_channel_key(key: Optional[str]) -> Optional[str]:
    """
    Return the provided channel key unchanged (no aliasing).
    """
    return key


def canonicalize_channel_config(channel_config: Dict[str, str],
                                verbose: bool = False) -> Dict[str, str]:
    """
    Normalize channel config keys while preserving user-specified filenames.

    Currently a pass-through aside from copying; no alias mapping is performed.
    """
    return dict(channel_config) if channel_config else {}


def canonicalize_channel_list(channel_list: Optional[Iterable[str]],
                              verbose: bool = False) -> Optional[List[str]]:
    """Return the list unchanged (deduplicated in order)."""
    if channel_list is None:
        return None

    seen = set()
    normalized_list: List[str] = []
    for key in channel_list:
        if key not in seen:
            normalized_list.append(key)
            seen.add(key)
    return normalized_list


def canonicalize_null_map(null_map: Dict[int, Iterable[str]],
                          verbose: bool = False) -> Dict[int, List[str]]:
    """Normalize the channel lists inside the per-sample null map."""
    if not null_map:
        return {}

    normalized = {}
    for sample, channels in null_map.items():
        normalized[sample] = canonicalize_channel_list(channels, verbose=verbose) or []
    return normalized
