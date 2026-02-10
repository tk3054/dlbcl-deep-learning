#!/usr/bin/env python3
"""
File Finder Utilities
Intelligently finds files in image directories by looking at what actually exists
instead of hardcoding filename patterns.

This solves the recurring issue of filename mismatches across different scripts.
"""

from pathlib import Path
from typing import Optional, List


def find_raw_image(base_dir: str, image_number: str = None, verbose: bool = False) -> Optional[Path]:
    """
    Find the raw image file in a directory.

    Args:
        base_dir: Directory to search in
        image_number: Optional image number to verify match (e.g., "2[CAR]")
        verbose: Print debug info

    Returns:
        Path to raw image file, or None if not found
    """
    base_path = Path(base_dir)

    # Look for *_raw.jpg files
    raw_files = list(base_path.glob("*_raw.jpg"))

    if not raw_files:
        if verbose:
            print(f"No *_raw.jpg files found in {base_dir}")
        return None

    if len(raw_files) == 1:
        if verbose:
            print(f"Found raw image: {raw_files[0].name}")
        return raw_files[0]

    # Multiple files - try to match image_number if provided
    if image_number:
        # Extract just the numeric part for matching
        num_part = image_number.split('[')[0]

        for raw_file in raw_files:
            # Check if filename contains the image number
            if f"_{num_part.zfill(2)}" in raw_file.name or f"_{num_part}_" in raw_file.name:
                if verbose:
                    print(f"Found matching raw image: {raw_file.name}")
                return raw_file

    # Fallback: return first one
    if verbose:
        print(f"Multiple raw files found, using first: {raw_files[0].name}")
    return raw_files[0]


def find_original_image(base_dir: str, verbose: bool = False) -> Optional[Path]:
    """
    Find the original image file (TIF format).
    Checks for original_image.tif or Actin-FITC.tif

    Args:
        base_dir: Directory to search in
        verbose: Print debug info

    Returns:
        Path to original image, or None if not found
    """
    base_path = Path(base_dir)

    # Priority order
    candidates = [
        base_path / "original_image.tif",
        base_path / "Actin-FITC.tif",
    ]

    for candidate in candidates:
        if candidate.exists():
            if verbose:
                print(f"Found original image: {candidate.name}")
            return candidate

    if verbose:
        print(f"No original TIF image found in {base_dir}")
    return None


def find_channel_files(base_dir: str, verbose: bool = False) -> dict:
    """
    Find all channel files in a directory.

    Args:
        base_dir: Directory to search in
        verbose: Print debug info

    Returns:
        Dictionary mapping channel names to file paths
    """
    base_path = Path(base_dir)

    # Common channel patterns
    channel_patterns = {
        'actin': ['Actin-FITC.tif', 'actin*.tif'],
        'cd19': ['CD19-*.tif', 'cd19*.tif'],
        'cd3': ['CD3-*.tif', 'cd3*.tif'],
        'cd5': ['CD5-*.tif', 'cd5*.tif'],
        'ccr7': ['CCR7-*.tif', 'ccr7*.tif'],
    }

    channels = {}

    for channel_name, patterns in channel_patterns.items():
        for pattern in patterns:
            matches = list(base_path.glob(pattern))
            if matches:
                channels[channel_name] = matches[0]
                if verbose:
                    print(f"Found {channel_name}: {matches[0].name}")
                break

    return channels


def list_all_images(base_dir: str) -> List[Path]:
    """
    List all image files in a directory.

    Args:
        base_dir: Directory to search in

    Returns:
        List of image file paths
    """
    base_path = Path(base_dir)

    # Common image extensions
    extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']

    images = []
    for ext in extensions:
        images.extend(base_path.glob(ext))

    return sorted(images)


def get_roi_dir(base_dir: str, shrunk: bool = False) -> Optional[Path]:
    """
    Get the ROI directory path.

    Args:
        base_dir: Base directory
        shrunk: If True, return shrunk ROI directory

    Returns:
        Path to ROI directory, or None if doesn't exist
    """
    base_path = Path(base_dir)

    if shrunk:
        roi_dir = base_path / "cell_rois_shrunk"
    else:
        roi_dir = base_path / "cell_rois"

    if roi_dir.exists():
        return roi_dir

    return None


def find_best_image_for_visualization(base_dir: str, image_number: str = None, verbose: bool = False) -> Optional[Path]:
    """
    Find the best image to use for visualization.
    Prefers raw JPG > Actin TIF > original_image.tif

    Args:
        base_dir: Directory to search in
        image_number: Optional image number for matching
        verbose: Print debug info

    Returns:
        Path to best image file
    """
    # Try raw image first
    raw_img = find_raw_image(base_dir, image_number, verbose)
    if raw_img:
        return raw_img

    # Try Actin-FITC
    actin_path = Path(base_dir) / "Actin-FITC.tif"
    if actin_path.exists():
        if verbose:
            print(f"Using Actin-FITC.tif for visualization")
        return actin_path

    # Try original_image
    original_img = find_original_image(base_dir, verbose)
    if original_img:
        return original_img

    if verbose:
        print("No suitable image found for visualization")
    return None
