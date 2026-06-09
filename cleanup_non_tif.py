#!/usr/bin/env python3
"""
Deep clean: delete everything except original .tif files.

Removes generated files while keeping raw TIF images.

Usage:
    python cleanup_non_tif.py
    python cleanup_non_tif.py --base-path /path/to/donor-folder
"""

from pathlib import Path
import argparse
import shutil

BASE_PATH = '/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241'


GENERATED_FORMATTED_PREFIXES = (
    "formatted_",
    "formattd_",
)


def is_generated_formatted_dir(path):
    """Return True for generated formatted cell/channel image folders."""
    name = path.name.lower()
    return name == "formatted_cells" or any(
        name.startswith(prefix) for prefix in GENERATED_FORMATTED_PREFIXES
    )


def cleanup_non_tif_files(base_path, verbose=True):
    """
    Delete all files and folders except original .tif/.TIF files.

    Processed TIF outputs are removed. Raw TIF inputs are kept.
    """
    base_path_obj = Path(base_path)

    if not base_path_obj.exists():
        print(f"✗ ERROR: Base path not found: {base_path}")
        return

    print("\n" + "=" * 80)
    print("DEEP CLEANUP: DELETING EVERYTHING EXCEPT ORIGINAL .TIF FILES")
    print("=" * 80)
    print(f"Base path: {base_path}")
    print("=" * 80 + "\n")

    deleted_files = 0
    deleted_dirs = 0
    kept_tifs = 0

    sample_folders = [
        item
        for item in base_path_obj.iterdir()
        if item.is_dir() and item.name.lower().startswith("sample")
    ]

    for sample_folder in sorted(sample_folders):
        if verbose:
            print(f"\nCleaning {sample_folder.name}...")

        for item in sample_folder.iterdir():
            if item.is_dir() and is_generated_formatted_dir(item):
                shutil.rmtree(item)
                deleted_dirs += 1
                if verbose:
                    print(f"  ✗ Deleted dir: {item.name}/")

        image_folders = [item for item in sample_folder.iterdir() if item.is_dir()]

        for image_folder in sorted(image_folders):
            if verbose:
                print(f"  {image_folder.name}/")

            for item in image_folder.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                    deleted_dirs += 1
                    if verbose:
                        print(f"    ✗ Deleted dir: {item.name}/")

            for item in image_folder.iterdir():
                if item.is_file():
                    keep_tif = (
                        item.suffix.lower() == ".tif"
                        and not item.name.startswith("processed_")
                        and item.name != "original_image.tif"
                        and item.name != "cellpose_prob_map.tif"
                    )

                    if keep_tif:
                        kept_tifs += 1
                        if verbose:
                            print(f"    ✓ Keeping: {item.name}")
                        continue

                    item.unlink()
                    deleted_files += 1
                    if verbose:
                        print(f"    ✗ Deleted: {item.name}")

        for item in sample_folder.iterdir():
            if item.is_file():
                item.unlink()
                deleted_files += 1
                if verbose:
                    print(f"  ✗ Deleted: {item.name}")

    for item in base_path_obj.iterdir():
        if item.is_file():
            item.unlink()
            deleted_files += 1
            if verbose:
                print(f"\n✗ Deleted base-level: {item.name}")

    for item in base_path_obj.iterdir():
        if item.is_dir() and is_generated_formatted_dir(item):
            shutil.rmtree(item)
            deleted_dirs += 1
            if verbose:
                print(f"\n✗ Deleted base-level dir: {item.name}/")

    print("\n" + "=" * 80)
    print("CLEANUP COMPLETE")
    print("=" * 80)
    print(f"✓ Kept {kept_tifs} .tif files")
    print(f"✗ Deleted {deleted_files} files")
    print(f"✗ Deleted {deleted_dirs} directories")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Delete generated outputs while keeping original .tif files."
    )
    parser.add_argument(
        "--base-path",
        default=BASE_PATH,
        help="Donor folder to clean. Defaults to BASE_PATH near the top of this file.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce cleanup logging")
    args = parser.parse_args()

    cleanup_non_tif_files(args.base_path, verbose=not args.quiet)


if __name__ == "__main__":
    main()
