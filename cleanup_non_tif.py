#!/usr/bin/env python3
"""
Deep Clean: Delete everything except .tif files
Removes all generated files, keeping only original TIF images

Usage:
    python cleanup_non_tif.py
"""

from pathlib import Path
import shutil
BASE_PATH = '/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241'

def cleanup_non_tif_files(base_path, verbose=True):
    """
    Delete all files and folders except .tif/.TIF files

    Args:
        base_path: Base directory path
        verbose: Print progress messages
    """
    base_path_obj = Path(base_path)

    if not base_path_obj.exists():
        print(f"✗ ERROR: Base path not found: {base_path}")
        return

    print("\n" + "="*80)
    print("DEEP CLEANUP: DELETING EVERYTHING EXCEPT .TIF FILES")
    print("="*80)
    print(f"Base path: {base_path}")
    print("="*80 + "\n")

    deleted_files = 0
    deleted_dirs = 0
    kept_tifs = 0

    # Find all sample folders
    sample_folders = [item for item in base_path_obj.iterdir()
                     if item.is_dir() and item.name.lower().startswith('sample')]

    for sample_folder in sorted(sample_folders):
        if verbose:
            print(f"\nCleaning {sample_folder.name}...")

        # Delete formatted_cells folder at the sample/patient level
        formatted_cells_dir = sample_folder / "formatted_cells"
        if formatted_cells_dir.exists() and formatted_cells_dir.is_dir():
            shutil.rmtree(formatted_cells_dir)
            deleted_dirs += 1
            if verbose:
                print(f"  ✗ Deleted dir: {formatted_cells_dir.name}/")

        # Find all image subdirectories
        image_folders = [item for item in sample_folder.iterdir() if item.is_dir()]

        for image_folder in sorted(image_folders):
            if verbose:
                print(f"  {image_folder.name}/")

            # First, delete all subdirectories (recursively)
            for item in image_folder.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                    deleted_dirs += 1
                    if verbose:
                        print(f"    ✗ Deleted dir: {item.name}/")

            # Then, delete non-TIF files in the image folder itself
            for item in image_folder.iterdir():
                if item.is_file():
                    keep_tif = (
                        item.suffix.lower() == '.tif'
                        and not item.name.startswith('processed_')
                        and item.name != 'original_image.tif'
                        and item.name != 'cellpose_prob_map.tif'
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

        # Delete sample-level files (like combined_measurements.csv)
        for item in sample_folder.iterdir():
            if item.is_file():
                item.unlink()
                deleted_files += 1
                if verbose:
                    print(f"  ✗ Deleted: {item.name}")

    # Delete base-level files (like all_samples_combined.csv)
    for item in base_path_obj.iterdir():
        if item.is_file():
            item.unlink()
            deleted_files += 1
            if verbose:
                print(f"\n✗ Deleted base-level: {item.name}")

    # Delete base-level formatted_cells directory if present
    formatted_cells_dir = base_path_obj / "formatted_cells"
    if formatted_cells_dir.exists() and formatted_cells_dir.is_dir():
        shutil.rmtree(formatted_cells_dir)
        deleted_dirs += 1
        if verbose:
            print(f"\n✗ Deleted base-level dir: {formatted_cells_dir.name}/")

    print("\n" + "="*80)
    print("CLEANUP COMPLETE")
    print("="*80)
    print(f"✓ Kept {kept_tifs} .tif files")
    print(f"✗ Deleted {deleted_files} files")
    print(f"✗ Deleted {deleted_dirs} directories")
    print("="*80 + "\n")


if __name__ == "__main__":
    cleanup_non_tif_files(BASE_PATH, verbose=True)
