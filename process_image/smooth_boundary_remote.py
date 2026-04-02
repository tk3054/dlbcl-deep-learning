#!/usr/bin/env python3
"""
Iterate through all patients in DLBCL_processed and generate smoothed
224x224 Actin crops for every sample/image folder.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.name_builder import extract_image_number


DEFAULT_ROOT = Path("/mnt/HDD16TB/LanceKam_Lab/Daizong/Project/DLBCL/DLBCL/DLBCL_processed")
DEFAULT_SOURCE_IMAGE = "processed_Actin-FITC.tif"
DEFAULT_ROI_DIR = "cell_rois"
DEFAULT_OUTPUT_DIR = "visualize smoothing"
DEFAULT_SIGMA_VALUES = [1, 3, 5]
DEFAULT_TARGET_SIZE = 224


def append_log(log_path, patient, sample, image_number, issue):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"patient={patient}\tsample={sample}\timage={image_number}\tissue={issue}\n"
        )


def extract_centered_crop(image, center_y, center_x, target_size, fill_value=0.0):
    target_size = int(target_size)
    half_size = target_size // 2

    y_start = int(round(center_y)) - half_size
    x_start = int(round(center_x)) - half_size
    y_end = y_start + target_size
    x_end = x_start + target_size

    src_y_start = max(0, y_start)
    src_x_start = max(0, x_start)
    src_y_end = min(image.shape[0], y_end)
    src_x_end = min(image.shape[1], x_end)

    dst_y_start = src_y_start - y_start
    dst_x_start = src_x_start - x_start
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    crop = np.full((target_size, target_size), fill_value, dtype=image.dtype)
    crop[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[
        src_y_start:src_y_end, src_x_start:src_x_end
    ]
    return crop


def save_smoothed_crop(cell_dir, final_crop, sigma):
    sigma_label = str(int(sigma)) if float(sigma).is_integer() else str(sigma).replace(".", "p")
    tifffile.imwrite(cell_dir / f"sigma_{sigma_label}.tif", final_crop.astype(np.float32))


def iter_patient_dirs(root_dir):
    for patient_dir in sorted(root_dir.iterdir()):
        if patient_dir.is_dir():
            yield patient_dir


def iter_sample_dirs(patient_dir):
    for sample_dir in sorted(patient_dir.glob("sample*")):
        if sample_dir.is_dir():
            yield sample_dir


def iter_image_dirs(sample_dir):
    for image_dir in sorted(
        sample_dir.iterdir(),
        key=lambda path: (
            extract_image_number(path.name) is None,
            extract_image_number(path.name) if extract_image_number(path.name) is not None else path.name,
        ),
    ):
        if image_dir.is_dir() and extract_image_number(image_dir.name) is not None:
            yield image_dir


def process_image_dir(
    image_dir,
    patient_name,
    sample_name,
    sigma_values,
    target_size,
    source_image_name,
    roi_dir_name,
    output_dir_name,
    log_path,
):
    source_path = image_dir / source_image_name
    roi_dir = image_dir / roi_dir_name
    output_dir = image_dir / output_dir_name
    image_number = image_dir.name

    if not source_path.exists():
        append_log(log_path, patient_name, sample_name, image_number, f"missing source image: {source_image_name}")
        return False

    if not roi_dir.exists():
        append_log(log_path, patient_name, sample_name, image_number, f"missing ROI directory: {roi_dir_name}")
        return False

    try:
        source_img = tifffile.imread(source_path)
        if source_img.ndim != 2:
            raise ValueError(f"Expected 2D source image, got shape {source_img.shape}")

        roi_files = sorted(roi_dir.glob("*.tif"))
        output_dir.mkdir(exist_ok=True)

        processed_any = False
        for cell_index, roi_file in enumerate(tqdm(roi_files, desc="Cells", leave=False), start=1):
            roi_mask = tifffile.imread(roi_file)
            if roi_mask.ndim != 2:
                raise ValueError(f"Expected 2D ROI mask for {roi_file.name}, got shape {roi_mask.shape}")

            binary_mask = roi_mask > 0
            coords = np.where(binary_mask)
            if len(coords[0]) == 0:
                append_log(log_path, patient_name, sample_name, image_number, f"empty ROI mask: {roi_file.name}")
                continue

            center_y = (float(coords[0].min()) + float(coords[0].max())) / 2.0
            center_x = (float(coords[1].min()) + float(coords[1].max())) / 2.0

            cell_dir = output_dir / f"cell_{cell_index}"
            cell_dir.mkdir(exist_ok=True)

            for sigma in sigma_values:
                soft_mask = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma)
                weighted_image = source_img.astype(np.float32) * soft_mask
                final_crop = extract_centered_crop(
                    weighted_image,
                    center_y=center_y,
                    center_x=center_x,
                    target_size=target_size,
                    fill_value=0.0,
                )

                save_smoothed_crop(cell_dir=cell_dir, final_crop=final_crop, sigma=sigma)

            processed_any = True

        if not processed_any:
            append_log(log_path, patient_name, sample_name, image_number, "no usable ROI files")
        return processed_any
    except Exception as exc:
        append_log(log_path, patient_name, sample_name, image_number, str(exc))
        return False


def main():
    parser = argparse.ArgumentParser(description="Iterate through DLBCL_processed and create smoothed Actin TIFFs.")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT), help="DLBCL_processed root directory")
    parser.add_argument("--source-image", default=DEFAULT_SOURCE_IMAGE, help="Processed Actin image filename")
    parser.add_argument("--roi-dir-name", default=DEFAULT_ROI_DIR, help="ROI directory name")
    parser.add_argument("--output-dir-name", default=DEFAULT_OUTPUT_DIR, help="Output directory name")
    parser.add_argument("--sigma-values", nargs="+", type=float, default=DEFAULT_SIGMA_VALUES, help="Sigma values to save")
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE, help="Fixed centered crop size")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    sigma_values = [float(sigma) for sigma in args.sigma_values]
    log_path = root_dir / "log.txt"

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    patient_dirs = list(iter_patient_dirs(root_dir))
    print("Patients discovered:", flush=True)
    for patient_dir in patient_dirs:
        print(f"  - {patient_dir.name}", flush=True)
        sample_dirs = list(iter_sample_dirs(patient_dir))
        if sample_dirs:
            print("    Samples discovered:", flush=True)
            for sample_dir in sample_dirs:
                image_dirs = list(iter_image_dirs(sample_dir))
                image_names = ", ".join(image_dir.name for image_dir in image_dirs) or "(none)"
                print(f"      - {sample_dir.name}: images [{image_names}]", flush=True)
        else:
            print("    Samples discovered: (none)", flush=True)

    total_images = 0
    processed_images = 0

    for patient_dir in patient_dirs:
        sample_dirs = list(iter_sample_dirs(patient_dir))
        for sample_dir in sample_dirs:
            image_dirs = list(iter_image_dirs(sample_dir))
            for image_dir in image_dirs:
                print(
                    f"Processing {patient_dir.name}/{sample_dir.name}/{image_dir.name}",
                    flush=True,
                )
                total_images += 1
                ok = process_image_dir(
                    image_dir=image_dir,
                    patient_name=patient_dir.name,
                    sample_name=sample_dir.name,
                    sigma_values=sigma_values,
                    target_size=args.target_size,
                    source_image_name=args.source_image,
                    roi_dir_name=args.roi_dir_name,
                    output_dir_name=args.output_dir_name,
                    log_path=log_path,
                )
                if ok:
                    processed_images += 1

    print(f"Processed {processed_images} image folders out of {total_images}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
