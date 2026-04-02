#!/usr/bin/env python3
"""
Iterate through all patients in DLBCL_processed and generate smoothing
visualizations for every sample/image folder.
"""

from pathlib import Path
import argparse

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter


DEFAULT_ROOT = Path("/mnt/HDD16TB/LanceKam_Lab/Daizong/Project/DLBCL/DLBCL/DLBCL_processed")
DEFAULT_SOURCE_IMAGE = "processed_Actin-FITC.tif"
DEFAULT_ROI_DIR = "cell_rois"
DEFAULT_OUTPUT_DIR = "visualize smoothing"
DEFAULT_SIGMA_VALUES = [1, 2, 3, 4, 5]
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


def get_centered_crop_bounds(center_y, center_x, target_size):
    half_size = int(target_size) // 2
    y_start = int(round(center_y)) - half_size
    x_start = int(round(center_x)) - half_size
    y_end = y_start + int(target_size)
    x_end = x_start + int(target_size)
    return y_start, y_end, x_start, x_end


def normalize_to_uint8(image):
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    elif max_value > 0:
        image = image / max_value

    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def make_labeled_tile(image_array, label, tile_size=220, label_height=24):
    image = Image.fromarray(normalize_to_uint8(image_array))
    image = ImageOps.contain(image, (tile_size, tile_size))

    canvas = Image.new("L", (tile_size, tile_size + label_height), color=255)
    x_offset = (tile_size - image.width) // 2
    canvas.paste(image, (x_offset, 0))

    draw = ImageDraw.Draw(canvas)
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    draw.text(((tile_size - text_width) // 2, tile_size + 4), label, fill=0)
    return canvas


def make_operator_tile(symbol, tile_size=90, label_height=24):
    canvas = Image.new("RGB", (tile_size, tile_size + label_height), color="white")
    draw = ImageDraw.Draw(canvas)
    text_bbox = draw.textbbox((0, 0), symbol)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.text(
        ((tile_size - text_width) // 2, (tile_size - text_height) // 2 - 4),
        symbol,
        fill="black",
    )
    return canvas


def make_crop_box_tile(image_array, crop_bounds, label, tile_size=220, label_height=24):
    image = Image.fromarray(normalize_to_uint8(image_array)).convert("RGB")
    original_width, original_height = image.size
    image = ImageOps.contain(image, (tile_size, tile_size))

    scale_x = image.width / original_width
    scale_y = image.height / original_height
    y_start, y_end, x_start, x_end = crop_bounds

    draw = ImageDraw.Draw(image)
    left = max(0, x_start) * scale_x
    top = max(0, y_start) * scale_y
    right = min(original_width, x_end) * scale_x
    bottom = min(original_height, y_end) * scale_y
    draw.rectangle([left, top, right, bottom], outline=(255, 64, 64), width=3)

    canvas = Image.new("RGB", (tile_size, tile_size + label_height), color="white")
    x_offset = (tile_size - image.width) // 2
    canvas.paste(image, (x_offset, 0))

    draw = ImageDraw.Draw(canvas)
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    draw.text(((tile_size - text_width) // 2, tile_size + 4), label, fill="black")
    return canvas


def save_process_schematic(
    cell_dir,
    source_img,
    soft_mask,
    weighted_image,
    crop_bounds,
    final_crop,
    sigma,
    padded,
):
    tiles = [
        make_labeled_tile(soft_mask, f"blurred ROI (sigma={sigma:g})"),
        make_operator_tile("x"),
        make_labeled_tile(source_img, "processed Actin"),
        make_operator_tile("="),
        make_crop_box_tile(weighted_image, crop_bounds, "weighted image + 224 crop box"),
        make_operator_tile("->"),
        make_labeled_tile(final_crop, f"final 224x224 crop{' (padded)' if padded else ''}"),
    ]

    padding = 12
    tile_widths = [tile.width for tile in tiles]
    tile_heights = [tile.height for tile in tiles]
    canvas_width = padding + sum(tile_widths) + padding * len(tiles)
    canvas_height = max(tile_heights) + (2 * padding)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")

    x_offset = padding
    for tile in tiles:
        y_offset = padding + (max(tile_heights) - tile.height) // 2
        canvas.paste(tile, (x_offset, y_offset))
        x_offset += tile.width + padding

    sigma_label = str(int(sigma)) if float(sigma).is_integer() else str(sigma).replace(".", "p")
    canvas.save(cell_dir / f"sigma_{sigma_label}.png")


def iter_patient_dirs(root_dir):
    for patient_dir in sorted(root_dir.iterdir()):
        if patient_dir.is_dir():
            yield patient_dir


def iter_sample_dirs(patient_dir):
    for sample_dir in sorted(patient_dir.glob("sample*")):
        if sample_dir.is_dir():
            yield sample_dir


def iter_image_dirs(sample_dir):
    for image_dir in sorted(sample_dir.iterdir(), key=lambda path: (not path.name.isdigit(), path.name)):
        if image_dir.is_dir() and image_dir.name.isdigit():
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
        for cell_index, roi_file in enumerate(roi_files, start=1):
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
                crop_bounds = get_centered_crop_bounds(center_y, center_x, target_size)
                padded = (
                    crop_bounds[0] < 0
                    or crop_bounds[2] < 0
                    or crop_bounds[1] > source_img.shape[0]
                    or crop_bounds[3] > source_img.shape[1]
                )

                final_crop = extract_centered_crop(
                    weighted_image,
                    center_y=center_y,
                    center_x=center_x,
                    target_size=target_size,
                    fill_value=0.0,
                )

                save_process_schematic(
                    cell_dir=cell_dir,
                    source_img=source_img,
                    soft_mask=soft_mask,
                    weighted_image=weighted_image,
                    crop_bounds=crop_bounds,
                    final_crop=final_crop,
                    sigma=sigma,
                    padded=padded,
                )

            processed_any = True

        if not processed_any:
            append_log(log_path, patient_name, sample_name, image_number, "no usable ROI files")
        return processed_any
    except Exception as exc:
        append_log(log_path, patient_name, sample_name, image_number, str(exc))
        return False


def main():
    parser = argparse.ArgumentParser(description="Iterate through DLBCL_processed and create smoothing schematics.")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT), help="DLBCL_processed root directory")
    parser.add_argument("--source-image", default=DEFAULT_SOURCE_IMAGE, help="Processed Actin image filename")
    parser.add_argument("--roi-dir-name", default=DEFAULT_ROI_DIR, help="ROI directory name")
    parser.add_argument("--output-dir-name", default=DEFAULT_OUTPUT_DIR, help="Visualization output directory name")
    parser.add_argument("--sigma-start", type=int, default=DEFAULT_SIGMA_VALUES[0], help="First sigma value")
    parser.add_argument("--sigma-end", type=int, default=DEFAULT_SIGMA_VALUES[-1], help="Last sigma value")
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE, help="Fixed centered crop size")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    sigma_values = list(range(args.sigma_start, args.sigma_end + 1))
    log_path = root_dir / "log.txt"

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    total_images = 0
    processed_images = 0

    for patient_dir in iter_patient_dirs(root_dir):
        for sample_dir in iter_sample_dirs(patient_dir):
            for image_dir in iter_image_dirs(sample_dir):
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
