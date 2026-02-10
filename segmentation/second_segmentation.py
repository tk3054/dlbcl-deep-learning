#!/usr/bin/env python3
"""
Run Cellpose on raw crop TIFs to get tighter per-cell masks.

Example:
  python second_segmentation.py \
    --input-dir "/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241/sample1/1/raw_crops"

If --input-dir is omitted, the script will use DEFAULT_INPUT_DIR.
"""

import argparse
from pathlib import Path

import numpy as np

# EDIT THESE DEFAULTS AS NEEDED
DEFAULT_INPUT_DIR = "/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241/sample1/1/raw_crops"
DEFAULT_OUTPUT_DIR_NAME = "second_segmentation"

# Cellpose parameters (edit in code, not CLI flags)
MODEL_TYPE = "cyto2"
DIAMETER = 50  # int or None for auto
FLOW_THRESHOLD = 0.2
CELLPROB_THRESHOLD = -2.0
USE_GPU = True
RESCALE = None  # None lets Cellpose decide
CHANNELS = [0, 0]  # grayscale
MIN_SIZE = None
MAX_SIZE = None

# Output / visualization
SAVE_OVERLAY = True
OUTLINE_COLOR = (255, 0, 0)  # red

# Debug visualization for a single image (set True + path)
DEBUG_VIS = True
DEBUG_IMAGE_PATH = "/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241/sample1/1/raw_crops/cell_01_raw.tif"
DEBUG_OUTPUT_DIR = "/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241/sample1/1/debug"

# Simple normalization (improves Cellpose on raw grayscale)
ENABLE_SIMPLE_NORM = True
NORM_LOW_PCT = 1
NORM_HIGH_PCT = 99

try:
    from cellpose import models, io, plot
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    models = None



def load_image(image_path: Path) -> np.ndarray:
    """Load an image as numpy array, preserving bit depth when possible."""
    import cv2
    img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        from PIL import Image
        pil_img = Image.open(image_path)
        if pil_img.mode == "RGB":
            return np.array(pil_img.convert("L"))
        return np.array(pil_img)
    return img


def simple_normalize(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    low = np.percentile(img, NORM_LOW_PCT)
    high = np.percentile(img, NORM_HIGH_PCT)
    if high <= low:
        return img
    img = (img - low) / (high - low)
    img = np.clip(img, 0.0, 1.0)
    return img


def to_uint8(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    low = np.percentile(img, 1)
    high = np.percentile(img, 99)
    if high > low:
        img = (img - low) / (high - low)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def segment_one_image(model, image: np.ndarray, diameter, flow_threshold, cellprob_threshold) -> np.ndarray:
    img_input = simple_normalize(image) if ENABLE_SIMPLE_NORM else image
    result = model.eval(
        x=img_input,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=CHANNELS,
        rescale=RESCALE,
        resample=True,
    )
    try:
        masks, flows, styles, diams = result
    except ValueError:
        masks, flows, styles = result
    return masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cellpose on raw crop TIFs.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Path to folder containing raw crop .tif files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder for masks (default: sibling 'second_segmentation').",
    )
    # Cellpose parameters are set as constants above (not CLI flags)
    return parser.parse_args()


def main() -> int:
    if not CELLPOSE_AVAILABLE:
        print("❌ Cellpose not installed. Install with: pip install cellpose")
        return 1

    if DEBUG_VIS and DEBUG_IMAGE_PATH:
        import matplotlib.pyplot as plt

        debug_path = Path(DEBUG_IMAGE_PATH)
        if not debug_path.is_file():
            print(f"❌ DEBUG_IMAGE_PATH must be a file, got: {debug_path}")
            return 1

        img = io.imread(str(debug_path))
        if img is None:
            print(f"❌ Could not read image: {debug_path}")
            return 1

        # Coerce to 2D grayscale to avoid channel deprecation warnings
        if img.ndim > 2:
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                img = img[..., 0]
            elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = img[0]
            else:
                img = np.squeeze(img)
                if img.ndim > 2:
                    img = img[..., 0]

        model = models.CellposeModel(gpu=USE_GPU, model_type=MODEL_TYPE)
        result = model.eval(
            img,
            diameter=0 if DIAMETER is None else DIAMETER,
            channels=None,
            normalize=True,
            cellprob_threshold=CELLPROB_THRESHOLD,
            flow_threshold=FLOW_THRESHOLD,
        )
        try:
            masks, flows, styles, diams = result
        except ValueError:
            masks, flows, styles = result

        dP = flows[0]
        cellprob = flows[1]
        if hasattr(cellprob, "ndim") and cellprob.ndim == 3:
            # Handle unexpected (2, H, W) shapes by collapsing channel axis
            cellprob = np.max(cellprob, axis=0)
        flow_mag = np.sqrt(dP[0] ** 2 + dP[1] ** 2)
        flow_rgb = plot.dx_to_circ(dP)

        debug_out_dir = Path(DEBUG_OUTPUT_DIR)
        debug_out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(14, 4))
        plt.subplot(1, 4, 1)
        plt.title("image")
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("cellprob")
        plt.imshow(cellprob, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("flow magnitude")
        plt.imshow(flow_mag, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("flow (dx_to_circ)")
        plt.imshow(flow_rgb)
        plt.axis("off")

        plt.tight_layout()
        panel_path = debug_out_dir / "debug_panel.png"
        plt.savefig(str(panel_path), dpi=200, bbox_inches="tight")
        plt.close()

        step = max(1, img.shape[0] // 40)
        Y, X = np.mgrid[0:img.shape[0]:step, 0:img.shape[1]:step]
        U = dP[1, ::step, ::step]
        V = -dP[0, ::step, ::step]

        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap="gray")
        plt.quiver(X, Y, U, V, color="yellow", angles="xy", scale_units="xy", scale=1)
        plt.title("flow vectors (quiver)")
        plt.axis("off")
        quiver_path = debug_out_dir / "debug_quiver.png"
        plt.savefig(str(quiver_path), dpi=200, bbox_inches="tight")
        plt.close()
        return 0

    args = parse_args()

    input_dir_value = args.input_dir or DEFAULT_INPUT_DIR
    if not input_dir_value:
        print("❌ Input dir is required.")
        return 1
    input_dir = Path(input_dir_value)
    if not input_dir.exists():
        print(f"❌ Input dir not found: {input_dir}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / "second_segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    diameter = DIAMETER

    print("Initializing Cellpose model...")
    model = models.CellposeModel(gpu=USE_GPU, model_type=MODEL_TYPE)
    print(f"  model_type={MODEL_TYPE}, gpu={USE_GPU}")
    print(f"  diameter={'auto' if diameter is None else diameter}, "
          f"flow_threshold={FLOW_THRESHOLD}, cellprob_threshold={CELLPROB_THRESHOLD}")

    from skimage import io as skio
    from skimage.segmentation import find_boundaries
    tif_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {'.tif', '.tiff'}]
    )
    if not tif_files:
        print(f"❌ No .tif files found in {input_dir}")
        return 1

    processed = 0
    empty = 0

    for tif_path in tif_files:
        image = load_image(tif_path)
        masks = segment_one_image(
            model,
            image,
            diameter=diameter,
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=CELLPROB_THRESHOLD,
        )
        out_name = tif_path.stem + "_mask.tif"
        out_path = output_dir / out_name
        if masks is None or masks.size == 0:
            empty += 1
            masks = np.zeros_like(image, dtype=np.uint16)
        skio.imsave(str(out_path), masks.astype(np.uint16), check_contrast=False)

        if SAVE_OVERLAY:
            outline = find_boundaries(masks, mode="outer")
            base = to_uint8(image)
            if base.ndim == 2:
                overlay = np.stack([base, base, base], axis=-1)
            else:
                overlay = base[..., :3].copy()
            overlay[outline] = OUTLINE_COLOR
            overlay_path = output_dir / f"{tif_path.stem}_overlay.png"
            skio.imsave(str(overlay_path), overlay, check_contrast=False)
        processed += 1

    print(f"✅ Done. Processed {processed} crops.")
    if empty:
        print(f"⚠️  {empty} crops had no detected masks (saved empty masks).")
    print(f"Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
