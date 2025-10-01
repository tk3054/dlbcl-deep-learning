#!/usr/bin/env python3
"""
Visualize Cellpose Segmentation Results
Shows the segmentation overlaid on the original image
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.segmentation import find_boundaries


SAMPLE_FOLDER = "sample2"
IMAGE_NUMBER = "2"
BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"

base_dir = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"
roi_dir = Path(base_dir) / "cell_rois"

if roi_dir.exists():
    # Load original image
    raw_path = list(Path(base_dir).glob("*_raw.jpg"))[0]
    raw_img = np.array(Image.open(raw_path).convert('L'))

    # Load all ROI masks and combine
    roi_files = sorted(roi_dir.glob("*.tif"))
    combined_mask = np.zeros_like(raw_img)

    for i, roi_file in enumerate(roi_files):
        roi_mask = np.array(Image.open(roi_file).convert('L'))
        combined_mask[roi_mask > 0] = i + 1

    # Create visualization with boundaries
    boundaries = find_boundaries(combined_mask, mode='inner')

    # Overlay boundaries on raw image
    overlay = np.stack([raw_img] * 3, axis=-1)
    overlay[boundaries] = [255, 0, 0]  # Red boundaries

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(combined_mask, cmap='nipy_spectral')
    axes[1].set_title(f'Cellpose Segmentation ({len(roi_files)} cells)')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Boundaries Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path(base_dir) / f"{SAMPLE_FOLDER}_{IMAGE_NUMBER}_segmentation_viz.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved: {output_path}")
    print(f"✓ Segmented {len(roi_files)} cells")
else:
    print(f"⚠️  No segmentation outputs found in {roi_dir}")
