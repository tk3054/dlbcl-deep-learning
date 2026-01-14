#!/usr/bin/env python3
"""
Compare Multiple Edge Softening Methods
Creates side-by-side comparison images showing different edge softening techniques.

Uses the EDGE_METHODS registry from process_image.soften_edges for all softening algorithms.
This module focuses on orchestrating comparisons and generating visualizations.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io
from process_image.soften_edges import EDGE_METHODS, create_beta_soft_mask


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_edge_methods(image_path, roi_mask_path, output_path, padding=10,
                        cellpose_prob_map=None):
    """
    Create a comparison image showing all edge softening methods.

    Parameters:
    -----------
    image_path : str or Path
        Path to the fluorescence image
    roi_mask_path : str or Path
        Path to the ROI mask
    output_path : str or Path
        Path to save the comparison image
    padding : int
        Padding around the bounding box
    cellpose_prob_map : numpy.ndarray, optional
        Cellpose probability map (flows[2]). If provided, beta transformation
        methods will be included in the comparison.
    """
    # Load image and mask
    image = np.array(Image.open(image_path))
    roi_mask = np.array(Image.open(roi_mask_path).convert('L'))

    # Find bounding box
    coords = np.where(roi_mask > 0)
    if len(coords[0]) == 0:
        print(f"Empty mask in {roi_mask_path}")
        return None

    y_min, y_max = coords[0].min(), coords[0].max() + 1
    x_min, x_max = coords[1].min(), coords[1].max() + 1

    # Add padding
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)

    # Crop image and mask
    cropped_img = image[y_min:y_max, x_min:x_max]
    cropped_mask = roi_mask[y_min:y_max, x_min:x_max]

    # Crop probability map if provided
    cropped_prob_map = None
    if cellpose_prob_map is not None:
        cropped_prob_map = cellpose_prob_map[y_min:y_max, x_min:x_max]

    # Normalize for visualization (16-bit to 8-bit)
    def normalize_to_8bit(img):
        if img.dtype == np.uint16 or img.max() > 255:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                return ((img.astype(float) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return img.astype(np.uint8)

    # Apply all edge methods from registry
    results = {}
    for method_name, method_func in EDGE_METHODS.items():
        # Get the soft mask using this method
        soft_mask = method_func(cropped_mask)

        # Apply to image - convert to float first to preserve precision
        masked_img = cropped_img.astype(float) * soft_mask

        # Store results
        results[method_name] = {
            'image': normalize_to_8bit(masked_img),
            'mask': (soft_mask * 255).astype(np.uint8)
        }

    # Add beta transformation methods if probability map is available
    if cropped_prob_map is not None:
        beta_values = [
            ('Beta β=0.5', 0.5),
            ('Beta β=0.7', 0.7),
            ('Beta β=0.9', 0.9),
        ]

        for method_name, beta in beta_values:
            try:
                # Get the soft mask using beta transformation
                soft_mask = create_beta_soft_mask(
                    cropped_mask,
                    beta=beta,
                    cellpose_prob_map=cropped_prob_map
                )

                # Apply to image
                masked_img = cropped_img.astype(float) * soft_mask

                # Store results
                results[method_name] = {
                    'image': normalize_to_8bit(masked_img),
                    'mask': (soft_mask * 255).astype(np.uint8)
                }
            except Exception as e:
                print(f"  ⚠️  Failed to compute {method_name}: {e}")
                continue

    # Create comparison figure
    num_methods = len(results) + 1  # +1 for original
    ncols = min(4, num_methods)  # Max 4 columns
    nrows = ((num_methods - 1) // ncols + 1) * 2  # 2 rows per method set

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    # Handle single row case
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    # Row 1: Show original
    axes[0, 0].imshow(cropped_img, cmap='gray')
    axes[0, 0].set_title('Original Crop', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Fill in the methods (images)
    for idx, (method_name, result) in enumerate(results.items(), start=1):
        row = (idx // ncols) * 2
        col = idx % ncols

        axes[row, col].imshow(result['image'], cmap='gray')
        axes[row, col].set_title(method_name, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')

    # Next row: Show masks
    axes[1, 0].imshow(cropped_img, cmap='gray', alpha=0.5)
    axes[1, 0].imshow((cropped_mask > 0).astype(np.uint8) * 255, cmap='Reds', alpha=0.5)
    axes[1, 0].set_title('Original + ROI Overlay', fontsize=10)
    axes[1, 0].axis('off')

    for idx, (method_name, result) in enumerate(results.items(), start=1):
        row = (idx // ncols) * 2 + 1
        col = idx % ncols

        axes[row, col].imshow(result['mask'], cmap='hot', vmin=0, vmax=255)
        axes[row, col].set_title(f'{method_name} (Mask)', fontsize=10)
        axes[row, col].axis('off')

    # Hide unused subplots
    for i in range(nrows):
        for j in range(ncols):
            if i == 0 and j >= len(results) + 1:
                axes[i, j].axis('off')
            elif i == 1 and j >= len(results) + 1:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comparison to: {output_path}")

    return results


def compare_multiple_cells(sample_folder, image_number, base_path, num_cells=3):
    """
    Create comparison images for multiple cells from a sample.

    Parameters:
    -----------
    sample_folder : str
        Sample folder name (e.g., "sample2")
    image_number : str
        Image number (e.g., "1")
    base_path : str
        Base directory path
    num_cells : int
        Number of cells to compare (default: 3)
    """
    base_dir = Path(base_path) / sample_folder / image_number
    roi_dir = base_dir / "cell_rois"

    # Find the actin image
    actin_candidates = ['Actin-FITC.tif', 'processed_Actin-FITC.tif']
    actin_path = None
    for candidate in actin_candidates:
        candidate_path = base_dir / candidate
        if candidate_path.exists():
            actin_path = candidate_path
            break

    if actin_path is None:
        print(f"✗ Could not find Actin-FITC.tif in {base_dir}")
        return

    if not roi_dir.exists():
        print(f"✗ ROI directory not found: {roi_dir}")
        return

    # Try to load Cellpose probability map
    prob_map_path = base_dir / "cellpose_prob_map.tif"
    cellpose_prob_map = None
    if prob_map_path.exists():
        try:
            cellpose_prob_map = io.imread(str(prob_map_path))
            print(f"✓ Loaded Cellpose probability map from {prob_map_path.name}")
            print("  → Beta transformation methods will be included in comparison")
        except Exception as e:
            print(f"⚠️  Failed to load probability map: {e}")
            cellpose_prob_map = None
    else:
        print(f"ℹ️  No Cellpose probability map found ({prob_map_path.name})")
        print("  → Beta transformation will not be included in comparison")
        print("  → To include beta methods, modify segment_cells_cellpose.py to save flows[2]")

    # Get ROI files
    roi_files = sorted(roi_dir.glob("*.tif"))[:num_cells]

    if len(roi_files) == 0:
        print(f"✗ No ROI files found in {roi_dir}")
        return

    # Count methods
    num_base_methods = len(EDGE_METHODS)
    num_beta_methods = 3 if cellpose_prob_map is not None else 0
    total_methods = num_base_methods + num_beta_methods

    print(f"\nCreating edge comparison for {len(roi_files)} cells from {sample_folder}/{image_number}")
    print(f"Using source image: {actin_path.name}")
    print(f"Comparing {total_methods} methods:")
    print(f"  • {num_base_methods} standard methods: {', '.join(EDGE_METHODS.keys())}")
    if num_beta_methods > 0:
        print(f"  • {num_beta_methods} beta methods: Beta β=0.5, Beta β=0.7, Beta β=0.9")
    print()

    # Create output directory
    output_dir = base_dir / "edge_comparisons"
    output_dir.mkdir(exist_ok=True)

    # Process each cell
    for idx, roi_file in enumerate(roi_files, 1):
        output_filename = f"comparison_{roi_file.stem}.png"
        output_path = output_dir / output_filename

        print(f"[{idx}/{len(roi_files)}] Processing {roi_file.name}...")

        compare_edge_methods(
            image_path=actin_path,
            roi_mask_path=roi_file,
            output_path=output_path,
            padding=10,
            cellpose_prob_map=cellpose_prob_map
        )

    print(f"\n✓ All comparisons saved to: {output_dir}")


if __name__ == "__main__":
    # Configuration
    SAMPLE_FOLDER = "sample2"
    IMAGE_NUMBER = "1"
    BASE_PATH = "/Users/taeeonkong/Desktop/113614(FITC-500ms)"
    NUM_CELLS = 5  # Number of cells to compare

    # To add new edge softening methods, edit the EDGE_METHODS dictionary at the top!

    compare_multiple_cells(
        sample_folder=SAMPLE_FOLDER,
        image_number=IMAGE_NUMBER,
        base_path=BASE_PATH,
        num_cells=NUM_CELLS
    )
