#!/usr/bin/env python3
"""
Visualize ROIs on JPG Images
Creates visualizations showing both original and shrunk ROIs overlaid on JPG images.

Usage:
    python visualize_rois_on_jpg.py
    (Edit SAMPLE_FOLDER and IMAGE_NUMBER below to change inputs)
"""

import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from skimage import morphology


def create_roi_visualization(sample_folder, image_number, base_path,
                             shrink_pixels=3, verbose=True):
    """
    Create visualization of ROIs (original and shrunk) overlaid on JPG images.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        shrink_pixels: Number of pixels to shrink ROIs (default: 3)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'output_files': List of created visualization files
    """
    from scipy import ndimage as ndi

    # Build paths
    base_dir = Path(f"{base_path}/{sample_folder}/{image_number}")
    roi_dir = base_dir / "cell_rois"
    output_dir = base_dir / "roi_visualizations"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Check if ROI directory exists
    if not roi_dir.exists():
        return {
            'success': False,
            'error': f"ROI directory not found: {roi_dir}",
            'output_files': []
        }

    # Find all JPG files (raw channel images)
    jpg_files = list(base_dir.glob("*_raw.jpg"))

    if not jpg_files:
        return {
            'success': False,
            'error': f"No *_raw.jpg files found in {base_dir}",
            'output_files': []
        }

    if verbose:
        print(f"Creating ROI visualizations for {len(jpg_files)} images...")

    # Load all ROI masks
    roi_files = sorted(roi_dir.glob("*.tif"))

    if not roi_files:
        return {
            'success': False,
            'error': f"No ROI .tif files found in {roi_dir}",
            'output_files': []
        }

    output_files = []

    # Process each JPG file
    for jpg_file in jpg_files:
        if verbose:
            print(f"  Processing {jpg_file.name}...")

        # Load the JPG image
        img = cv2.imread(str(jpg_file))
        if img is None:
            img = np.array(Image.open(jpg_file))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Create a copy for overlay
        overlay_img = img.copy()

        # Create combined masks for original and shrunk ROIs
        original_combined = np.zeros(img.shape[:2], dtype=np.uint8)
        shrunk_combined = np.zeros(img.shape[:2], dtype=np.uint8)

        # Process each ROI
        for i, roi_file in enumerate(roi_files):
            # Load original ROI mask
            roi_mask = np.array(Image.open(roi_file).convert('L'))

            # Add to original combined mask
            original_combined[roi_mask > 0] = i + 1

            # Create shrunk version
            if shrink_pixels > 0:
                binary_mask = roi_mask > 0

                # Create circular structuring element
                y, x = np.ogrid[-shrink_pixels:shrink_pixels+1, -shrink_pixels:shrink_pixels+1]
                structuring_element = x**2 + y**2 <= shrink_pixels**2

                # Apply erosion
                eroded_mask = ndi.binary_erosion(binary_mask, structure=structuring_element)

                # Add to shrunk combined mask (skip if disappeared)
                if np.sum(eroded_mask) > 0:
                    shrunk_combined[eroded_mask] = i + 1

        # Extract boundaries
        original_boundaries = _extract_boundaries(original_combined)
        shrunk_boundaries = _extract_boundaries(shrunk_combined)

        # Draw boundaries on overlay image
        # Original ROIs in RED
        overlay_img[original_boundaries] = [0, 0, 255]  # BGR format: RED
        # Shrunk ROIs in GREEN
        overlay_img[shrunk_boundaries] = [0, 255, 0]  # BGR format: GREEN

        # Add cell number labels
        for i in range(len(roi_files)):
            # Find centroid of this ROI
            roi_pixels = np.where(original_combined == i + 1)
            if len(roi_pixels[0]) > 0:
                centroid_y = int(np.mean(roi_pixels[0]))
                centroid_x = int(np.mean(roi_pixels[1]))

                # Position text above the ROI
                text_y = max(centroid_y - 15, 20)  # Place above, but not off-screen
                text_x = centroid_x

                # Draw cell number with thick yellow and black in middle
                cell_num = str(i + 1)
                # Yellow text (thicker)
                cv2.putText(overlay_img, cell_num, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 3, cv2.LINE_AA)
                # Black text (thinner, on top)
                cv2.putText(overlay_img, cell_num, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Add text label
        label_text = f"Red=Original ({len(roi_files)} ROIs), Green=Shrunk ({shrink_pixels}px)"
        cv2.putText(overlay_img, label_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Save overlay image
        output_filename = jpg_file.stem + "_roi_overlay.jpg"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), overlay_img)

        output_files.append(str(output_path))

        if verbose:
            print(f"    ✓ Saved: {output_filename}")

    if verbose:
        print(f"  ✓ Created {len(output_files)} visualization files")

    return {
        'success': True,
        'output_files': output_files,
        'output_dir': str(output_dir)
    }


def visualize_roi_all_channels(sample_folder, image_number, base_path,
                               shrink_pixels=3, verbose=True):
    """
    Create a multi-panel comparison showing ROI overlays on all channel JPGs.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        shrink_pixels: Number of pixels to shrink ROIs (default: 3)
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
            - 'output_file': Path to created comparison image
    """
    from scipy import ndimage as ndi
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Build paths
    base_dir = Path(f"{base_path}/{sample_folder}/{image_number}")
    roi_dir = base_dir / "cell_rois"

    # Check if ROI directory exists
    if not roi_dir.exists():
        return {
            'success': False,
            'error': f"ROI directory not found: {roi_dir}",
            'output_file': None
        }

    # Find all JPG files (raw channel images)
    jpg_files = sorted(base_dir.glob("*_raw.jpg"))

    if not jpg_files:
        return {
            'success': False,
            'error': f"No *_raw.jpg files found in {base_dir}",
            'output_file': None
        }

    if verbose:
        print(f"Creating multi-channel ROI comparison ({len(jpg_files)} channels)...")

    # Load all ROI masks
    roi_files = sorted(roi_dir.glob("*.tif"))

    if not roi_files:
        return {
            'success': False,
            'error': f"No ROI .tif files found in {roi_dir}",
            'output_file': None
        }

    # Create combined masks for original and shrunk ROIs
    # (We'll reuse these for all channels)
    first_img = cv2.imread(str(jpg_files[0]))
    if first_img is None:
        first_img = np.array(Image.open(jpg_files[0]))

    original_combined = np.zeros(first_img.shape[:2], dtype=np.uint8)
    shrunk_combined = np.zeros(first_img.shape[:2], dtype=np.uint8)

    # Process each ROI
    for i, roi_file in enumerate(roi_files):
        # Load original ROI mask
        roi_mask = np.array(Image.open(roi_file).convert('L'))

        # Add to original combined mask
        original_combined[roi_mask > 0] = i + 1

        # Create shrunk version
        if shrink_pixels > 0:
            binary_mask = roi_mask > 0

            # Create circular structuring element
            y, x = np.ogrid[-shrink_pixels:shrink_pixels+1, -shrink_pixels:shrink_pixels+1]
            structuring_element = x**2 + y**2 <= shrink_pixels**2

            # Apply erosion
            eroded_mask = ndi.binary_erosion(binary_mask, structure=structuring_element)

            # Add to shrunk combined mask (skip if disappeared)
            if np.sum(eroded_mask) > 0:
                shrunk_combined[eroded_mask] = i + 1

    # Extract boundaries (once, reused for all channels)
    original_boundaries = _extract_boundaries(original_combined)
    shrunk_boundaries = _extract_boundaries(shrunk_combined)

    # Calculate grid layout
    num_channels = len(jpg_files)
    num_cols = min(3, num_channels)  # Max 3 columns
    num_rows = (num_channels + num_cols - 1) // num_cols

    # Create figure
    _, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 6*num_rows))
    if num_channels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Process each channel
    for idx, jpg_file in enumerate(jpg_files):
        if verbose:
            print(f"  Processing {jpg_file.name}...")

        # Load the JPG image
        img = cv2.imread(str(jpg_file))
        if img is None:
            img = np.array(Image.open(jpg_file))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Create overlay
        overlay_img = img.copy()

        # Draw boundaries
        # Original ROIs in RED
        overlay_img[original_boundaries] = [0, 0, 255]  # BGR format: RED
        # Shrunk ROIs in GREEN
        overlay_img[shrunk_boundaries] = [0, 255, 0]  # BGR format: GREEN

        # Add cell number labels
        for i in range(len(roi_files)):
            # Find centroid of this ROI
            roi_pixels = np.where(original_combined == i + 1)
            if len(roi_pixels[0]) > 0:
                centroid_y = int(np.mean(roi_pixels[0]))
                centroid_x = int(np.mean(roi_pixels[1]))

                # Position text above the ROI
                text_y = max(centroid_y - 15, 20)  # Place above, but not off-screen
                text_x = centroid_x

                # Draw cell number with thick yellow and black in middle
                cell_num = str(i + 1)
                # Yellow text (thicker)
                cv2.putText(overlay_img, cell_num, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 3, cv2.LINE_AA)
                # Black text (thinner, on top)
                cv2.putText(overlay_img, cell_num, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Convert BGR to RGB for matplotlib
        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

        # Extract channel name from filename
        # Example: CLLSaSa_07292025_1to10_40min_1_01[CAR]_Actin-FITC_raw.jpg -> Actin-FITC
        channel_name = jpg_file.stem
        if "_raw" in channel_name:
            # Extract the part before _raw
            parts = channel_name.split("_")
            # Find the channel name (usually second to last part before _raw)
            for i, part in enumerate(parts):
                if part == "raw" and i > 0:
                    channel_name = parts[i-1]
                    break

        # Plot on subplot
        axes[idx].imshow(overlay_rgb)
        axes[idx].set_title(f'{channel_name}\n(Red=Original, Green=Shrunk)', fontsize=12)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_channels, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'ROI Comparison - {sample_folder}/{image_number}\n({len(roi_files)} ROIs, shrink={shrink_pixels}px)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path = base_dir / "roi_shrink_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  ✓ Saved comparison: {output_path.name}")

    return {
        'success': True,
        'output_file': str(output_path)
    }


def _extract_boundaries(combined_mask):
    """
    Extract boundaries from a combined mask with multiple labels.

    Args:
        combined_mask: Labeled mask (0=background, 1,2,3...=ROI labels)

    Returns:
        Boolean array indicating boundary pixels
    """
    boundaries = np.zeros_like(combined_mask, dtype=bool)

    # Get unique labels (excluding background 0)
    labels = np.unique(combined_mask)
    labels = labels[labels > 0]

    # Extract boundary for each label
    for label in labels:
        mask = (combined_mask == label)
        eroded = morphology.binary_erosion(mask)
        boundary = mask & ~eroded
        boundaries |= boundary

    return boundaries


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    """Run ROI visualization with configuration"""

    # Configuration
    SAMPLE_FOLDER = "sample1"
    IMAGE_NUMBER = "1[CAR]"
    BASE_PATH = "/Users/taeeonkong/Desktop/2025 Fall Images/09-26-2025 DLBCL"
    SHRINK_PIXELS = 3

    print("Creating individual ROI visualizations...")
    result = create_roi_visualization(
        sample_folder=SAMPLE_FOLDER,
        image_number=IMAGE_NUMBER,
        base_path=BASE_PATH,
        shrink_pixels=SHRINK_PIXELS,
        verbose=True
    )

    if result['success']:
        print(f"\n✓ Successfully created {len(result['output_files'])} visualizations")
        print(f"Output directory: {result['output_dir']}")
    else:
        print(f"\n✗ Error: {result['error']}")

    print("\n" + "="*60)
    print("Creating multi-channel comparison...")
    result_comparison = visualize_roi_all_channels(
        sample_folder=SAMPLE_FOLDER,
        image_number=IMAGE_NUMBER,
        base_path=BASE_PATH,
        shrink_pixels=SHRINK_PIXELS,
        verbose=True
    )

    if result_comparison['success']:
        print(f"\n✓ Successfully created comparison: {result_comparison['output_file']}")
    else:
        print(f"\n✗ Error: {result_comparison['error']}")


if __name__ == "__main__":
    main()
