#!/usr/bin/env python3
"""
Find optimal Cellpose parameters without GUI
Edit SAMPLE_FOLDER and IMAGE_NUMBER below, then run: python find_cellpose_params.py
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import transforms
from cellpose.models import CellposeModel
from skimage.segmentation import find_boundaries

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

SAMPLE_FOLDER = "sample2"  # Options: "sample1", "sample2", "sample3"
IMAGE_NUMBER = "2"         # Options: "1", "2", "3", "4", etc.
BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"

# ============================================================================

def find_parameters(image_path):
    """Test different parameter combinations"""

    print(f"Loading image: {image_path}")

    # Load image (handle different formats)
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if image is None:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    # Normalize like Cellpose web tool (handle 2D grayscale)
    print("Normalizing image...")
    if len(image.shape) == 2:
        img_norm = transforms.normalize_img(image[..., np.newaxis], axis=-1).squeeze()
    else:
        img_norm = transforms.normalize_img(image, axis=-1)

    # Initialize model
    print("Loading Cellpose model...")
    try:
        model = CellposeModel(gpu=True, model_type='cyto2')
        print("Using GPU (MPS)")
    except:
        model = CellposeModel(gpu=False, model_type='cyto2')
        print("Using CPU")

    # Test configurations (added more permissive options for dim cells)
    configs = [
        {'name': 'Auto (Default)', 'diam': None, 'prob': 0.0},
        {'name': 'Small cells', 'diam': 25, 'prob': 0.0},
        {'name': 'Medium cells', 'diam': 30, 'prob': 0.0},
        {'name': 'Medium + Permissive', 'diam': 30, 'prob': -2.0},
        {'name': 'Medium + Very Permissive', 'diam': 30, 'prob': -4.0},
        {'name': 'Medium + Ultra Permissive', 'diam': 30, 'prob': -6.0},
        {'name': 'Large + Permissive', 'diam': 35, 'prob': -2.0},
        {'name': 'Large + Very Permissive', 'diam': 35, 'prob': -4.0},
    ]

    print(f"\nTesting {len(configs)} configurations...")
    print("="*60)

    # Create output directory for individual results
    output_dir = Path('cellpose_test_results')
    output_dir.mkdir(exist_ok=True)
    print(f"Saving individual results to: {output_dir}/")

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    results = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {cfg['name']}")
        print(f"    Parameters: diameter={cfg['diam']}, cellprob={cfg['prob']}")

        try:
            masks, flows, styles = model.eval(
                img_norm,
                diameter=cfg['diam'],
                cellprob_threshold=cfg['prob'],
                flow_threshold=0.4,
                channels=None,
                rescale=None,
                resample=True
            )

            # Get diameter from styles (handle both dict and ndarray formats)
            if isinstance(styles, dict):
                detected_diam = styles.get('diam', 30.0)
            elif hasattr(styles, '__getitem__') and len(styles) > 0:
                detected_diam = styles[0] if isinstance(styles[0], (int, float)) else 30.0
            else:
                detected_diam = 30.0

            # If still None or invalid, use default
            if detected_diam is None or not isinstance(detected_diam, (int, float)):
                detected_diam = 30.0

            # Use provided diameter if specified
            if cfg['diam'] is not None:
                detected_diam = cfg['diam']

            ncells = int(masks.max())

            print(f"    Result: {ncells} cells detected (diameter={detected_diam:.1f})")

            results.append({
                'config': cfg['name'],
                'diameter': cfg['diam'],
                'detected_diam': detected_diam,
                'cellprob': cfg['prob'],
                'ncells': ncells
            })

            # Create overlay with boundaries
            boundaries = find_boundaries(masks, mode='inner')

            # Normalize image to 8-bit for display
            img_display = image.astype(float)
            img_min, img_max = img_display.min(), img_display.max()
            if img_max > img_min:
                img_display = (img_display - img_min) / (img_max - img_min) * 255
            img_display = img_display.astype(np.uint8)

            # Create RGB overlay
            if len(image.shape) == 2:
                overlay = np.stack([img_display]*3, axis=-1)
            else:
                overlay = img_display.copy()

            overlay[boundaries] = [255, 0, 0]  # Red boundaries

            # Plot in main comparison grid
            axes[i].imshow(overlay)

            title = f"{cfg['name']}\n"
            if cfg['diam'] is None:
                title += f"Auto diam={detected_diam:.0f}"
            else:
                title += f"Diam={cfg['diam']}"
            title += f", Prob={cfg['prob']}\n{ncells} cells"

            axes[i].set_title(title, fontsize=10, pad=10, fontweight='bold')
            axes[i].axis('off')

            # Save individual result for closer inspection
            try:
                config_name = cfg['name'].replace(' ', '_').replace('+', '').lower()
                individual_fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(overlay)
                ax.set_title(f"{cfg['name']}: {ncells} cells\nDiameter={cfg['diam']}, CellProb={cfg['prob']}",
                            fontsize=14, fontweight='bold')
                ax.axis('off')
                individual_path = output_dir / f'{i+1:02d}_{config_name}.png'
                individual_fig.savefig(individual_path, dpi=150, bbox_inches='tight')
                plt.close(individual_fig)
                print(f"    Saved: {individual_path.name}")
            except Exception as save_error:
                print(f"    Warning: Failed to save individual result: {save_error}")

        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            print(f"    Full traceback:")
            traceback.print_exc()
            axes[i].text(0.5, 0.5, f'ERROR\n{str(e)[:50]}', ha='center', va='center',
                        fontsize=10, color='red', transform=axes[i].transAxes)
            axes[i].set_title(f"{cfg['name']}\nFAILED", fontsize=10, color='red')
            axes[i].axis('off')

    # Hide unused subplots
    for j in range(len(configs), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    main_output = 'cellpose_parameter_comparison.png'
    plt.savefig(main_output, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"✓ Saved main comparison: {main_output}")
    print(f"{'='*60}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY (sorted by cell count)")
    print("="*60)
    print(f"{'Config':<30} {'Cells':<8} {'Diameter':<12} {'CellProb'}")
    print("-"*60)

    for r in sorted(results, key=lambda x: x['ncells'], reverse=True):
        diam_str = f"{r['detected_diam']:.0f} (auto)" if r['diameter'] is None else str(r['diameter'])
        print(f"{r['config']:<30} {r['ncells']:<8} {diam_str:<12} {r['cellprob']}")

    # Find best config (most cells detected)
    if results:
        best = max(results, key=lambda x: x['ncells'])
        print("\n" + "="*60)
        print("RECOMMENDED PARAMETERS (most cells detected):")
        print("="*60)
        print(f"Configuration: {best['config']}")
        print(f"Cells detected: {best['ncells']}")
        print("\nUse these in your pipeline:")
        print("\nPARAMS = {")
        print("    'cellpose_model': 'cyto2',")
        print(f"    'cellpose_diameter': {best['diameter']},")
        print(f"    'cellpose_cellprob_threshold': {best['cellprob']},")
        print("    'cellpose_flow_threshold': 0.4,")
        print("    'cellpose_use_gpu': True,")
        print("    'min_size': 500,")
        print("    'max_size': 9000,")
        print("}")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"1. Open: {main_output}")
    print(f"2. Check individual results in: {output_dir}/")
    print("3. Choose the config with:")
    print("   - All cells detected (including dim ones)")
    print("   - Good boundary accuracy")
    print("   - No over-segmentation")
    print("4. Update your pipeline with those parameters")
    print("="*60)

if __name__ == "__main__":
    # Build path from configuration
    base_dir = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"

    # Try to find raw.jpg first, fall back to Actin-FITC.tif
    raw_files = list(Path(base_dir).glob("*_raw.jpg"))

    if raw_files:
        image_path = str(raw_files[0])
        print(f"Testing image: {SAMPLE_FOLDER}/{IMAGE_NUMBER}")
        print(f"Using preprocessed raw: {image_path}\n")
    else:
        # Fall back to original TIF
        actin_path = Path(base_dir) / "Actin-FITC.tif"
        if actin_path.exists():
            image_path = str(actin_path)
            print(f"Testing image: {SAMPLE_FOLDER}/{IMAGE_NUMBER}")
            print(f"No raw.jpg found, using original: {image_path}\n")
        else:
            print(f"ERROR: No image found in {base_dir}")
            print(f"  Looked for: *_raw.jpg or Actin-FITC.tif")
            print(f"\nMake sure you've set:")
            print(f"  SAMPLE_FOLDER = \"{SAMPLE_FOLDER}\"")
            print(f"  IMAGE_NUMBER = \"{IMAGE_NUMBER}\"")
            sys.exit(1)

    find_parameters(image_path)
