#!/usr/bin/env python3
"""
Soft Edge Extraction for Cell ROIs
Applies Gaussian blur or sigmoid distance transform to masks to create gradual
transitions at cell boundaries.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


def apply_gaussian_soft_mask(image, binary_mask, sigma=2.0):
    """
    Apply soft edge masking using Gaussian blur on the mask.

    The key insight: blur the MASK, not the image!
    This creates a gradual transition from cell to background.

    Parameters:
    -----------
    image : numpy.ndarray
        Original fluorescence image (can be 2D grayscale or 3D multi-channel)
    binary_mask : numpy.ndarray
        Binary mask for the cell (1 = cell, 0 = background)
    sigma : float
        Gaussian blur strength for the mask edges (default: 2.0)
        - Larger sigma = softer/wider transition
        - Smaller sigma = sharper transition

    Returns:
    --------
    soft_masked_image : numpy.ndarray
        Image multiplied by the soft mask (same shape as input image)
    soft_mask : numpy.ndarray
        The soft mask used (for debugging/visualization)

    Example:
    --------
    # Binary mask (from segmentation)
    binary_mask = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]])

    soft_image, soft_mask = apply_gaussian_soft_mask(
        original_image, binary_mask, sigma=2.0
    )
    """
    binary = binary_mask.astype(float)
    print(f"[DEBUG] apply_gaussian_soft_mask called with sigma={sigma}")

    # Apply Gaussian blur to the binary mask
    soft_mask = gaussian_filter(binary, sigma=sigma)
    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    # Apply soft mask to image
    original_dtype = image.dtype
    image_float = image.astype(float)

    if image.ndim == 2:
        # Grayscale image
        soft_masked_image = image_float * soft_mask
    else:
        # Multi-channel image
        soft_masked_image = image_float * soft_mask[:, :, np.newaxis]

    # Convert back to original dtype
    if original_dtype in [np.uint8, np.uint16]:
        soft_masked_image = np.clip(soft_masked_image, 0, np.iinfo(original_dtype).max)
        soft_masked_image = soft_masked_image.astype(original_dtype)

    return soft_masked_image, soft_mask


def apply_sigmoid_soft_mask(image, binary_mask, k=2.0, d0=0.0):
    """
    Apply soft edge masking using sigmoid distance transform.

    Uses distance transform to create a smooth transition based on distance
    from cell boundary, shaped by a sigmoid function.

    Parameters:
    -----------
    image : numpy.ndarray
        Original fluorescence image (can be 2D grayscale or 3D multi-channel)
    binary_mask : numpy.ndarray
        Binary mask for the cell (1 = cell, 0 = background)
    k : float
        Sigmoid steepness parameter (default: 2.0)
        - Higher values = sharper transition
        - Lower values = softer transition
    d0 : float
        Sigmoid distance threshold (default: 0.0)
        - 0.0 = transition at original edge
        - Negative = expand mask into background
        - Positive = shrink mask from edge

    Returns:
    --------
    soft_masked_image : numpy.ndarray
        Image multiplied by the soft mask (same shape as input image)
    soft_mask : numpy.ndarray
        The soft mask used (for debugging/visualization)

    Example:
    --------
    soft_image, soft_mask = apply_sigmoid_soft_mask(
        original_image, binary_mask, k=2.0, d0=0.0
    )
    """
    binary = binary_mask.astype(float)
    print(f"[DEBUG] apply_sigmoid_soft_mask called with k={k}, d0={d0}")

    # Use distance transform + sigmoid
    dist_from_inside = distance_transform_edt(binary)
    dist_from_outside = distance_transform_edt(1 - binary)
    signed_distance = dist_from_inside - dist_from_outside

    # Apply sigmoid: 1 / (1 + exp(-k * (d - d0)))
    soft_mask = 1.0 / (1.0 + np.exp(-k * (signed_distance - d0)))
    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    # Apply soft mask to image
    original_dtype = image.dtype
    image_float = image.astype(float)

    if image.ndim == 2:
        # Grayscale image
        soft_masked_image = image_float * soft_mask
    else:
        # Multi-channel image
        soft_masked_image = image_float * soft_mask[:, :, np.newaxis]

    # Convert back to original dtype
    if original_dtype in [np.uint8, np.uint16]:
        soft_masked_image = np.clip(soft_masked_image, 0, np.iinfo(original_dtype).max)
        soft_masked_image = soft_masked_image.astype(original_dtype)

    return soft_masked_image, soft_mask


def apply_soft_mask(image, binary_mask, method='gaussian', sigma=2.0, k=2.0, d0=0.0,
                   beta=0.5, cellpose_prob_map=None):
    """
    Apply soft edge masking by blurring the mask and using it as weights.

    Dispatcher function that calls the appropriate softening method.

    Parameters:
    -----------
    image : numpy.ndarray
        Original fluorescence image (can be 2D grayscale or 3D multi-channel)
    binary_mask : numpy.ndarray
        Binary mask for the cell (1 = cell, 0 = background)
    method : str
        Softening method: 'gaussian', 'sigmoid', or 'beta' (default: 'gaussian')
    sigma : float
        Gaussian blur strength (only used with method='gaussian')
    k : float
        Sigmoid steepness parameter (only used with method='sigmoid')
    d0 : float
        Sigmoid distance threshold (only used with method='sigmoid')
    beta : float
        Beta transformation parameter (only used with method='beta')
    cellpose_prob_map : numpy.ndarray
        Cellpose probability map (REQUIRED for method='beta')

    Returns:
    --------
    soft_masked_image : numpy.ndarray
        Image multiplied by the soft mask (same shape as input image)
    soft_mask : numpy.ndarray
        The soft mask used (for debugging/visualization)
    """
    if method == 'gaussian':
        return apply_gaussian_soft_mask(image, binary_mask, sigma=sigma)
    elif method == 'sigmoid':
        return apply_sigmoid_soft_mask(image, binary_mask, k=k, d0=d0)
    elif method == 'beta':
        return apply_beta_soft_mask(image, binary_mask, beta=beta,
                                    cellpose_prob_map=cellpose_prob_map)
    else:
        raise ValueError(f"Method must be 'gaussian', 'sigmoid', or 'beta', got '{method}'")


def create_gaussian_soft_mask(binary_mask, sigma=2.0):
    """
    Create a Gaussian soft mask from a binary mask (mask generation only).

    Parameters:
    -----------
    binary_mask : numpy.ndarray
        Binary mask for the cell (1 = cell, 0 = background)
    sigma : float
        Gaussian blur strength for the mask edges (default: 2.0)

    Returns:
    --------
    soft_mask : numpy.ndarray
        The soft mask (values 0.0 to 1.0)
    """
    binary = binary_mask.astype(float)
    soft_mask = gaussian_filter(binary, sigma=sigma)
    return np.clip(soft_mask, 0.0, 1.0)


def create_sigmoid_soft_mask(binary_mask, k=2.0, d0=0.0, debug=False):
    """
    Create a sigmoid soft mask from a binary mask (mask generation only).

    Parameters:
    -----------
    binary_mask : numpy.ndarray
        Binary mask for the cell (1 = cell, 0 = background)
    k : float
        Sigmoid steepness parameter (default: 2.0)
    d0 : float
        Sigmoid distance threshold (default: 0.0)
    debug : bool
        If True, return intermediate values for debugging (default: False)

    Returns:
    --------
    soft_mask : numpy.ndarray
        The soft mask (values 0.0 to 1.0)

    Or if debug=True:
        dict with keys: 'soft_mask', 'dist_from_inside', 'dist_from_outside',
                       'signed_distance', 'exponent', 'sigmoid_raw'
    """
    binary = binary_mask.astype(float)
    dist_from_inside = distance_transform_edt(binary)
    dist_from_outside = distance_transform_edt(1 - binary)
    signed_distance = dist_from_inside - dist_from_outside

    exponent = -k * (signed_distance - d0)
    sigmoid_raw = 1.0 / (1.0 + np.exp(exponent))
    soft_mask = np.clip(sigmoid_raw, 0.0, 1.0)

    if debug:
        return {
            'soft_mask': soft_mask,
            'dist_from_inside': dist_from_inside,
            'dist_from_outside': dist_from_outside,
            'signed_distance': signed_distance,
            'exponent': exponent,
            'sigmoid_raw': sigmoid_raw,
            'k': k,
            'd0': d0
        }

    return soft_mask


def extract_soft_cell_crop(image, binary_mask, sigma=2.0, padding=10):
    """
    Extract single cell with soft edges and crop to bounding box.

    This is a complete extraction function that:
    1. Applies soft masking using Gaussian blur
    2. Finds the bounding box
    3. Crops both the image and mask

    Parameters:
    -----------
    image : numpy.ndarray
        Original fluorescence image
    binary_mask : numpy.ndarray
        Binary mask for ONE cell (single ROI)
    sigma : float
        Gaussian blur strength for mask edges (default: 2.0)
    padding : int
        Pixels to add around bounding box (default: 10)

    Returns:
    --------
    cropped_soft_cell : numpy.ndarray
        Cropped image with soft edges applied
    cropped_soft_mask : numpy.ndarray
        Cropped soft mask
    bbox : dict
        Bounding box coordinates {'r_min', 'r_max', 'c_min', 'c_max'}
    """
    # 1. Create soft mask
    soft_mask = gaussian_filter(binary_mask.astype(float), sigma=sigma)

    # 2. Find bounding box of the cell (use binary mask for bbox)
    rows, cols = np.where(binary_mask > 0)

    if len(rows) == 0:
        # Empty mask
        return None, None, None

    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()

    # Add padding
    r_min = max(0, r_min - padding)
    r_max = min(image.shape[0], r_max + padding + 1)
    c_min = max(0, c_min - padding)
    c_max = min(image.shape[1], c_max + padding + 1)

    # 3. Crop both image and soft mask
    cropped_image = image[r_min:r_max, c_min:c_max]
    cropped_soft_mask = soft_mask[r_min:r_max, c_min:c_max]

    # 4. Apply soft mask to cropped image
    if cropped_image.ndim == 2:
        cropped_soft_cell = cropped_image * cropped_soft_mask
    else:  # Multi-channel
        cropped_soft_cell = cropped_image * cropped_soft_mask[:, :, np.newaxis]

    bbox = {
        'r_min': r_min,
        'r_max': r_max,
        'c_min': c_min,
        'c_max': c_max
    }

    return cropped_soft_cell, cropped_soft_mask, bbox


# ============================================================================
# BETA TRANSFORMATION FOR EDGE SOFTENING
# ============================================================================

def create_beta_soft_mask(binary_mask, beta=0.5, cellpose_prob_map=None):
    """
    Create a soft mask using beta transformation on Cellpose probability map.

    Beta transformation applies a power function to probability values:
        soft_mask = prob^beta

    - beta > 1: Sharpens edges (pushes values toward 0 or 1)
    - beta < 1: Softens edges (makes transition more gradual)
    - beta = 1: No change

    Parameters:
    -----------
    binary_mask : numpy.ndarray
        Binary mask for the cell (1 = cell, 0 = background)
    beta : float
        Beta transformation parameter (default: 0.5)
        Recommended range: 0.3 (very soft) to 0.9 (slightly soft)
    cellpose_prob_map : numpy.ndarray
        Cellpose probability map (flows[2] from model.eval())
        REQUIRED - will error if not provided

    Returns:
    --------
    soft_mask : numpy.ndarray
        Soft mask with beta-transformed edges (values 0.0 to 1.0)

    Example:
    --------
    masks, flows, styles = model.eval(...)
    soft_mask = create_beta_soft_mask(
        binary_mask,
        beta=0.6,
        cellpose_prob_map=flows[2]
    )
    """
    if cellpose_prob_map is None:
        raise ValueError(
            "cellpose_prob_map is required for beta transformation. "
            "Pass flows[2] from Cellpose model.eval(). "
            "If you don't have the probability map, use gaussian or sigmoid methods instead."
        )

    binary = (binary_mask > 0).astype(float)

    # Use actual Cellpose probability map
    # Normalize to [0, 1] range
    prob_map = cellpose_prob_map.astype(float)
    prob_min, prob_max = prob_map.min(), prob_map.max()
    if prob_max > prob_min:
        prob_normalized = (prob_map - prob_min) / (prob_max - prob_min)
    else:
        prob_normalized = prob_map

    # Mask to cell region only
    prob_normalized = prob_normalized * binary

    # Apply beta transformation
    soft_mask = np.power(prob_normalized, beta)

    # Ensure values are in [0, 1]
    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    return soft_mask


def create_beta_distance_soft_mask(binary_mask, beta=0.6, edge_width=5):
    """
    Beta transformation with distance-based probability map.

    Creates a smooth probability map based on distance from edge,
    then applies beta transformation for additional control.

    This method combines:
    1. Distance transform (smooth gradient)
    2. Beta transformation (edge softness control)

    Parameters:
    -----------
    binary_mask : numpy.ndarray
        Binary mask for the cell
    beta : float
        Beta parameter for softening (default: 0.6)
    edge_width : float
        Width of transition zone in pixels (default: 5)

    Returns:
    --------
    soft_mask : numpy.ndarray
        Soft mask with beta-transformed edges
    """
    binary = (binary_mask > 0).astype(float)

    # Calculate signed distance transform
    dist_inside = distance_transform_edt(binary)
    dist_outside = distance_transform_edt(1 - binary)
    signed_distance = dist_inside - dist_outside

    # Create smooth probability map using sigmoid on distance
    # This creates a smooth 0-1 transition across the edge
    prob_map = 1.0 / (1.0 + np.exp(-signed_distance / edge_width))

    # Apply beta transformation to adjust edge softness
    soft_mask = np.power(prob_map, beta)

    return np.clip(soft_mask, 0.0, 1.0)


def apply_beta_soft_mask(image, binary_mask, beta=0.5, cellpose_prob_map=None):
    """
    Apply beta transformation soft masking to an image.

    Parameters:
    -----------
    image : numpy.ndarray
        Original fluorescence image
    binary_mask : numpy.ndarray
        Binary mask for the cell
    beta : float
        Beta transformation parameter (0.3-0.9 recommended)
    cellpose_prob_map : numpy.ndarray
        Cellpose probability map (flows[2]) - REQUIRED

    Returns:
    --------
    soft_masked_image : numpy.ndarray
        Image with soft mask applied
    soft_mask : numpy.ndarray
        The soft mask used
    """
    # Create beta soft mask
    soft_mask = create_beta_soft_mask(
        binary_mask,
        beta=beta,
        cellpose_prob_map=cellpose_prob_map
    )

    # Apply to image
    original_dtype = image.dtype
    image_float = image.astype(float)

    if image.ndim == 2:
        soft_masked_image = image_float * soft_mask
    else:
        soft_masked_image = image_float * soft_mask[:, :, np.newaxis]

    # Convert back to original dtype
    if original_dtype in [np.uint8, np.uint16]:
        soft_masked_image = np.clip(soft_masked_image, 0, np.iinfo(original_dtype).max)
        soft_masked_image = soft_masked_image.astype(original_dtype)

    return soft_masked_image, soft_mask


# ============================================================================
# EDGE METHOD REGISTRY
# ============================================================================

def method_hard_cutoff(mask):
    """Binary mask - sharp edges (original method)"""
    return (mask > 0).astype(float)


def method_gaussian_sharp(mask):
    """Gaussian blur with lower sigma (sharper)"""
    binary = (mask > 0).astype(float)
    return create_gaussian_soft_mask(binary, sigma=1.0)


def method_gaussian_standard(mask):
    """Gaussian blur with standard sigma"""
    binary = (mask > 0).astype(float)
    return create_gaussian_soft_mask(binary, sigma=2.0)


def method_gaussian_soft(mask):
    """Gaussian blur with higher sigma (softer)"""
    binary = (mask > 0).astype(float)
    return create_gaussian_soft_mask(binary, sigma=3.5)


def method_sigmoid_soft(mask):
    """Sigmoid with softer transition"""
    binary = (mask > 0).astype(float)
    return create_sigmoid_soft_mask(binary, k=0.1, d0=0.0)


def method_sigmoid_standard(mask):
    """Sigmoid distance transform with standard steepness"""
    binary = (mask > 0).astype(float)
    return create_sigmoid_soft_mask(binary, k=1.0, d0=0.0)


def method_sigmoid_sharp(mask):
    """Sigmoid with sharper transition"""
    binary = (mask > 0).astype(float)
    return create_sigmoid_soft_mask(binary, k=2.0, d0=0.0)


def method_beta_distance(mask):
    """Beta transformation with distance-based probability (fallback when no Cellpose prob map)"""
    binary = (mask > 0).astype(float)
    return create_beta_distance_soft_mask(binary, beta=0.6, edge_width=5)


# Registry of all edge softening methods for comparison
# Note: Pure beta transformation (create_beta_soft_mask) requires Cellpose probability map
# and is not included here. Use create_beta_soft_mask() directly with flows[2].
EDGE_METHODS = {
    'Hard Cutoff': method_hard_cutoff,
    'Gaussian σ=1.0': method_gaussian_sharp,
    'Gaussian σ=2.0': method_gaussian_standard,
    'Gaussian σ=3.5': method_gaussian_soft,
    'Sigmoid k=0.3': method_sigmoid_soft,
    'Sigmoid k=1.0': method_sigmoid_standard,
    'Sigmoid k=2.0': method_sigmoid_sharp,
    'Beta+Distance': method_beta_distance,
}
