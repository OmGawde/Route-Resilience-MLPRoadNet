"""
Centerline-Guided Morphological Refinement & Speckle Removal.
Cleans raw probability maps and bridges minor micro-occlusions using dual-head synergy.
"""

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes


def refine_road_mask(
    mask_prob: np.ndarray,
    cline_prob: np.ndarray,
    mask_thresh: float = 0.50,
    cline_thresh: float = 0.40,
    min_area_pixels: int = 100,
    close_kernel_size: int = 3,
) -> np.ndarray:
    """
    Refines raw road mask by fusing centerline prior, morphological closing,
    and connected component area filtering.

    Args:
        mask_prob: (H, W) float in [0, 1] from mask head / fused head
        cline_prob: (H, W) float in [0, 1] from centerline head
        mask_thresh: Threshold for base road mask (0.50 for exact boundary alignment)
        cline_thresh: Threshold for centerline skeleton prior
        min_area_pixels: Minimum connected component area to keep (removes noise speckles)
        close_kernel_size: Kernel radius for bridging micro gaps

    Returns:
        Clean binary mask (H, W) uint8 in {0, 1}.
    """
    # 1. Base segmentation with exact 0.50 threshold
    base_mask = (mask_prob > mask_thresh).astype(bool)

    # 2. Centerline prior: boost road areas that have strong centerline support
    strong_cline = (cline_prob > cline_thresh).astype(bool)
    dilated_cline = cv2.dilate(
        strong_cline.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)

    # Combined: rescue faint road pixels along the skeleton without bloating the edges
    rescue_candidates = (mask_prob > 0.40) & dilated_cline
    fused_binary = base_mask | rescue_candidates

    # 3. Micro morphological closing (3x3 kernel)
    if close_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        closed = cv2.morphologyEx(fused_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    else:
        closed = fused_binary

    # 4. Remove isolated false-positive speckles
    cleaned = remove_small_objects(closed, min_size=min_area_pixels)
    
    # 5. Fill small pinhole gaps
    filled = remove_small_holes(cleaned, area_threshold=48)

    return filled.astype(np.uint8)
