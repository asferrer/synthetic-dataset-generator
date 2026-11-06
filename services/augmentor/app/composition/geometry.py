"""Real-world size tables and pure geometry/mask helpers (no GPU needed)."""

from typing import Tuple

import cv2
import numpy as np
from loguru import logger


REAL_WORLD_SIZES = {
    # Marine life
    'fish': 0.25,              # Average small-medium fish
    'small_fish': 0.10,
    'large_fish': 0.60,
    'shark': 2.5,              # Average shark
    'whale': 12.0,
    'dolphin': 2.5,
    'turtle': 0.8,
    'jellyfish': 0.30,
    'octopus': 0.50,
    'squid': 0.40,
    'crab': 0.15,
    'lobster': 0.35,
    'starfish': 0.20,
    'seahorse': 0.15,
    'ray': 1.5,
    'eel': 0.80,
    'coral': 0.40,

    # Debris and trash (common sizes)
    'plastic': 0.15,           # Generic plastic piece
    'plastic_bag': 0.30,
    'plastic_bottle': 0.25,
    'bottle': 0.25,
    'can': 0.12,               # Soda/beer can
    'glass': 0.20,             # Glass bottle
    'glass_bottle': 0.25,
    'tire': 0.65,              # Car tire diameter
    'rope': 0.03,              # Rope diameter (length varies)
    'net': 0.50,               # Fishing net section
    'fishing_net': 1.0,
    'debris': 0.25,            # Generic debris
    'metal_debris': 0.30,
    'wood': 0.40,
    'paper': 0.20,
    'cloth': 0.30,
    'shoe': 0.30,
    'bucket': 0.35,
    'container': 0.50,
    'barrel': 0.90,

    # Equipment
    'buoy': 0.50,
    'anchor': 0.80,
    'chain': 0.05,             # Link diameter

    # Default for unknown objects
    'default': 0.25,
}

# Reference distance for "natural" object appearance (in meters)
# Objects photographed at this distance look "normal sized" in their source image
REFERENCE_CAPTURE_DISTANCE = 2.0

# Underwater visibility parameters
UNDERWATER_VISIBILITY = {
    'clear': 30.0,      # meters - very clear tropical water
    'moderate': 15.0,   # meters - typical ocean
    'murky': 5.0,       # meters - low visibility
}


def get_real_world_size(object_class: str, config=None) -> float:
    """
    Get the real-world size of an object in meters.

    Args:
        object_class: Object class name
        config: Optional ObjectSizeConfig instance. If None, uses global config.

    Returns:
        Size in meters
    """
    if config is None:
        from app.config_manager import get_object_size_config
        config = get_object_size_config()

    return config.get_size(object_class)


def compute_iou(boxA: Tuple, boxB: Tuple) -> float:
    """Calculate Intersection over Union of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea != 0 else 0


def check_mask_overlap(
    placement_mask: np.ndarray,
    candidate_mask: np.ndarray,
    candidate_pos: Tuple[int, int],
    threshold: float = 0.05
) -> bool:
    """Check overlap between masks with robust validation."""
    try:
        if placement_mask is None or candidate_mask is None:
            return False

        if len(placement_mask.shape) != 2 or len(candidate_mask.shape) != 2:
            return False

        x, y = candidate_pos
        cand_h, cand_w = candidate_mask.shape
        bg_h, bg_w = placement_mask.shape

        if cand_h == 0 or cand_w == 0 or bg_h == 0 or bg_w == 0:
            return False

        roi_x_start = max(0, x)
        roi_y_start = max(0, y)
        roi_x_end = min(bg_w, x + cand_w)
        roi_y_end = min(bg_h, y + cand_h)

        if roi_x_start >= roi_x_end or roi_y_start >= roi_y_end:
            return False

        cand_roi_x_start = max(0, roi_x_start - x)
        cand_roi_y_start = max(0, roi_y_start - y)
        cand_roi_x_end = min(cand_roi_x_start + (roi_x_end - roi_x_start), cand_w)
        cand_roi_y_end = min(cand_roi_y_start + (roi_y_end - roi_y_start), cand_h)

        roi_placement = placement_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_candidate = candidate_mask[cand_roi_y_start:cand_roi_y_end, cand_roi_x_start:cand_roi_x_end]

        if roi_placement.shape != roi_candidate.shape:
            min_h = min(roi_placement.shape[0], roi_candidate.shape[0])
            min_w = min(roi_placement.shape[1], roi_candidate.shape[1])
            roi_placement = roi_placement[:min_h, :min_w]
            roi_candidate = roi_candidate[:min_h, :min_w]

        intersection = cv2.bitwise_and(roi_placement, roi_candidate)
        candidate_area = cv2.countNonZero(roi_candidate)

        if candidate_area == 0:
            return False

        overlap_area = cv2.countNonZero(intersection)
        overlap_ratio = overlap_area / candidate_area

        return overlap_ratio > threshold

    except Exception as e:
        logger.error(f"Error in check_mask_overlap: {e}")
        return True


def refine_object_mask(
    alpha_channel: np.ndarray,
    mask_bin: np.ndarray,
    feather_radius: int = 3,
    use_binary_alpha: bool = False,
    smooth_gradual: bool = True,
    gradual_mask_threshold: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine binary mask for smoother edges with configurable feathering.

    Creates a gradual transition at object edges. Can use either binary alpha
    (minimal feathering) or gradient alpha (more blending) based on configuration.

    Args:
        alpha_channel: Original alpha channel (0-255)
        mask_bin: Binary mask (0 or 255)
        feather_radius: Pixels to feather at edges (default: 3, recommended: 1-2 for binary)
        use_binary_alpha: If True, use minimal feathering to prevent color bleeding

    Returns:
        Tuple of (refined_alpha as float 0-1, binary_mask)
    """
    unique_vals = np.unique(alpha_channel)
    has_transparency = len(unique_vals) > 5

    if has_transparency and not use_binary_alpha:
        # Image already has gradual alpha - enhance edges slightly
        alpha_refined = alpha_channel.astype(np.float32) / 255.0
        alpha_refined[alpha_refined < 0.05] = 0.0

        # Apply slight blur to smooth any jagged edges (`smooth_gradual=False`: matte of edge_matting, already fractional)
        if smooth_gradual:
            alpha_refined = cv2.GaussianBlur(alpha_refined, (5, 5), 1.0)

        # `gradual_mask_threshold` 127 with edge_matting: the annotated mask is the matte's 0.5 alpha contour, not its weak halo
        _, mask_refined_bin = cv2.threshold(
            (alpha_refined * 255).astype(np.uint8), gradual_mask_threshold, 255, cv2.THRESH_BINARY
        )
        return alpha_refined, mask_refined_bin

    # For binary masks OR if binary_alpha is enforced
    # Calculate distance from edge (inward)
    dist_inside = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)

    if use_binary_alpha:
        # BINARY ALPHA MODE: Minimal feathering to prevent color bleeding
        feather = max(feather_radius, 1)  # 1-2px recommended

        # Core region: fully opaque
        core_mask = dist_inside > feather
        # Feather zone: gradient over 1-2px
        feather_zone = (dist_inside > 0) & (dist_inside <= feather)

        # Create alpha
        alpha_refined = np.zeros_like(dist_inside, dtype=np.float32)
        alpha_refined[core_mask] = 1.0
        alpha_refined[feather_zone] = dist_inside[feather_zone] / feather

        # Minimal smoothing (1px) for anti-aliasing
        alpha_refined = cv2.GaussianBlur(alpha_refined, (3, 3), 0.5)

        # Ensure core stays fully opaque
        alpha_refined[core_mask] = 1.0

        return alpha_refined, mask_bin

    else:
        # GRADIENT ALPHA MODE (legacy behavior)
        mask_float = mask_bin.astype(np.float32) / 255.0
        feather = max(feather_radius, 2)

        # Normalize distances to feather zone
        alpha_inside = np.clip(dist_inside / feather, 0, 1)
        alpha_refined = alpha_inside * mask_float

        # Apply smoothing for natural falloff
        alpha_refined = cv2.GaussianBlur(alpha_refined, (5, 5), 1.5)

        # Ensure core of object remains fully opaque
        core_mask = dist_inside > feather
        alpha_refined[core_mask] = 1.0

        return alpha_refined, mask_bin
