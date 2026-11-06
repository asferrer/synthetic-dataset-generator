"""
Physically-based underwater effects.

Implements a simplified underwater image formation model per BGR channel:

    I(d) = J * exp(-c * d) + B_inf * (1 - exp(-c * d))

where J is the unattenuated scene radiance, c the per-channel attenuation
coefficient (red absorbs fastest, blue slowest), d the normalized distance
and B_inf the veiling light (water color). Also provides seeded caustics
generation with optional depth modulation, and depth occlusion of
composited objects.

Depth convention (same as the composer): normalized 0..1 where 0 = near
camera and 1 = far (~max visible distance for the given clarity).
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# Total optical depth c*d_max per BGR channel over the visible range,
# keyed by WaterClarity values. Dimensionless: exp(-value) is the channel
# transmission at the farthest visible point.
ATTENUATION_BGR = {
    "clear": (0.4, 0.7, 1.6),
    "murky": (0.9, 1.3, 2.6),
    "very_murky": (1.6, 2.2, 3.8),
}


def _depth_at(image_shape: Tuple[int, int], depth_map: np.ndarray) -> np.ndarray:
    """Depth map resized to the image resolution, clipped to 0..1."""
    h, w = image_shape
    depth = depth_map.astype(np.float32)
    if depth.max() > 1.0:
        depth = depth / depth.max()
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(depth, 0.0, 1.0)


def apply_beer_lambert(
    image: np.ndarray,
    depth_map: np.ndarray,
    water_color: Tuple[int, int, int] = (180, 120, 30),
    water_clarity: str = "clear",
    strength: float = 1.0,
) -> np.ndarray:
    """Depth-dependent color attenuation with veiling light.

    Args:
        image: BGR uint8 image
        depth_map: normalized depth (0=near, 1=far), any resolution
        water_color: veiling light B_inf in BGR
        water_clarity: key into ATTENUATION_BGR
        strength: blend between original (0) and fully attenuated (1)

    Returns:
        BGR uint8 image
    """
    try:
        coeffs = np.asarray(
            ATTENUATION_BGR.get(water_clarity, ATTENUATION_BGR["clear"]),
            dtype=np.float32,
        )
        depth = _depth_at(image.shape[:2], depth_map)

        transmission = np.exp(-depth[:, :, np.newaxis] * coeffs[np.newaxis, np.newaxis, :])
        veil = np.asarray(water_color, dtype=np.float32)

        attenuated = image.astype(np.float32) * transmission + veil * (1.0 - transmission)
        strength = float(np.clip(strength, 0.0, 1.0))
        result = image.astype(np.float32) * (1.0 - strength) + attenuated * strength
        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.warning(f"Beer-Lambert attenuation failed: {e}")
        return image


def generate_caustics_pattern(
    shape: Tuple[int, int],
    seed: Optional[int] = None,
) -> np.ndarray:
    """Seeded multi-wave interference pattern normalized to 0..1.

    Random phases and frequencies per seed give per-image variety while
    staying reproducible when the caller controls the seed.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, w, dtype=np.float32)
    y = np.linspace(0, 4 * np.pi, h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    fx, fy = rng.uniform(0.7, 1.5, size=2)
    p1, p2, p3 = rng.uniform(0, 2 * np.pi, size=3)

    pattern = np.sin(fx * X + p1)
    pattern += np.sin(fy * Y + p2)
    pattern += np.sin((fx * X + fy * Y) / 2 + p3)
    pattern_min, pattern_max = pattern.min(), pattern.max()
    if pattern_max - pattern_min < 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    return ((pattern - pattern_min) / (pattern_max - pattern_min)).astype(np.float32)


def apply_caustics(
    image: np.ndarray,
    intensity: float,
    seed: Optional[int] = None,
    depth_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multiplicative caustics, optionally attenuated with distance.

    With a depth map, caustic strength fades linearly with distance —
    light ripples concentrate on nearby/shallow geometry and wash out in
    the far field, as in real shallow-water footage.
    """
    try:
        pattern = generate_caustics_pattern(image.shape[:2], seed=seed)
        if depth_map is not None:
            proximity = 1.0 - _depth_at(image.shape[:2], depth_map)
            pattern = pattern * proximity
        result = image.astype(np.float32) * (1.0 + pattern[:, :, np.newaxis] * intensity)
        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.warning(f"Caustics application failed: {e}")
        return image


def occlude_object(
    mask: np.ndarray,
    alpha: np.ndarray,
    depth_map: np.ndarray,
    position: Tuple[int, int],
    depth_margin: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Hide object pixels where the background is closer to the camera.

    The object's depth is sampled at its bottom-center anchor. Background
    pixels whose depth is smaller (closer) by more than depth_margin
    occlude the object: mask and alpha are zeroed there.

    Args:
        mask: HxW binary object mask (uint8, at object patch resolution)
        alpha: HxW float alpha factor aligned with mask
        depth_map: background depth at background resolution (0..1)
        position: (x, y) top-left paste position of the patch
        depth_margin: tolerance before a pixel counts as an occluder

    Returns:
        (mask, alpha, visibility) — visibility is the fraction of the
        original mask that remains visible (1.0 when nothing occludes).
    """
    try:
        original_area = cv2.countNonZero(mask)
        if original_area == 0:
            return mask, alpha, 0.0

        bg_h, bg_w = depth_map.shape[:2]
        p_h, p_w = mask.shape[:2]
        x, y = position

        # Intersection of the patch with the background
        x0_bg, y0_bg = max(x, 0), max(y, 0)
        x1_bg, y1_bg = min(x + p_w, bg_w), min(y + p_h, bg_h)
        if x0_bg >= x1_bg or y0_bg >= y1_bg:
            return mask, alpha, 1.0

        depth = depth_map.astype(np.float32)
        if depth.max() > 1.0:
            depth = depth / depth.max()

        roi_depth = depth[y0_bg:y1_bg, x0_bg:x1_bg]

        # Object depth at its bottom-center anchor
        anchor_x = int(np.clip(x + p_w // 2, 0, bg_w - 1))
        anchor_y = int(np.clip(y + p_h - 1, 0, bg_h - 1))
        object_depth = float(depth[anchor_y, anchor_x])

        occluders = roi_depth < (object_depth - depth_margin)
        if not np.any(occluders):
            return mask, alpha, 1.0

        x0_p, y0_p = x0_bg - x, y0_bg - y
        mask_view = mask[y0_p:y0_p + (y1_bg - y0_bg), x0_p:x0_p + (x1_bg - x0_bg)]
        alpha_view = alpha[y0_p:y0_p + (y1_bg - y0_bg), x0_p:x0_p + (x1_bg - x0_bg)]
        mask_view[occluders] = 0
        alpha_view[occluders] = 0

        visibility = cv2.countNonZero(mask) / original_area
        return mask, alpha, visibility
    except Exception as e:
        logger.warning(f"Depth occlusion failed: {e}")
        return mask, alpha, 1.0
