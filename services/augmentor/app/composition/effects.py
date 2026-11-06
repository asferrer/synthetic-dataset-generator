"""Realism effects and harmonization applied to composed objects."""

import random
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np
from loguru import logger

from app import copy_paste
from app import edge_matting
from app import medium_transfer
from app.models.schemas import LightingInfo
from app.composition.results import BlurBudget


class EffectsMixin:
    """Realism effects and harmonization applied to composed objects."""

    def _matte_crop(self, path: str, obj_img: np.ndarray, class_name: str, profile, augmented: bool) -> np.ndarray:
        """Matted crop (fractional alpha + F, enlarged by a few px per side), cached per crop and parameters."""
        key = (path, class_name, profile.edge_matting_alpha, profile.edge_matting_foreground, profile.edge_matting_a0)
        out = None if augmented else self._matte_cache.get(key)
        if out is None:
            out, _ = edge_matting.matte_bgra(obj_img, class_name, profile.edge_matting_alpha,
                                             profile.edge_matting_foreground, profile.edge_matting_a0)
            if not augmented:
                self._matte_cache.put(key, out)
        return out

    def _medium_source(self, path: str, obj_img: np.ndarray, profile, augmented: bool):
        """Ring statistics of the crop in its source (medium_transfer.MediumStats, or None if they cannot be measured)."""
        key = (path, "crop")
        cached = None if augmented else self._medium_src_cache.get(key)
        if cached is not None:      # (stats,): None is a valid value (ring too small)
            return cached[0]
        mask = (obj_img[:, :, 3] > 10).astype(np.uint8) if obj_img.shape[2] == 4 else None
        stats = None if mask is None else medium_transfer.medium_stats(obj_img[:, :, :3], mask)
        if not augmented:
            self._medium_src_cache.put(key, (stats,))
        return stats

    @staticmethod
    def _medium_context(composite: np.ndarray, existing_bboxes, plan, profile) -> Dict[str, Any]:
        """Destination data that does not change between attempts: dark floor and boxes (already pasted or real) that are not medium."""
        h, w = composite.shape[:2]
        exclude = np.zeros((h, w), dtype=bool)
        boxes = list(existing_bboxes) + (list(plan.real_boxes) if plan is not None and profile.copy_paste_real else [])
        for x1, y1, x2, y2 in boxes:
            exclude[max(0, int(y1)):max(0, int(y2)) + 1, max(0, int(x1)):max(0, int(x2)) + 1] = True
        floor = (medium_transfer.dark_floor(composite) if profile.medium_dark_floor
                 else np.zeros(3, dtype=np.float32))
        return {"exclude": exclude, "floor": floor}

    @staticmethod
    def _apply_medium_transfer(obj: np.ndarray, mask: np.ndarray, origin: Tuple[int, int], composite: np.ndarray,
                               src, ctx: Dict[str, Any], profile) -> np.ndarray:
        """Swaps the source medium for that of the destination ring (M1); identity if the destination ring cannot be measured."""
        h, w = composite.shape[:2]
        full = np.zeros((h, w), dtype=np.uint8)
        x0, y0 = origin
        full[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = mask
        dst = medium_transfer.medium_stats(composite, full, ctx["exclude"], floor=ctx["floor"])
        return medium_transfer.transfer(obj, src, dst, profile.medium_transfer, profile.medium_lambda,
                                        profile.medium_dark_floor)

    def _apply_color_correction(
        self,
        obj: np.ndarray,
        roi: np.ndarray,
        mask: np.ndarray,
        intensity: float = 0.15,
    ) -> np.ndarray:
        """
        Transfer color statistics from background to object using Reinhard color transfer.

        Uses proper LAB color space transfer that preserves object texture/detail
        while adapting colors to match the background environment.

        Args:
            obj: Object image (BGR)
            roi: Background region of interest (BGR)
            mask: Object mask (unused but kept for interface compatibility)
            intensity: Blend intensity (0=no change, 1=full transfer).
                      Low values (0.1-0.2) recommended for synthetic data.
        """
        try:
            # Convert to LAB color space
            obj_lab = cv2.cvtColor(obj, cv2.COLOR_BGR2LAB).astype(np.float32)
            roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

            # Calculate statistics for each channel
            obj_mean = obj_lab.mean(axis=(0, 1))
            obj_std = obj_lab.std(axis=(0, 1)) + 1e-6
            roi_mean = roi_lab.mean(axis=(0, 1))
            roi_std = roi_lab.std(axis=(0, 1)) + 1e-6

            # Reinhard color transfer: normalize then apply target statistics
            # Step 1: Normalize object (center and scale)
            normalized = (obj_lab - obj_mean) / obj_std

            # Step 2: Apply background statistics (full transfer)
            transferred = normalized * roi_std + roi_mean

            # Step 3: Blend with original based on intensity
            # Low intensity = preserve object colors, high = adapt to background
            result = obj_lab * (1 - intensity) + transferred * intensity

            # Preserve more of the original luminance (L channel) to maintain object details
            # Only transfer 50% of L channel change to keep object recognizable
            l_blend = 0.5 * intensity
            result[:, :, 0] = obj_lab[:, :, 0] * (1 - l_blend) + transferred[:, :, 0] * l_blend

            result = np.clip(result, 0, 255).astype(np.uint8)
            return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        except Exception as e:
            logger.warning(f"Color correction failed: {e}")
            return obj

    def _apply_blur_matching(
        self,
        obj: np.ndarray,
        roi: np.ndarray,
        mask: np.ndarray,
        strength: float = 1.0,
        blur_budget: Optional[BlurBudget] = None,
        blur_max: Optional[int] = None,
    ) -> np.ndarray:
        """
        Subtly match blur level of object to background.

        `blur_max` (copy-paste profile): kernel cap; if it does not even allow 3 px, no blur is applied.

        Uses conservative blur to avoid destroying object details.
        Respects blur budget to prevent over-processing small objects.
        """
        try:
            # Estimate blur from Laplacian variance (higher = sharper)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bg_sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

            gray_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
            obj_sharpness = cv2.Laplacian(gray_obj, cv2.CV_64F).var()

            # Only apply blur if object is SIGNIFICANTLY sharper than background
            # Higher threshold (3x) to preserve object details
            if obj_sharpness > bg_sharpness * 3.0 and bg_sharpness > 10:
                # Calculate blur ratio with logarithmic dampening
                # This prevents excessive blur even with large differences
                ratio = obj_sharpness / max(bg_sharpness, 100)
                dampened_ratio = np.log2(max(ratio, 1)) + 1  # Log dampening

                # Very conservative blur amount (max 5px kernel)
                blur_amount = int(dampened_ratio * strength * 0.5)
                blur_amount = max(3, min(blur_amount, 5))
                if copy_paste.cap_kernel(blur_amount, blur_max) == 0:
                    return obj

                # Apply blur budget if provided
                if blur_budget:
                    actual_blur = blur_budget.apply(blur_amount)
                    if actual_blur < 3:
                        logger.debug(f"Blur budget exceeded, skipping blur matching")
                        return obj
                    blur_amount = int(actual_blur)

                if blur_amount % 2 == 0:
                    blur_amount += 1
                blur_amount = copy_paste.cap_kernel(blur_amount, blur_max)

                # Only apply if blur would be noticeable
                if blur_amount >= 3:
                    logger.debug(f"Blur matching: obj_sharp={obj_sharpness:.0f}, bg_sharp={bg_sharpness:.0f}, blur={blur_amount}")
                    return cv2.GaussianBlur(obj, (blur_amount, blur_amount), 0)

            return obj

        except Exception as e:
            logger.warning(f"Blur matching failed: {e}")
            return obj

    def _apply_lighting_effect(
        self,
        obj: np.ndarray,
        light_type: str = "ambient",
        intensity: float = 0.5,
    ) -> np.ndarray:
        """Apply lighting effect to object."""
        try:
            h, w = obj.shape[:2]

            if light_type == "spotlight":
                Y, X = np.ogrid[:h, :w]
                cx, cy = w // 2, h // 2
                dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                mask = 1 - (dist / (max(h, w) * 0.7))
                mask = np.clip(mask, 0, 1)
            elif light_type == "gradient":
                mask = np.linspace(1, 0.5, h)[:, np.newaxis]
                mask = np.repeat(mask, w, axis=1)
            else:  # ambient
                mask = np.ones((h, w))

            # Apply lighting
            mask = mask[:, :, np.newaxis]
            adjusted = obj.astype(np.float32)
            adjusted = adjusted * (1 - intensity) + adjusted * mask * intensity * 1.2
            return np.clip(adjusted, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Lighting effect failed: {e}")
            return obj

    def _apply_motion_blur(
        self,
        obj: np.ndarray,
        kernel_size: int = 15,
        blur_budget: Optional[BlurBudget] = None,
    ) -> np.ndarray:
        """
        Apply motion blur to object.

        Respects blur budget - motion blur uses ~1/3 of kernel size as effective blur.
        """
        try:
            # Apply blur budget if provided
            if blur_budget:
                # Motion blur effective blur is ~1/3 of kernel size
                effective_blur = kernel_size / 3.0
                actual_blur = blur_budget.apply(effective_blur)
                if actual_blur < 2:
                    logger.debug(f"Blur budget exceeded, skipping motion blur")
                    return obj
                kernel_size = int(actual_blur * 3)
                if kernel_size < 3:
                    return obj
                if kernel_size % 2 == 0:
                    kernel_size += 1

            angle = random.uniform(0, 360)
            k = np.zeros((kernel_size, kernel_size))
            k[kernel_size // 2, :] = 1
            M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
            k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
            k = k / k.sum()
            return cv2.filter2D(obj, -1, k)
        except Exception as e:
            logger.warning(f"Motion blur failed: {e}")
            return obj

    def _apply_shadow(
        self,
        bg: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        lighting_info: Optional[LightingInfo],
        opacity: float = 0.12,
        blur_size: int = 31,
        depth_map: Optional[np.ndarray] = None,
        position: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Apply subtle, realistic depth-aware shadow from object mask.

        Underwater shadows are naturally very soft and diffuse due to:
        - Light scattering in water
        - Multiple light sources (surface + ambient)
        - Reduced contrast at depth

        Args:
            bg: Background image
            mask: Object binary mask
            bbox: Bounding box (x, y, w, h)
            lighting_info: Light source information
            opacity: Base shadow opacity (kept low for realism)
            blur_size: Base blur kernel size (larger for soft shadows)
            depth_map: Optional depth map for depth-aware shadows
            position: Object position (x, y)
        """
        try:
            x, y, w, h = bbox
            bg_h, bg_w = bg.shape[:2]

            # =================================================================
            # Step 1: Calculate depth-based shadow parameters
            # =================================================================
            depth_value = 0.5  # Default mid-depth

            if depth_map is not None and position is not None:
                depth_normalized = depth_map.astype(np.float32)
                if depth_normalized.max() > 1:
                    depth_normalized = depth_normalized / depth_normalized.max()

                px, py = position
                px = max(0, min(px, bg_w - 1))
                py = max(0, min(py, bg_h - 1))
                depth_value = depth_normalized[py, px]

            # Underwater shadow physics:
            # - Near objects: slightly more defined shadows
            # - Mid/Far objects: very diffuse, almost invisible shadows
            # - All shadows are soft due to light scattering

            if depth_value < 0.3:
                # Near: slightly more visible but still soft
                depth_opacity_factor = 1.0
                depth_blur_factor = 0.8
                depth_offset_factor = 1.0
            elif depth_value < 0.6:
                # Mid: very soft shadows
                t = (depth_value - 0.3) / 0.3
                depth_opacity_factor = 1.0 - t * 0.5  # 1.0 -> 0.5
                depth_blur_factor = 0.8 + t * 0.4     # 0.8 -> 1.2
                depth_offset_factor = 1.0 - t * 0.5  # 1.0 -> 0.5
            else:
                # Far: almost no shadow (light too diffuse)
                depth_opacity_factor = 0.2
                depth_blur_factor = 1.5
                depth_offset_factor = 0.2

            # Adjust parameters based on depth
            adjusted_opacity = opacity * depth_opacity_factor
            adjusted_blur = int(blur_size * depth_blur_factor)
            if adjusted_blur % 2 == 0:
                adjusted_blur += 1
            adjusted_blur = max(5, min(adjusted_blur, 61))

            # =================================================================
            # Step 2: Determine shadow direction and offset (very subtle)
            # =================================================================
            # Base offset is small - underwater shadows are close to objects
            base_offset = 5 * depth_offset_factor

            # Check if object is near bottom (potential seafloor contact)
            relative_y = y / bg_h
            is_near_bottom = relative_y > 0.75

            if is_near_bottom:
                # Contact shadow: directly underneath, minimal offset
                offset_x = int(base_offset * 0.2)
                offset_y = int(base_offset * 0.3)
                # Contact shadows slightly more visible but still subtle
                adjusted_opacity = min(adjusted_opacity * 1.3, 0.18)
                adjusted_blur = max(adjusted_blur, 15)  # Keep soft
            elif lighting_info and lighting_info.light_sources:
                # Use lighting info for shadow direction
                ls = lighting_info.light_sources[0]
                azimuth = ls.position[0]
                offset_x = int(-np.cos(np.radians(azimuth)) * base_offset)
                offset_y = int(-np.sin(np.radians(azimuth)) * base_offset)
            else:
                # Default: light from above (typical underwater)
                offset_x = int(base_offset * 0.3)
                offset_y = int(base_offset * 0.8)

            # =================================================================
            # Step 3: Create and position shadow mask
            # =================================================================
            shadow_mask = np.zeros(bg.shape[:2], dtype=np.float32)

            sy = max(0, y + offset_y)
            sx = max(0, x + offset_x)
            ey = min(bg_h, sy + mask.shape[0])
            ex = min(bg_w, sx + mask.shape[1])

            mask_h = ey - sy
            mask_w = ex - sx

            if mask_h > 0 and mask_w > 0:
                shadow_mask[sy:ey, sx:ex] = mask[:mask_h, :mask_w].astype(np.float32) / 255.0

            # =================================================================
            # Step 4: Apply multi-pass blur for natural falloff
            # =================================================================
            # First pass: main blur
            shadow_mask = cv2.GaussianBlur(shadow_mask, (adjusted_blur, adjusted_blur), 0)

            # Second pass: additional softening for natural gradient
            extra_blur = adjusted_blur + 10
            if extra_blur % 2 == 0:
                extra_blur += 1
            shadow_mask = cv2.GaussianBlur(shadow_mask, (extra_blur, extra_blur), 0)

            # Apply opacity with soft falloff at edges
            shadow_mask = shadow_mask * adjusted_opacity

            # Feather edges for more natural look
            shadow_mask = np.power(shadow_mask, 0.8)

            # =================================================================
            # Step 5: Apply shadow as subtle darkening
            # =================================================================
            # Simple darkening without color tint (more natural underwater)
            shadow_3ch = shadow_mask[:, :, np.newaxis]

            # Apply shadow as multiplicative darkening
            result = bg.astype(np.float32) * (1.0 - shadow_3ch * 0.7)

            return np.clip(result, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Shadow application failed: {e}")
            return bg

    def _apply_underwater_effect(
        self,
        image: np.ndarray,
        intensity: float = 0.25,
        water_color: Tuple[int, int, int] = (120, 80, 20),
    ) -> np.ndarray:
        """Apply underwater color cast."""
        try:
            overlay = np.full_like(image, water_color, dtype=np.uint8)
            result = cv2.addWeighted(image, 1 - intensity, overlay, intensity, 0)
            return result
        except Exception as e:
            logger.warning(f"Underwater effect failed: {e}")
            return image

    def _apply_poisson_blend(
        self,
        obj: np.ndarray,
        roi: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Apply Poisson seamless blending."""
        try:
            if cv2.countNonZero(mask) == 0:
                return roi

            # Center on the mask's bounding rect so seamlessClone keeps the
            # object where it was composited (a fixed ROI center shifts the
            # object and desynchronizes its bbox annotation)
            x, y, w, h = cv2.boundingRect(mask)
            center = (x + w // 2, y + h // 2)
            result = cv2.seamlessClone(obj, roi, mask, center, cv2.NORMAL_CLONE)
            return result
        except Exception as e:
            logger.warning(f"Poisson blending failed: {e}")
            return roi

    def _apply_edge_smoothing(
        self,
        blended: np.ndarray,
        roi: np.ndarray,
        mask: np.ndarray,
        feather: int = 3,
    ) -> np.ndarray:
        """
        Contrast-adaptive edge feathering for seamless compositing.

        Reduces feather radius where object/background contrast is high to
        prevent dark halos (e.g. dark debris on bright sand). Keeps full
        feathering where contrast is low for smooth transitions.

        Args:
            blended: Object already blended onto background
            roi: Original background region
            mask: Binary mask of object
            feather: Max feather radius in pixels

        Returns:
            Smoothly blended result
        """
        try:
            h, w = mask.shape[:2]
            if h < 3 or w < 3:
                return blended

            mask_bin = (mask > 127).astype(np.uint8) * 255
            dist_inside = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
            feather_px = max(feather, 2)

            # --- Contrast-adaptive feather map ---
            # Compute per-pixel luminance contrast between object and background
            obj_gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY).astype(np.float32)
            bg_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            contrast = np.abs(obj_gray - bg_gray) / 255.0  # 0..1

            # Smooth contrast map to avoid pixel-level noise
            contrast = cv2.GaussianBlur(contrast, (15, 15), 4.0)

            # Map contrast to effective feather: high contrast -> small feather
            # contrast=0 -> feather_px, contrast=1 -> 2px (minimum)
            min_feather = 2.0
            effective_feather = feather_px - (feather_px - min_feather) * contrast
            effective_feather = np.clip(effective_feather, min_feather, feather_px)

            # Per-pixel alpha: dist_inside / effective_feather
            alpha = np.where(
                effective_feather > 0,
                np.clip(dist_inside / effective_feather, 0, 1),
                1.0,
            )

            # Smooth alpha for natural falloff
            alpha = cv2.GaussianBlur(alpha.astype(np.float32), (5, 5), 1.5)

            # Core stays fully opaque
            core_mask = dist_inside > feather_px * 1.5
            alpha[core_mask] = 1.0

            alpha_3ch = alpha[:, :, np.newaxis]

            result = (
                blended.astype(np.float32) * alpha_3ch
                + roi.astype(np.float32) * (1.0 - alpha_3ch)
            )

            # Light color harmonization in transition zone
            edge_zone = (dist_inside > 0) & (dist_inside < feather_px)
            if np.any(edge_zone):
                edge_3ch = edge_zone[:, :, np.newaxis]
                edge_blend = (
                    result * 0.85
                    + cv2.GaussianBlur(result.astype(np.float32), (3, 3), 0.5) * 0.15
                )
                result = np.where(edge_3ch, edge_blend, result)

            return np.clip(result, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Edge smoothing failed: {e}")
            return blended
