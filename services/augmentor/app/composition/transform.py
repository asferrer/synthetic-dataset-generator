"""Object scale/rotation/flip transforms, depth-aware scaling and bbox bookkeeping."""

import random
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from app import faithful_scale
from app import resample
from app.composition.results import BlurBudget
from app.composition.geometry import REFERENCE_CAPTURE_DISTANCE, UNDERWATER_VISIBILITY, get_real_world_size


class TransformMixin:
    """Object scale/rotation/flip transforms, depth-aware scaling and bbox bookkeeping."""

    def _transform_object(
        self,
        obj_img: np.ndarray,
        bg_shape: Tuple[int, int],
        depth_map: Optional[np.ndarray],
        position: Optional[Tuple[int, int]],
        scale: Optional[float],
        rotation: Optional[float],
        object_class: str = "default",
        blur_budget: Optional[BlurBudget] = None,
        faithful: Optional[faithful_scale.FaithfulScale] = None,
        single_resample: bool = False,
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int], float]:
        """
        Apply geometric transformations to object with high-quality interpolation.

        Uses appropriate interpolation methods to prevent aliasing artifacts:
        - INTER_LANCZOS4 for upscaling (best quality, prevents jaggies)
        - INTER_AREA for downscaling (prevents moire patterns)
        - INTER_CUBIC for rotation (good balance of quality and speed)

        Also applies pre-filtering to prevent aliasing in low-resolution objects.
        """
        if single_resample:
            return self._transform_single(obj_img, bg_shape, depth_map, position, scale, rotation, object_class,
                                          faithful)
        bg_h, bg_w = bg_shape
        transformed = obj_img.copy()

        # Pre-processing: Apply slight blur to very small objects to reduce aliasing
        h, w = transformed.shape[:2]
        min_dim = min(h, w)
        if min_dim < 100:
            # Small objects need anti-aliasing to prevent jaggies
            blur_size = 3 if min_dim > 50 else 5

            # Apply blur budget if provided
            if blur_budget:
                actual_blur = blur_budget.apply(blur_size)
                if actual_blur >= 3:
                    blur_size = int(actual_blur)
                    if blur_size % 2 == 0:
                        blur_size += 1
                else:
                    # Skip pre-blur if budget exceeded
                    blur_size = 0

            if blur_size >= 3:
                # Only blur the RGB channels, not alpha
                if transformed.shape[2] == 4:
                    rgb = transformed[:, :, :3]
                    alpha = transformed[:, :, 3:4]
                    rgb = cv2.GaussianBlur(rgb, (blur_size, blur_size), 0.5)
                    transformed = np.concatenate([rgb, alpha], axis=2)
                else:
                    transformed = cv2.GaussianBlur(transformed, (blur_size, blur_size), 0.5)

        # Apply rotation with high-quality interpolation
        if rotation is None:
            rotation = random.uniform(-45, 45)

        if rotation != 0:
            h, w = transformed.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            # Use INTER_CUBIC for rotation - better quality than LINEAR
            transformed = cv2.warpAffine(
                transformed, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0) if transformed.shape[2] == 4 else (0, 0, 0)
            )

        p_h, p_w = transformed.shape[:2]
        if p_h == 0 or p_w == 0:
            return None, (0, 0), 1.0

        # Calculate scale using physics-based approach
        if scale is None:
            scale = self._pick_scale((p_w, p_h), lambda: faithful_scale.alpha_bbox_wh(transformed), bg_shape,
                                     depth_map, position, object_class, faithful)

        # Apply scale with appropriate interpolation
        if scale != 1.0:
            new_w = int(p_w * scale)
            new_h = int(p_h * scale)
            if new_w > 0 and new_h > 0:
                # Choose interpolation based on scaling direction
                if scale > 1.0:
                    # Upscaling: Use LANCZOS4 for best quality (prevents jaggies)
                    interpolation = cv2.INTER_LANCZOS4
                else:
                    # Downscaling: Use AREA for best quality (prevents moire)
                    interpolation = cv2.INTER_AREA

                transformed = cv2.resize(transformed, (new_w, new_h), interpolation=interpolation)

                # For significant upscaling, apply subtle smoothing to blend edges
                if scale > 2.0:
                    if transformed.shape[2] == 4:
                        rgb = transformed[:, :, :3]
                        alpha = transformed[:, :, 3:4]
                        rgb = cv2.GaussianBlur(rgb, (3, 3), 0.3)
                        transformed = np.concatenate([rgb, alpha], axis=2)
                    else:
                        transformed = cv2.GaussianBlur(transformed, (3, 3), 0.3)

        return self._finalize_transform(transformed, bg_shape, position, scale)

    def _pick_scale(self, canvas_wh, bbox_wh_fn, bg_shape, depth_map, position, object_class, faithful) -> float:
        """Scale of the object: faithful (real size distribution), depth-aware or random (original order)."""
        bg_h, bg_w = bg_shape
        p_w, p_h = canvas_wh
        if faithful is not None:
            floor_px = max(int(min(bg_h, bg_w) * 0.025), 15)
            return faithful.resolve((p_w, p_h), bbox_wh_fn(), (bg_w, bg_h), floor_px)
        if depth_map is not None:
            return self._calculate_depth_aware_scale((p_h, p_w), (bg_h, bg_w), depth_map, position, object_class)
        max_scale = min((bg_w * 0.7) / p_w, (bg_h * 0.7) / p_h, self.max_upscale_ratio)
        return random.uniform(0.1, max_scale) if max_scale > 0.1 else 0.5

    @staticmethod
    def _finalize_transform(transformed, bg_shape, position, scale):
        """Minimum-size check and random fallback position of a transformed object."""
        bg_h, bg_w = bg_shape
        p_h, p_w = transformed.shape[:2]
        if p_h == 0 or p_w == 0:
            return None, (0, 0), scale

        # Adaptive minimum size check
        # Calculate minimum based on image size (2.5% of min dimension)
        min_dimension = int(min(bg_h, bg_w) * 0.025)  # 2.5% = 16px for 640x640
        min_dimension = max(min_dimension, 15)  # Absolute minimum: 15px

        if p_h < min_dimension or p_w < min_dimension:
            logger.debug(f"Object too small after scaling ({p_w}x{p_h} < {min_dimension}), skipping")
            return None, (0, 0), scale

        # Calculate position
        if position is None:
            max_x = max(bg_w - p_w, 1)
            max_y = max(bg_h - p_h, 1)
            position = (random.randrange(0, max_x), random.randrange(0, max_y))

        return transformed, position, scale

    def _transform_single(self, obj_img, bg_shape, depth_map, position, scale, rotation, object_class, faithful):
        """`_transform_object` with a single resampling (profile `single_resample`, app/resample.py).

        No pre-blur and one warpAffine for rotation + scale. Same random calls, in the same order, as the old path.
        """
        if rotation is None:
            rotation = random.uniform(-45, 45)
        canvas_wh, bbox_wh = resample.rotated_geometry(obj_img, rotation)
        if min(canvas_wh) <= 0:
            return None, (0, 0), 1.0
        if scale is None:
            scale = self._pick_scale(canvas_wh, lambda: bbox_wh, bg_shape, depth_map, position, object_class, faithful)
        transformed = resample.warp_single(obj_img, rotation, scale) if scale > 0 else None
        if transformed is None:
            return None, (0, 0), scale
        return self._finalize_transform(transformed, bg_shape, position, scale)

    def _calculate_depth_aware_scale(
        self,
        obj_dims: Tuple[int, int],
        bg_dims: Tuple[int, int],
        depth_map: np.ndarray,
        position: Optional[Tuple[int, int]] = None,
        object_class: str = "default",
    ) -> float:
        """
        Calculate scale factor using physics-based perspective model.

        This algorithm considers:
        1. Real-world size of the object type (shark=2.5m, can=0.12m, etc.)
        2. Depth/distance in the scene (perspective scaling)
        3. Source image resolution vs object's real size
        4. Natural size constraints to avoid unrealistic compositions

        The model assumes:
        - Underwater visibility of ~15m (moderate clarity)
        - Source images captured at ~2m reference distance
        - Linear perspective scaling with distance

        Args:
            obj_dims: (height, width) of the object in pixels
            bg_dims: (height, width) of the background
            depth_map: Normalized depth map (0=near, 1=far)
            position: Optional (x, y) position for depth sampling
            object_class: Type of object for real-world size lookup

        Returns:
            Scale factor to apply to the object
        """
        obj_h, obj_w = obj_dims
        bg_h, bg_w = bg_dims

        # =================================================================
        # Step 1: Get real-world size and estimate source image scale
        # =================================================================
        real_size_meters = get_real_world_size(object_class)

        obj_max_dim = max(obj_h, obj_w)

        # =================================================================
        # Step 2: Get depth value and convert to estimated distance
        # =================================================================
        depth_normalized = depth_map.astype(np.float32)
        if depth_normalized.max() > 1:
            depth_normalized = depth_normalized / depth_normalized.max()

        if position:
            x, y = position
            x = max(0, min(x, bg_w - 1))
            y = max(0, min(y, bg_h - 1))
            depth_value = depth_normalized[y, x]
        else:
            # Random depth selection with preference for mid-range
            depth_value = random.triangular(0.2, 0.8, 0.5)

        # Convert normalized depth (0-1) to estimated distance in meters
        # Using moderate underwater visibility (~15m)
        max_visible_distance = UNDERWATER_VISIBILITY.get('moderate', 15.0)
        min_distance = 0.5  # Objects closer than 0.5m are too close
        estimated_distance = min_distance + depth_value * (max_visible_distance - min_distance)

        # =================================================================
        # Step 3: Calculate perspective scale factor
        # =================================================================
        # Objects appear smaller as distance increases (1/d relationship)
        # Scale relative to reference capture distance
        perspective_scale = REFERENCE_CAPTURE_DISTANCE / estimated_distance

        # =================================================================
        # Step 4: Calculate target size in background image
        # =================================================================
        # Estimate background's pixels-per-meter at different depths
        # For training data, we want objects to be clearly visible
        # Using 2.5m scene width for larger, more detectable objects
        bg_scene_width_meters = 2.5  # Smaller scene = larger objects
        bg_pixels_per_meter = bg_w / bg_scene_width_meters

        # Clamp perspective scale to avoid objects being too small at distance
        # Objects beyond ~10m shouldn't shrink much more
        clamped_perspective = max(perspective_scale, 0.35)

        # Target size = real_size * bg_resolution * perspective
        # Apply boost (2.5x) to ensure objects are large enough for detection
        target_size_pixels = real_size_meters * bg_pixels_per_meter * clamped_perspective * 2.5

        # Calculate scale to achieve target size
        scale = target_size_pixels / obj_max_dim

        # =================================================================
        # Step 5: Apply relative size constraints
        # =================================================================
        obj_area = obj_h * obj_w
        bg_area = bg_h * bg_w
        relative_size = obj_area / bg_area

        # Size thresholds (relative to background area)
        SIZE_SMALL = 0.08
        SIZE_MEDIUM = 0.20
        SIZE_LARGE = 0.35

        # Determine max upscale based on current relative size
        # These are more permissive to allow physics-based sizing to work
        if relative_size < 0.01:
            max_upscale_for_size = 4.0
            size_category = "tiny"
        elif relative_size < SIZE_SMALL:
            max_upscale_for_size = 3.0
            size_category = "small"
        elif relative_size < SIZE_MEDIUM:
            max_upscale_for_size = 2.0
            size_category = "medium"
        elif relative_size < SIZE_LARGE:
            max_upscale_for_size = 1.5
            size_category = "large"
        else:
            max_upscale_for_size = 1.0
            size_category = "huge"

        # Cap upscaling based on relative size
        if scale > 1.0:
            scale = min(scale, max_upscale_for_size)

        # =================================================================
        # Step 6: Apply size moderation for large objects
        # =================================================================
        if relative_size > SIZE_SMALL and scale > 1.0:
            # Smoothly reduce upscaling as object gets larger
            moderation = 1.0 - (relative_size - SIZE_SMALL) / (1.0 - SIZE_SMALL)
            moderation = np.clip(moderation, 0.3, 1.0)
            scale = 1.0 + (scale - 1.0) * moderation

        # =================================================================
        # Step 7: Final constraints
        # =================================================================
        # Max 50% of any background dimension (more conservative for realism)
        max_scale_w = (bg_w * 0.5) / obj_w
        max_scale_h = (bg_h * 0.5) / obj_h
        scale = min(scale, max_scale_w, max_scale_h)

        # Apply global max upscale ratio
        scale = min(scale, self.max_upscale_ratio)

        # Minimum scale to keep object visible and recognizable for training
        # At least 150px for good detectability in object detection models
        min_size_pixels = 150
        min_scale = min_size_pixels / obj_max_dim
        # Minimum 15% of original size to preserve enough detail
        scale = max(scale, min_scale, 0.15)

        logger.debug(
            f"Physics scale: {object_class} ({real_size_meters:.2f}m), "
            f"obj={obj_w}x{obj_h}px, depth={depth_value:.2f} (~{estimated_distance:.1f}m), "
            f"perspective={perspective_scale:.2f}, size_cat={size_category}, "
            f"final_scale={scale:.3f}"
        )

        return scale

    def _clip_bbox_to_image(
        self,
        bbox: Tuple[int, int, int, int],
        img_width: int,
        img_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Clip bounding box to image bounds.

        Args:
            bbox: (x1, y1, x2, y2)
            img_width, img_height: Image dimensions

        Returns:
            Clipped (x1, y1, x2, y2) within [0, img_width) x [0, img_height)
        """
        x1, y1, x2, y2 = bbox

        # Clip to image bounds
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(x1 + 1, min(x2, img_width))
        y2 = max(y1 + 1, min(y2, img_height))

        return (x1, y1, x2, y2)

    def _recalculate_bbox_after_global_effects(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        initial_bbox: Tuple[int, int, int, int],
        threshold: int = 3
    ) -> Tuple[int, int, int, int]:
        """
        Recalculate bbox to include visual changes from global effects.

        Uses image differencing to detect caustics/underwater color shifts.
        More conservative threshold than shadow detection (3 vs 5).

        Args:
            before_image: Composite before global effects
            after_image: Composite after global effects
            initial_bbox: Original bbox (x1, y1, x2, y2)
            threshold: Change detection threshold (default: 3)

        Returns:
            Expanded bbox (x1, y1, x2, y2) including global effect changes
        """
        x1, y1, x2, y2 = initial_bbox
        bg_h, bg_w = before_image.shape[:2]

        # Expand search region (20px - less than shadows: 80px)
        expansion = 20
        search_x1 = max(0, x1 - expansion)
        search_y1 = max(0, y1 - expansion)
        search_x2 = min(bg_w, x2 + expansion)
        search_y2 = min(bg_h, y2 + expansion)

        # Extract ROIs
        roi_before = before_image[search_y1:search_y2, search_x1:search_x2]
        roi_after = after_image[search_y1:search_y2, search_x1:search_x2]

        # Detect changes
        diff = cv2.absdiff(roi_after, roi_before)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, visible_mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

        # Clean noise
        kernel = np.ones((3, 3), np.uint8)
        visible_mask = cv2.morphologyEx(visible_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x_rel, y_rel, w_rel, h_rel = cv2.boundingRect(largest)

            # Convert to global coordinates
            x_new = search_x1 + x_rel
            y_new = search_y1 + y_rel

            # Union with original bbox
            x1 = min(x1, x_new)
            y1 = min(y1, y_new)
            x2 = max(x2, x_new + w_rel)
            y2 = max(y2, y_new + h_rel)

        return (x1, y1, x2, y2)
