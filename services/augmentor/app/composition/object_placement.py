"""Object placement: position selection, attempts and per-object trace."""

import os
import random
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np
from loguru import logger

from app import placement
from app import effects_underwater
from app import crop_augment
from app import faithful_scale
from app import background_match
from app import copy_paste
from app import harmonize
from app import position_stats
from app import sharpness
from app.synth_context import ImagePlan
from app.models.schemas import ObjectPlacement, EffectsConfig, EffectType, LightingInfo
from app.composition.results import BlurBudget
from app.composition.geometry import compute_iou, check_mask_overlap, refine_object_mask


class PlacementMixin:
    """Object placement: position selection, attempts and per-object trace."""

    async def _place_object(
        self,
        composite: np.ndarray,
        obj_placement: ObjectPlacement,
        depth_map: Optional[np.ndarray],
        lighting_info: Optional[LightingInfo],
        effects: List[EffectType],
        effects_config: EffectsConfig,
        placement_mask: np.ndarray,
        existing_bboxes: List[Tuple],
        debug_output_dir: Optional[str] = None,
        object_index: int = 0,
        region_map: Optional[np.ndarray] = None,
        plan: Optional[ImagePlan] = None,
        trace_out: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[np.ndarray, Tuple, np.ndarray]]:
        """Place a single object on the composite image.

        With a synthesis plan (profile v10) the pre-sampled paste augmentation is
        applied to the crop, the scale follows the real size distribution of the
        class and `trace_out` receives scale, target size and augmentations.
        """
        # Load object image (cached: batches reuse the same object crops;
        # _transform_object copies before mutating)
        obj_img = self._object_cache.get(obj_placement.image_path)
        if obj_img is None:
            obj_img = cv2.imread(obj_placement.image_path, cv2.IMREAD_UNCHANGED)
            if obj_img is None:
                logger.warning(f"Failed to load object: {obj_placement.image_path}")
                return None
            self._object_cache.put(obj_placement.image_path, obj_img)

        profile = plan.profile if plan is not None else None
        if obj_placement.augment:
            obj_img = crop_augment.apply_augment(obj_img, obj_placement.augment)
        # M1/M2 (v11): the source ring is measured on the original crop and matting is applied to the crop
        # BEFORE scaling/rotating (fractional alpha + F); the medium is changed afterwards, on F
        medium_src, medium_ctx = None, None
        if profile is not None and profile.medium_transfer:
            medium_src = self._medium_source(obj_placement.image_path, obj_img, profile, bool(obj_placement.augment))
            medium_ctx = self._medium_context(composite, existing_bboxes, plan, profile)
        if profile is not None and profile.edge_matting:
            obj_img = self._matte_crop(obj_placement.image_path, obj_img, obj_placement.class_name, profile,
                                       bool(obj_placement.augment))
        min_area_ratio = self.min_area_ratio
        max_area_ratio = self.max_area_ratio
        try_count = self.try_count
        if profile is not None:
            if profile.min_mask_area_ratio is not None:
                min_area_ratio = profile.min_mask_area_ratio
            if profile.max_mask_area_ratio is not None:
                max_area_ratio = profile.max_mask_area_ratio
            try_count = profile.placement_try_count or try_count

        # Debug: Save original object
        if debug_output_dir:
            cv2.imwrite(os.path.join(debug_output_dir, "03_object_original.png"), obj_img)
            logger.info(f"Pipeline debug: saved 03_object_original.png")

        bg_h, bg_w = composite.shape[:2]
        bg_area = bg_h * bg_w

        # Create blur budget based on object size
        # Will be initialized properly after first transform
        blur_budget = None
        visibility_rejects = 0  # attempts discarded by the visibility filter (profile v11)

        # Does this object touch the image border? decided once per object (profile position_from_stats)
        stats_touch = None
        if profile is not None and profile.position_from_stats and obj_placement.position is None:
            stats_touch = position_stats.decide_touch(
                position_stats.load_position_stats(profile.position_stats_path), profile, obj_placement.class_name,
                random)

        # local_patch_match (v11): scores the ring of each candidate position (once per object)
        patch_score = None
        if stats_touch is not None and profile.local_patch_match:
            patch_score = background_match.patch_scorer(profile, obj_placement.image_path, composite)

        for attempt in range(try_count):
            # Apply transformations with physics-based scaling
            faithful = None
            if obj_placement.scale is None:
                quantile = obj_placement.size_quantile if attempt == 0 else random.random()
                faithful = faithful_scale.make_faithful(
                    profile, plan.stats if plan else None, obj_placement.class_name,
                    quantile if obj_placement.size_quantile is not None else None)
            transformed, pos, scale = self._transform_object(
                obj_img,
                (bg_h, bg_w),
                depth_map,
                obj_placement.position,
                obj_placement.scale,
                obj_placement.rotation,
                object_class=obj_placement.class_name,
                blur_budget=blur_budget,
                faithful=faithful,
                single_resample=bool(profile is not None and profile.single_resample),
            )

            if transformed is None:
                continue

            # Semantic placement: replace the random position with one whose
            # anchor lies in a region compatible with the class buoyancy
            semantic_used = False
            if obj_placement.position is None and region_map is not None:
                semantic_pos = placement.sample_position(
                    region_map,
                    obj_placement.class_name,
                    (transformed.shape[1], transformed.shape[0]),
                    behavior=obj_placement.behavior_override,
                    center_weight=self._center_weight(profile, obj_placement.class_name, (bg_w, bg_h)),
                )
                if semantic_pos is not None:
                    pos, semantic_used = semantic_pos, True

            # Position from the real training distribution with a border margin (profile v11)
            if obj_placement.position is None and stats_touch is not None:
                again = (lambda: placement.sample_position(
                    region_map, obj_placement.class_name, (transformed.shape[1], transformed.shape[0]),
                    behavior=obj_placement.behavior_override,
                    center_weight=self._center_weight(profile, obj_placement.class_name, (bg_w, bg_h))))
                pos = self._stats_position(profile, obj_placement.class_name, transformed, pos, (bg_w, bg_h),
                                           semantic_used, stats_touch, again if semantic_used else None,
                                           self._position_avoider(profile, plan, existing_bboxes), patch_score)

            # Initialize blur budget after first successful transform
            if blur_budget is None:
                p_h, p_w = transformed.shape[:2]
                min_dim = min(p_h, p_w)
                blur_budget = BlurBudget(
                    max_blur_budget=effects_config.max_blur_budget,
                    object_size=min_dim
                )

            # Debug: Save transformed object (only once)
            if debug_output_dir and attempt == 0:
                cv2.imwrite(os.path.join(debug_output_dir, "04_object_transformed.png"), transformed)
                logger.info(f"Pipeline debug: saved 04_object_transformed.png")

            # Extract alpha channel
            if transformed.shape[2] == 4:
                alpha_channel = transformed[:, :, 3]
                obj_bgr = transformed[:, :, :3]
            else:
                alpha_channel = np.ones(transformed.shape[:2], dtype=np.uint8) * 255
                obj_bgr = transformed

            # Create mask
            _, mask_bin = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
            alpha_factor, mask_bin = refine_object_mask(
                alpha_channel,
                mask_bin,
                feather_radius=effects_config.alpha_feather_radius,
                use_binary_alpha=effects_config.use_binary_alpha,
                smooth_gradual=not (profile is not None and profile.edge_matting),
                gradual_mask_threshold=127 if (profile is not None and profile.edge_matting) else 10,
            )

            # Depth occlusion: hide object pixels behind closer background
            if effects_config.depth_occlusion and depth_map is not None:
                mask_bin, alpha_factor, visibility = effects_underwater.occlude_object(
                    mask_bin, alpha_factor, depth_map, pos
                )
                if visibility < 0.25:
                    logger.debug(
                        f"Object {obj_placement.class_name} only {visibility:.0%} visible "
                        f"after depth occlusion, retrying placement"
                    )
                    continue

            # Check area constraints
            object_area = cv2.countNonZero(mask_bin)
            if object_area == 0:
                continue

            object_ratio = object_area / bg_area
            if not (min_area_ratio <= object_ratio <= max_area_ratio):
                continue

            # Check overlap
            if check_mask_overlap(placement_mask, mask_bin, pos, self.overlap_threshold):
                continue

            # Calculate bounding box
            x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(mask_bin)
            bbox = (
                pos[0] + x_bb,
                pos[1] + y_bb,
                pos[0] + x_bb + w_bb,
                pos[1] + y_bb + h_bb,
            )

            # Check IoU with existing boxes
            if any(compute_iou(bbox, bb) > self.overlap_threshold for bb in existing_bboxes):
                continue

            # Copy-paste: do not cover the kept real boxes beyond the profile limit
            if profile is not None and profile.copy_paste_real and copy_paste.exceeds_real_overlap(
                    bbox, plan.real_boxes, profile.real_overlap_max):
                continue

            # Calculate paste region
            p_h, p_w = transformed.shape[:2]
            x_paste, y_paste = pos

            x_start_bg = max(x_paste, 0)
            y_start_bg = max(y_paste, 0)
            x_end_bg = min(x_paste + p_w, bg_w)
            y_end_bg = min(y_paste + p_h, bg_h)
            x_start_patch = max(0, -x_paste)
            y_start_patch = max(0, -y_paste)
            eff_w = x_end_bg - x_start_bg
            eff_h = y_end_bg - y_start_bg

            if eff_w <= 0 or eff_h <= 0:
                continue

            # Extract effective regions
            obj_bgr_eff = obj_bgr[y_start_patch:y_start_patch+eff_h, x_start_patch:x_start_patch+eff_w]
            mask_bin_eff = mask_bin[y_start_patch:y_start_patch+eff_h, x_start_patch:x_start_patch+eff_w]
            alpha_eff = alpha_factor[y_start_patch:y_start_patch+eff_h, x_start_patch:x_start_patch+eff_w]
            alpha_eff_3ch = alpha_eff[:, :, np.newaxis] if len(alpha_eff.shape) == 2 else alpha_eff

            roi = composite[y_start_bg:y_end_bg, x_start_bg:x_end_bg]

            # Apply per-object effects
            obj_bgr_eff_original = obj_bgr_eff.copy()  # Save original for identity validation
            obj_processed = obj_bgr_eff.copy()

            if EffectType.COLOR_CORRECTION in effects:
                obj_processed = self._apply_color_correction(
                    obj_processed, roi, mask_bin_eff, effects_config.color_intensity
                )
                # Debug: Save after color correction
                if debug_output_dir:
                    debug_img = roi.copy()
                    alpha_3ch = alpha_eff_3ch if len(alpha_eff_3ch.shape) == 3 else alpha_eff_3ch[:, :, np.newaxis]
                    debug_blend = (obj_processed.astype(np.float32) * alpha_3ch +
                                   debug_img.astype(np.float32) * (1.0 - alpha_3ch)).astype(np.uint8)
                    cv2.imwrite(os.path.join(debug_output_dir, "05_color_correction.jpg"), debug_blend)
                    logger.info(f"Pipeline debug: saved 05_color_correction.jpg")

            if medium_src is not None:
                obj_processed = self._apply_medium_transfer(
                    obj_processed, mask_bin_eff, (x_start_bg, y_start_bg), composite, medium_src, medium_ctx, profile)

            if profile is not None and profile.harmonize:
                obj_processed = harmonize.harmonize_object(
                    obj_processed, mask_bin_eff, composite, (x_start_bg, y_start_bg, eff_w, eff_h),
                    profile.harmonize_strength, profile.harmonize_l_weight, profile.harmonize_ring)

            if profile is not None and profile.match_sharpness:
                obj_processed, _ = sharpness.match_sharpness(
                    obj_processed, mask_bin_eff, composite, (x_start_bg, y_start_bg, eff_w, eff_h),
                    obj_placement.class_name, profile, random.Random(random.getrandbits(32)))
            elif EffectType.BLUR_MATCHING in effects:
                obj_processed = self._apply_blur_matching(
                    obj_processed, roi, mask_bin_eff, effects_config.blur_strength, blur_budget,
                    blur_max=copy_paste.blur_kernel_cap(profile) if profile is not None else None,
                )
                # Debug: Save after blur matching
                if debug_output_dir:
                    debug_img = roi.copy()
                    alpha_3ch = alpha_eff_3ch if len(alpha_eff_3ch.shape) == 3 else alpha_eff_3ch[:, :, np.newaxis]
                    debug_blend = (obj_processed.astype(np.float32) * alpha_3ch +
                                   debug_img.astype(np.float32) * (1.0 - alpha_3ch)).astype(np.uint8)
                    cv2.imwrite(os.path.join(debug_output_dir, "06_blur_matching.jpg"), debug_blend)
                    logger.info(f"Pipeline debug: saved 06_blur_matching.jpg")

            if EffectType.LIGHTING in effects:
                obj_processed = self._apply_lighting_effect(
                    obj_processed, effects_config.lighting_type, effects_config.lighting_intensity
                )

            # Apply motion blur with probability
            if EffectType.MOTION_BLUR in effects:
                if random.random() < effects_config.motion_blur_probability:
                    obj_processed = self._apply_motion_blur(
                        obj_processed, effects_config.motion_blur_kernel, blur_budget
                    )

            obj_processed = np.clip(obj_processed, 0, 255).astype(np.uint8)

            # Validate object identity after effects
            if effects_config.validate_identity:
                is_valid, metrics = self._validate_object_identity(
                    obj_bgr_eff_original,
                    obj_processed,
                    mask_bin_eff,
                    effects_config
                )
                if not is_valid:
                    logger.warning(
                        f"Object identity validation failed for {obj_placement.class_name} "
                        f"(attempt {attempt+1}/{self.try_count}): "
                        f"color_shift={metrics['color_shift']:.1f}, "
                        f"sharpness={metrics['sharpness_ratio']:.2f}, "
                        f"contrast={metrics['contrast_ratio']:.2f}"
                    )
                    continue  # Try next placement attempt

            # Generate depth-aware shadow
            bg_with_shadow = composite.copy()
            if EffectType.SHADOWS in effects:
                bg_with_shadow = self._apply_shadow(
                    bg_with_shadow,
                    mask_bin_eff,
                    (x_start_bg, y_start_bg, eff_w, eff_h),
                    lighting_info,
                    effects_config.shadow_opacity,
                    effects_config.shadow_blur,
                    depth_map=depth_map,
                    position=pos,
                )
                # Debug: Save after shadow application
                if debug_output_dir:
                    cv2.imwrite(os.path.join(debug_output_dir, "07_shadows.jpg"), bg_with_shadow)
                    logger.info(f"Pipeline debug: saved 07_shadows.jpg")

            # Blend object onto background
            roi_with_shadow = bg_with_shadow[y_start_bg:y_end_bg, x_start_bg:x_end_bg]

            if EffectType.POISSON_BLEND in effects:
                blended = self._apply_poisson_blend(
                    obj_processed, roi_with_shadow, mask_bin_eff
                )
            else:
                # Alpha blending
                blended = (
                    obj_processed.astype(np.float32) * alpha_eff_3ch +
                    roi_with_shadow.astype(np.float32) * (1.0 - alpha_eff_3ch)
                ).astype(np.uint8)

            # Apply edge smoothing
            if EffectType.EDGE_SMOOTHING in effects and not (
                    profile is not None and profile.edge_matting and not profile.edge_matting_edge_smoothing):
                blended = self._apply_edge_smoothing(
                    blended, roi_with_shadow, mask_bin_eff, effects_config.edge_feather
                )

            # Update composite
            result = bg_with_shadow.copy()
            result[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = blended

            # Create object mask in full image coordinates
            obj_mask_full = np.zeros((bg_h, bg_w), dtype=np.uint8)
            obj_mask_full[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = mask_bin_eff

            # Visibility filter: an object that blends into its surroundings is not pasted (nor annotated)
            if self._fails_visibility(profile, obj_placement.class_name, result, bbox):
                visibility_rejects += 1
                if trace_out is not None:
                    trace_out["visibility_rejects"] = visibility_rejects
                continue

            # Recalculate bounding box to include shadows and all visual effects
            # This ensures annotations cover the entire visible region (object + shadows + blur)
            # (profile `tight_bbox`: the annotation stays the box fitted to the object mask)
            tight_bbox = profile is not None and profile.tight_bbox
            if not tight_bbox and (EffectType.SHADOWS in effects or EffectType.MOTION_BLUR in effects):
                try:
                    # Define search region (expand bbox by shadow blur size)
                    x1_old, y1_old, x2_old, y2_old = bbox
                    max_shadow_extent = 80  # Max shadow blur + offset

                    # Expand search region
                    search_x1 = max(0, x1_old - max_shadow_extent)
                    search_y1 = max(0, y1_old - max_shadow_extent)
                    search_x2 = min(bg_w, x2_old + max_shadow_extent)
                    search_y2 = min(bg_h, y2_old + max_shadow_extent)

                    # Extract search regions from both images
                    roi_result = result[search_y1:search_y2, search_x1:search_x2]
                    roi_composite = composite[search_y1:search_y2, search_x1:search_x2]

                    # Detect changes in the search region
                    diff = cv2.absdiff(roi_result, roi_composite)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

                    # Use adaptive threshold to handle varying shadow intensities
                    # Threshold at 5 (shadows are subtle, typically 3-15 intensity change)
                    _, visible_mask = cv2.threshold(diff_gray, 5, 255, cv2.THRESH_BINARY)

                    # Apply morphological operations to clean up noise and connect shadow regions
                    kernel_small = np.ones((3, 3), np.uint8)
                    visible_mask = cv2.morphologyEx(visible_mask, cv2.MORPH_OPEN, kernel_small)  # Remove noise
                    kernel_large = np.ones((7, 7), np.uint8)
                    visible_mask = cv2.morphologyEx(visible_mask, cv2.MORPH_CLOSE, kernel_large)  # Connect regions

                    # Find contours in the search region
                    contours, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Get bounding box of the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        x_rel, y_rel, w_rel, h_rel = cv2.boundingRect(largest_contour)

                        # Convert back to full image coordinates
                        x_new = search_x1 + x_rel
                        y_new = search_y1 + y_rel

                        # Expand bbox to include detected region (union)
                        x1_new = min(x1_old, x_new)
                        y1_new = min(y1_old, y_new)
                        x2_new = max(x2_old, x_new + w_rel)
                        y2_new = max(y2_old, y_new + h_rel)

                        # Sanity check: new bbox shouldn't be more than 2x larger
                        old_area = (x2_old - x1_old) * (y2_old - y1_old)
                        new_area = (x2_new - x1_new) * (y2_new - y1_new)
                        if new_area < old_area * 2.5:  # Reasonable expansion
                            bbox = (x1_new, y1_new, x2_new, y2_new)
                        else:
                            logger.debug(f"Bbox expansion too large ({new_area/old_area:.1f}x), keeping original")

                except Exception as e:
                    logger.debug(f"Bbox recalculation failed: {e}, using original bbox")

            if trace_out is not None:
                trace_out.update(self._object_trace(
                    obj_placement, scale, faithful, bbox, (bg_w, bg_h)))
            return result, bbox, obj_mask_full

        return None

    @staticmethod
    def _center_weight(profile, class_name: str, bg_wh: Tuple[int, int]):
        """Weight of the semantic anchors by the real density of the box center (None = uniform, as always)."""
        if profile is None or not profile.position_from_stats:
            return None
        stats = position_stats.load_position_stats(profile.position_stats_path)
        return position_stats.center_weight_fn(stats, class_name, bg_wh)

    def _stats_position(self, profile, class_name: str, transformed: np.ndarray, pos: Tuple[int, int],
                        bg_wh: Tuple[int, int], semantic: bool, touch: Optional[bool] = None,
                        next_base=None, avoid=None, score=None) -> Tuple[int, int]:
        """New top-left of the canvas so that the ALPHA box follows the real border/center distribution."""
        stats = position_stats.load_position_stats(profile.position_stats_path)
        if transformed.shape[2] == 4:
            bx, by, bw, bh = cv2.boundingRect((transformed[:, :, 3] > 10).astype(np.uint8))
        else:
            bx, by, bw, bh = 0, 0, transformed.shape[1], transformed.shape[0]
        if bw == 0 or bh == 0:
            return pos
        return position_stats.search_position(
            stats, profile, class_name, (bx, by), (bw, bh), bg_wh, random, first_base=pos if semantic else None,
            next_base=next_base if semantic else None, touch=touch, avoid=avoid, score=score,
            n_candidates=profile.local_patch_candidates)

    def _position_avoider(self, profile, plan, existing_bboxes):
        """f(box) -> True if a box would overlap a kept real box or an already placed object (rejected later anyway)."""
        real = plan.real_boxes if profile.copy_paste_real else []

        def avoid(box) -> bool:
            return (copy_paste.exceeds_real_overlap(box, real, profile.real_overlap_max)
                    or any(compute_iou(box, bb) > self.overlap_threshold for bb in existing_bboxes))
        return avoid

    @staticmethod
    def _object_trace(obj_placement, scale, faithful, bbox, bg_wh) -> Dict[str, Any]:
        """Traceability of a placed object: scale, target size, augmentations and placement."""
        augmentations = dict(obj_placement.augment or {})
        if obj_placement.behavior_override:
            augmentations["behavior"] = obj_placement.behavior_override
        return {
            "scale": round(float(scale), 4),
            "target_rel_size": round(faithful.target, 4) if faithful and faithful.target is not None else None,
            "augmentations": augmentations or None,
        }
