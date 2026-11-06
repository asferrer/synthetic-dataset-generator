"""Single-image composition entry point."""

import os
import json
import time
import random
from typing import List, Dict, Optional, Any

import cv2
import numpy as np
from loguru import logger

from app import placement
from app import effects_underwater
from app import output_format
from app import copy_paste
from app.mask_rle import encode_rle
from app.synth_context import ImagePlan
from app.models.schemas import ObjectPlacement, AnnotationBox, EffectsConfig, EffectType, LightingInfo, TRACE_FIELDS
from app.composition.results import ComposeResult


class SingleImageMixin:
    """Single-image composition entry point."""

    async def compose(
        self,
        background_path: str,
        objects: List[ObjectPlacement],
        depth_map_path: Optional[str] = None,
        depth_map: Optional[np.ndarray] = None,
        depth_source: Optional[str] = None,
        lighting_info: Optional[LightingInfo] = None,
        effects: List[EffectType] = None,
        effects_config: EffectsConfig = None,
        output_path: str = None,
        save_annotations: bool = True,
        debug_output_dir: Optional[str] = None,
        semantic_placement: bool = True,
        plan: Optional[ImagePlan] = None,
    ) -> ComposeResult:
        """
        Compose a single synthetic image.

        Args:
            background_path: Path to background image
            objects: List of objects to place
            depth_map_path: Optional pre-computed depth map (path to .npy file)
            depth_map: Optional pre-computed depth map (numpy array, takes precedence over depth_map_path)
            lighting_info: Optional pre-computed lighting
            effects: Effects to apply
            effects_config: Effect configuration
            output_path: Where to save result
            save_annotations: Whether to save annotations JSON
            debug_output_dir: Optional directory to save intermediate pipeline images

        Returns:
            ComposeResult with annotations and metadata
        """
        effects = effects or [EffectType.COLOR_CORRECTION, EffectType.BLUR_MATCHING]
        effects_config = effects_config or EffectsConfig()

        # Load background
        bg_image = cv2.imread(background_path)
        if bg_image is None:
            raise FileNotFoundError(f"Background not found: {background_path}")

        bg_h, bg_w = bg_image.shape[:2]
        result = ComposeResult()
        result.effects_applied = [e.value for e in effects]

        # Debug: Save background
        if debug_output_dir:
            cv2.imwrite(os.path.join(debug_output_dir, "00_background.jpg"), bg_image)
            logger.info(f"Pipeline debug: saved 00_background.jpg")

        # Load or compute depth map (depth_map numpy array takes precedence over depth_map_path)
        if depth_map is None and depth_map_path:
            depth_map = np.load(depth_map_path)
            result.depth_used = True
            result.depth_source = depth_source or "precomputed"
        elif depth_map is not None:
            result.depth_used = True
            result.depth_source = depth_source or "precomputed"

        # If no depth map provided but needed for effects, compute it
        needs_depth = (
            EffectType.BLUR_MATCHING in effects
            or EffectType.SHADOWS in effects
            or effects_config.underwater_physical
            or effects_config.caustics_depth_aware
            or effects_config.depth_occlusion
        )
        if depth_map is None and needs_depth:
            depth_map, result.depth_source = await self._get_depth_map(
                bg_image, cache_key=background_path
            )
            result.depth_used = True

        # Debug: Save depth map as colored heatmap
        if debug_output_dir and depth_map is not None:
            depth_normalized = (depth_map * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
            cv2.imwrite(os.path.join(debug_output_dir, "01_depth_map.png"), depth_colored)
            logger.info(f"Pipeline debug: saved 01_depth_map.png")

        # Profile with its own resolution: the background is resized before pasting the objects
        if plan is not None and plan.output_size is not None:
            bg_image, depth_map = output_format.resize_to(bg_image, depth_map, plan.output_size)
            bg_h, bg_w = bg_image.shape[:2]
        if plan is not None:
            result.image_size = (bg_w, bg_h)

        # Estimate or use provided lighting
        if lighting_info is None and EffectType.SHADOWS in effects:
            lighting_info = await self._estimate_lighting(bg_image)

        # Debug: Save lighting analysis visualization
        if debug_output_dir and lighting_info is not None:
            lighting_vis = self._visualize_lighting(bg_image.copy(), lighting_info)
            cv2.imwrite(os.path.join(debug_output_dir, "02_lighting_analysis.jpg"), lighting_vis)
            logger.info(f"Pipeline debug: saved 02_lighting_analysis.jpg")

        # Scene analysis for semantic understanding (ALWAYS enabled when segmentation service available)
        scene_analysis = None
        debug_info = None
        placement_decisions = []
        temp_bg_path = None  # Will be set if we save background for segmentation

        # Perform scene analysis for intelligent placement (not just debug mode)
        cached_scene = None if debug_output_dir else self._scene_cache.get(background_path)
        if cached_scene is not None:
            scene_analysis = cached_scene
        elif self.segmentation_client is not None:
            try:
                # Save background to shared volume for segmentation service
                temp_bg_path = f"/shared/temp/pipeline_bg_{int(time.time() * 1000)}_{random.randrange(1 << 20)}.jpg"
                os.makedirs(os.path.dirname(temp_bg_path), exist_ok=True)
                cv2.imwrite(temp_bg_path, bg_image)

                if debug_output_dir:
                    # Full debug analysis with visualization
                    scene_analysis, debug_info = await self.segmentation_client.analyze_scene_with_debug(
                        bg_image,
                        image_path=temp_bg_path,
                        save_visualization=False,  # We'll create our own
                        image_id="pipeline_debug",
                    )
                    # Create scene analysis visualization
                    scene_vis = self._visualize_scene_analysis(bg_image.copy(), scene_analysis, debug_info)
                    cv2.imwrite(os.path.join(debug_output_dir, "02b_scene_analysis.jpg"), scene_vis)
                    logger.info(f"Pipeline debug: saved 02b_scene_analysis.jpg")
                else:
                    # Normal mode: just analyze scene (no debug overhead)
                    scene_analysis = await self.segmentation_client.analyze_scene(
                        bg_image,
                        image_path=temp_bg_path,
                    )
                    logger.debug(f"Scene analysis: dominant={scene_analysis.dominant_region}")
                if scene_analysis is not None:
                    self._scene_cache.put(background_path, scene_analysis)
            except Exception as e:
                logger.warning(f"Scene analysis failed: {e}, creating heuristic analysis")
                scene_analysis = self._create_heuristic_scene_analysis(bg_image)

        # Region map at background resolution for semantic placement
        region_map_full = None
        if semantic_placement and scene_analysis is not None:
            region_map_full = placement.region_map_at(
                getattr(scene_analysis, "region_map", None), (bg_w, bg_h)
            )
            if region_map_full is None:
                logger.debug("Semantic placement enabled but scene analysis has no region map")

        # Initialize placement tracking
        placement_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        existing_bboxes = []
        composite = bg_image.copy()

        # Place each object - track all objects in debug mode
        object_debug_results = []  # Collect debug info for all objects
        placed_objects_metadata = []  # Store metadata to create annotations AFTER global effects
        visibility_log: Dict[str, Dict[str, int]] = {"rejected": {}, "dropped": {}}
        for obj_idx, obj_placement in enumerate(objects):
            # Create per-object debug directory if in debug mode
            obj_debug_dir = None
            if debug_output_dir:
                obj_debug_dir = os.path.join(debug_output_dir, f"object_{obj_idx:02d}_{obj_placement.class_name}")
                os.makedirs(obj_debug_dir, exist_ok=True)

            # Track placement decision for debug visualizations only —
            # semantic placement itself uses the local region map (no HTTP per object)
            placement_decision = None
            if debug_output_dir and scene_analysis is not None and self.segmentation_client is not None:
                try:
                    # Determine position for compatibility check
                    check_pos = obj_placement.position or (bg_w // 2, bg_h // 2)
                    _, reason, placement_decision = await self.segmentation_client.check_compatibility_with_debug(
                        obj_placement.class_name,
                        check_pos,
                        bg_image,
                        image_path=temp_bg_path,
                    )
                except Exception as e:
                    logger.debug(f"Placement decision tracking failed: {e}")

            trace_out: Dict[str, Any] = {}
            obj_result = await self._place_object(
                plan=plan,
                trace_out=trace_out,
                composite=composite,
                obj_placement=obj_placement,
                depth_map=depth_map,
                lighting_info=lighting_info,
                effects=effects,
                effects_config=effects_config,
                placement_mask=placement_mask,
                existing_bboxes=existing_bboxes,
                debug_output_dir=obj_debug_dir,
                object_index=obj_idx,  # Pass object index for debug naming
                region_map=region_map_full,
            )

            self._log_visibility(visibility_log, obj_placement.class_name, trace_out, obj_result is None)
            if obj_result is not None:
                composite, bbox, obj_mask = obj_result

                # Collect debug info
                if debug_output_dir:
                    object_debug_results.append({
                        'index': obj_idx,
                        'class_name': obj_placement.class_name,
                        'bbox': bbox,
                        'placement_decision': placement_decision,
                        'debug_dir': obj_debug_dir,
                    })
                existing_bboxes.append(bbox)

                # Update placement decision with actual position
                if placement_decision is not None:
                    placement_decision.requested_position = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    placement_decisions.append(placement_decision)

                # Update placement mask
                if obj_mask is not None:
                    kernel = np.ones((5, 5), np.uint8)
                    dilated = cv2.dilate(obj_mask, kernel, iterations=1)
                    placement_mask = cv2.bitwise_or(placement_mask, dilated)

                # Store metadata for annotation creation AFTER global effects
                placed_objects_metadata.append({
                    'bbox': bbox,
                    'class_name': obj_placement.class_name,
                    'mask': obj_mask,
                    'source_object': obj_placement.image_path,
                    'trace': trace_out,
                })
                result.objects_placed += 1

        # Clean up temp background shared with the segmentation service
        if temp_bg_path:
            try:
                os.remove(temp_bg_path)
            except OSError:
                pass

        # Debug: Save composite after object placement (before global effects)
        if debug_output_dir:
            cv2.imwrite(os.path.join(debug_output_dir, "08_objects_placed.jpg"), composite)
            logger.info(f"Pipeline debug: saved 08_objects_placed.jpg")

            # Save placement decisions visualization
            if placement_decisions and scene_analysis is not None:
                placement_vis = self._visualize_placement_decisions(
                    composite.copy(),
                    scene_analysis,
                    placement_decisions,
                    existing_bboxes,
                )
                cv2.imwrite(os.path.join(debug_output_dir, "08b_placement_decisions.jpg"), placement_vis)
                logger.info(f"Pipeline debug: saved 08b_placement_decisions.jpg")

            # Create consolidated multi-object summary visualization
            if object_debug_results:
                summary_vis = self._create_multi_object_summary(
                    bg_image.copy(),
                    composite.copy(),
                    object_debug_results,
                    scene_analysis,
                    depth_map,
                )
                cv2.imwrite(os.path.join(debug_output_dir, "08c_multi_object_summary.jpg"), summary_vis)
                logger.info(f"Pipeline debug: saved 08c_multi_object_summary.jpg with {len(object_debug_results)} objects")

        # Save composite BEFORE global effects for bbox recalculation
        composite_before_global = composite.copy()
        global_effects_applied = EffectType.CAUSTICS in effects or EffectType.UNDERWATER in effects

        # Apply global effects
        if EffectType.CAUSTICS in effects:
            composite = effects_underwater.apply_caustics(
                composite,
                effects_config.caustics_intensity,
                seed=random.getrandbits(32),
                depth_map=depth_map if effects_config.caustics_depth_aware else None,
            )
            # Debug: Save after caustics
            if debug_output_dir:
                cv2.imwrite(os.path.join(debug_output_dir, "09_caustics.jpg"), composite)
                logger.info(f"Pipeline debug: saved 09_caustics.jpg")

        underwater_intensity = effects_config.underwater_intensity
        if EffectType.UNDERWATER in effects and plan is not None and plan.profile.haze_from_stats:
            underwater_intensity = output_format.matched_haze_intensity(
                composite, plan.target_l_std, copy_paste.haze_cap(plan.profile), underwater_intensity
            )
            result.image_trace["underwater_intensity"] = round(underwater_intensity, 4)
        if EffectType.UNDERWATER in effects:
            if effects_config.underwater_physical and depth_map is not None:
                composite = effects_underwater.apply_beer_lambert(
                    composite,
                    depth_map,
                    effects_config.water_color,
                    water_clarity=effects_config.water_clarity.value
                    if hasattr(effects_config.water_clarity, "value")
                    else str(effects_config.water_clarity),
                    strength=underwater_intensity,
                )
            else:
                composite = self._apply_underwater_effect(
                    composite,
                    underwater_intensity,
                    effects_config.water_color,
                )
            # Debug: Save after underwater effect
            if debug_output_dir:
                cv2.imwrite(os.path.join(debug_output_dir, "10_underwater.jpg"), composite)
                logger.info(f"Pipeline debug: saved 10_underwater.jpg")

        # Create annotations AFTER global effects with recalculated bboxes
        save_masks = plan is not None and plan.profile.save_masks
        for obj_meta in placed_objects_metadata:
            bbox = obj_meta['bbox']

            # Recalculate bbox to include global effects if enabled
            # (profile `tight_bbox` skips it: a global colour change makes the diff cover the whole +-20 px window)
            tight_bbox = plan is not None and plan.profile.tight_bbox
            if global_effects_applied and effects_config.recalculate_bbox_after_global and not tight_bbox:
                bbox = self._recalculate_bbox_after_global_effects(
                    composite_before_global,
                    composite,
                    bbox
                )

            # Clip bbox to image bounds
            bbox = self._clip_bbox_to_image(bbox, bg_w, bg_h)

            # Create annotation with final bbox
            x1, y1, x2, y2 = bbox
            annotation = AnnotationBox(
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1,
                class_name=obj_meta['class_name'],
                confidence=1.0,
                area=(x2 - x1) * (y2 - y1),
                source_object=obj_meta.get('source_object'),
                mask_rle=(encode_rle(obj_meta['mask']) if save_masks and obj_meta.get('mask') is not None else None),
                **(self._trace_fields(obj_meta.get('trace')) if plan is not None and plan.profile.trace else {}),
            )
            result.annotations.append(annotation)

        if plan is not None and plan.profile.copy_paste_real:
            result.real_annotations = copy_paste.real_annotation_boxes(plan.real_annotations, (bg_w, bg_h))
        if any(visibility_log.values()):
            result.image_trace["visibility"] = visibility_log

        # Debug: Save final result
        if debug_output_dir:
            cv2.imwrite(os.path.join(debug_output_dir, "11_final.jpg"), composite)
            logger.info(f"Pipeline debug: saved 11_final.jpg")

        # Save output
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if plan is not None and plan.jpeg_quality is not None:
                cv2.imwrite(output_path, composite, [cv2.IMWRITE_JPEG_QUALITY, plan.jpeg_quality])
                result.image_trace["jpeg_quality"] = plan.jpeg_quality
            else:
                cv2.imwrite(output_path, composite)

            if save_annotations and result.annotations:
                ann_path = output_path.rsplit('.', 1)[0] + '_annotations.json'
                annotations_dict = [ann.model_dump(exclude={k for k in (*TRACE_FIELDS, 'mask_rle') if getattr(ann, k) is None})
                                    for ann in result.annotations]
                with open(ann_path, 'w') as f:
                    json.dump(annotations_dict, f, indent=2)

        return result

    @staticmethod
    def _trace_fields(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Traceability fields (scale, target size, augmentations) of a placed object."""
        trace = trace or {}
        return {k: trace.get(k) for k in TRACE_FIELDS}
