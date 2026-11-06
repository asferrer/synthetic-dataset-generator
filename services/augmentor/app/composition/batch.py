"""Sequential batch planning and execution (compose_batch) plus helpers shared with the parallel path."""

import os
import shutil
import json
import gc
import time
import glob
import random
from typing import List, Dict, Optional, Callable

import cv2
import torch
from loguru import logger

from app.object_planner import plan_objects
from app.crop_filter import apply_library_filters
from app.synth_context import SynthContext, merge_extra_libraries
from app.synth_profile import SynthProfile
from app.batch_recovery import ANN_SUFFIX, COCO_NAME, FLUSH_EVERY
from app.models.schemas import ObjectPlacement, EffectsConfig, EffectType
from app.composition.results import BatchResult


class BatchMixin:
    """Sequential batch planning and execution (compose_batch) plus helpers shared with the parallel path."""

    async def compose_batch(
        self,
        backgrounds_dir: str,
        objects_dir: str,
        output_dir: str,
        num_images: int,
        targets_per_class: Optional[Dict[str, int]] = None,
        max_objects_per_image: int = 5,
        effects: List[EffectType] = None,
        effects_config: EffectsConfig = None,
        depth_aware: bool = True,
        depth_service_url: Optional[str] = None,
        validate_quality: bool = False,
        validate_physics: bool = False,
        reject_invalid: bool = True,
        save_pipeline_debug: bool = False,
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        resume_from_index: int = 0,
        dataset_info: Optional[Dict] = None,
        semantic_placement: bool = True,
        synth_profile: Optional[SynthProfile] = None,
        seed: Optional[int] = None,
    ) -> BatchResult:
        """
        Compose multiple synthetic images in batch.

        Args:
            backgrounds_dir: Directory with background images
            objects_dir: Directory with object images by class
            output_dir: Output directory
            num_images: Number of images to generate
            targets_per_class: Optional target count per class
            max_objects_per_image: Max objects per image
            effects: Effects to apply
            effects_config: Effect configuration
            depth_aware: Use depth-aware placement
            depth_service_url: URL to depth service
            validate_quality: Run quality validation
            validate_physics: Run physics validation
            reject_invalid: Reject images that fail validation
            save_pipeline_debug: Save intermediate pipeline images for first iteration
            progress_callback: Callback for progress updates
            cancel_check: Callback that returns True if job should be cancelled
            resume_from_index: Starting image index for resume (0 = start fresh)
            dataset_info: Dataset metadata (name, description, version, license, etc.)
            synth_profile: Synthesis profile (v10/v11) or None for the legacy behavior (no synthesis profile)
            seed: Batch seed; per-image seeds derive from it (profile with per_image_seed)

        Returns:
            BatchResult with generation statistics
        """
        effects = effects or [
            EffectType.COLOR_CORRECTION,
            EffectType.BLUR_MATCHING,
            EffectType.CAUSTICS,
        ]
        effects_config = effects_config or EffectsConfig()

        if depth_service_url:
            self.depth_service_url = depth_service_url

        self._sweep_temp_backgrounds()

        # Create output directories
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Create pipeline debug directory if requested
        debug_dir = None
        if save_pipeline_debug:
            debug_dir = os.path.join(output_dir, "pipeline_debug")
            os.makedirs(debug_dir, exist_ok=True)
            logger.info(f"Pipeline debug enabled, saving to: {debug_dir}")

        # Load backgrounds
        bg_files = glob.glob(os.path.join(backgrounds_dir, "*.*"))
        bg_files = [f for f in bg_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not bg_files:
            raise FileNotFoundError(f"No backgrounds found in {backgrounds_dir}")

        # Load objects by class
        objects_by_class = self._load_objects_from_directory(objects_dir)
        merge_extra_libraries(objects_by_class, synth_profile)
        rect_report = apply_library_filters(objects_by_class, synth_profile)
        if not objects_by_class:
            raise FileNotFoundError(f"No objects found in {objects_dir}")

        available_classes = list(objects_by_class.keys())
        logger.info(f"Found {len(available_classes)} classes: {available_classes}")
        synth = self._make_synth(synth_profile, objects_by_class, backgrounds_dir, bg_files, seed)
        if synth:
            synth.rect_report = rect_report
            effects_config = effects_config.model_copy(update=synth.profile.effects_overrides)

        # Initialize counters
        result = BatchResult()
        result.synthetic_counts = {cls: 0 for cls in available_classes}

        # Calculate required instances
        if targets_per_class:
            required = {cls: targets_per_class.get(cls, 0) for cls in available_classes}
        else:
            per_class = max(1, num_images // len(available_classes))
            required = {cls: per_class for cls in available_classes}

        total_required = sum(required.values())
        # Calculate effective target: use minimum of num_images and total targets
        effective_target = min(num_images, total_required) if targets_per_class else num_images
        logger.info(f"Effective target: {effective_target} images (num_images={num_images}, total_required={total_required})")
        all_annotations = []

        # Load existing COCO if resuming
        existing_coco = None
        if resume_from_index > 0:
            coco_path = os.path.join(output_dir, "synthetic_dataset.json")
            if os.path.exists(coco_path):
                try:
                    with open(coco_path, 'r') as f:
                        existing_coco = json.load(f)
                    logger.info(f"Loaded existing COCO with {len(existing_coco.get('images', []))} images for resume")
                except Exception as e:
                    logger.warning(f"Failed to load existing COCO: {e}")
        if synth and synth.usage:
            synth.usage.preload_from_coco(existing_coco)

        # Generation loop
        generated = 0
        image_index = resume_from_index  # Start naming from this index
        rejected = 0
        max_iterations = num_images * 5  # Safety limit

        for iteration in range(max_iterations):
            # Periodic GC/CUDA cleanup (every iteration would be too aggressive:
            # each empty_cache forces a sync and re-allocation)
            if iteration > 0 and iteration % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    logger.info(f"GPU cleanup at iteration {iteration}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            # Check for cancellation
            if cancel_check and cancel_check():
                logger.info(f"Batch composition cancelled after {generated} images")
                break

            # Check if effective target reached (considers both num_images and per-class targets)
            if generated >= effective_target:
                logger.info(f"Effective target reached: {generated}/{effective_target} images")
                break

            # Check if any class still needs instances
            remaining = {cls: required[cls] - result.synthetic_counts[cls]
                        for cls in available_classes}
            available = [cls for cls, cnt in remaining.items() if cnt > 0 and objects_by_class.get(cls)]
            if synth and synth.usage:
                available = [cls for cls in available if synth.usage.has_free(cls)]

            if not available:
                logger.info(f"All per-class targets satisfied (or crops exhausted) after {generated} images")
                break

            # Select background
            attempt = resume_from_index + iteration
            rng = synth.make_rng(attempt) if synth else random
            bg_file = synth.pick_background(attempt, rng) if synth else random.choice(bg_files)
            bg_image = cv2.imread(bg_file)
            if bg_image is None:
                continue
            if synth and synth.profile.per_image_seed:
                random.seed(synth.seed_for(attempt))

            # Get depth map if needed
            depth_map = None
            depth_source = None
            if depth_aware:
                depth_map, depth_source = await self._get_depth_map(bg_image, cache_key=bg_file)

            # Select objects to place
            objects, reserved, image_plan = self._select_objects(
                synth, rng, attempt, bg_file, bg_image, available, remaining,
                objects_by_class, max_objects_per_image)
            if not objects:
                continue

            # Compose image - use image_index for naming
            output_path = os.path.join(images_dir, f"synthetic_{image_index:05d}.jpg")

            # Pass debug_dir only until first successful image
            current_debug_dir = debug_dir if (save_pipeline_debug and generated == 0) else None
            # Clean stale object dirs from previous failed attempts
            if current_debug_dir and os.path.exists(current_debug_dir):
                for entry in os.listdir(current_debug_dir):
                    entry_path = os.path.join(current_debug_dir, entry)
                    if os.path.isdir(entry_path) and entry.startswith("object_"):
                        shutil.rmtree(entry_path)

            try:
                compose_result = await self.compose(
                    background_path=bg_file,
                    objects=objects,
                    depth_map=depth_map,
                    depth_source=depth_source,
                    lighting_info=None,
                    effects=effects,
                    effects_config=effects_config,
                    output_path=output_path,
                    save_annotations=True,
                    debug_output_dir=current_debug_dir,
                    semantic_placement=semantic_placement,
                    plan=image_plan,
                )

                if compose_result.objects_placed > 0:
                    is_valid, quality_info = await self._validate_composed(
                        output_path,
                        compose_result.annotations,
                        self._output_hw(compose_result, bg_image)[0],
                        validate_quality,
                        validate_physics,
                    )

                    if reject_invalid and not is_valid:
                        self._discard_output(output_path)
                        rejected += 1
                        self._settle(synth, reserved, [])
                        logger.info(
                            f"Rejected synthetic_{image_index:05d}.jpg by validation, "
                            f"retrying with a new composition"
                        )
                    else:
                        # Collect annotations for COCO (use current image_index before incrementing)
                        self._settle(synth, reserved, [a.source_object for a in compose_result.annotations])
                        all_annotations.append(self._coco_entry(
                            f"synthetic_{image_index:05d}.jpg", bg_image, compose_result,
                            quality_info, image_plan, synth))

                        generated += 1
                        image_index += 1

                        # Update class counts
                        for ann in compose_result.annotations:
                            result.synthetic_counts[ann.class_name] = \
                                result.synthetic_counts.get(ann.class_name, 0) + 1

                        # Commit point (see batch_recovery)
                        if generated % FLUSH_EVERY == 0:
                            self._generate_coco_json(
                                all_annotations,
                                available_classes,
                                os.path.join(output_dir, COCO_NAME),
                                existing_coco=existing_coco,
                                dataset_info=dataset_info,
                                synth_info=synth.info() if synth else None,
                            )
                else:
                    self._settle(synth, reserved, [])

            except Exception as e:
                logger.warning(f"Failed to compose image: {e}")
                rejected += 1
                self._settle(synth, reserved, [])
                # Memory cleanup before continue
                if 'bg_image' in locals():
                    del bg_image
                if 'depth_map' in locals() and depth_map is not None:
                    del depth_map
                if 'objects' in locals():
                    del objects
                continue

            # Progress callback with effective pending calculation
            if progress_callback:
                # Calculate pending based on effective target (considers per-class targets)
                effective_pending = max(0, effective_target - generated)
                progress_callback({
                    'generated': generated,
                    'rejected': rejected,
                    'pending': effective_pending,
                    'counts': result.synthetic_counts,
                    'total_target': effective_target,
                    'next_image_index': image_index,
                })

            # Memory cleanup at the end of each successful iteration
            # (GC/empty_cache run periodically at the start of the loop, not per image)
            if 'compose_result' in locals():
                del compose_result
            if 'objects' in locals():
                del objects
            if 'bg_image' in locals():
                del bg_image
            if 'depth_map' in locals() and depth_map is not None:
                del depth_map

        # Final cleanup for the whole batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Generate COCO JSON (merge with existing if resuming)
        coco_path = os.path.join(output_dir, "synthetic_dataset.json")
        if synth:
            result.exhausted_classes = synth.exhausted_report(result.synthetic_counts, required)
            result.synth_profile = synth.profile.name
        self._generate_coco_json(
            all_annotations,
            available_classes,
            coco_path,
            existing_coco=existing_coco,
            dataset_info=dataset_info,
            synth_info=self._synth_info(synth, result, all_annotations),
        )
        self._write_rect_report(output_dir, synth)
        self._write_visibility_report(output_dir, synth)

        result.images_generated = generated
        result.images_rejected = rejected
        result.output_coco_path = coco_path

        logger.info(f"Batch complete: {generated} generated, {rejected} rejected (resume_from_index={resume_from_index})")
        return result

    @staticmethod
    def _discard_output(output_path: str) -> None:
        """Remove a rejected image and the annotation file compose() wrote for it."""
        ann_path = output_path.rsplit('.', 1)[0] + ANN_SUFFIX
        for path in (output_path, ann_path):
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _classes_still_needed(
        counts: Dict[str, int], required: Dict[str, int], usage=None
    ) -> List[str]:
        """Classes whose per-class instance target has not been reached yet.

        With a crop-reuse cap (`usage`), a class whose crops are exhausted and
        has no reservation in flight is final: it no longer counts as needed.
        """
        needed = [cls for cls, target in required.items() if counts.get(cls, 0) < target]
        if usage is None:
            return needed
        return [cls for cls in needed if usage.has_free(cls) or usage.pending > 0]

    @staticmethod
    def _make_synth(profile, objects_by_class, backgrounds_dir, bg_files, seed) -> Optional[SynthContext]:
        """Batch synthesis context, or None for the legacy behavior."""
        if profile is None:
            return None
        ctx = SynthContext(profile, objects_by_class, backgrounds_dir, bg_files, base_seed=seed)
        logger.info(f"Synthesis profile {profile.name} active (batch seed {ctx.base_seed})")
        return ctx

    @staticmethod
    def _select_objects(
        synth, rng, attempt, bg_file, bg_image, available, remaining, objects_by_class, max_objects
    ):
        """Objects to paste: (placements, reserved crop paths, image plan).

        Without a profile this is exactly the original selection (same random
        calls in the same order).
        """
        if synth is None:
            num_objects = random.randint(1, min(max_objects, len(available)))
            selected_classes = random.sample(available, k=num_objects)
            objects = []
            for cls in selected_classes:
                obj_data = random.choice(objects_by_class[cls])
                objects.append(ObjectPlacement(
                    image_path=obj_data['path'],
                    class_name=cls,
                    position=None,  # Auto-place
                    scale=None,     # Depth-aware
                    rotation=None,  # Random
                ))
            return objects, [], None
        objects, reserved, requested = plan_objects(
            synth, rng, available, remaining, objects_by_class, max_objects,
            bg_wh=(bg_image.shape[1], bg_image.shape[0]), bg_file=bg_file)
        plan = synth.make_plan(attempt, bg_file, (bg_image.shape[1], bg_image.shape[0]), rng)
        plan.objects_requested = requested
        return objects, reserved, plan

    @staticmethod
    def _settle(synth, reserved: List[str], kept: List[Optional[str]]) -> None:
        """Confirm or return the crop reservations of a finished image."""
        if synth is not None and synth.usage is not None and reserved:
            synth.usage.settle(reserved, kept)

    @staticmethod
    def _sweep_temp_backgrounds(max_age_seconds: int = 3600):
        """Remove stale /shared/temp files left behind by crashed jobs."""
        temp_dir = "/shared/temp"
        if not os.path.isdir(temp_dir):
            return
        now = time.time()
        removed = 0
        for fname in os.listdir(temp_dir):
            if not (fname.startswith("pipeline_bg_") or fname.startswith("scene_")):
                continue
            path = os.path.join(temp_dir, fname)
            try:
                if now - os.path.getmtime(path) > max_age_seconds:
                    os.remove(path)
                    removed += 1
            except OSError:
                pass
        if removed:
            logger.info(f"Swept {removed} stale temp background(s) from {temp_dir}")
