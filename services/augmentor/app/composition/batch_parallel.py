"""Parallel batch execution with a semaphore and VRAM-triggered cleanup."""

import os
import json
import gc
import glob
import random
import asyncio
from typing import List, Dict, Optional, Any, Callable
from collections import defaultdict

import cv2
import torch
from loguru import logger

from app.crop_filter import apply_library_filters
from app.synth_context import SynthContext, merge_extra_libraries
from app.synth_profile import SynthProfile
from app.batch_recovery import COCO_NAME
from app.models.schemas import EffectsConfig, EffectType
from app.composition.results import BatchResult, VRAMMonitor


class ParallelBatchMixin:
    """Parallel batch execution with a semaphore and VRAM-triggered cleanup."""

    async def _cleanup_vram(self):
        """Aggressive VRAM cleanup for parallel processing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            stats = VRAMMonitor().get_vram_stats()
            logger.info(
                f"VRAM cleanup executed: "
                f"{stats.get('allocated_gb', 0):.2f}GB / {stats.get('total_gb', 32):.0f}GB"
            )

    async def _compose_single_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        counters: Dict[str, int],
        counters_lock: asyncio.Lock,
        vram_monitor: VRAMMonitor,
        bg_files: List[str],
        objects_by_class: Dict[str, List[Dict]],
        available_classes: List[str],
        required: Dict[str, int],
        max_objects_per_image: int,
        output_dir: str,
        images_dir: str,
        depth_aware: bool,
        effects: List[EffectType],
        effects_config: Optional[EffectsConfig],
        debug_dir: Optional[str],
        cancel_check: Optional[Callable[[], bool]],
        validate_quality: bool = False,
        validate_physics: bool = False,
        reject_invalid: bool = True,
        semantic_placement: bool = True,
        synth: Optional[SynthContext] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Composes a single image with semaphore control for parallel processing.

        This worker method handles resource locking, error recovery, and VRAM management.

        Returns:
            Dict with compose_result and image_index, or None if failed
        """
        reserved: List[str] = []
        kept: List[Optional[str]] = []
        async with semaphore:  # Max N concurrent (configurable)
            try:
                # Check remaining work and get unique image index atomically.
                # The per-class check goes first so that no depth map is
                # requested (and no index consumed) once every class is covered.
                usage = synth.usage if synth else None
                while True:
                    async with counters_lock:
                        if counters['generated'] >= counters['target']:
                            return None  # Already reached target
                        still_needed = self._classes_still_needed(counters['counts'], required)
                        if usage is not None:
                            still_needed = [c for c in still_needed if usage.has_free(c)]
                        if still_needed:
                            image_index = counters['image_index']
                            counters['image_index'] += 1
                            break
                        if usage is None or usage.pending == 0:
                            return None  # All classes satisfied (or crops exhausted)
                    await asyncio.sleep(0.05)  # reservations in flight may be returned

                # Check for cancellation
                if cancel_check and cancel_check():
                    return None

                # Select background
                rng = synth.make_rng(image_index) if synth else random
                bg_file = synth.pick_background(image_index, rng) if synth else random.choice(bg_files)
                bg_image = cv2.imread(bg_file)
                if bg_image is None:
                    async with counters_lock:
                        counters['rejected'] += 1
                    return None

                # Get depth map if needed
                depth_map = None
                depth_source = None
                if depth_aware:
                    depth_map, depth_source = await self._get_depth_map(bg_image, cache_key=bg_file)

                # Select objects to place (synchronous: crop reservation is atomic)
                remaining = {c: required[c] - counters['counts'].get(c, 0) for c in required}
                objects, reserved, image_plan = self._select_objects(
                    synth, rng, image_index, bg_file, bg_image, still_needed, remaining,
                    objects_by_class, max_objects_per_image)
                if not objects:
                    return None

                # Compose image
                output_path = os.path.join(images_dir, f"synthetic_{image_index:05d}.jpg")

                # Pass debug_dir only for the first successfully generated image
                current_debug_dir = None
                async with counters_lock:
                    if debug_dir and counters['generated'] == 0:
                        current_debug_dir = debug_dir

                compose_result = await self.compose(
                    background_path=bg_file,
                    objects=objects,
                    output_path=output_path,
                    depth_map=depth_map,
                    depth_source=depth_source,
                    effects=effects,
                    effects_config=effects_config,
                    debug_output_dir=current_debug_dir,
                    semantic_placement=semantic_placement,
                    plan=image_plan,
                )

                if compose_result.objects_placed == 0:
                    async with counters_lock:
                        counters['rejected'] += 1
                    return None

                # Optional quality/physics validation
                is_valid, quality_info = await self._validate_composed(
                    output_path,
                    compose_result.annotations,
                    self._output_hw(compose_result, bg_image)[0],
                    validate_quality,
                    validate_physics,
                )
                if reject_invalid and not is_valid:
                    self._discard_output(output_path)
                    async with counters_lock:
                        counters['rejected'] += 1
                    logger.info(f"Rejected {os.path.basename(output_path)} by validation")
                    return None

                # Update counters atomically
                async with counters_lock:
                    counters['generated'] += 1
                    for ann in compose_result.annotations:
                        counters['counts'][ann.class_name] += 1
                kept = [a.source_object for a in compose_result.annotations]

                # Check VRAM after each image
                if vram_monitor.should_cleanup():
                    await self._cleanup_vram()

                # Get image dimensions
                bg_h, bg_w = self._output_hw(compose_result, bg_image)

                return {
                    'compose_result': compose_result,
                    'image_index': image_index,
                    'image_name': os.path.basename(output_path),
                    'width': bg_w,
                    'height': bg_h,
                    'quality': quality_info,
                    'plan': image_plan,
                }

            except Exception as e:
                logger.warning(f"Failed to compose image in parallel worker: {e}")
                async with counters_lock:
                    counters['rejected'] += 1
                return None

            finally:
                self._settle(synth, reserved, kept)
                # Per-task cleanup — GC/empty_cache is handled by the VRAM
                # monitor under pressure and once at end of batch
                if 'bg_image' in locals():
                    del bg_image
                if 'depth_map' in locals() and depth_map is not None:
                    del depth_map
                if 'objects' in locals():
                    del objects
                if 'compose_result' in locals():
                    del compose_result

    async def compose_batch_parallel(
        self,
        num_images: int,
        backgrounds_dir: str,
        objects_dir: str,
        output_dir: str,
        concurrent_limit: int = 4,
        vram_threshold: float = 0.5,
        depth_aware: bool = True,
        max_objects_per_image: int = 10,
        targets_per_class: Optional[Dict[str, int]] = None,
        resume_from_index: int = 0,
        effects: List[EffectType] = None,
        effects_config: Optional[EffectsConfig] = None,
        save_pipeline_debug: bool = False,
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        dataset_info: Optional[Dict] = None,
        validate_quality: bool = False,
        validate_physics: bool = False,
        reject_invalid: bool = True,
        semantic_placement: bool = True,
        synth_profile: Optional[SynthProfile] = None,
        seed: Optional[int] = None,
    ) -> BatchResult:
        """
        Composes a batch of synthetic images with PARALLEL processing.

        This method processes multiple images concurrently to maximize CPU/GPU utilization.

        Args:
            concurrent_limit: Max images to process simultaneously (default 4)
            vram_threshold: VRAM usage % to trigger cleanup (default 0.5 = 50%)
            dataset_info: Dataset metadata (name, description, version, license, etc.)
            ... (other args same as compose_batch)

        Returns:
            BatchResult with generation statistics
        """
        logger.info(
            f"Starting PARALLEL batch composition: {num_images} images, "
            f"{concurrent_limit} concurrent, {vram_threshold*100:.0f}% VRAM threshold"
        )

        # Initialize effects if not provided
        effects = effects or [
            EffectType.COLOR_CORRECTION,
            EffectType.BLUR_MATCHING,
            EffectType.CAUSTICS,
        ]
        effects_config = effects_config or EffectsConfig()

        self._sweep_temp_backgrounds()

        # Setup
        result = BatchResult()
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Setup pipeline debug directory if requested
        debug_dir = None
        if save_pipeline_debug:
            debug_dir = os.path.join(output_dir, "pipeline_debug")
            os.makedirs(debug_dir, exist_ok=True)
            logger.info(f"Pipeline debug enabled: {debug_dir}")

        # Load backgrounds and objects
        bg_files = glob.glob(os.path.join(backgrounds_dir, "**/*.jpg"), recursive=True) + \
                   glob.glob(os.path.join(backgrounds_dir, "**/*.png"), recursive=True)

        if not bg_files:
            raise ValueError(f"No background images found in {backgrounds_dir}")

        # Group objects by class
        objects_by_class = defaultdict(list)
        for obj_file in glob.glob(os.path.join(objects_dir, "**/*.png"), recursive=True):
            class_name = os.path.basename(os.path.dirname(obj_file))
            objects_by_class[class_name].append({'path': obj_file})
        merge_extra_libraries(objects_by_class, synth_profile)
        if synth_profile is not None and synth_profile.copy_paste_real:
            raise ValueError("copy_paste_real is only supported in sequential mode (parallel=false)")
        rect_report = apply_library_filters(objects_by_class, synth_profile)

        available_classes = list(objects_by_class.keys())
        if not available_classes:
            raise ValueError(f"No object classes found in {objects_dir}")

        # Determine required counts per class
        if targets_per_class:
            # Filter to only classes that have objects available
            required = {
                cls: count for cls, count in targets_per_class.items()
                if cls in available_classes
            }
            # Warn about missing classes
            missing_classes = set(targets_per_class.keys()) - set(available_classes)
            if missing_classes:
                logger.warning(
                    f"Requested classes not found in objects directory: {missing_classes}. "
                    f"Available classes: {available_classes}"
                )
        else:
            count_per_class = max(1, num_images // len(available_classes))
            required = {cls: count_per_class for cls in available_classes}

        if not required:
            raise ValueError(
                f"No valid object classes available. "
                f"Requested: {list(targets_per_class.keys()) if targets_per_class else 'auto'}, "
                f"Available: {available_classes}"
            )

        synth = self._make_synth(synth_profile, objects_by_class, backgrounds_dir, bg_files, seed)
        if synth:
            synth.rect_report = rect_report
            effects_config = effects_config.model_copy(update=synth.profile.effects_overrides)

        # Calculate effective target: use minimum of num_images and total per-class targets
        total_required = sum(required.values())
        effective_target = min(num_images, total_required) if targets_per_class else num_images
        logger.info(f"Parallel mode - Effective target: {effective_target} images (num_images={num_images}, total_required={total_required})")

        # Resume support - load existing COCO if resuming
        existing_coco = None
        if resume_from_index > 0:
            coco_path = os.path.join(output_dir, "synthetic_dataset.json")
            if os.path.exists(coco_path):
                with open(coco_path, 'r') as f:
                    existing_coco = json.load(f)
                logger.info(f"Resuming from image {resume_from_index}, loaded existing COCO")
        if synth and synth.usage:
            synth.usage.preload_from_coco(existing_coco)

        # Shared state with locks
        counters = {
            'generated': 0,
            'rejected': 0,
            'image_index': resume_from_index,
            'target': effective_target,  # Use effective target (considers per-class targets)
            'counts': defaultdict(int),
        }
        counters_lock = asyncio.Lock()

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent_limit)

        # VRAM monitor
        vram_monitor = VRAMMonitor(threshold=vram_threshold)

        # Task tracking
        tasks = []
        all_results = []
        all_annotations = []  # Process annotations incrementally to free memory
        results_batch_size = 50  # Process every 50 results to avoid memory build-up

        max_iterations = effective_target * 5  # Safety limit based on effective target

        for iteration in range(max_iterations):
            # Check if effective target reached (considers per-class targets)
            # effective_target counts images while the per-class targets count
            # instances (several per image), so the class check is what usually
            # ends the batch.
            async with counters_lock:
                if counters['generated'] >= effective_target:
                    logger.info(f"Effective target reached: {counters['generated']}/{effective_target} images")
                    break
                if not self._classes_still_needed(counters['counts'], required, synth.usage if synth else None):
                    logger.info(f"All per-class targets satisfied (or crops exhausted) after {counters['generated']} images")
                    break

            # Create worker task
            task = asyncio.create_task(
                self._compose_single_with_semaphore(
                    semaphore=semaphore,
                    counters=counters,
                    counters_lock=counters_lock,
                    vram_monitor=vram_monitor,
                    bg_files=bg_files,
                    objects_by_class=objects_by_class,
                    available_classes=available_classes,
                    required=required,
                    max_objects_per_image=max_objects_per_image,
                    output_dir=output_dir,
                    images_dir=images_dir,
                    depth_aware=depth_aware,
                    effects=effects,
                    effects_config=effects_config,
                    debug_dir=debug_dir,
                    cancel_check=cancel_check,
                    validate_quality=validate_quality,
                    validate_physics=validate_physics,
                    reject_invalid=reject_invalid,
                    semantic_placement=semantic_placement,
                    synth=synth,
                )
            )
            tasks.append(task)

            # Wait for some tasks to complete if queue is full
            if len(tasks) >= concurrent_limit * 2:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    result_data = await task
                    if result_data:
                        all_results.append(result_data)
                tasks = list(pending)

                # Incremental processing: every 50 results, convert to annotations and free memory
                if len(all_results) >= results_batch_size:
                    for result_data in all_results:
                        all_annotations.append(self._entry_from_result(result_data))
                    # Free memory of the processed results
                    all_results.clear()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.debug(f"Processed {results_batch_size} results incrementally, freed memory")

                    # Commit point (see batch_recovery)
                    self._generate_coco_json(
                        all_annotations,
                        available_classes,
                        os.path.join(output_dir, COCO_NAME),
                        existing_coco=existing_coco,
                        dataset_info=dataset_info,
                        synth_info=synth.info() if synth else None,
                    )

                # Progress callback with effective pending calculation
                if progress_callback:
                    async with counters_lock:
                        effective_pending = max(0, effective_target - counters['generated'])
                        progress_callback({
                            'generated': counters['generated'],
                            'rejected': counters['rejected'],
                            'pending': effective_pending,
                            'counts': dict(counters['counts']),
                            'total_target': effective_target,
                            'next_image_index': counters['image_index'],
                        })

        # Wait for remaining tasks
        if tasks:
            done, _ = await asyncio.wait(tasks)
            for task in done:
                result_data = await task
                if result_data:
                    all_results.append(result_data)

        # Final progress: when the class targets are met nothing is pending,
        # although fewer images than effective_target were needed.
        if progress_callback:
            async with counters_lock:
                classes_done = not self._classes_still_needed(counters['counts'], required, synth.usage if synth else None)
                progress_callback({
                    'generated': counters['generated'],
                    'rejected': counters['rejected'],
                    'pending': 0 if classes_done else max(0, effective_target - counters['generated']),
                    'counts': dict(counters['counts']),
                    'total_target': counters['generated'] if classes_done else effective_target,
                    'next_image_index': counters['image_index'],
                })

        # Process the remaining results (those that did not reach the batch size)
        for result_data in all_results:
            all_annotations.append(self._entry_from_result(result_data))
        # Free memory of the last results
        all_results.clear()
        gc.collect()

        # Generate COCO JSON
        coco_path = os.path.join(output_dir, "synthetic_dataset.json")
        async with counters_lock:
            result.images_generated = counters['generated']
            result.images_rejected = counters['rejected']
            result.synthetic_counts = dict(counters['counts'])
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
        result.output_coco_path = coco_path

        logger.info(
            f"PARALLEL batch complete: {result.images_generated} generated, "
            f"{result.images_rejected} rejected in {concurrent_limit}x parallel mode"
        )

        # Final cleanup for the whole batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
