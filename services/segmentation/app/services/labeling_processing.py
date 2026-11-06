"""
Procesamiento en segundo plano de los trabajos de etiquetado.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from app.detection_validator import deduplicate_annotations, get_detection_validator
from app.prompt_optimizer import get_prompt_optimizer
from app.service_state import VRAMMonitor, VRAM_MONITOR_AVAILABLE, labeling_jobs, state
from app.services.labeling_export import _export_to_voc, _export_to_yolo
from app.services.labeling_images import _apply_padding_to_bbox, _save_preview_image

logger = logging.getLogger(__name__)


async def _process_labeling_job(job_id: str, resume_from: int = 0):
    """Process a labeling job - detect and segment objects in images.

    Args:
        job_id: The job identifier
        resume_from: Image index to resume from (0 = start fresh)
    """
    import gc
    import torch
    from PIL import Image as PILImage

    job = labeling_jobs[job_id]
    start_time = time.time()

    image_paths = job["_image_paths"]
    classes = job["_classes"]  # These are the search prompts
    class_mapping = job.get("_class_mapping")  # Maps prompts to final class names
    min_confidence = job["_min_confidence"]
    min_area = job["_min_area"]
    max_instances = job["_max_instances"]
    simplify_polygons = job["_simplify_polygons"]
    simplify_tolerance = job["_simplify_tolerance"]
    task_type = job["_task_type"]
    padding = job.get("_padding", 0)  # Pixels of padding around bboxes
    save_visualizations = job.get("_save_visualizations", True)  # Save preview images
    deduplication_strategy = job.get("_deduplication_strategy", "confidence")  # Deduplication strategy
    output_dir = Path(job["output_dir"])
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint_interval = 10  # Save checkpoint every N images
    max_previews = 50  # Maximum number of preview images to keep
    # Calculate dynamic preview interval based on total images (aim for ~30-50 previews)
    total_images = len(image_paths)
    preview_interval = max(1, total_images // max_previews) if total_images > max_previews else 1
    gc_interval = 5  # Run garbage collection every N images
    yield_interval = 1  # Yield to event loop every N images

    # Initialize preview tracking
    preview_paths = job.get("_preview_paths", [])
    preview_count = len(preview_paths)
    images_with_detections = 0

    # Initialize quality metrics tracking
    all_scores = []  # Track all detection scores for avg_confidence
    low_confidence_count = 0  # Detections with score < 0.5
    images_without_detections = 0
    consecutive_errors = 0
    max_consecutive_errors = 10  # Abort if this many consecutive errors
    max_error_rate = 0.1  # Abort if error rate exceeds 10%

    # Initialize optimization modules
    prompt_optimizer = get_prompt_optimizer()
    detection_validator = get_detection_validator()

    # Initialize warnings list for class-level issues
    job["warnings"] = job.get("warnings", [])

    # Initialize VRAM monitor if available
    vram_monitor = None
    if VRAM_MONITOR_AVAILABLE and VRAMMonitor is not None:
        vram_monitor = VRAMMonitor(threshold=0.7, check_interval=2)
        logger.info(f"VRAMMonitor initialized for job {job_id}")

    # Determine final category names
    # If class_mapping exists, use unique mapped values as categories
    # Otherwise, use the original class names
    if class_mapping:
        # Get unique final class names (values from the mapping)
        # Preserve order by using the first appearance
        final_classes = []
        seen = set()
        for cls in classes:
            final_name = class_mapping.get(cls, cls)
            if final_name not in seen:
                final_classes.append(final_name)
                seen.add(final_name)
    else:
        final_classes = classes

    # Initialize or load COCO structure
    coco_result = None
    annotation_id = 1

    # Try to load checkpoint if resuming
    if resume_from > 0 and checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            coco_result = checkpoint.get("coco_data")
            annotation_id = checkpoint.get("next_annotation_id", 1)
            # Restore objects_by_class counts
            if "objects_by_class" in checkpoint:
                job["objects_by_class"] = checkpoint["objects_by_class"]
                job["total_objects_found"] = sum(checkpoint["objects_by_class"].values())
            logger.info(f"Resuming job {job_id} from image {resume_from}, annotation_id {annotation_id}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
            coco_result = None

    # Initialize fresh if no checkpoint loaded
    if coco_result is None:
        coco_result = {
            "info": {
                "description": "Auto-labeled dataset",
                "date_created": datetime.now().isoformat(),
                "version": "1.0"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": i + 1, "name": cls, "supercategory": ""} for i, cls in enumerate(final_classes)]
        }
        resume_from = 0  # Reset if no valid checkpoint

    # Map final class names to category IDs
    category_map = {cls: i + 1 for i, cls in enumerate(final_classes)}

    # Process images starting from resume_from
    for img_idx, img_path in enumerate(image_paths):
        # Skip already processed images when resuming
        if img_idx < resume_from:
            continue

        job["current_image"] = Path(img_path).name
        job["processed_images"] = img_idx

        # Log progress every 10 images
        if img_idx == 0 or (img_idx + 1) % 10 == 0:
            logger.info(f"[LABEL] Job {job_id}: Processing image {img_idx + 1}/{len(image_paths)} ({100*(img_idx+1)/len(image_paths):.1f}%)")

        try:
            # Timeout per image to prevent indefinite blocking (30 seconds)
            async with asyncio.timeout(30.0):
                # Load image with validation
                image = cv2.imread(img_path)
                if image is None:
                    job["errors"].append(f"Failed to load: {img_path}")
                    consecutive_errors += 1
                    continue

                h, w = image.shape[:2]

                # Validate image dimensions (skip if too small or too large)
                if h < 10 or w < 10:
                    job["errors"].append(f"Image too small ({w}x{h}): {img_path}")
                    consecutive_errors += 1
                    continue
                if h > 8192 or w > 8192:
                    job["errors"].append(f"Image too large ({w}x{h}), may cause OOM: {img_path}")
                    consecutive_errors += 1
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(image_rgb)

                # Add image to COCO
                image_id = img_idx + 1
                coco_result["images"].append({
                    "id": image_id,
                    "file_name": Path(img_path).name,
                    "width": w,
                    "height": h
                })

                # Track annotations added for this image (for deduplication)
                image_annotations_start_idx = len(coco_result["annotations"])

                # Process each class (cls is the search prompt)
                for cls in classes:
                    # Get final class name (may be different if mapping exists)
                    final_class = class_mapping.get(cls, cls) if class_mapping else cls

                    # Get optimized prompt for better detection
                    optimized_prompt = prompt_optimizer.get_primary_prompt(cls)

                    # Run SAM3 text prompt segmentation using optimized prompt
                    inputs = state.sam3_processor(
                        images=pil_image,
                        text=optimized_prompt,  # Use optimized prompt for detection
                        return_tensors="pt"
                    ).to(state.device)

                    with torch.no_grad():
                        outputs = state.sam3_model(**inputs)

                    # Post-process results
                    target_sizes = inputs.get("original_sizes")
                    if target_sizes is not None:
                        target_sizes = target_sizes.tolist()
                    else:
                        target_sizes = [(h, w)]

                    results = state.sam3_processor.post_process_instance_segmentation(
                        outputs,
                        threshold=min_confidence,
                        mask_threshold=min_confidence,
                        target_sizes=target_sizes
                    )[0]

                    if 'masks' not in results or len(results['masks']) == 0:
                        continue

                    # Process each detected instance
                    instances_added = 0
                    for mask, score in zip(results['masks'], results['scores']):
                        if instances_added >= max_instances:
                            break

                        score_val = score.cpu().item()
                        if score_val < min_confidence:
                            continue

                        mask_np = mask.cpu().numpy().astype(np.uint8)
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                        # Calculate area
                        area = int(np.sum(mask_np > 0))
                        if area < min_area:
                            continue

                        # Get bounding box
                        contours, _ = cv2.findContours(
                            (mask_np * 255).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )

                        if not contours:
                            continue

                        x, y, bw, bh = cv2.boundingRect(np.concatenate(contours))
                        bbox = [float(x), float(y), float(bw), float(bh)]

                        # Validate detection using DetectionValidator
                        is_valid, rejection_reason, adjusted_score = detection_validator.validate_detection(
                            mask=mask_np,
                            bbox=bbox,
                            class_name=final_class,
                            image_size=(w, h),
                            score=score_val,
                        )

                        if not is_valid:
                            logger.debug(f"Rejected detection for '{final_class}': {rejection_reason}")
                            continue

                        # Track quality metrics
                        all_scores.append(adjusted_score)
                        if adjusted_score < 0.5:
                            low_confidence_count += 1

                        # Apply padding if configured
                        if padding > 0:
                            bbox = _apply_padding_to_bbox(bbox, padding, w, h)

                        # Build annotation (use final_class for category lookup)
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_map[final_class],
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                            "_score": adjusted_score,  # Track score for potential dedup
                        }

                        # Add segmentation if task requires it
                        if task_type in ["segmentation", "both"]:
                            polygons = state.object_extractor.mask_to_polygon(
                                (mask_np * 255).astype(np.uint8),
                                simplify=simplify_polygons,
                                tolerance=simplify_tolerance
                            )
                            if polygons:
                                annotation["segmentation"] = polygons

                        coco_result["annotations"].append(annotation)
                        annotation_id += 1
                        instances_added += 1

                        # Track by final class name
                        job["objects_by_class"][final_class] = job["objects_by_class"].get(final_class, 0) + 1
                        job["total_objects_found"] += 1

            # Deduplicate annotations for this image (remove overlapping detections)
            image_annotations = coco_result["annotations"][image_annotations_start_idx:]
            if len(image_annotations) > 1:
                deduped = deduplicate_annotations(image_annotations, iou_threshold=0.5, strategy=deduplication_strategy)
                removed_count = len(image_annotations) - len(deduped)
                if removed_count > 0:
                    # Update the annotations list
                    coco_result["annotations"] = coco_result["annotations"][:image_annotations_start_idx] + deduped
                    # Adjust counts
                    job["total_objects_found"] -= removed_count
                    # Re-assign annotation IDs for this image's annotations
                    for i, ann in enumerate(coco_result["annotations"][image_annotations_start_idx:]):
                        ann["id"] = image_annotations_start_idx + i + 1
                    annotation_id = len(coco_result["annotations"]) + 1
                    logger.debug(f"Deduplicated {removed_count} overlapping annotations for {Path(img_path).name}")

            # Track images without detections
            current_image_detections = len(coco_result["annotations"]) - image_annotations_start_idx
            if current_image_detections == 0:
                images_without_detections += 1

            # Reset consecutive error counter on successful processing
            consecutive_errors = 0

            # Save preview image if this image had detections
            if save_visualizations and current_image_detections > 0:
                images_with_detections += 1
                # Save preview periodically (every N images with detections, up to max_previews)
                if images_with_detections % preview_interval == 0 and preview_count < max_previews:
                    # Get annotations for this image
                    image_annotations = [
                        ann for ann in coco_result["annotations"]
                        if ann.get("image_id") == image_id
                    ]
                    # Create reverse category map (id -> name)
                    category_id_to_name = {i + 1: cls for i, cls in enumerate(final_classes)}

                    preview_path = _save_preview_image(
                        output_dir, image, Path(img_path).name,
                        image_annotations, category_id_to_name, preview_count
                    )
                    if preview_path:
                        preview_paths.append(preview_path)
                        preview_count += 1
                        job["_preview_paths"] = preview_paths
                        logger.debug(f"Saved preview {preview_count}/{max_previews} at image {img_idx + 1}")

            # Save checkpoint periodically and update database
            if (img_idx + 1) % checkpoint_interval == 0:
                _save_labeling_checkpoint(
                    checkpoint_path, coco_result, annotation_id,
                    img_idx + 1, job["objects_by_class"]
                )
                logger.debug(f"Saved checkpoint at image {img_idx + 1}")

                # Clear retry count on successful processing
                if "_retry_count" in job and img_path in job["_retry_count"]:
                    del job["_retry_count"][img_path]

                # Update quality metrics in real-time (so frontend can display them)
                job["quality_metrics"] = {
                    "avg_confidence": sum(all_scores) / len(all_scores) if all_scores else 0.0,
                    "images_with_detections": images_with_detections,
                    "images_without_detections": images_without_detections,
                    "low_confidence_count": low_confidence_count,
                    "total_detections": len(all_scores),
                }

                # Update progress in database
                if state.db:
                    try:
                        state.db.update_job_progress(
                            job_id,
                            processed_items=img_idx + 1,
                            current_item=Path(img_path).name,
                            progress_details={
                                "total_objects_found": job["total_objects_found"],
                                "objects_by_class": job["objects_by_class"],
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Failed to update progress in database: {e}")

            # Garbage collection - use VRAMMonitor if available, otherwise periodic
            should_cleanup = False
            if vram_monitor is not None:
                should_cleanup = vram_monitor.should_cleanup()
            else:
                should_cleanup = (img_idx + 1) % gc_interval == 0

            if should_cleanup:
                # Clear image references
                del image, image_rgb, pil_image
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                if vram_monitor is not None:
                    stats = vram_monitor.get_vram_stats()
                    logger.debug(f"VRAM cleanup at image {img_idx + 1}: {stats.get('allocated_gb', 0):.2f}GB")

                # Yield to event loop to allow health checks and other operations
                if (img_idx + 1) % yield_interval == 0:
                    await asyncio.sleep(0)

        except asyncio.TimeoutError:
            # Image processing took too long (>30s), skip it
            consecutive_errors += 1
            error_msg = f"Timeout processing {img_path} (>30s), skipping"
            job["errors"].append(error_msg)
            logger.warning(f"Job {job_id}: {error_msg}")
            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            # OOM: Clean memory and retry with exponential backoff
            consecutive_errors += 1
            torch.cuda.empty_cache()
            gc.collect()

            retry_count = job.get("_retry_count", {}).get(img_path, 0)
            if retry_count < 3:
                # Set up retry
                job.setdefault("_retry_count", {})[img_path] = retry_count + 1
                wait_time = 2 ** retry_count  # Exponential backoff: 1, 2, 4 seconds
                logger.warning(f"OOM on {img_path}, retry {retry_count + 1}/3 after {wait_time}s")
                await asyncio.sleep(wait_time)
                # Re-add to process queue (will be picked up next iteration)
                image_paths.insert(img_idx + 1, img_path)
                continue
            else:
                job["errors"].append(f"OOM error after 3 retries: {img_path}")
                logger.error(f"OOM error after 3 retries: {img_path}")

        except Exception as e:
            consecutive_errors += 1
            job["errors"].append(f"Error processing {img_path}: {str(e)}")
            logger.error(f"Error processing {img_path}: {e}")
            # Save checkpoint on error so we can resume
            _save_labeling_checkpoint(
                checkpoint_path, coco_result, annotation_id,
                img_idx, job["objects_by_class"]
            )
            # Clean up on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Check abort conditions
        processed = img_idx + 1
        error_count = len(job.get("errors", []))

        # Abort if too many consecutive errors
        if consecutive_errors >= max_consecutive_errors:
            job["status"] = "failed"
            job["error"] = f"Too many consecutive errors ({consecutive_errors}). Check input data or GPU memory."
            logger.error(f"Job {job_id} aborted: {job['error']}")
            break

        # Abort if error rate exceeds threshold (after processing enough images)
        if processed >= 50 and error_count / processed > max_error_rate:
            job["status"] = "failed"
            job["error"] = f"Error rate too high: {error_count}/{processed} ({100*error_count/processed:.1f}%). Check class names or image quality."
            logger.error(f"Job {job_id} aborted: {job['error']}")
            break

        # Add warning if a class has no detections after significant processing
        if processed >= 20 and processed % 20 == 0:
            for cls in classes:
                final_cls = class_mapping.get(cls, cls) if class_mapping else cls
                if job["objects_by_class"].get(final_cls, 0) == 0:
                    warning = f"No detections for '{cls}' after {processed} images"
                    if warning not in job.get("warnings", []):
                        job.setdefault("warnings", []).append(warning)
                        logger.warning(f"Job {job_id}: {warning}")

    # Update quality metrics
    job["quality_metrics"] = {
        "avg_confidence": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "images_with_detections": images_with_detections,
        "images_without_detections": images_without_detections,
        "low_confidence_count": low_confidence_count,
        "total_detections": len(all_scores),
    }

    # Check if job was aborted
    if job.get("status") == "failed":
        # Save partial results even on failure
        coco_path = output_dir / "annotations_partial.json"
        with open(coco_path, 'w') as f:
            json.dump(coco_result, f, indent=2)
        logger.info(f"Saved partial results to {coco_path}")
        return

    # Save results
    job["processed_images"] = len(image_paths)
    job["processing_time_ms"] = (time.time() - start_time) * 1000

    # Save COCO JSON
    coco_path = output_dir / "annotations.json"
    with open(coco_path, 'w') as f:
        json.dump(coco_result, f, indent=2)

    # Save in other formats if requested
    if "yolo" in job["output_formats"]:
        _export_to_yolo(coco_result, output_dir, image_paths)

    if "voc" in job["output_formats"]:
        _export_to_voc(coco_result, output_dir, image_paths)

    # Remove checkpoint file on successful completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.debug("Removed checkpoint file after successful completion")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint file: {e}")

    logger.info(f"Labeling job {job_id} completed: {job['total_objects_found']} objects found")


def _save_labeling_checkpoint(checkpoint_path: Path, coco_data: Dict, next_annotation_id: int,
                               last_processed_idx: int, objects_by_class: Dict[str, int]):
    """Save checkpoint for labeling job recovery."""
    checkpoint = {
        "coco_data": coco_data,
        "next_annotation_id": next_annotation_id,
        "last_processed_idx": last_processed_idx,
        "objects_by_class": objects_by_class,
        "saved_at": datetime.now().isoformat()
    }
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
