"""
Procesamiento en segundo plano de los trabajos de reetiquetado.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.detection_validator import deduplicate_annotations
from app.prompt_optimizer import get_prompt_optimizer
from app.service_state import labeling_jobs, state
from app.services.labeling_export import _export_to_voc, _export_to_yolo
from app.utils.bbox_iou import _calculate_iou

logger = logging.getLogger(__name__)


async def _process_relabeling_job(job_id: str):
    """Process a relabeling job."""
    import gc
    import torch
    from PIL import Image as PILImage

    logger.info(f"[RELABEL] _process_relabeling_job started for {job_id}")

    job = labeling_jobs[job_id]
    start_time = time.time()

    relabel_mode = job["_relabel_mode"]
    logger.info(f"[RELABEL] Mode: {relabel_mode}, Classes: {job.get('_classes', [])[:5]}...")
    coco_data = job.get("_coco_data")
    image_paths = job["_image_paths"]
    image_lookup = job["_image_lookup"]
    image_id_map = job.get("_image_id_map", {})  # filename -> original_image_id
    classes = job["_classes"]
    min_confidence = job["_min_confidence"]
    simplify_polygons = job["_simplify_polygons"]
    deduplication_strategy = job.get("_deduplication_strategy", "confidence")  # Deduplication strategy
    output_dir = Path(job["output_dir"])
    gc_interval = 5  # Run garbage collection every N images

    # Initialize result based on mode
    if relabel_mode == "add" and coco_data:
        # Start with existing data
        coco_result = coco_data.copy()
        # Add new categories
        existing_cats = {c["name"] for c in coco_result.get("categories", [])}
        max_cat_id = max([c["id"] for c in coco_result.get("categories", [])], default=0)
        for cls in classes:
            if cls not in existing_cats:
                max_cat_id += 1
                coco_result["categories"].append({"id": max_cat_id, "name": cls, "supercategory": ""})
        annotation_id = max([a["id"] for a in coco_result.get("annotations", [])], default=0) + 1
    else:
        # Start fresh
        coco_result = {
            "info": {
                "description": "Relabeled dataset",
                "date_created": datetime.now().isoformat(),
                "version": "1.0"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": i + 1, "name": cls, "supercategory": ""} for i, cls in enumerate(classes)]
        }
        annotation_id = 1

    category_map = {c["name"]: c["id"] for c in coco_result.get("categories", [])}
    prompt_optimizer = get_prompt_optimizer()
    logger.info(f"[RELABEL] Category map: {category_map}")
    logger.info(f"[RELABEL] Starting to process {len(image_paths)} images")

    # Process images
    for img_idx, img_path in enumerate(image_paths):
        job["current_image"] = Path(img_path).name
        job["processed_images"] = img_idx

        if img_idx == 0 or (img_idx + 1) % 10 == 0:
            logger.info(f"[RELABEL] Job {job_id}: Processing image {img_idx + 1}/{len(image_paths)} ({100*(img_idx+1)/len(image_paths):.1f}%) - {Path(img_path).name}")

        try:
            # Timeout per image to prevent indefinite blocking (30 seconds)
            async with asyncio.timeout(30.0):
                image = cv2.imread(img_path)
                if image is None:
                    continue

            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)

            # Use original image ID if available, otherwise generate sequential ID
            filename = Path(img_path).name
            image_id = image_id_map.get(filename, img_idx + 1)

            # Add image if not already in coco_result
            existing_image_ids = {img["id"] for img in coco_result.get("images", [])}
            if image_id not in existing_image_ids:
                coco_result["images"].append({
                    "id": image_id,
                    "file_name": filename,
                    "width": w,
                    "height": h
                })

            # Track annotations added for this image (for deduplication)
            image_annotations_start_idx = len(coco_result["annotations"])

            # Handle improve_segmentation mode differently
            if relabel_mode == "improve_segmentation" and coco_data:
                # Get existing annotations for this image (bbox-only annotations)
                img_anns = [a for a in coco_data.get("annotations", [])
                           if a.get("image_id") == image_id and not a.get("segmentation")]

                for ann in img_anns:
                    bbox = ann.get("bbox", [0, 0, 0, 0])
                    if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                        # Use SAM3 to generate segmentation from bbox
                        x, y, bw, bh = bbox
                        input_box = [[x, y, x + bw, y + bh]]

                        inputs = state.sam3_processor(
                            images=pil_image,
                            input_boxes=[input_box],
                            return_tensors="pt"
                        ).to(state.device)

                        with torch.no_grad():
                            outputs = state.sam3_model(**inputs)

                        masks = state.sam3_processor.post_process_masks(
                            outputs.pred_masks,
                            inputs["original_sizes"],
                            inputs["reshaped_input_sizes"]
                        )

                        if len(masks) > 0 and len(masks[0]) > 0:
                            mask = masks[0][0]
                            mask_np = mask.cpu().numpy().astype(np.uint8) * 255

                            if mask_np.shape != (h, w):
                                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                            polygons = state.object_extractor.mask_to_polygon(
                                mask_np,
                                simplify=simplify_polygons,
                                tolerance=2.0
                            )

                            if polygons:
                                # Update annotation with segmentation
                                ann["segmentation"] = polygons
                                ann["area"] = int(np.sum(mask_np > 0))
                                job["total_objects_found"] += 1

            else:
                # Get existing annotations for this image (for deduplication in "add" mode)
                existing_anns_for_image = []
                if relabel_mode == "add" and coco_data:
                    existing_anns_for_image = [a for a in coco_data.get("annotations", [])
                                               if a.get("image_id") == image_id]

                # Regular labeling with text prompts
                for cls in classes:
                    if cls not in category_map:
                        continue

                    optimized_prompt = prompt_optimizer.get_primary_prompt(cls)
                    inputs = state.sam3_processor(
                        images=pil_image,
                        text=optimized_prompt,
                        return_tensors="pt"
                    ).to(state.device)

                    with torch.no_grad():
                        outputs = state.sam3_model(**inputs)

                    target_sizes = [(h, w)]
                    results = state.sam3_processor.post_process_instance_segmentation(
                        outputs,
                        threshold=min_confidence,
                        mask_threshold=min_confidence,
                        target_sizes=target_sizes
                    )[0]

                    if 'masks' not in results:
                        continue

                    for mask, score in zip(results['masks'], results['scores']):
                        score_val = score.cpu().item()
                        if score_val < min_confidence:
                            continue

                        mask_np = mask.cpu().numpy().astype(np.uint8)
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                        area = int(np.sum(mask_np > 0))
                        if area < 100:
                            continue

                        contours, _ = cv2.findContours(
                            (mask_np * 255).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )

                        if not contours:
                            continue

                        x, y, bw, bh = cv2.boundingRect(np.concatenate(contours))

                        polygons = state.object_extractor.mask_to_polygon(
                            (mask_np * 255).astype(np.uint8),
                            simplify=simplify_polygons,
                            tolerance=2.0
                        )

                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_map[cls],
                            "bbox": [float(x), float(y), float(bw), float(bh)],
                            "area": area,
                            "iscrowd": 0,
                            "_score": score_val,
                        }

                        if polygons:
                            annotation["segmentation"] = polygons

                        # Check for duplicates in "add" mode (IoU threshold 0.5)
                        is_duplicate = False
                        if relabel_mode == "add" and existing_anns_for_image:
                            new_bbox = annotation["bbox"]
                            for exist_ann in existing_anns_for_image:
                                exist_bbox = exist_ann.get("bbox", [])
                                if len(exist_bbox) == 4:
                                    # Calculate IoU
                                    iou = _calculate_iou(new_bbox, exist_bbox)
                                    if iou > 0.5:
                                        is_duplicate = True
                                        break

                        if not is_duplicate:
                            coco_result["annotations"].append(annotation)
                            annotation_id += 1

                            job["objects_by_class"][cls] = job["objects_by_class"].get(cls, 0) + 1
                            job["total_objects_found"] += 1

            # Deduplicate annotations for this image (remove overlapping detections across all classes)
            image_anns = coco_result["annotations"][image_annotations_start_idx:]
            if len(image_anns) > 1:
                deduped = deduplicate_annotations(image_anns, iou_threshold=0.5, strategy=deduplication_strategy)
                removed_count = len(image_anns) - len(deduped)
                if removed_count > 0:
                    coco_result["annotations"] = coco_result["annotations"][:image_annotations_start_idx] + deduped
                    job["total_objects_found"] -= removed_count
                    for i, ann in enumerate(coco_result["annotations"][image_annotations_start_idx:]):
                        ann["id"] = image_annotations_start_idx + i + 1
                    annotation_id = len(coco_result["annotations"]) + 1
                    logger.debug(f"[RELABEL] Deduplicated {removed_count} overlapping annotations for {Path(img_path).name}")

            # Garbage collection and yielding
            if (img_idx + 1) % gc_interval == 0:
                del image, image_rgb, pil_image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Yield to event loop
            await asyncio.sleep(0)

        except asyncio.TimeoutError:
            # Image processing took too long (>30s), skip it
            error_msg = f"[RELABEL] Timeout processing {img_path} (>30s), skipping"
            job["errors"].append(error_msg)
            logger.warning(error_msg)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            job["errors"].append(f"Error processing {img_path}: {str(e)}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save results
    job["processed_images"] = len(image_paths)
    job["processing_time_ms"] = (time.time() - start_time) * 1000

    coco_path = output_dir / "annotations.json"
    with open(coco_path, 'w') as f:
        json.dump(coco_result, f, indent=2)

    if "yolo" in job["output_formats"]:
        _export_to_yolo(coco_result, output_dir, image_paths)

    if "voc" in job["output_formats"]:
        _export_to_voc(coco_result, output_dir, image_paths)

    logger.info(f"Relabeling job {job_id} completed")
