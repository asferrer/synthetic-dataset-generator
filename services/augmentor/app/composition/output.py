"""Output I/O: COCO entries and files, object loading and per-image reports."""

import os
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
from loguru import logger

from app import copy_paste
from app import visibility
from app.batch_recovery import write_json_atomic
from app.models.schemas import AnnotationBox, TRACE_FIELDS
from app.composition.results import ComposeResult, BatchResult


class OutputMixin:
    """Output I/O: COCO entries and files, object loading and per-image reports."""

    @staticmethod
    def _output_hw(compose_result: ComposeResult, bg_image: np.ndarray) -> Tuple[int, int]:
        """(height, width) of the saved image (the profile may change the resolution)."""
        if compose_result.image_size is not None:
            return compose_result.image_size[1], compose_result.image_size[0]
        if bg_image is None:
            return 0, 0  # the caller overrides width/height (parallel workers)
        return bg_image.shape[0], bg_image.shape[1]

    @classmethod
    def _coco_entry(cls, image_name, bg_image, compose_result, quality, plan, synth) -> dict:
        """Per-image record that _generate_coco_json writes to the COCO."""
        height, width = cls._output_hw(compose_result, bg_image)
        entry = {
            'image_name': image_name,
            'width': width,
            'height': height,
            'annotations': cls._entry_annotations(compose_result, plan),
            'depth_source': compose_result.depth_source,
            'quality': quality,
        }
        if plan is not None and plan.profile.trace:
            entry['synth'] = dict(
                background=plan.background,
                seed=plan.seed,
                objects_requested=plan.objects_requested,
                **compose_result.image_trace,
            )
        return entry

    @staticmethod
    def _entry_annotations(compose_result: ComposeResult, plan) -> List[AnnotationBox]:
        """Annotations of the COCO record: pasted objects (+ the kept real boxes with copy-paste)."""
        if plan is not None and plan.profile.copy_paste_real:
            return copy_paste.merge_annotations(compose_result.real_annotations, compose_result.annotations)
        return compose_result.annotations

    @classmethod
    def _entry_from_result(cls, result_data: Dict[str, Any]) -> dict:
        """COCO record of a finished parallel worker result."""
        plan = result_data.get('plan')
        entry = cls._coco_entry(result_data['image_name'], None, result_data['compose_result'],
                                result_data.get('quality'), plan, None)
        entry['width'], entry['height'] = result_data['width'], result_data['height']
        return entry

    @staticmethod
    def _write_rect_report(output_dir: str, synth) -> None:
        """Writes the per-class count of crops dropped by the rectangular filter (v11) next to the COCO."""
        if synth is not None and synth.rect_report is not None:
            write_json_atomic(os.path.join(output_dir, "rect_filter_report.json"), synth.rect_report)

    @staticmethod
    def _write_visibility_report(output_dir: str, synth) -> None:
        """Objects discarded by the visibility filter (v11), per class, next to the COCO."""
        if synth is not None and synth.visibility_report is not None:
            write_json_atomic(os.path.join(output_dir, "visibility_report.json"), synth.visibility_report)

    @staticmethod
    def _synth_info(synth, result: BatchResult, entries: Optional[List[dict]] = None) -> Optional[dict]:
        """`synth_generator` block of the COCO (profile, seed, exhausted classes, visibility filter)."""
        if synth is None:
            return None
        if synth.profile.visibility_filter:
            synth.visibility_report = visibility.build_report(entries or [], result.synthetic_counts)
        return dict(synth.info(), exhausted_classes=result.exhausted_classes)

    def _load_objects_from_directory(self, objects_dir: str) -> Dict[str, List[dict]]:
        """Load objects from directory organized by class."""
        objects_by_class = defaultdict(list)

        if not os.path.isdir(objects_dir):
            return objects_by_class

        for class_name in os.listdir(objects_dir):
            class_dir = os.path.join(objects_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(class_dir, file_name)
                    objects_by_class[class_name].append({
                        'path': file_path,
                        'name': file_name,
                    })

        return objects_by_class

    def _generate_coco_json(
        self,
        annotations_data: List[dict],
        classes: List[str],
        output_path: str,
        existing_coco: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
        synth_info: Optional[dict] = None,
    ):
        """
        Generate COCO format JSON from annotations.

        Args:
            annotations_data: New annotations to add
            classes: List of class names
            output_path: Path to save COCO JSON
            existing_coco: Existing COCO data to merge with (for resume)
            dataset_info: Dataset metadata (name, description, version, license, etc.)
        """
        category_map = {cls: idx for idx, cls in enumerate(sorted(classes))}

        # Start with existing data if provided. Copy the lists: the batch
        # loops flush the COCO several times with the same existing_coco.
        if existing_coco:
            images = list(existing_coco.get("images", []))
            annotations = list(existing_coco.get("annotations", []))
            # Get next IDs
            img_id_start = max((img["id"] for img in images), default=-1) + 1
            ann_id = max((ann["id"] for ann in annotations), default=0) + 1
            logger.info(f"Merging with existing COCO: {len(images)} images, {len(annotations)} annotations")
        else:
            images = []
            annotations = []
            img_id_start = 0
            ann_id = 1

        # Add new annotations
        for idx, img_data in enumerate(annotations_data):
            img_id = img_id_start + idx
            image_entry = {
                "id": img_id,
                "file_name": img_data['image_name'],
                "width": img_data['width'],
                "height": img_data['height'],
            }
            if img_data.get('depth_source'):
                image_entry["depth_source"] = img_data['depth_source']
            if img_data.get('quality') is not None:
                image_entry["quality"] = img_data['quality']
            if img_data.get('synth'):
                image_entry.update({k: v for k, v in img_data['synth'].items() if v is not None})
            images.append(image_entry)

            for ann in img_data['annotations']:
                coco_ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_map.get(ann.class_name, -1),
                    "bbox": [ann.x, ann.y, ann.width, ann.height],
                    "area": ann.area,
                    "segmentation": [],
                    "iscrowd": 0,
                }
                if getattr(ann, "source_object", None):
                    coco_ann["source_object"] = ann.source_object
                if getattr(ann, "origin", None):
                    coco_ann["origin"] = ann.origin
                if getattr(ann, "mask_rle", None):
                    coco_ann["mask_rle"] = ann.mask_rle
                for key in TRACE_FIELDS:
                    if getattr(ann, key, None) is not None:
                        coco_ann[key] = getattr(ann, key)
                annotations.append(coco_ann)
                ann_id += 1

        categories = [
            {"id": cat_id, "name": name, "supercategory": "none"}
            for name, cat_id in category_map.items()
        ]

        # Build info section from dataset_info or use defaults
        if dataset_info:
            from datetime import datetime
            info = {
                "description": dataset_info.get("name", "Synthetic Dataset"),
                "version": dataset_info.get("version", "1.0"),
                "year": dataset_info.get("year", datetime.now().year),
                "contributor": dataset_info.get("contributor", ""),
                "url": dataset_info.get("url", ""),
                "date_created": dataset_info.get("date_created") or datetime.now().isoformat(),
            }
            # Add full description if provided
            if dataset_info.get("description"):
                info["description"] = f"{dataset_info.get('name', 'Synthetic Dataset')} - {dataset_info.get('description')}"
        else:
            from datetime import datetime
            info = {
                "description": "Synthetic Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "",
                "date_created": datetime.now().isoformat(),
            }

        # Build licenses section
        licenses = []
        if dataset_info and dataset_info.get("license_name"):
            licenses.append({
                "id": 1,
                "name": dataset_info.get("license_name", ""),
                "url": dataset_info.get("license_url", ""),
            })

        if synth_info:
            info["synth_generator"] = synth_info

        coco_data = {
            "info": info,
            "licenses": licenses,
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        write_json_atomic(output_path, coco_data)

        logger.info(f"COCO JSON saved to {output_path} ({len(images)} total images)")
