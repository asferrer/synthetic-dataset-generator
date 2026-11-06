"""
Utilidades de imagen para el etiquetado: previsualizaciones, rutas y relleno de cajas.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _draw_annotations_on_image(
    image: np.ndarray,
    annotations: List[Dict],
    category_map: Dict[int, str],
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True
) -> np.ndarray:
    """Draw annotations (bboxes, masks, labels) on an image for visualization.

    Args:
        image: BGR image array
        annotations: List of COCO-format annotations for this image
        category_map: Dict mapping category_id to category name
        draw_masks: Whether to draw segmentation masks
        draw_boxes: Whether to draw bounding boxes
        draw_labels: Whether to draw class labels

    Returns:
        Annotated image (BGR)
    """
    vis_image = image.copy()
    h, w = vis_image.shape[:2]

    # Generate distinct colors for each category
    np.random.seed(42)  # For consistent colors
    colors = {}
    for cat_id in category_map.keys():
        colors[cat_id] = tuple(int(c) for c in np.random.randint(50, 255, 3))

    for ann in annotations:
        cat_id = ann.get("category_id", 1)
        cat_name = category_map.get(cat_id, f"class_{cat_id}")
        color = colors.get(cat_id, (0, 255, 0))

        # Draw mask if available
        if draw_masks and "segmentation" in ann and ann["segmentation"]:
            overlay = vis_image.copy()
            for seg in ann["segmentation"]:
                if isinstance(seg, list) and len(seg) >= 6:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)

        # Draw bounding box
        if draw_boxes and "bbox" in ann:
            x, y, bw, bh = [int(v) for v in ann["bbox"]]
            cv2.rectangle(vis_image, (x, y), (x + bw, y + bh), color, 2)

        # Draw label
        if draw_labels and "bbox" in ann:
            x, y, bw, bh = [int(v) for v in ann["bbox"]]
            label = f"{cat_name}"
            if "score" in ann:
                label += f" {ann['score']:.2f}"

            # Background for text
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x, y - text_h - 4), (x + text_w + 4, y), color, -1)
            cv2.putText(vis_image, label, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_image


def _save_preview_image(
    output_dir: Path,
    image: np.ndarray,
    image_name: str,
    annotations: List[Dict],
    category_map: Dict[int, str],
    preview_idx: int
) -> Optional[str]:
    """Save a preview image with annotations drawn.

    Returns the relative path to the saved preview, or None if failed.
    """
    try:
        previews_dir = output_dir / "previews"
        previews_dir.mkdir(exist_ok=True)

        # Draw annotations
        vis_image = _draw_annotations_on_image(
            image, annotations, category_map,
            draw_masks=True, draw_boxes=True, draw_labels=True
        )

        # Save with a consistent naming scheme
        preview_name = f"preview_{preview_idx:04d}_{Path(image_name).stem}.jpg"
        preview_path = previews_dir / preview_name

        # Resize if too large (max 1024px on longest side) for faster loading
        h, w = vis_image.shape[:2]
        max_size = 1024
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            vis_image = cv2.resize(vis_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(preview_path), vis_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return str(preview_path)
    except Exception as e:
        logger.warning(f"Failed to save preview image: {e}")
        return None


def _get_image_paths_from_output_dir(output_dir: Path) -> List[str]:
    """Reconstruct image paths from checkpoint/coco file in output directory.

    This is used when resuming a job after service restart.
    """
    image_paths = []

    # Try to load from annotations.json or checkpoint
    coco_path = output_dir / "annotations.json"
    checkpoint_path = output_dir / "checkpoint.json"

    coco_data = None
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            coco_data = checkpoint.get("coco_data")
        except Exception:
            pass

    if coco_data is None and coco_path.exists():
        try:
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
        except Exception:
            pass

    if coco_data and "images" in coco_data:
        # Get image directory from first image path or use output_dir parent
        images_dir = output_dir.parent / "images"
        if not images_dir.exists():
            images_dir = output_dir.parent

        for img_info in coco_data["images"]:
            file_name = img_info.get("file_name", "")
            # Try to find the image file
            for search_dir in [images_dir, output_dir.parent, output_dir]:
                potential_path = search_dir / file_name
                if potential_path.exists():
                    image_paths.append(str(potential_path))
                    break

    return image_paths


def _apply_padding_to_bbox(bbox: List[float], padding: int, img_width: int, img_height: int) -> List[float]:
    """Apply padding to a bounding box while keeping it within image bounds.

    Args:
        bbox: [x, y, width, height] format bounding box
        padding: Pixels of padding to add on each side
        img_width: Image width for bounds checking
        img_height: Image height for bounds checking

    Returns:
        Padded bbox [x, y, width, height] clamped to image bounds
    """
    if padding <= 0:
        return bbox

    x, y, w, h = bbox
    # Expand the bbox by padding on each side
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_width, x + w + padding)
    y2 = min(img_height, y + h + padding)

    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
