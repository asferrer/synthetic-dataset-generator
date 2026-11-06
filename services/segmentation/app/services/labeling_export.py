"""
Exportación de anotaciones de etiquetado a formatos YOLO y VOC.
"""

from pathlib import Path
from typing import Dict, List


def _export_to_yolo(coco_data: Dict, output_dir: Path, image_paths: List[str]):
    """Export COCO data to YOLO format."""
    yolo_dir = output_dir / "yolo"
    yolo_dir.mkdir(exist_ok=True)

    labels_dir = yolo_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    # Build image lookup
    image_dims = {img["id"]: (img["width"], img["height"]) for img in coco_data.get("images", [])}
    image_names = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}

    # Category mapping (YOLO uses 0-indexed)
    categories = coco_data.get("categories", [])
    cat_map = {c["id"]: i for i, c in enumerate(categories)}

    # Group annotations by image
    anns_by_image = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Write labels
    for img_id, anns in anns_by_image.items():
        if img_id not in image_dims:
            continue

        w, h = image_dims[img_id]
        filename = Path(image_names[img_id]).stem + ".txt"

        lines = []
        for ann in anns:
            cat_idx = cat_map.get(ann["category_id"], 0)
            bbox = ann.get("bbox", [0, 0, 0, 0])

            # Convert to YOLO format (center x, center y, width, height) normalized
            x_center = (bbox[0] + bbox[2] / 2) / w
            y_center = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h

            lines.append(f"{cat_idx} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        with open(labels_dir / filename, 'w') as f:
            f.write('\n'.join(lines))

    # Write classes.txt
    with open(yolo_dir / "classes.txt", 'w') as f:
        for cat in categories:
            f.write(f"{cat['name']}\n")


def _export_to_voc(coco_data: Dict, output_dir: Path, image_paths: List[str]):
    """Export COCO data to Pascal VOC format."""
    voc_dir = output_dir / "voc"
    voc_dir.mkdir(exist_ok=True)

    annotations_dir = voc_dir / "Annotations"
    annotations_dir.mkdir(exist_ok=True)

    # Build lookups
    image_info = {img["id"]: img for img in coco_data.get("images", [])}
    cat_names = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

    # Group annotations by image
    anns_by_image = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Write XML files
    for img_id, anns in anns_by_image.items():
        if img_id not in image_info:
            continue

        img = image_info[img_id]
        filename = Path(img["file_name"]).stem + ".xml"

        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<annotation>',
            f'  <filename>{img["file_name"]}</filename>',
            '  <size>',
            f'    <width>{img["width"]}</width>',
            f'    <height>{img["height"]}</height>',
            '    <depth>3</depth>',
            '  </size>',
        ]

        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            cat_name = cat_names.get(ann["category_id"], "unknown")

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[0] + bbox[2])
            ymax = int(bbox[1] + bbox[3])

            xml_lines.extend([
                '  <object>',
                f'    <name>{cat_name}</name>',
                '    <bndbox>',
                f'      <xmin>{xmin}</xmin>',
                f'      <ymin>{ymin}</ymin>',
                f'      <xmax>{xmax}</xmax>',
                f'      <ymax>{ymax}</ymax>',
                '    </bndbox>',
                '  </object>',
            ])

        xml_lines.append('</annotation>')

        with open(annotations_dir / filename, 'w') as f:
            f.write('\n'.join(xml_lines))
