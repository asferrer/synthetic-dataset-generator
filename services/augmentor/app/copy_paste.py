"""Copy-paste onto real images of the training split (profile v11; Ghiasi et al., 2021).

The background is a real training image whose real annotations are KEPT: the COCO output includes the original boxes
(`origin = "real"`, no `source_object`) plus the pasted ones (`origin = "pasted"`). The original resolution and the
JPEG quality estimated from its quantisation table are preserved. A pasted box may not overlap a real box by more
than `real_overlap_max` (relative to the area of the smaller of the two boxes).

The backgrounds are a flat folder with an `annotations.json` (COCO with `file_name` relative to that folder), as
written by `scripts/build_real_train.py`.
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from app.models.schemas import AnnotationBox

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


class RealAnnotations:
    """Real boxes per image (key = file_name relative to the backgrounds folder)."""

    def __init__(self, by_name: Dict[str, List[dict]]):
        self.by_name = by_name

    @classmethod
    def load(cls, path: str) -> "RealAnnotations":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"copy_paste_real: real-background COCO file not found: {path}")
        with open(path, encoding="utf8") as f:
            coco = json.load(f)
        names = {c["id"]: c["name"] for c in coco["categories"]}
        files = {im["id"]: im["file_name"].replace("\\", "/") for im in coco["images"]}
        by_name: Dict[str, List[dict]] = {name: [] for name in files.values()}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0) or ann["image_id"] not in files:
                continue
            x, y, w, h = ann["bbox"]
            by_name[files[ann["image_id"]]].append({"x": x, "y": y, "w": w, "h": h, "cls": names[ann["category_id"]]})
        return cls(by_name)

    def get(self, name: str) -> List[dict]:
        return self.by_name.get(name.replace("\\", "/"), [])


def annotation_path(profile, backgrounds_dir: str) -> str:
    return profile.real_annotations_path or os.path.join(backgrounds_dir, "annotations.json")


def real_annotation_boxes(real: Sequence[dict], image_wh: Tuple[int, int]) -> List[AnnotationBox]:
    """Kept real annotations, clipped to the image (empty ones are dropped)."""
    w_img, h_img = image_wh
    out: List[AnnotationBox] = []
    for r in real:
        x1, y1 = max(0, int(round(r["x"]))), max(0, int(round(r["y"])))
        x2, y2 = min(w_img, int(round(r["x"] + r["w"]))), min(h_img, int(round(r["y"] + r["h"])))
        if x2 > x1 and y2 > y1:
            out.append(AnnotationBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1, class_name=r["cls"],
                                     area=(x2 - x1) * (y2 - y1), origin="real"))
    return out


def to_xyxy(anns: Sequence[AnnotationBox]) -> List[Box]:
    return [(a.x, a.y, a.x + a.width, a.y + a.height) for a in anns]


def overlap_with_real(bbox: Box, real_boxes: Sequence[Box]) -> float:
    """Largest overlap of `bbox` with a real box: intersection / area of the smaller of the two."""
    worst = 0.0
    area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
    for rb in real_boxes:
        inter = max(0, min(bbox[2], rb[2]) - max(bbox[0], rb[0])) * max(0, min(bbox[3], rb[3]) - max(bbox[1], rb[1]))
        smaller = min(area, max(0, rb[2] - rb[0]) * max(0, rb[3] - rb[1]))
        if smaller > 0:
            worst = max(worst, inter / smaller)
    return worst


def exceeds_real_overlap(bbox: Box, real_boxes: Sequence[Box], limit: float) -> bool:
    return bool(real_boxes) and overlap_with_real(bbox, real_boxes) > limit


def merge_annotations(real: Sequence[AnnotationBox], pasted: Sequence[AnnotationBox]) -> List[AnnotationBox]:
    """Kept real + pasted annotations (pasted ones tagged `origin = "pasted"`); inputs are not modified."""
    tagged = [a.model_copy(update={"origin": "pasted"}) for a in pasted]
    return list(real) + tagged


def original_jpeg_quality(path: str) -> Optional[int]:
    """Estimated JPEG quality of the original (None if not a JPEG)."""
    from app.output_format import estimate_jpeg_quality
    q = estimate_jpeg_quality(path)
    return None if q is None else int(round(q))


def haze_cap(profile) -> float:
    """Effective maximum haze strength: with copy-paste, its own cap if lower than the global one."""
    cap = profile.haze_max_intensity
    if profile.copy_paste_real and profile.copy_paste_haze_max is not None:
        cap = min(cap, profile.copy_paste_haze_max)
    return cap


def blur_kernel_cap(profile) -> Optional[int]:
    """Cap on the blend blur kernel (None = no dedicated cap)."""
    return profile.copy_paste_blur_max if profile.copy_paste_real else None


def cap_kernel(kernel: int, cap: Optional[int]) -> int:
    """Odd Gaussian kernel limited by `cap`; 0 if the cap does not even allow the minimum of 3 (no blur)."""
    if cap is None or kernel <= cap:
        return kernel
    capped = cap if cap % 2 == 1 else cap - 1
    return capped if capped >= 3 else 0
