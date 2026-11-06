"""Refit the boxes of a synthetic COCO produced by the generator WITHOUT overwriting the original (writes a new COCO).

Two methods:
  segmentation  box fitted to the `segmentation` polygon/mask (exact; only if the COCO carries it)
  undo-margin   undo the +-N px margin (20 by default) that `_recalculate_bbox_after_global_effects` adds on each side
                of pasted boxes after a global colour change. Approximate: on sides clipped by the image border the
                original margin is unknown (between 0 and N px) and the box is left at the border; it does not undo
                the widening caused by shadows (about 2 px on average). Only valid for COCOs generated with
                `recalculate_bbox_after_global` on and a noticeable global colour change (profile v10 and earlier);
                do not apply it to profile v11 batches without checking.
Boxes with `origin == "real"` (copy-paste) are left untouched. Pasted objects without `origin` are all synthetic.

Usage:
  python -X utf8 tools/refit_coco_boxes.py IN.json OUT.json --method undo-margin [--margin 20]
  python -X utf8 tools/refit_coco_boxes.py IN.json OUT.json --method segmentation
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def box_from_segmentation(seg):
    """Box (x, y, w, h) of a COCO polygon (list of lists), or None if empty."""
    pts = [p for poly in (seg or []) if isinstance(poly, list) and len(poly) >= 6
           for p in zip(poly[0::2], poly[1::2])]
    if not pts:
        return None
    xs, ys = zip(*pts)
    x1, y1, x2, y2 = int(np.floor(min(xs))), int(np.floor(min(ys))), int(np.ceil(max(xs))), int(np.ceil(max(ys)))
    return [x1, y1, x2 - x1, y2 - y1]


def undo_margin(bbox, img_w, img_h, margin):
    """Shrink every side that does not touch the image border by `margin` px; border sides are kept."""
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h
    x1, y1 = (x1 + margin if x1 > 0 else 0), (y1 + margin if y1 > 0 else 0)
    x2, y2 = (x2 - margin if x2 < img_w else img_w), (y2 - margin if y2 < img_h else img_h)
    if x2 - x1 < 2 or y2 - y1 < 2:       # too small for the margin: it was not inflated by the effect
        return list(bbox)
    return [x1, y1, x2 - x1, y2 - y1]


def refit(coco, method, margin):
    sizes = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    changed = skipped = 0
    for ann in coco["annotations"]:
        if ann.get("origin") == "real":
            continue
        old = ann["bbox"]
        if method == "segmentation":
            new = box_from_segmentation(ann.get("segmentation"))
        else:
            new = undo_margin(old, *sizes[ann["image_id"]], margin)
        if new is None:
            skipped += 1
            continue
        ann["bbox_original"], ann["bbox"], ann["area"] = old, new, new[2] * new[3]
        changed += new != old
    return changed, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("coco_in")
    ap.add_argument("coco_out")
    ap.add_argument("--method", choices=("segmentation", "undo-margin"), required=True)
    ap.add_argument("--margin", type=int, default=20)
    a = ap.parse_args()
    if Path(a.coco_in).resolve() == Path(a.coco_out).resolve():
        sys.exit("The output must not be the input file")
    coco = json.load(open(a.coco_in, encoding="utf8"))
    changed, skipped = refit(coco, a.method, a.margin)
    coco.setdefault("info", {})["box_refit"] = {"method": a.method, "margin": a.margin, "source": Path(a.coco_in).name}
    json.dump(coco, open(a.coco_out, "w", encoding="utf8"))
    print(f"{len(coco['annotations'])} annotations: {changed} boxes changed, {skipped} without mask; wrote {a.coco_out}")


if __name__ == "__main__":
    main()
