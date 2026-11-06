"""Precompute the POSITION distribution of the REAL training objects (profile v11, `position_from_stats`).

Output (small, versionable JSON read by app/position_stats.py), global (`ALL`) and per class:
  - n: boxes counted.
  - center_hist: GRID x GRID histogram (rows = y, columns = x) of the normalised box-centre position,
    normalised to sum 1.
  - touch_fraction: fraction of boxes that touch the image border (within touch_px px of some side).
  - touch_sides: among those that touch, the frequency of each side (left/right/top/bottom; a box may touch several).
  - margin_quantiles: 11 quantiles (0..100 %) of the distance to the border (fraction of the shorter image side) of
    the boxes that do NOT touch the border.

Only the TRAIN COCO is accepted (the path must contain "train" and the name must not indicate val or test).

Usage:
  python -X utf8 scripts/build_position_stats.py \
      --train gen_inputs/real_train/annotations.json \
      --out services/augmentor/app/data/position_stats_train.json
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

GRID = 10
TOUCH_PX = 3
SIDES = ("left", "right", "top", "bottom")


def check_is_train(path: str) -> None:
    low = path.replace("\\", "/").lower()
    if "train" not in low or any(t in os.path.basename(low) for t in ("val", "test")):
        raise SystemExit(f"Only the TRAIN COCO is accepted (got: {path})")


def box_record(ann: dict, wh) -> dict:
    """Normalised centre, touched sides and normalised border distance of a box clipped to the image."""
    w_img, h_img = wh
    x, y, w, h = ann["bbox"]
    x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(w_img), x + w), min(float(h_img), y + h)
    dists = {"left": x1, "right": w_img - x2, "top": y1, "bottom": h_img - y2}
    touched = [s for s, d in dists.items() if d <= TOUCH_PX]
    return {"cx": (x1 + x2) / 2 / w_img, "cy": (y1 + y2) / 2 / h_img, "touched": touched,
            "margin": min(dists.values()) / min(w_img, h_img)}


def summarize(recs: List[dict]) -> dict:
    hist = np.zeros((GRID, GRID))
    for r in recs:
        i, j = min(GRID - 1, int(r["cy"] * GRID)), min(GRID - 1, int(r["cx"] * GRID))
        hist[i, j] += 1
    touching = [r for r in recs if r["touched"]]
    sides = {s: sum(s in r["touched"] for r in touching) for s in SIDES}
    side_total = max(1, sum(sides.values()))
    free = [r["margin"] for r in recs if not r["touched"]]
    return {
        "n": len(recs),
        "center_hist": np.round(hist / max(1.0, hist.sum()), 6).tolist(),
        "touch_fraction": round(len(touching) / max(1, len(recs)), 4),
        "touch_sides": {s: round(n / side_total, 4) for s, n in sides.items()},
        "margin_quantiles": [round(float(v), 5) for v in np.quantile(free, np.linspace(0, 1, 11))] if free else [],
    }


def collect(coco: dict) -> Dict[str, List[dict]]:
    sizes = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_class: Dict[str, List[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd") or ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
            continue
        by_class[names[ann["category_id"]]].append(box_record(ann, sizes[ann["image_id"]]))
    return by_class


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    check_is_train(a.train)
    with open(a.train, encoding="utf8") as f:
        coco = json.load(f)
    by_class = collect(coco)
    pooled = [r for recs in by_class.values() for r in recs]
    out = {"source": os.path.basename(os.path.dirname(a.train)) + "/" + os.path.basename(a.train),
           "generated_by": "scripts/build_position_stats.py", "grid": GRID, "touch_px": TOUCH_PX,
           "classes": {"ALL": summarize(pooled), **{c: summarize(r) for c, r in sorted(by_class.items())}}}
    with open(a.out, "w", encoding="utf8") as f:
        json.dump(out, f, indent=1)
    for c, v in out["classes"].items():
        print(f"{c:15s} n={v['n']:6d} touch_fraction={v['touch_fraction']:.3f} sides={v['touch_sides']}")


if __name__ == "__main__":
    main()
