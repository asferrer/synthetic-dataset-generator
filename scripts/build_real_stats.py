"""Precompute the empirical statistics of the REAL training data.

Output (small, versionable JSON read by app/real_stats.py):
  - classes[<class>].sqrt_area_rel_quantiles: 101 quantiles (0..100 %) of the square root
    of the relative box area (w*h / (W*H) of its image).
  - objects_per_image: histogram {n_objects: n_images} of the training
    images with at least one object.

Only the TRAIN COCO is accepted: the script refuses to read files whose name
indicates val or test, so those statistics cannot end up in the generator.

Usage:
  python -X utf8 scripts/build_real_stats.py \
      --train gen_inputs/real_train/annotations.json \
      --out services/augmentor/app/data/real_stats_train.json
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List

NUM_QUANTILES = 101


def quantiles(values: List[float], n: int = NUM_QUANTILES) -> List[float]:
    """n evenly spaced quantiles (0..1) with linear interpolation."""
    ordered = sorted(values)
    last = len(ordered) - 1
    out = []
    for i in range(n):
        pos = last * i / (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, last)
        out.append(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))
    return [round(v, 6) for v in out]


def check_is_train(path: str) -> None:
    name = os.path.basename(path).lower()
    if "val" in name or "test" in name or "train" not in name:
        raise SystemExit(f"Only the TRAIN COCO is accepted (got: {name})")


def collect(coco: dict):
    """Return (square roots of the relative area per class, objects per image)."""
    sizes = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_class: Dict[str, List[float]] = defaultdict(list)
    per_image: Counter = Counter()
    for ann in coco["annotations"]:
        if ann.get("iscrowd"):
            continue
        w, h = ann["bbox"][2], ann["bbox"][3]
        img_w, img_h = sizes[ann["image_id"]]
        if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
            continue
        rel = min(1.0, (w * h) / (img_w * img_h))
        by_class[names[ann["category_id"]]].append(math.sqrt(rel))
        per_image[ann["image_id"]] += 1
    return by_class, per_image


def build(coco: dict, source: str) -> dict:
    by_class, per_image = collect(coco)
    hist = Counter(per_image.values())
    return {
        "source": source,
        "generated_by": "scripts/build_real_stats.py",
        "n_images_total": len(coco["images"]),
        "n_images_with_objects": len(per_image),
        "classes": {
            cls: {"n": len(vals), "sqrt_area_rel_quantiles": quantiles(vals)}
            for cls, vals in sorted(by_class.items())
        },
        "objects_per_image": {str(k): hist[k] for k in sorted(hist)},
        "image_sizes": dict(Counter(
            f"{im['width']}x{im['height']}" for im in coco["images"]).most_common(40)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", required=True, help="real training COCO")
    parser.add_argument("--out", required=True, help="output JSON")
    args = parser.parse_args()
    check_is_train(args.train)
    with open(args.train, "r", encoding="utf-8") as f:
        coco = json.load(f)
    stats = build(coco, os.path.basename(args.train))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(f"{args.out}: {len(stats['classes'])} classes, "
          f"{stats['n_images_with_objects']} images with objects")


if __name__ == "__main__":
    main()
