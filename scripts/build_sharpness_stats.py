"""Precompute the sharpness of the REAL training objects (profile v11, `match_sharpness`).

For every real training box (clipped to the image) it measures, over the central 50 % of the box, the `hf_ratio` and
the Laplacian variance at 64 x 64 (`app/sharpness.py`) and the noise in flat regions at native resolution. It writes
11 quantiles (0..100 %, 10 % steps) per class and globally (`all`).

Only the TRAIN COCO is accepted (the path must contain "train" and the name must not indicate val or test).
Image paths are resolved relative to the folder of the COCO file.

Usage:
  python -X utf8 scripts/build_sharpness_stats.py \
      --train gen_inputs/real_train/annotations.json \
      --out services/augmentor/app/data/real_sharpness_train.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "services", "augmentor"))
from app import sharpness  # noqa: E402

MIN_SIDE = 16


def check_is_train(path: str) -> None:
    low = path.replace("\\", "/").lower()
    if "train" not in low or any(t in os.path.basename(low) for t in ("val", "test")):
        raise SystemExit(f"Only the TRAIN COCO is accepted (got: {path})")


def measure_box(gray: np.ndarray, box) -> dict:
    h, w = gray.shape
    x, y, bw, bh = box
    x1, y1, x2, y2 = max(0, int(x)), max(0, int(y)), min(w, int(round(x + bw))), min(h, int(round(y + bh)))
    if min(x2 - x1, y2 - y1) < MIN_SIDE:
        return {}
    region = sharpness.central_region(gray, (x1, y1, x2, y2))
    if region is None:
        return {}
    g64 = sharpness.to_measure_size(region)
    return {"hf_ratio": sharpness.hf_ratio(g64), "lapvar64": float(cv2.Laplacian(g64, cv2.CV_32F).var()),
            "noise_flat": sharpness.flat_noise(region)}


def quantiles(values) -> list:
    return [round(float(v), 5) for v in np.quantile(values, np.linspace(0, 1, 11))]


def summarize(rows: list) -> dict:
    out = {"n": len(rows)}
    for key in ("hf_ratio", "lapvar64", "noise_flat"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[f"{key}_quantiles"] = quantiles(vals)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    check_is_train(a.train)
    coco = json.load(open(a.train, encoding="utf8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    files = {i["id"]: i["file_name"] for i in coco["images"]}
    by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if not ann.get("iscrowd"):
            by_img[ann["image_id"]].append(ann)
    folder = os.path.dirname(a.train)
    rows = defaultdict(list)
    for k, (img_id, anns) in enumerate(by_img.items()):
        img = cv2.imread(os.path.join(folder, files[img_id]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gray = img.astype(np.float32)
        for ann in anns:
            m = measure_box(gray, ann["bbox"])
            if m:
                rows[names[ann["category_id"]]].append(m)
        if k % 1000 == 0:
            print(f"  {k}/{len(by_img)} images", flush=True)
    pooled = [r for v in rows.values() for r in v]
    out = {"source": os.path.basename(folder) + "/" + os.path.basename(a.train),
           "generated_by": "scripts/build_sharpness_stats.py",
           "classes": {"all": summarize(pooled), **{c: summarize(v) for c, v in sorted(rows.items())}}}
    json.dump(out, open(a.out, "w", encoding="utf8"), indent=1)
    for c, v in out["classes"].items():
        print(f"{c:15s} n={v['n']:6d} hf p10/p50/p90 = {v['hf_ratio_quantiles'][1]}/{v['hf_ratio_quantiles'][5]}/{v['hf_ratio_quantiles'][9]}")


if __name__ == "__main__":
    main()
