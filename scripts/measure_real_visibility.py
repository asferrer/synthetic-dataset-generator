"""Visibility of the real training objects: basis of the generator's visibility-filter thresholds.

For every box of the real-background COCO (`gen_inputs/real_train/annotations.json`, training split only) it measures
the local contrast C against the surrounding ring (`services/augmentor/app/visibility.py`: mean LAB distance between
the box pixels and the mean LAB colour of the ring) and the luminance standard deviation inside the box, and writes
per-class percentiles. `visibility_min_contrast` and `visibility_percentile` of the synthesis profile come from a low
percentile of this distribution (see `services/augmentor/app/data/visibility_percentiles_train.json`).

Usage:
    python scripts/measure_real_visibility.py [--coco gen_inputs/real_train/annotations.json] [--out out_dir]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "augmentor"))
import cv2  # noqa: E402

from app import visibility as vis  # noqa: E402

GEN_INPUTS = Path(os.environ.get("GEN_INPUTS", "gen_inputs"))
PERCENTILES = (1, 5, 10, 15, 25, 50)


def load_boxes(coco_path: Path):
    """{file_name: [(class, x1, y1, x2, y2)]} from a training COCO file."""
    if any(t in coco_path.parent.name.lower() + coco_path.name.lower() for t in ("val", "test")):
        raise SystemExit(f"Training split only: {coco_path}")
    coco = json.load(open(coco_path, encoding="utf8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    files = {im["id"]: im["file_name"] for im in coco["images"]}
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) or a["image_id"] not in files:
            continue
        x, y, w, h = a["bbox"]
        by_img[files[a["image_id"]]].append((names[a["category_id"]], int(x), int(y), int(round(x + w)), int(round(y + h))))
    return by_img


def measure(coco_path: Path, max_images=None):
    by_img = load_boxes(coco_path)
    contrast, lstd = defaultdict(list), defaultdict(list)
    for i, (name, boxes) in enumerate(by_img.items()):
        if max_images and i >= max_images:
            break
        img = cv2.imread(str(coco_path.parent / name))
        if img is None:
            continue
        for cls, x1, y1, x2, y2 in boxes:
            c = vis.local_contrast(img, (x1, y1, x2, y2))
            s = vis.box_luminance_std(img, (x1, y1, x2, y2))
            if c is not None:
                contrast[cls].append(c)
            if s is not None:
                lstd[cls].append(s)
        if i % 1000 == 0:
            print(f"  {i}/{len(by_img)} images", flush=True)
    return contrast, lstd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coco", default=str(GEN_INPUTS / "real_train" / "annotations.json"))
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--out", default=".", help="folder for real_visibility_train.json")
    a = ap.parse_args()
    contrast, lstd = measure(Path(a.coco), a.max_images)
    rep = {"coco": a.coco, "C": vis.percentile_table(contrast, PERCENTILES),
           "lstd": vis.percentile_table(lstd, PERCENTILES)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out / "real_visibility_train.json", "w", encoding="utf8"), indent=1)
    print(json.dumps(rep["C"], indent=1))


if __name__ == "__main__":
    main()
