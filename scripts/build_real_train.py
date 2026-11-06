"""Build the real-background folder used by copy-paste from the training split.

Copies the images listed in a training-split COCO file into a flat folder and writes `annotations.json` there:
a COCO file whose `file_name` is `<id:05d>_<original name>`, with the original boxes (no polygons) and, for each
image, `source` and `orig_file_name`. The generator reads this folder as `/app/datasets/real_train`.

Only the training split is accepted: a COCO file whose name contains "val" or "test" is refused. Existing files of
the same size are not copied again.

Usage:
    python scripts/build_real_train.py --train <dataset>/annotations/train.json --images-root <dataset> \
        --out gen_inputs/real_train
    # optional: --ids-from <coco.json> keeps only the training images whose ids appear in that file
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def load_train(path: Path) -> dict:
    if any(word in path.name.lower() for word in ("val", "test")):
        sys.exit(f"Training split only: {path.name} looks like validation/test")
    return json.loads(path.read_text(encoding="utf8"))


def subset_by_ids(train: dict, ref: dict) -> dict:
    """Training COCO restricted to the image ids of `ref` (all of them must exist in `train`)."""
    ids = {im["id"] for im in ref["images"]}
    keep = [im for im in train["images"] if im["id"] in ids]
    if len(keep) != len(ids):
        sys.exit(f"{len(ids) - len(keep)} ids of --ids-from are not in the training split")
    return {**train, "images": keep, "annotations": [a for a in train["annotations"] if a["image_id"] in ids]}


def new_name(image: dict) -> str:
    return f"{image['id']:05d}_{Path(image['file_name']).name}"


def copy_images(train: dict, images_root: Path, out: Path):
    """Copy the missing images. Returns (copied, already_there, missing, size_mismatch, by_source)."""
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    copied = present = 0
    missing, bad_size, by_source = [], [], Counter()
    for im in train["images"]:
        src, dst = images_root / im["file_name"], out / new_name(im)
        if not src.exists():
            missing.append(im["file_name"])
            continue
        by_source[im.get("source", "unknown")] += 1
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            present += 1
        else:
            shutil.copy2(src, dst)
            copied += 1
        with Image.open(dst) as pil:
            if pil.size != (im["width"], im["height"]):
                bad_size.append(new_name(im))
    return copied, present, missing, bad_size, by_source


def write_annotations(train: dict, out: Path) -> dict:
    images = [{"id": im["id"], "file_name": new_name(im), "width": im["width"], "height": im["height"],
               "source": im.get("source", "unknown"), "orig_file_name": im["file_name"]} for im in train["images"]]
    anns = [{"id": a["id"], "image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"],
             "area": a["area"], "iscrowd": a.get("iscrowd", 0), "segmentation": []} for a in train["annotations"]]
    coco = {"info": {"description": "Training split for copy-paste (boxes only, no polygons)"},
            "licenses": [], "images": images, "annotations": anns, "categories": train["categories"]}
    (out / "annotations.json").write_text(json.dumps(coco), encoding="utf8")
    return coco


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True, help="training-split COCO file")
    ap.add_argument("--images-root", required=True, help="folder that `file_name` entries are relative to")
    ap.add_argument("--out", default="gen_inputs/real_train")
    ap.add_argument("--ids-from", help="COCO file: keep only the training images with these ids")
    a = ap.parse_args()
    train = load_train(Path(a.train))
    if a.ids_from:
        train = subset_by_ids(train, load_train(Path(a.ids_from)))
    out = Path(a.out)
    copied, present, missing, bad_size, by_source = copy_images(train, Path(a.images_root), out)
    if missing:
        sys.exit(f"{len(missing)} images are missing (e.g. {missing[:3]}); annotations.json not written")
    coco = write_annotations(train, out)
    print(json.dumps({"images": len(coco["images"]), "annotations": len(coco["annotations"]), "copied": copied,
                      "already_there": present, "by_source": dict(by_source), "size_mismatch": len(bad_size)},
                     indent=1))


if __name__ == "__main__":
    main()
