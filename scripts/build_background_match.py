"""Colour descriptors of the training images and the source of every crop (profile v11, `background_match`).

Per `real_train` image (training split only): global LAB mean and deviation (image at 64 px), coarse LAB histogram
(4x4x4) and the image group of the partition. Per crop (`Extracted_objects_train` and `crops_poly`): the source image
(manifest) and the LAB mean/deviation of the ring around its annotated box in that image (`local_patch_match`).
Output: one small compressed NPZ (`services/augmentor/app/data/background_match_train.npz`, < 5 MB).

CPU only (OpenCV), low priority, N processes. It never uses val or test: it refuses to run if the COCO has images that
are not in the training split according to the groups file.

Environment:
  GEN_INPUTS   directory with real_train, Extracted_objects_train and crops_poly (default `gen_inputs`)
  GROUPS_JSON  REQUIRED. JSON list of `{"group": int, "split": "train"|"val"|"test", "image_ids": [...]}` describing
               the dataset partition (image groups, e.g. frames of the same sequence). It comes from the partitioning
               step of the dataset and is not part of this repository.

Usage (from the repository root):
  GROUPS_JSON=path/to/groups.json python -X utf8 scripts/build_background_match.py --workers 4
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "augmentor"))
from app import background_match as bm  # noqa: E402

GEN_INPUTS = Path(os.environ.get("GEN_INPUTS", "gen_inputs"))
RING_FRAC = 0.25   # ring width as a fraction of the longer box side (stored in the NPZ)


def _low_priority() -> None:
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    except Exception:  # noqa: BLE001
        pass


def _describe(job):
    """(index, descriptor, histogram, [(crop key, ring)]) of one image and of the crops cut from it."""
    idx, path, boxes = job
    img = cv2.imread(path)
    if img is None:
        return idx, None, None, []
    meter = bm.RingMeter(img)
    rings = []
    for key, (x, y, w, h) in boxes:
        r = meter.stats((x, y, x + w, y + h), RING_FRAC)
        rings.append((key, None if r is None else r))
    return idx, bm.lab_descriptor(img), bm.lab_histogram(img), rings


def crop_sources(real_dir: Path, anns: dict, files: dict):
    """[(crop key, image id, annotated box)] of the ext and poly crops whose manifest agrees with the COCO."""
    out, dropped = [], 0
    for lib, manifest, name in (("ext", GEN_INPUTS / "Extracted_objects_train" / "manifest_train.json",
                                 lambda c: f"{c['cls']}/{c['ann']}.png"),
                                ("poly", GEN_INPUTS / "crops_poly" / "manifest_poly.json",
                                 lambda c: f"{c['cls']}/poly_{c['ann']}.png")):
        for c in json.load(open(manifest, encoding="utf8"))["crops"]:
            a = anns.get(c["ann"])
            if a is None or a["image_id"] != c["image_id"] or c["image_id"] not in files:
                dropped += 1
                continue
            out.append((name(c), c["image_id"], a["bbox"]))
    return out, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real-dir", default=str(GEN_INPUTS / "real_train"))
    ap.add_argument("--out", default=str(ROOT / "services" / "augmentor" / "app" / "data" / "background_match_train.npz"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    groups_path = os.environ.get("GROUPS_JSON")
    if not groups_path:
        raise SystemExit("Set GROUPS_JSON to the partition groups file (see the module docstring)")
    real_dir = Path(a.real_dir)
    coco = json.load(open(real_dir / "annotations.json", encoding="utf8"))
    files = {im["id"]: im["file_name"].replace("\\", "/") for im in coco["images"]}
    anns = {x["id"]: x for x in coco["annotations"]}
    groups = {}
    splits = set()
    for g in json.load(open(groups_path, encoding="utf8")):
        for i in g["image_ids"]:
            groups[i] = g["group"]
            splits.add((i, g["split"]))
    not_train = [i for i, s in splits if s != "train" and i in files]
    if not_train:
        raise SystemExit(f"{len(not_train)} COCO images are not in the training split according to the groups file: aborting")
    ids = sorted(files)
    pos = {iid: k for k, iid in enumerate(ids)}
    crops, dropped = crop_sources(real_dir, anns, files)
    by_img = {}
    for key, iid, box in crops:
        by_img.setdefault(iid, []).append((key, box))
    jobs = [(pos[i], str(real_dir / files[i]), by_img.get(i, [])) for i in ids]

    n = len(ids)
    lab, hist = np.full((n, 6), np.nan, np.float32), np.zeros((n, bm.HIST_BINS ** 3), np.uint8)
    ring = {}
    with Pool(max(1, a.workers), initializer=_low_priority) as pool:
        for done, (k, d, h, rings) in enumerate(pool.imap_unordered(_describe, jobs, chunksize=16), 1):
            if d is not None:
                lab[k], hist[k] = d, np.clip(np.round(h * 255), 0, 255).astype(np.uint8)
            ring.update(rings)
            if done % 1000 == 0:
                print(f"{done}/{n}", flush=True)
    bad = np.where(~np.isfinite(lab).all(axis=1))[0]
    if len(bad):
        raise SystemExit(f"{len(bad)} unreadable images (e.g. {files[ids[bad[0]]]})")
    keys = [c[0] for c in crops]
    crop_img = np.array([pos[c[1]] for c in crops], dtype=np.int32)
    crop_ring = np.stack([ring[k] if ring.get(k) is not None else np.full(6, np.nan, np.float32) for k in keys])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(
        a.out, files=np.array([files[i] for i in ids]), image_id=np.array(ids, np.int32),
        group=np.array([groups.get(i, -1) for i in ids], np.int32), lab=lab, hist=hist,
        crop_keys=np.array(keys), crop_img=crop_img, crop_ring=crop_ring.astype(np.float32),
        ring_frac=np.float32(RING_FRAC))
    print(f"{n} images, {len(keys)} crops ({dropped} without a matching source), "
          f"{int(np.isnan(crop_ring).any(axis=1).sum())} without a measurable ring, {os.path.getsize(a.out) / 1e6:.2f} MB -> {a.out}")


if __name__ == "__main__":
    main()
