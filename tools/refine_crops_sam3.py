"""Refine the generator's crop masks with SAM3 (this script itself needs no GPU).

Three phases, one seed (`--seed`, 21 by default: sample selection and SAM3):

  1. `plan`   (host, CPU)       choose the annotated training objects that have a crop in the libraries
                                (`Extracted_objects_train` and `crops_poly`) and write `jobs.json`.
  2. SAM3     (container)       `tools/sam3_candidates.py` stores the candidate masks of every object (box prompt,
                                box + text concept prompt and text-only prompt; image encoded once). See its header.
  3. `build`  (host, CPU)       pick the best candidate per object, apply the quality filters, handle nets with a fine
                                alpha and write the NEW library `<out>/library/<Class>/sam3r_<ann>.png|json` (same
                                structure as `crops_poly`; existing libraries are never touched), the manifest with
                                per-object measurements and, optionally, a COCO of refined training masks (`--coco-out`).

Selection: score = w_ref * agreement + (1 - w_ref) * SAM3 score. Agreement is the mask IoU with the original polygon
or, if the object has none, the IoU between the mask box and the annotated box. A candidate is rejected if it
overflows the box, has anomalous coverage, does not belong to the class concept or falls below the minimum agreement.
Fishing_Net: fine alpha (what is indistinguishable from the surrounding water is removed), holes are not filled.

Training split only: `plan` and `build` refuse to read a COCO whose name indicates val or test. Idle priority.

Environment:
  GEN_INPUTS   directory with Extracted_objects_train, crops_poly and real_train (default `gen_inputs`)
  TRAIN_JSON   training COCO used by `plan` (default `$GEN_INPUTS/real_train/annotations.json`)

Example (a 40-object sample):
    python -X utf8 tools/refine_crops_sam3.py plan  --out gen_inputs/crops_sam3
    docker exec -i -e CUDA_VISIBLE_DEVICES= synth-segmentation nice -n 19 python - \\
        --jobs /app/datasets/crops_sam3/jobs.json --out /app/datasets/crops_sam3 --device cpu < tools/sam3_candidates.py
    python -X utf8 tools/refine_crops_sam3.py build --out gen_inputs/crops_sam3
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import refine_crops_lib as rl  # noqa: E402

GEN_INPUTS = Path(os.environ.get("GEN_INPUTS", "gen_inputs"))
DEFAULT_QUOTA = {"Fishing_Net": 12, "Plastic_Debris": 6, "Bottle": 4, "Can": 5, "Metal_Debris": 4, "Glove": 3,
                 "Tire": 3, "Mask": 3}
PREFIX = "sam3r_"


def set_idle_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001  (no psutil, or not on Windows)
        pass


def read_train_only(path):
    """Training COCO; refuses to read one whose name indicates val or test."""
    low = Path(path).name.lower()
    if any(t in low for t in ("val", "test")):
        raise SystemExit(f"Training split only: {path} looks like val/test")
    return json.loads(Path(path).read_text(encoding="utf8"))


def crop_candidates(train, ext_manifest, poly_manifest):
    """Training objects with a crop in some library: list of dicts (lib, cls, ann, ...). `ext` and `poly` do not overlap."""
    anns = {a["id"]: a for a in train["annotations"]}
    out = []
    for lib, man in (("ext", ext_manifest), ("poly", poly_manifest)):
        for c in man["crops"]:
            a = anns.get(c["ann"])
            if a is None:
                continue
            name = f"{'poly_' if lib == 'poly' else ''}{c['ann']}.png"
            poly = a.get("segmentation") if isinstance(a.get("segmentation"), list) and a["segmentation"] else None
            out.append({"lib": lib, "cls": c["cls"], "ann": c["ann"], "image_id": a["image_id"], "box": a["bbox"],
                        "poly": poly, "crop": f"{c['cls']}/{name}"})
    return out


def stratified_sample(cands, quota, seed):
    """Seeded sample stratified by class; within a class, objects with a polygon are preferred (half of the quota)."""
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for c in sorted(cands, key=lambda c: (c["lib"], c["ann"])):
        by_cls[c["cls"]].append(c)
    chosen = []
    for cls, q in quota.items():
        with_poly = [c for c in by_cls.get(cls, []) if c["poly"]]
        without = [c for c in by_cls.get(cls, []) if not c["poly"]]
        rng.shuffle(with_poly), rng.shuffle(without)
        take = with_poly[: math.ceil(q / 2)] + without
        take += [c for c in with_poly[math.ceil(q / 2):]]
        chosen += take[:q]
    return chosen


def cmd_plan(a):
    train = read_train_only(a.train_json)
    real = json.loads((Path(a.real_root) / "annotations.json").read_text(encoding="utf8"))
    files = {i["id"]: i["file_name"] for i in real["images"]}
    ext = json.loads((Path(a.crops_root) / "Extracted_objects_train" / "manifest_train.json").read_text(encoding="utf8"))
    poly = json.loads((Path(a.crops_root) / "crops_poly" / "manifest_poly.json").read_text(encoding="utf8"))
    cands = crop_candidates(train, ext, poly)
    if a.objects_json:
        keep = {(o["lib"], o["ann"]) for o in json.loads(Path(a.objects_json).read_text(encoding="utf8"))["objects"]}
        cands = [c for c in cands if (c["lib"], c["ann"]) in keep]
    quota = json.loads(a.quota) if a.quota else DEFAULT_QUOTA
    chosen = sorted(cands, key=lambda c: (c["lib"], c["ann"])) if a.all else stratified_sample(cands, quota, a.seed)
    objs = [{**c, "image": files[c["image_id"]]} for c in chosen]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {"seed": a.seed, "images_root": a.container_images_root, "quota": quota, "n": len(objs), "objects": objs}
    (out / "jobs.json").write_text(json.dumps(jobs), encoding="utf8")
    print(json.dumps({"n": len(objs), "per_class": dict(Counter(o["cls"] for o in objs)),
                      "with_polygon": sum(1 for o in objs if o["poly"])}, ensure_ascii=False))


# ------------------------------------------------------------------ build
def original_mask(job, crops_root, rgb):
    """(original mask in the image or None, reason if it could not be placed, fit rmse)."""
    shape = rgb.shape[:2]
    if job["lib"] == "poly":
        return rl.rings_mask(job["poly"], shape), None, None
    path = Path(crops_root) / "Extracted_objects_train" / job["crop"]
    if not path.exists():
        return None, "crop_missing", None
    m, rmse = rl.original_mask_ext(rgb, np.asarray(Image.open(path).convert("RGBA")), job["box"])
    return m, (None if m is not None else "crop_not_located"), rmse


def overflow(mask, box, dilate=0.05):
    """Fraction of the mask outside the grown annotated box."""
    near = np.logical_and(mask, rl.box_mask(box, mask.shape, dilate)).sum()
    return float(1.0 - near / max(1, mask.sum()))


def object_metrics(job, rgb, orig, alpha, ref_mask):
    """Original vs refined mask measurements (coverage, IoU, background inside, box fill)."""
    barea = max(1.0, job["box"][2] * job["box"][3])
    ref = alpha > 0.5
    m = {"cov_ref": float(ref.sum() / barea), "rect_fill_ref": rl.rect_fill(alpha), "overflow_ref": overflow(ref, job["box"])}
    if ref_mask is not None and job["lib"] == "ext":
        m["iou_poly_ref"] = rl.mask_iou(ref, ref_mask)
    if orig is None:
        return m
    common = orig | ref
    m.update(cov_orig=float(orig.sum() / barea), overflow_orig=overflow(orig, job["box"]), rect_fill_orig=rl.rect_fill(orig.astype(np.float32)),
             iou_orig_ref=rl.mask_iou(orig, ref),
             bg_orig=rl.background_fraction(rgb, orig.astype(np.float32), common),
             bg_ref=rl.background_fraction(rgb, alpha, common))
    if ref_mask is not None and job["lib"] == "ext":
        m["iou_poly_orig"] = rl.mask_iou(orig, ref_mask)
    return m


def finalize_alpha(job, rgb, cand_mask, p):
    """(float alpha in [0, 1], notes): cleaned mask; fine alpha for Fishing_Net."""
    box_m = rl.box_mask(job["box"], cand_mask.shape, p.box_dilate)
    mask = rl.clean_mask(cand_mask & box_m, p, fill_holes=job["cls"] != rl.NET)
    notes = []
    alpha = mask.astype(np.float32)
    if job["cls"] == rl.NET:
        fine = rl.net_alpha(rgb, mask)
        if fine is None or (fine > 0.5).sum() < p.net_alpha_min * mask.sum():
            notes.append("fine_alpha_dropped")
        else:
            alpha = fine
    return alpha, notes


def save_crop(lib_dir, job, rgb, alpha, row):
    """Write the RGBA PNG (box + 5 px padding, RGB kept outside the mask) and a minimal COCO JSON."""
    rgba = rl.make_rgba(rgb, alpha, pad=rl.CROP_PAD)
    cdir = Path(lib_dir) / job["cls"]
    cdir.mkdir(parents=True, exist_ok=True)
    name = f"{PREFIX}{job['ann']}"
    Image.fromarray(rgba, "RGBA").save(cdir / f"{name}.png")
    h, w = rgba.shape[:2]
    meta = {"info": {"description": "Extracted object refined with SAM3", "version": "1.0", "source": "sam3_refined",
                     "extracted_from": f"image_{job['image_id']}", "original_annotation_id": job["ann"],
                     "prompt_mode": row["mode"], "sam3_score": row["score"], "extraction_date": datetime.now().isoformat()},
            "licenses": [], "images": [{"id": 1, "file_name": f"{name}.png", "width": w, "height": h}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, w, h], "area": w * h, "iscrowd": 0,
                             "segmentation": [[0, 0, w, 0, w, h, 0, h]]}],
            "categories": [{"id": 1, "name": job["cls"], "supercategory": None}]}
    (cdir / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    return f"{job['cls']}/{name}.png", [w, h]


def refine_one(job, a, p, lib_dir):
    """Refine one object; return (manifest row, alpha or None, mask or None)."""
    row = {"cls": job["cls"], "ann": job["ann"], "lib": job["lib"], "has_poly": bool(job["poly"])}
    cpath = Path(a.out) / "cand" / f"{job['ann']}.npz"
    if not cpath.exists():
        return {**row, "status": "no_sam3_candidates"}, None, None
    rgb = np.asarray(Image.open(Path(a.real_root) / job["image"]).convert("RGB"))
    shape = rgb.shape[:2]
    cands = rl.load_candidates(cpath, shape)
    ref_mask = rl.rings_mask(job["poly"], shape) if job["poly"] else None
    orig, why_orig, rmse = original_mask(job, a.crops_root, rgb)
    weak = orig if (ref_mask is None and orig is not None) else None
    row["ref_kind"] = "polygon" if ref_mask is not None else ("original" if weak is not None else "box")
    best, diag = rl.select_best(cands, ref_mask, job["box"], job["cls"], p, weak)
    row["candidates"] = sorted(diag, key=lambda r: (len(r["reasons"]), -r["comb"]))[:4]
    if why_orig:
        row["original_note"] = why_orig
    elif rmse is not None:
        row["orig_rmse"] = round(float(rmse), 3)
    if best is None:
        return {**row, "status": "rejected", "reason": rl.main_reason(diag)}, None, None
    alpha, notes = finalize_alpha(job, rgb, cands[best].mask, p)
    if job["cls"] != rl.NET and rl.rect_fill(alpha) > p.rect_fill_max:
        return {**row, "status": "rejected", "reason": "rectangular"}, None, None
    row.update(status="accepted", mode=cands[best].mode, score=round(cands[best].score, 3), notes=notes)
    row.update({k: (round(v, 4) if v is not None else None) for k, v in object_metrics(job, rgb, orig, alpha, ref_mask).items()})
    row["file"], row["wh"] = save_crop(lib_dir, job, rgb, alpha, row)
    return row, alpha, cands[best].mask


def median(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.median(v)), 3) if v else None


def summarize(rows):
    """Per class: accepted, rejected with reason, and original vs refined medians."""
    out = {}
    for cls in sorted({r["cls"] for r in rows}):
        rs = [r for r in rows if r["cls"] == cls]
        ok = [r for r in rs if r["status"] == "accepted"]
        out[cls] = {"objects": len(rs), "accepted": len(ok),
                    "reasons": dict(Counter(r.get("reason", r["status"]) for r in rs if r["status"] != "accepted")),
                    "mode": dict(Counter(r["mode"] for r in ok)),
                    **{f"med_{k}": median(ok, k) for k in ("cov_orig", "cov_ref", "iou_poly_orig", "iou_poly_ref",
                                                          "iou_orig_ref", "bg_orig", "bg_ref", "rect_fill_orig", "rect_fill_ref",
                                                          "overflow_orig", "overflow_ref")}}
    return out


def write_coco(a, jobs, rows, alphas):
    """Training COCO with only the accepted refined masks (mask polygons; fitted box)."""
    if any(t in Path(a.coco_out).name.lower() for t in ("val", "test")):
        raise SystemExit("Training split only: the output COCO name looks like val/test")
    real = json.loads((Path(a.real_root) / "annotations.json").read_text(encoding="utf8"))
    imgs = {i["id"]: i for i in real["images"]}
    ann_cat = {x["id"]: x["category_id"] for x in real["annotations"]}
    anns, used = [], set()
    for job, row in zip(jobs, rows):
        m = alphas.get(job["ann"])
        if row["status"] != "accepted" or m is None:
            continue
        bb = rl.mask_bbox(m > 0.5)
        anns.append({"id": job["ann"], "image_id": job["image_id"], "category_id": ann_cat[job["ann"]],
                     "bbox": list(bb), "area": int((m > 0.5).sum()), "iscrowd": 0, "segmentation": rl.mask_to_polygons(m > 0.5),
                     "sam3_score": row["score"], "prompt_mode": row["mode"]})
        used.add(job["image_id"])
    coco = {"info": {"description": "Masks refined with SAM3 (training split only, sample)", "seed": a.seed},
            "images": [imgs[i] for i in sorted(used)], "annotations": anns, "categories": real["categories"]}
    Path(a.coco_out).write_text(json.dumps(coco), encoding="utf8")


def cmd_build(a):
    p = rl.Params(**json.loads(a.params)) if a.params else rl.Params()
    jobs = json.loads((Path(a.out) / "jobs.json").read_text(encoding="utf8"))
    lib_dir = Path(a.out) / "library"
    rows, alphas = [], {}
    for job in jobs["objects"]:
        job["box"] = [float(v) for v in job["box"]]
        row, alpha, _ = refine_one(job, a, p, lib_dir)
        rows.append(row)
        if alpha is not None:
            alphas[job["ann"]] = alpha
    summ = summarize(rows)
    (Path(a.out) / "manifest_sam3r.json").write_text(json.dumps(
        {"source": "sam3_refined", "seed": jobs["seed"], "params": asdict(p), "summary": summ, "objects": rows},
        indent=1, ensure_ascii=False), encoding="utf8")
    if a.coco_out:
        write_coco(a, jobs["objects"], rows, alphas)
    print(json.dumps(summ, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "build"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=str(GEN_INPUTS / "crops_sam3"))
        s.add_argument("--real-root", default=str(GEN_INPUTS / "real_train"))
        s.add_argument("--crops-root", default=str(GEN_INPUTS))
        s.add_argument("--seed", type=int, default=21)
    s = sub.choices["plan"]
    s.add_argument("--train-json", default=os.environ.get("TRAIN_JSON", str(GEN_INPUTS / "real_train" / "annotations.json")),
                   help="training COCO (env TRAIN_JSON)")
    s.add_argument("--objects-json", help="restrict the candidates to the objects of this JSON")
    s.add_argument("--all", action="store_true", help="every object with a crop, not a sample")
    s.add_argument("--quota", help='JSON {"class": n}; default 40 objects')
    s.add_argument("--container-images-root", default="/app/datasets/real_train")
    s = sub.choices["build"]
    s.add_argument("--params", help="JSON with refine_crops_lib.Params thresholds")
    s.add_argument("--coco-out", help="COCO of refined training masks (optional)")
    a = ap.parse_args()
    set_idle_priority()
    random.seed(a.seed), np.random.seed(a.seed)
    {"plan": cmd_plan, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
