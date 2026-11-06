"""Clean the crop library: measure how far each mask overflows its annotated box and write a NEW library.

For every crop of `Extracted_objects_train` and `crops_poly` that has a source annotation (the `jobs.json` written by
`refine_crops_sam3.py plan --all`), the crop is located in its training image by template matching
(`refine_crops_lib.locate_crop`), its overflow, coverage and components are measured (`clean_crops_lib.measure`) and
the alpha channel is cleaned (`clean_crops_lib.clean_alpha`): clip to the annotated box + margin, remove small
components, and drop the crop if almost nothing remains or the coverage is anomalous.

The output (`--out-root/<library>_clean`) mirrors the original structure (`<Class>/<name>.png|json`, manifest) so the
generator reads it the same way; RGB is kept outside the mask. It never writes into a folder that already has content
and never touches the original libraries. CPU only (idle priority, empty `CUDA_VISIBLE_DEVICES`), training images only.

Crops that cannot be located cannot be measured or cleaned: by default they are copied unchanged (`--unlocated drop`
removes them) and counted separately in the report.

Environment: GEN_INPUTS (default `gen_inputs`) holds the libraries, `real_train` and `crops_sam3_full/jobs.json`.

Usage (from the repository root):
    python -X utf8 tools/clean_crops.py --workers 10                 # writes both _clean libraries and the report
    python -X utf8 tools/clean_crops.py --dry-run --workers 10       # only measures and writes the report
    python -X utf8 tools/clean_crops.py --mode drop --workers 10 --report report.json
        # _drop libraries: same drops, original masks (not clipped to the box)
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import clean_crops_lib as cl  # noqa: E402
import refine_crops_lib as rl  # noqa: E402

GEN_INPUTS = Path(os.environ.get("GEN_INPUTS", "gen_inputs"))
LIBS = {"ext": "Extracted_objects_train", "poly": "crops_poly"}
MANIFESTS = {"ext": "manifest_train.json", "poly": "manifest_poly.json"}


def set_idle_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass


def _init_worker():
    set_idle_priority()


def sidecar(src_json: Path, name: str, w: int, h: int, cls: str, row: dict) -> dict:
    """Sidecar JSON of the clean crop: the original with size, box and rectangular polygon updated."""
    meta = json.loads(src_json.read_text(encoding="utf8")) if src_json.exists() else {"info": {}, "licenses": [], "categories": [
        {"id": 1, "name": cls, "supercategory": None}]}
    meta.setdefault("info", {})["cleaned"] = {"area_orig": row.get("area_orig"), "area_final": row.get("area_final"),
                                              "tool": "clean_crops.py"}
    meta["images"] = [{"id": 1, "file_name": name, "width": w, "height": h}]
    meta["annotations"] = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, w, h], "area": w * h, "iscrowd": 0,
                            "segmentation": [[0, 0, w, 0, w, h, 0, h]]}]
    return meta


def process_image(args):
    """Process every crop of one training image; return the report rows."""
    jobs, real_root, in_roots, out_roots, p, unlocated = args[:6]
    mode = args[6] if len(args) > 6 else "clean"
    rgb = np.asarray(Image.open(Path(real_root) / jobs[0]["image"]).convert("RGB"))
    rows = []
    for job in jobs:
        box = [float(v) for v in job["box"]]
        src = Path(in_roots[job["lib"]]) / job["crop"]
        row = {"cls": job["cls"], "lib": job["lib"], "ann": job["ann"], "crop": job["crop"]}
        if not src.exists():
            rows.append({**row, "status": "unlocated", "note": "crop_missing"})
            continue
        rgba = np.asarray(Image.open(src).convert("RGBA"))
        found = rl.locate_crop(rgb, rgba, box)
        if found is None:
            rows.append({**row, "status": "unlocated", "note": "no_match"})
            _write_unlocated(job, src, out_roots, unlocated)
            continue
        pos, rmse = found
        row.update(cl.measure(rgba[:, :, 3], box, pos, p))
        row["rmse"] = round(float(rmse), 3)
        new_alpha, info = cl.clean_alpha(rgba[:, :, 3], box, pos, job["cls"], p)
        row.update(info)
        if new_alpha is None:
            row["status"] = "dropped"
        else:
            row["status"] = "clean"
            if mode == "drop":                  # drop-only: the original crop passes unchanged, not clipped to the box
                row["modified"] = False
                row["removed_frac"] = 0.0
            if out_roots:
                if mode == "drop":
                    _write_unlocated(job, src, out_roots, "keep")
                else:
                    _write_clean(job, src, rgba, new_alpha, out_roots[job["lib"]], row)
        rows.append(row)
    return rows


def _write_unlocated(job, src, out_roots, mode):
    if not out_roots or mode != "keep":
        return
    dst = Path(out_roots[job["lib"]]) / job["crop"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if src.with_suffix(".json").exists():
        shutil.copy2(src.with_suffix(".json"), dst.with_suffix(".json"))


def _write_clean(job, src, rgba, new_alpha, out_root, row):
    pad = cl.original_pad(rgba[:, :, 3])
    out = rgba.copy()
    out[:, :, 3] = new_alpha
    out, _ = cl.shrink_rgba(out, pad)
    dst = Path(out_root) / job["crop"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGBA").save(dst)
    h, w = out.shape[:2]
    meta = sidecar(src.with_suffix(".json"), dst.name, w, h, job["cls"], row)
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    row["wh"] = [w, h]


def write_manifests(rows, in_roots, out_roots, p, unlocated):
    """Manifests of the clean libraries: the source ones without the dropped crops (plus the cleaning info)."""
    gone = defaultdict(set)
    for r in rows:
        if r["status"] == "dropped" or (r["status"] == "unlocated" and unlocated == "drop"):
            gone[r["lib"]].add(r["ann"])
    for lib, root in out_roots.items():
        man = json.loads((Path(in_roots[lib]) / MANIFESTS[lib]).read_text(encoding="utf8"))
        man["crops"] = [c for c in man["crops"] if c["ann"] not in gone[lib]]
        if lib == "ext":
            man["n_crops"] = len(man["crops"])
            man["destination"] = str(root)
        man["cleaning"] = {"params": asdict(p), "source": str(in_roots[lib]), "dropped": len(gone[lib]),
                           "unlocated": unlocated}
        (Path(root) / MANIFESTS[lib]).write_text(json.dumps(man), encoding="utf8")


def prepare_out(out_roots, dry_run):
    """Create the output folders; refuse to use one that already has content (nothing is overwritten)."""
    if dry_run:
        return
    for root in out_roots.values():
        r = Path(root)
        if r.exists() and any(r.iterdir()):
            raise SystemExit(f"{r} already exists and is not empty: refusing to overwrite (delete it by hand if it is an incomplete output)")
        r.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", default=str(GEN_INPUTS / "crops_sam3_full" / "jobs.json"))
    ap.add_argument("--real-root", default=str(GEN_INPUTS / "real_train"))
    ap.add_argument("--in-root", default=str(GEN_INPUTS))
    ap.add_argument("--out-root", default=str(GEN_INPUTS))
    ap.add_argument("--mode", choices=["clean", "drop"], default="clean",
                    help="clean: alpha clipped to the box; drop: copy unchanged only the crops that are not dropped")
    ap.add_argument("--suffix", default=None, help="default _clean (clean mode) or _drop (drop mode)")
    ap.add_argument("--report", default=None, help="JSON report (default <out-root>/clean_crops_report.json)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--unlocated", choices=["keep", "drop"], default="keep")
    ap.add_argument("--params", help="JSON with CleanParams thresholds")
    ap.add_argument("--limit", type=int, help="only the first N images (testing)")
    ap.add_argument("--dry-run", action="store_true", help="only measure and write the report; write no crops")
    a = ap.parse_args()
    set_idle_priority()
    a.suffix = a.suffix or ("_drop" if a.mode == "drop" else "_clean")
    p = cl.CleanParams(**json.loads(a.params)) if a.params else cl.CleanParams()
    if "val" in Path(a.jobs).name.lower() or "test" in Path(a.jobs).name.lower():
        raise SystemExit("Training split only")
    jobs = json.loads(Path(a.jobs).read_text(encoding="utf8"))["objects"]
    in_roots = {k: str(Path(a.in_root) / v) for k, v in LIBS.items()}
    out_roots = {k: str(Path(a.out_root) / (v + a.suffix)) for k, v in LIBS.items()}
    prepare_out(out_roots, a.dry_run)
    by_img = defaultdict(list)
    for j in jobs:
        by_img[j["image"]].append(j)
    groups = list(by_img.values())[: a.limit]
    todo = [(g, a.real_root, in_roots, None if a.dry_run else out_roots, p, a.unlocated, a.mode) for g in groups]
    t0, rows = time.time(), []
    with ProcessPoolExecutor(a.workers, initializer=_init_worker) as ex:
        for i, rs in enumerate(ex.map(process_image, todo, chunksize=4)):
            rows += rs
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(todo)} images, {time.time() - t0:.0f} s", flush=True)
    if not a.dry_run:
        write_manifests(rows, in_roots, out_roots, p, a.unlocated)
    report = {"params": asdict(p), "unlocated": a.unlocated, "seconds": round(time.time() - t0, 1), "n": len(rows),
              "dry_run": a.dry_run, "mode": a.mode, "drops_by_class": cl.drops_by_class(rows), "summary": cl.summarize(rows, p),
              "totals": {s: sum(r["status"] == s for r in rows) for s in ("clean", "dropped", "unlocated")},
              "objects": rows}
    rep = Path(a.report or Path(a.out_root) / "clean_crops_report.json")
    rep.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"totals": report["totals"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
