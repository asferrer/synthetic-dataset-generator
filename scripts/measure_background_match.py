"""Expected diversity of the crop-background pairing (`background_match`), measured on CPU WITHOUT generating images.

It reproduces only the PLANNING of the batch (round-robin backgrounds, classes, crops with a usage cap and the scale
filter) with the generator's own code (`SynthContext` + `plan_objects`) and counts where the crops come from. It does
not place or compose: failed placements (the compositor drops objects that do not fit) are absent, so these are
REQUESTED uses, not pasted ones.

Per configuration: objects, distinct crops, crops at the cap, distinct source images, usage spread over source images
(maximum, top 10, Gini index), how many objects come from the same image group as the background, mean LAB distance
background-source (against the random baseline) and distinct backgrounds per source image.

Environment:
  GEN_INPUTS   directory with real_train, Extracted_objects_train and crops_poly (default `gen_inputs`)
  JOB_REQUEST  job request JSON (default configs/batch_request.json)

Usage (from the repository root):
  python -X utf8 scripts/measure_background_match.py --n 300 3000 --out output/background_match_diversity
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "augmentor"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from app import background_match as bm  # noqa: E402
from app.crop_filter import apply_library_filters  # noqa: E402
from app.object_planner import plan_objects  # noqa: E402
from app.synth_context import SynthContext, merge_extra_libraries  # noqa: E402
from app.synth_profile import (V11_PROFILE, V11_Q_BASE_OVERRIDES, V11_Q_LAB_OVERRIDES,  # noqa: E402
                               V11_Q_LAB_SAME_OVERRIDES)

GEN_INPUTS = Path(os.environ.get("GEN_INPUTS", "gen_inputs"))
REQUEST = Path(os.environ.get("JOB_REQUEST", ROOT / "configs" / "batch_request.json"))
REAL_BASE = {"copy_paste_real": True, "harmonize": True}
CONFIGS = {
    "q_base (random)": V11_Q_BASE_OVERRIDES,
    "q_lab (k=20, T=1, excluding same group)": V11_Q_LAB_OVERRIDES,
    "q_lab_same (k=20, T=1)": V11_Q_LAB_SAME_OVERRIDES,
    "q_lab_same (k=50, T=1)": {**V11_Q_LAB_SAME_OVERRIDES, "background_match_k": 50},
    "q_lab_same (k=20, T=3)": {**V11_Q_LAB_SAME_OVERRIDES, "background_match_temperature": 3.0},
    "q_lab_same (k=20, T=0, nearest)": {**V11_Q_LAB_SAME_OVERRIDES, "background_match_temperature": 0.0},
}


def load_objects(profile, objects_dir: Path):
    objs = {}
    for cls in sorted(os.listdir(objects_dir)):
        cdir = objects_dir / cls
        if cdir.is_dir():
            objs[cls] = [{"path": str(cdir / f), "name": f} for f in sorted(os.listdir(cdir)) if f.lower().endswith(".png")]
    merge_extra_libraries(objs, profile)
    apply_library_filters(objs, profile)
    return objs


def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def simulate(profile, objects, bg_files, sizes, request, n_images: int, seed: int, table):
    ctx = SynthContext(profile, objects, str(GEN_INPUTS / "real_train"), bg_files, base_seed=seed)
    required = {c: request["targets_per_class"].get(c, 0) for c in objects}
    used_src, pairs, dists, same_group, per_crop = Counter(), Counter(), [], 0, Counter()
    n_obj = 0
    for attempt in range(n_images):
        rng = ctx.make_rng(attempt)
        bg = ctx.pick_background(attempt, rng)
        avail = [c for c in required if ctx.usage is None or ctx.usage.has_free(c)]
        if not avail:
            break
        objs, reserved, _ = plan_objects(ctx, rng, avail, required, objects, request["max_objects_per_image"],
                                         bg_wh=sizes[ctx.background_name(bg)], bg_file=bg)
        if ctx.usage is not None:
            ctx.usage.settle(reserved, [o.image_path for o in objs])   # all are pasted: the case most favourable to the cap
        bi = table.index[ctx.background_name(bg)]
        for o in objs:
            s = table.source_image(o.image_path)
            n_obj += 1
            per_crop[o.image_path] += 1
            if s < 0:
                continue
            used_src[s] += 1
            pairs[(s, bi)] += 1
            dists.append(float(bm.descriptor_distance(table.lab[bi], table.lab[s])))
            same_group += int(table.group[s] >= 0 and table.group[s] == table.group[bi])
    counts = np.array(sorted(used_src.values(), reverse=True))
    bgs_per_src = Counter(s for s, _ in pairs)
    return {
        "images": n_images, "objects": n_obj, "distinct_crops": len(per_crop),
        "crops_at_cap": sum(1 for v in per_crop.values() if profile.max_crop_reuse and v >= profile.max_crop_reuse),
        "max_crop_uses": max(per_crop.values()), "distinct_sources": len(used_src),
        "max_source_uses": int(counts[0]), "top10_sources_frac": round(float(counts[:10].sum() / counts.sum()), 4),
        "sources_gini": round(gini(counts), 3), "same_group_frac": round(same_group / max(1, len(dists)), 4),
        "lab_dist_mean": round(float(np.mean(dists)), 2), "lab_dist_median": round(float(np.median(dists)), 2),
        "backgrounds_per_source_mean": round(float(np.mean(list(bgs_per_src.values()))), 2),
        "match": dict(ctx.matcher.stats) if ctx.matcher else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, nargs="+", default=[300, 3000], help="simulated images per configuration")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default="output/background_match_diversity")
    a = ap.parse_args()
    request = json.load(open(REQUEST, encoding="utf8"))
    real = GEN_INPUTS / "real_train"
    coco = json.load(open(real / "annotations.json", encoding="utf8"))
    sizes = {im["file_name"].replace("\\", "/"): (im["width"], im["height"]) for im in coco["images"]}
    bg_files = [str(real / f) for f in sizes]
    table = bm.load_match_table(None)
    base = V11_PROFILE.model_copy(update={"extra_objects_dirs": [str(GEN_INPUTS / "crops_poly")]})
    objects = load_objects(base, GEN_INPUTS / "Extracted_objects_train")
    print({c: len(v) for c, v in objects.items()}, flush=True)
    results = {}
    for n in a.n:
        for name, ov in CONFIGS.items():
            profile = base.model_copy(update={"name": "m", **REAL_BASE, **ov})
            random.seed(a.seed)
            r = simulate(profile, objects, bg_files, sizes, request, n, a.seed, table)
            results[f"{name} | n={n}"] = r
            print(name, n, r, flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(results, open(Path(a.out) / "background_match_diversity.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
