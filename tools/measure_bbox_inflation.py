"""Measure how much the generator inflates the annotated box relative to the object's mask box (CPU only).

The generator's COCO does not store masks (`segmentation` is empty), so the measurement REPEATS the generation with the
real compositor, patched to record three boxes per pasted object:
  tight   box fitted to the object's binary mask (`cv2.boundingRect(mask_bin)`), the third value `_place_object` returns
          (mask in image coordinates)
  shadow  box after the shadow / motion-blur recomputation (inside `_place_object`, +-80 px)
  final   annotated box, after `_recalculate_bbox_after_global_effects` (+-20 px) and clipping to the image
Limitations: fallback depth (vertical gradient, no service), random placement (no SAM) and a different random-number
consumption than a real batch, so the boxes are not those of a real batch although the margin distribution is
the code's own. One variant = one profile: `legacy` (no profile) or any variant of run_variants.py.

Usage (from the repository root, CPU):
  python -X utf8 tools/measure_bbox_inflation.py --variant legacy --n 300 --out ./bbox_check
  python -X utf8 tools/measure_bbox_inflation.py --variant j_v11_visible --n 300 --out ./bbox_check
Output: <out>/<variant>_objects.json (one row per object) and a summary on stdout.

Environment:
  GEN_ROOT      repository root (default: parent of tools/)
  GEN_INPUTS    directory with Backgrounds_filtered, real_train, Extracted_objects_train and crops_poly
                (default `gen_inputs`)
  JOB_REQUEST   job request JSON (default configs/batch_request.json)
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

GEN_ROOT = Path(os.environ.get("GEN_ROOT", Path(__file__).resolve().parents[1]))
GEN_INPUTS = Path(os.environ.get("GEN_INPUTS", "gen_inputs"))
REQUEST = Path(os.environ.get("JOB_REQUEST", GEN_ROOT / "configs" / "batch_request.json"))
os.environ["AUGMENTOR_APP_ROOT"] = str(GEN_ROOT / "services" / "augmentor")
sys.path.insert(0, str(GEN_ROOT / "tools"))
import run_variants as variants  # noqa: E402

LEGACY = "legacy"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_box(mask):
    ys, xs = np.nonzero(mask > 127)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else None


def install_recorders(comp, rows):
    """Record tight/shadow in `_place_object` and final in the global recomputation (matched by the post-shadow box)."""
    pending = {}
    orig_place = comp._place_object
    orig_recalc = comp._recalculate_bbox_after_global_effects

    async def place(composite, obj_placement, *args, **kw):
        res = await orig_place(composite, obj_placement, *args, **kw)
        if res is not None:
            _, bbox, mask = res
            pending[tuple(int(v) for v in bbox)] = (mask_box(mask), obj_placement.class_name,
                                                    composite.shape[1], composite.shape[0])
        return res

    def recalc(before, after, bbox, *args, **kw):
        out = orig_recalc(before, after, bbox, *args, **kw)
        key = tuple(int(v) for v in bbox)
        tight, cls, w, h = pending.pop(key, (None, "?", before.shape[1], before.shape[0]))
        final = (max(0, out[0]), max(0, out[1]), min(w, out[2]), min(h, out[3]))
        rows.append({"class": cls, "tight": tight, "shadow": list(key), "final": list(final),
                     "img_w": w, "img_h": h})
        return out

    comp._place_object, comp._recalculate_bbox_after_global_effects = place, recalc

    def flush():
        """Objects whose box skips the global recomputation (`tight_bbox` profile): final = clipped post-shadow box."""
        for key, (tight, cls, w, h) in pending.items():
            final = [max(0, key[0]), max(0, key[1]), min(w, key[2]), min(h, key[3])]
            rows.append({"class": cls, "tight": tight, "shadow": list(key), "final": final, "img_w": w, "img_h": h})
        pending.clear()

    return flush


def build_spec(name):
    if name == LEGACY:
        return {"profile": None}
    return variants.VARIANTS[name]


async def run(a):
    from app.models.schemas import EffectsConfig, EffectType
    req = json.load(open(REQUEST, encoding="utf8"))
    req["semantic_placement"] = False
    spec = build_spec(a.variant)
    variants.install_patches()
    variants.FLAGS.update(aug=spec.get("aug", True))
    comp = variants.make_composer("http://127.0.0.1:9", None)

    async def fallback_depth(image, cache_key=None):
        h, w = image.shape[:2]
        return np.repeat(np.linspace(0, 1, h)[:, np.newaxis], w, axis=1).astype(np.float32), "fallback_gradient"

    comp._get_depth_map = fallback_depth
    rows = []
    flush = install_recorders(comp, rows)
    ec = dict(req["effects_config"])
    if a.no_recalc_flag:
        ec["recalculate_bbox_after_global"] = False
    real = a.variant != LEGACY and spec["real"]
    backgrounds = str(GEN_INPUTS / "real_train") if real else str(GEN_INPUTS / "Backgrounds_filtered")
    profile = (None if a.variant == LEGACY
               else variants.build_profile(spec, a.variant, [str(GEN_INPUTS / "crops_poly")]))
    out_dir = tempfile.mkdtemp(prefix="bbox_check_", dir=a.out)
    await comp.compose_batch(
        backgrounds_dir=backgrounds, objects_dir=str(GEN_INPUTS / "Extracted_objects_train"),
        output_dir=out_dir, num_images=a.n, targets_per_class=req["targets_per_class"],
        max_objects_per_image=req["max_objects_per_image"], effects=[EffectType(e) for e in req["effects"]],
        effects_config=EffectsConfig(**ec), depth_aware=req["depth_aware"], validate_quality=False,
        validate_physics=False, reject_invalid=False, semantic_placement=False, synth_profile=profile, seed=a.seed,
        dataset_info={"name": "bbox check"})
    flush()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", default=LEGACY, help="`legacy` (no profile) or a run_variants.py variant")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-recalc-flag", action="store_true", help="turn recalculate_bbox_after_global off (control)")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows = asyncio.run(run(a))
    tag = a.variant + ("_norecalc" if a.no_recalc_flag else "")
    json.dump(rows, open(Path(a.out) / f"{tag}_objects.json", "w"), indent=0)
    print(f"{tag}: {len(rows)} objects")


if __name__ == "__main__":
    main()
