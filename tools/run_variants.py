"""Run synthesis-profile variants (ablation arms and the final batch) over the same small job.

Every variant uses the same seed and cycles through the backgrounds round-robin, so variants are comparable.
Each one is a synthesis profile (`v10` or `v11`, see docs/SYNTHESIS_PROFILE.md) plus a set of overrides.

Variants:
  c_v10_noaug       reference: profile v10 with the augmentations neutralised AFTER sampling
  f_v11_rectfilter  profile v11 (no paste augmentations, rectangular-crop filter), synthetic backgrounds
  g_v11_copypaste   f + copy-paste onto real training images (real background, real boxes kept)
  h_v11_harmon      f + local colour harmonisation
  i_v11_all         f + copy-paste + harmonisation
  j_v11_visible     i + no global haze, small fitting blur, per-class visibility filter
  k_sharp           j + single resample (one warpAffine) + sharpness/grain matching
  l_scale           k + crop scale restricted to a range
  m_pos             l + object position drawn from the real position statistics
  n_vis15           m + visibility threshold at the 15th percentile of real objects
  o_tight           n + annotated box fitted to the mask (`tight_bbox`)
  p_pos_tight       m + `tight_bbox`
  s_clean           p_pos_tight with the cleaned crop library (tools/clean_crops.py, mask cropped to the box)
  s_drop            p_pos_tight with the drop-only crop library (tools/clean_crops.py --mode drop); the final batch
  q_base            p_pos_tight without sharpness/grain matching
  q_lab             q_base + `background_match` lab_knn excluding the group of the crop's source image
  q_lab_same        q_base + `background_match` lab_knn allowing the same group (never the same image)
  q_lab_local       q_lab_same + `local_patch_match` (position chosen by ring similarity)
  r_matte           q_lab_same + `edge_matting` (closed-form alpha from a trimap; needs pymatting)
  r_matte_fg        r_matte with Germer foreground decontamination
  r_medium          q_lab_same without harmonisation + `medium_transfer` (ring gain 0.5)
  r_medium_harm     q_lab_same + `medium_transfer` (ring gain 1) before harmonisation

With copy-paste the background is real, so per-image detectability metrics drop trivially; compare per-object
metrics instead (kept real boxes carry `origin = "real"` in the COCO and must be excluded).

Usage (needs the depth and segmentation services, i.e. GPU and Docker). Run it INSIDE the augmentor container:

  export MSYS_NO_PATHCONV=1        # Git Bash on Windows only: stops /tmp and /app being rewritten
  docker cp tools/run_variants.py synth-augmentor:/tmp/run_variants.py
  docker cp configs/batch_request.json synth-augmentor:/tmp/batch_request.json
  docker cp services/augmentor/app/. synth-augmentor:/app/app/        # if the image is older than the code
  docker exec synth-augmentor python /tmp/run_variants.py --n 20 --out /app/output/synthetic_batch_smoke
  docker exec synth-augmentor python /tmp/run_variants.py --n 300 --variants s_drop
  docker cp synth-augmentor:/app/output/synthetic_batch ./output/synthetic_batch

Arguments:
  --n               images per variant (default 300)
  --seed            random seed shared by all variants (default 21)
  --out             output directory inside the container (default /app/output/synthetic_batch)
  --request         JSON job request with effects and targets (default /tmp/batch_request.json,
                    i.e. configs/batch_request.json copied into the container; env JOB_REQUEST)
  --backgrounds     synthetic backgrounds directory (default: `backgrounds_dir` of the request)
  --real-dir        real training images with annotations.json for copy-paste (default /app/datasets/real_train)
  --objects         crop library (default: `objects_dir` of the request; some variants set their own)
  --extra-objects   additional polygon crop libraries (default /app/datasets/crops_poly; empty list: none)
  --variants        variants to run (default: all); a variant whose COCO is already complete is skipped
  --save-masks      store each pasted object's mask in the COCO (`mask_rle`)
  --depth-url, --seg-url   service URLs (env DEPTH_SERVICE_URL, SEGMENTATION_SERVICE_URL)

Environment: AUGMENTOR_APP_ROOT (default /app) is the directory that contains the `app` package.
Each variant writes `rect_filter_report.json` (discarded crops per class) next to its COCO and `variant.json`.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("AUGMENTOR_APP_ROOT", "/app"))
from app.synth_profile import (V11_POSITION_OVERRIDES, V11_Q_BASE_OVERRIDES, V11_Q_LAB_LOCAL_OVERRIDES,  # noqa: E402
                               V11_Q_LAB_OVERRIDES, V11_Q_LAB_SAME_OVERRIDES, V11_R_MATTE_FG_OVERRIDES,
                               V11_R_MATTE_OVERRIDES, V11_R_MEDIUM_HARM_OVERRIDES, V11_R_MEDIUM_OVERRIDES,
                               V11_SCALE_OVERRIDES, V11_SHARP_OVERRIDES, V11_TIGHT_OVERRIDES, V11_VIS15_OVERRIDES,
                               V11_VISIBLE_OVERRIDES)

# Copy-paste onto a real background + harmonisation: the base of i and of every later variant
REAL_BASE = {"copy_paste_real": True, "harmonize": True}
# Cleaned crop libraries (tools/clean_crops.py) as seen from the container; used by s_clean
CLEAN_OBJECTS = "/app/datasets/Extracted_objects_train_clean"
CLEAN_POLY = "/app/datasets/crops_poly_clean"
# Drop-only crop libraries (original masks); used by s_drop
DROP_OBJECTS = "/app/datasets/Extracted_objects_train_drop"
DROP_POLY = "/app/datasets/crops_poly_drop"
# `overrides`: profile fields to override; `real`: real training backgrounds (copy-paste)
VARIANTS = {
    "c_v10_noaug": {"profile": "v10", "aug": False, "real": False, "overrides": {}},
    "f_v11_rectfilter": {"profile": "v11", "aug": True, "real": False, "overrides": {}},
    "g_v11_copypaste": {"profile": "v11", "aug": True, "real": True, "overrides": {"copy_paste_real": True}},
    "h_v11_harmon": {"profile": "v11", "aug": True, "real": False, "overrides": {"harmonize": True}},
    "i_v11_all": {"profile": "v11", "aug": True, "real": True,
                  "overrides": {"copy_paste_real": True, "harmonize": True}},
    "j_v11_visible": {"profile": "v11", "aug": True, "real": True,
                      "overrides": {**REAL_BASE, **V11_VISIBLE_OVERRIDES}},
    "k_sharp": {"profile": "v11", "aug": True, "real": True, "overrides": {**REAL_BASE, **V11_SHARP_OVERRIDES}},
    "l_scale": {"profile": "v11", "aug": True, "real": True, "overrides": {**REAL_BASE, **V11_SCALE_OVERRIDES}},
    "m_pos": {"profile": "v11", "aug": True, "real": True, "overrides": {**REAL_BASE, **V11_POSITION_OVERRIDES}},
    "n_vis15": {"profile": "v11", "aug": True, "real": True, "overrides": {**REAL_BASE, **V11_VIS15_OVERRIDES}},
    "o_tight": {"profile": "v11", "aug": True, "real": True, "overrides": {**REAL_BASE, **V11_TIGHT_OVERRIDES}},
    "p_pos_tight": {"profile": "v11", "aug": True, "real": True,
                    "overrides": {**REAL_BASE, **V11_POSITION_OVERRIDES, "tight_bbox": True}},
    # p_pos_tight with the cleaned crop libraries (same file names)
    "s_clean": {"profile": "v11", "aug": True, "real": True,
                "overrides": {**REAL_BASE, **V11_POSITION_OVERRIDES, "tight_bbox": True},
                "objects": CLEAN_OBJECTS, "extra_objects": [CLEAN_POLY]},
    # p_pos_tight with libraries that only drop anomalous crops (original masks)
    "s_drop": {"profile": "v11", "aug": True, "real": True,
               "overrides": {**REAL_BASE, **V11_POSITION_OVERRIDES, "tight_bbox": True},
               "objects": DROP_OBJECTS, "extra_objects": [DROP_POLY]},
    # p_pos_tight without sharpness/grain matching, with tight_bbox
    "q_base": {"profile": "v11", "aug": True, "real": True, "overrides": {**REAL_BASE, **V11_Q_BASE_OVERRIDES}},
    "q_lab": {"profile": "v11", "aug": True, "real": True, "overrides": {**REAL_BASE, **V11_Q_LAB_OVERRIDES}},
    "q_lab_same": {"profile": "v11", "aug": True, "real": True, "overrides": {**REAL_BASE, **V11_Q_LAB_SAME_OVERRIDES}},
    "q_lab_local": {"profile": "v11", "aug": True, "real": True,
                    "overrides": {**REAL_BASE, **V11_Q_LAB_LOCAL_OVERRIDES}},
    # base q_lab_same + edge matting / ring-to-ring medium transfer
    "r_matte": {"profile": "v11", "aug": True, "real": True, "overrides": {**REAL_BASE, **V11_R_MATTE_OVERRIDES}},
    "r_matte_fg": {"profile": "v11", "aug": True, "real": True, "overrides": {**REAL_BASE, **V11_R_MATTE_FG_OVERRIDES}},
    "r_medium": {"profile": "v11", "aug": True, "real": True, "overrides": {**REAL_BASE, **V11_R_MEDIUM_OVERRIDES}},
    "r_medium_harm": {"profile": "v11", "aug": True, "real": True,
                      "overrides": {**REAL_BASE, **V11_R_MEDIUM_HARM_OVERRIDES}},
}
# State read by the augmentation patch; set before each variant.
FLAGS = {"aug": True}


def neutral_augment(aug: dict) -> dict:
    """No-op augmentation with the same shape as sample_augment (no rotation, unit colour gains, no blur)."""
    out = dict(aug)
    out.update(hflip=False, rotation=0.0, perspective=None, brightness=1.0, contrast=1.0, saturation=1.0,
               blur_sigma=None)
    return out


def install_patches():
    """Patch sample_augment to neutralise it after consuming the same rng (only for c_v10_noaug)."""
    from app import object_planner
    real_sample = object_planner.sample_augment

    def sample(rng, profile):
        aug = real_sample(rng, profile)
        return aug if FLAGS["aug"] else neutral_augment(aug)

    object_planner.sample_augment = sample


def build_profile(spec: dict, name: str, extra_dirs=(), save_masks: bool = False):
    """Variant profile: v10 or v11 with the polygon crop libraries and the variant overrides."""
    from app.synth_profile import V10_PROFILE, V11_PROFILE
    base = V11_PROFILE if spec["profile"] == "v11" else V10_PROFILE
    update = {"name": name, "extra_objects_dirs": list(extra_dirs), **spec["overrides"]}
    if save_masks:                      # only adds `mask_rle` to the COCO; images and boxes are unchanged
        update["save_masks"] = True
    return base.model_copy(update=update)


def make_composer(depth_url, seg_url):
    from app.caches import LRUCache
    from app.composer import ImageComposer
    from app.validators import PhysicsValidator, QualityValidator
    comp = ImageComposer(use_gpu=False, depth_service_url=depth_url)
    comp.quality_validator, comp.physics_validator = QualityValidator(use_gpu=False), PhysicsValidator()
    for attr in ("_depth_cache", "_scene_cache"):   # the same backgrounds repeat across variants
        setattr(comp, attr, LRUCache(maxsize=1024))
    if seg_url:
        comp.segmentation_service_url = seg_url
    return comp


def check_depth_service(comp):
    """Fail early without a depth service (otherwise a vertical-gradient fallback would be used)."""
    import httpx
    try:
        ok = httpx.get(f"{comp.depth_service_url}/health", timeout=10).status_code == 200
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        raise SystemExit(f"No depth service at {comp.depth_service_url}: refusing to use the fallback gradient.")


def is_done(vdir: Path, n: int) -> bool:
    f = vdir / "synthetic_dataset.json"
    return f.exists() and len(json.load(open(f, encoding="utf8")).get("images", [])) >= n


async def run_variant(comp, name, spec, req, a):
    from app.models.schemas import EffectsConfig, EffectType
    vdir = Path(a.out) / name
    if is_done(vdir, a.n):
        print(f"[{name}] already complete, skipping", flush=True)
        return
    FLAGS.update(aug=spec["aug"])
    backgrounds = a.real_dir if spec["real"] else (a.backgrounds or req["backgrounds_dir"])
    t0 = time.time()
    res = await comp.compose_batch(
        backgrounds_dir=backgrounds, objects_dir=spec.get("objects") or a.objects or req["objects_dir"],
        output_dir=str(vdir), num_images=a.n, targets_per_class=req["targets_per_class"],
        max_objects_per_image=req["max_objects_per_image"], effects=[EffectType(e) for e in req["effects"]],
        effects_config=EffectsConfig(**req["effects_config"]), depth_aware=req["depth_aware"],
        validate_quality=req["validate_quality"], validate_physics=req["validate_physics"],
        reject_invalid=req["reject_invalid"], semantic_placement=req["semantic_placement"],
        synth_profile=build_profile(spec, name, spec.get("extra_objects", a.extra_objects), a.save_masks),
        seed=a.seed, dataset_info={"name": f"synthetic {name}"})
    info = {"variant": name, **spec, "backgrounds": backgrounds, "n": a.n, "seed": a.seed, "generated": res.images_generated,
            "rejected": res.images_rejected, "seconds": round(time.time() - t0, 1)}
    json.dump(info, open(vdir / "variant.json", "w", encoding="utf8"), indent=1)
    print(f"[{name}] {info}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=300, help="images per variant")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default="/app/output/synthetic_batch")
    ap.add_argument("--request", default=os.environ.get("JOB_REQUEST", "/tmp/batch_request.json"),
                    help="job request JSON (effects, targets); configs/batch_request.json copied into the container")
    ap.add_argument("--backgrounds")
    ap.add_argument("--real-dir", default="/app/datasets/real_train",
                    help="real training images with annotations.json (copy-paste)")
    ap.add_argument("--objects")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--save-masks", action="store_true",
                    help="store each pasted object's mask in the COCO (`mask_rle`)")
    ap.add_argument("--extra-objects", nargs="*", default=["/app/datasets/crops_poly"],
                    help="extra crop libraries for all variants (empty: none)")
    ap.add_argument("--depth-url", default=os.environ.get("DEPTH_SERVICE_URL", "http://depth:8001"))
    ap.add_argument("--seg-url", default=os.environ.get("SEGMENTATION_SERVICE_URL"))
    a = ap.parse_args()
    req = json.load(open(a.request, encoding="utf8"))
    install_patches()
    comp = make_composer(a.depth_url, a.seg_url)
    check_depth_service(comp)
    for name in a.variants:
        asyncio.run(run_variant(comp, name, VARIANTS[name], req, a))


if __name__ == "__main__":
    main()
