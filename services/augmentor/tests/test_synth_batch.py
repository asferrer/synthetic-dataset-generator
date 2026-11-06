"""Batches with a synthesis profile: reuse cap, traceability, real compose and legacy parity."""

import asyncio
import json
import os
import random

import cv2
import numpy as np
import pytest

from app import composer as composer_mod
from app.composer import ComposeResult, ImageComposer
from app.models.schemas import EffectType, ObjectPlacement
from app.synth_context import ImagePlan
from app.synth_profile import SynthProfile
from tests.synth_helpers import fake_annotation, make_stats_file, rgba_crop

CLASSES = ["Bottle", "Can", "Tire"]


@pytest.fixture
def dirs(tmp_path):
    bg_dir = tmp_path / "bg"
    bg_dir.mkdir()
    for i in range(4):
        (bg_dir / f"bg_{i}.jpg").write_bytes(b"x")
    obj_dir = tmp_path / "obj"
    for cls in CLASSES:
        (obj_dir / cls).mkdir(parents=True)
        for i in range(2):
            (obj_dir / cls / f"crop_{i}.png").write_bytes(b"x")
    return str(bg_dir), str(obj_dir), str(tmp_path / "out")


@pytest.fixture
def composer(monkeypatch):
    comp = ImageComposer(use_gpu=False)
    seen = {"plans": [], "objects": []}

    async def fake_depth(bg_image, cache_key=None):
        return np.zeros((4, 4), dtype=np.float32), "fallback_gradient"

    async def fake_compose(objects, **kwargs):
        seen["plans"].append(kwargs.get("plan"))
        seen["objects"].append(list(objects))
        anns = [fake_annotation(o.class_name, o.image_path) for o in objects]
        return ComposeResult(annotations=anns, objects_placed=len(anns))

    async def fake_validate(*args, **kwargs):
        return True, None

    monkeypatch.setattr(comp, "_get_depth_map", fake_depth)
    monkeypatch.setattr(comp, "compose", fake_compose)
    monkeypatch.setattr(comp, "_validate_composed", fake_validate)
    monkeypatch.setattr(composer_mod.cv2, "imread", lambda p: np.zeros((8, 8, 3), np.uint8))
    comp.seen = seen
    return comp


def profile(tmp_path, **kw):
    kw.setdefault("stats_path", make_stats_file(tmp_path))
    return SynthProfile(name="t", trace=True, per_image_seed=True, background_round_robin=True, **kw)


def run_parallel(comp, dirs, targets, prof=None, seed=3, num_images=1000):
    bg_dir, obj_dir, out_dir = dirs
    return asyncio.run(comp.compose_batch_parallel(
        num_images=num_images, backgrounds_dir=bg_dir, objects_dir=obj_dir, output_dir=out_dir,
        concurrent_limit=2, targets_per_class=targets, max_objects_per_image=3,
        synth_profile=prof, seed=seed))


def run_sequential(comp, dirs, targets, prof=None, seed=3, num_images=1000):
    bg_dir, obj_dir, out_dir = dirs
    return asyncio.run(comp.compose_batch(
        num_images=num_images, backgrounds_dir=bg_dir, objects_dir=obj_dir, output_dir=out_dir,
        targets_per_class=targets, max_objects_per_image=3, synth_profile=prof, seed=seed))


def load_coco(result):
    with open(result.output_coco_path) as f:
        return json.load(f)


@pytest.mark.parametrize("runner", [run_parallel, run_sequential])
def test_exhausted_class_ends_without_error_and_is_reported(composer, dirs, tmp_path, runner):
    prof = profile(tmp_path, max_crop_reuse=3)
    result = runner(composer, dirs, {c: 100 for c in CLASSES}, prof)
    assert all(result.synthetic_counts[c] == 6 for c in CLASSES)  # 2 recortes x 3 usos
    assert set(result.exhausted_classes) == set(CLASSES)
    info = result.exhausted_classes["Can"]
    assert info["achieved"] == 6 and info["target"] == 100 and info["crops"] == 2
    coco = load_coco(result)
    gen = coco["info"]["synth_generator"]
    assert gen["profile"] == "t" and gen["batch_seed"] == 3
    assert set(gen["exhausted_classes"]) == set(CLASSES)


@pytest.mark.parametrize("runner", [run_parallel, run_sequential])
def test_no_exhaustion_when_cap_is_not_reached(composer, dirs, tmp_path, runner):
    prof = profile(tmp_path, max_crop_reuse=50)
    result = runner(composer, dirs, {c: 5 for c in CLASSES}, prof)
    assert result.exhausted_classes == {}
    assert all(result.synthetic_counts[c] >= 5 for c in CLASSES)


@pytest.mark.parametrize("runner", [run_parallel, run_sequential])
def test_cap_is_never_exceeded_per_crop(composer, dirs, tmp_path, runner):
    prof = profile(tmp_path, max_crop_reuse=4)
    result = runner(composer, dirs, {c: 100 for c in CLASSES}, prof)
    coco = load_coco(result)
    uses = {}
    for ann in coco["annotations"]:
        uses[ann["source_object"]] = uses.get(ann["source_object"], 0) + 1
    assert uses and max(uses.values()) <= 4


@pytest.mark.parametrize("runner", [run_parallel, run_sequential])
def test_trace_in_coco_images_background_seed_and_balance(composer, dirs, tmp_path, runner):
    prof = profile(tmp_path)
    result = runner(composer, dirs, {c: 40 for c in CLASSES}, prof, seed=11)
    coco = load_coco(result)
    names = [im["background"] for im in coco["images"]]
    assert all(n.startswith("bg_") for n in names)
    counts = [names.count(f"bg_{i}.jpg") for i in range(4)]
    assert max(counts) - min(counts) <= 1  # uniform spread of backgrounds
    seeds = [im["seed"] for im in coco["images"]]
    assert len(set(seeds)) == len(seeds)
    assert all("objects_requested" in im for im in coco["images"])


def test_sequential_batch_is_reproducible_with_same_seed(composer, dirs, tmp_path):
    prof = profile(tmp_path, faithful_count=True, max_objects_cap=3,
                   paste_augment=True, faithful_scale=True)
    run_sequential(composer, dirs, {c: 12 for c in CLASSES}, prof, seed=5)
    first = [[o.model_dump() for o in objs] for objs in composer.seen["objects"]]
    composer.seen["objects"].clear()
    import shutil
    shutil.rmtree(dirs[2])
    run_sequential(composer, dirs, {c: 12 for c in CLASSES}, prof, seed=5)
    second = [[o.model_dump() for o in objs] for objs in composer.seen["objects"]]
    assert first == second and len(first) > 0


def test_annotation_trace_fields_written_to_coco(tmp_path):
    comp = ImageComposer(use_gpu=False)
    ann = fake_annotation("Can", "/o/Can/a.png")
    ann.scale, ann.target_rel_size, ann.augmentations = 1.5, 0.21, {"hflip": True}
    entry = comp._coco_entry("synthetic_00000.jpg", np.zeros((10, 20, 3), np.uint8),
                             ComposeResult(annotations=[ann], image_size=(8, 6)), None, None, None)
    assert (entry["width"], entry["height"]) == (8, 6) and "synth" not in entry
    out = tmp_path / "c.json"
    comp._generate_coco_json([entry], ["Can"], str(out))
    coco = json.loads(out.read_text())
    a = coco["annotations"][0]
    assert a["scale"] == 1.5 and a["target_rel_size"] == 0.21 and a["augmentations"] == {"hflip": True}
    assert "synth_generator" not in coco["info"]


# ---------------------------------------------------------------- parity with the legacy generator
def test_legacy_selection_matches_original_algorithm():
    objs = {c: [{"path": f"/o/{c}/{i}.png"} for i in range(5)] for c in CLASSES}
    available = list(CLASSES)
    random.seed(99)
    got, reserved, plan = ImageComposer._select_objects(
        None, random, 0, "bg", None, available, {}, objs, 3)
    random.seed(99)
    n = random.randint(1, min(3, len(available)))
    expected = [(c, random.choice(objs[c])["path"]) for c in random.sample(available, k=n)]
    assert [(o.class_name, o.image_path) for o in got] == expected
    assert reserved == [] and plan is None
    assert all(o.scale is None and o.rotation is None and o.augment is None and o.size_quantile is None
               for o in got)


@pytest.mark.parametrize("runner", [run_parallel, run_sequential])
def test_legacy_batch_output_has_no_synth_fields(composer, dirs, runner):
    result = runner(composer, dirs, {c: 5 for c in CLASSES}, None)
    coco = load_coco(result)
    assert "synth_generator" not in coco["info"]
    assert result.exhausted_classes == {} and result.synth_profile is None
    for im in coco["images"]:
        assert not ({"background", "seed", "objects_requested", "synth"} & set(im))
    for ann in coco["annotations"]:
        assert not ({"scale", "target_rel_size", "augmentations"} & set(ann))
    assert all(p is None for p in composer.seen["plans"])


def test_inactive_profile_resolves_to_none_so_legacy_path_runs():
    from app.synth_profile import get_profile
    assert get_profile("legacy") is None and get_profile(None, {}) is None


# ---------------------------------------------------------------- real compose (no services)
@pytest.fixture
def real_composer(monkeypatch, tmp_path):
    comp = ImageComposer(use_gpu=False)
    monkeypatch.setattr(ImageComposer, "segmentation_client", property(lambda self: None))
    bg = np.random.RandomState(1).randint(60, 200, (150, 200, 3)).astype(np.uint8)
    bg_path = tmp_path / "bg.jpg"
    cv2.imwrite(str(bg_path), bg)
    crop_path = tmp_path / "crop.png"
    cv2.imwrite(str(crop_path), rgba_crop(80, 50))
    return comp, str(bg_path), str(crop_path), tmp_path


def make_plan(tmp_path, **kw):
    prof = SynthProfile(name="t", faithful_scale=True, paste_augment=True, trace=True,
                        stats_path=make_stats_file(tmp_path), max_crop_upscale=3.0,
                        min_mask_area_ratio=0.0002, max_mask_area_ratio=0.95, **kw)
    from app.real_stats import load_stats
    return ImagePlan(profile=prof, seed=1, background="bg.jpg", stats=load_stats(prof.stats_path))


def test_compose_with_plan_resizes_traces_and_compresses(real_composer):
    comp, bg, crop, tmp = real_composer
    plan = make_plan(tmp)
    plan.output_size, plan.jpeg_quality = (96, 96), 75
    obj = ObjectPlacement(image_path=crop, class_name="Bottle", size_quantile=0.1,
                          augment={"hflip": True, "brightness": 1.1, "contrast": 1.0, "saturation": 1.0,
                                   "blur_sigma": None, "perspective": None, "rotation": 5.0},
                          rotation=5.0)
    out = str(tmp / "out" / "img.jpg")
    res = asyncio.run(comp.compose(bg, [obj], effects=[EffectType.COLOR_CORRECTION], output_path=out,
                                   plan=plan))
    img = cv2.imread(out)
    assert img.shape[:2] == (96, 96) and res.image_size == (96, 96)
    assert res.objects_placed == 1
    ann = res.annotations[0]
    assert 0 <= ann.x and ann.x + ann.width <= 96 and 0 <= ann.y and ann.y + ann.height <= 96
    assert ann.scale > 0 and ann.target_rel_size is not None
    assert ann.augmentations["hflip"] is True and ann.source_object == crop
    assert res.image_trace["jpeg_quality"] == 75
    ann_json = json.load(open(out.replace(".jpg", "_annotations.json")))[0]
    assert ann_json["target_rel_size"] == ann.target_rel_size
    # the box size approaches the sampled target (physical limits aside)
    rel = (ann.width * ann.height) / (96 * 96)
    assert rel == pytest.approx(ann.target_rel_size ** 2, rel=0.6)


def test_compose_legacy_writes_no_trace_fields(real_composer):
    comp, bg, crop, tmp = real_composer
    obj = ObjectPlacement(image_path=crop, class_name="Bottle", scale=0.8, rotation=0)
    out = str(tmp / "out" / "legacy.jpg")
    res = asyncio.run(comp.compose(bg, [obj], effects=[EffectType.COLOR_CORRECTION], output_path=out))
    assert res.image_size is None and res.image_trace == {}
    ann_json = json.load(open(out.replace(".jpg", "_annotations.json")))[0]
    assert not ({"scale", "target_rel_size", "augmentations"} & set(ann_json))
    assert cv2.imread(out).shape[:2] == (150, 200)


def test_compose_haze_matches_target_contrast(real_composer):
    comp, bg, crop, tmp = real_composer
    plan = make_plan(tmp, haze_from_stats=True, haze_max_intensity=0.2)
    plan.target_l_std = 1.0  # very low contrast: the haze reaches its cap
    obj = ObjectPlacement(image_path=crop, class_name="Bottle", scale=0.8, rotation=0)
    res = asyncio.run(comp.compose(bg, [obj], effects=[EffectType.UNDERWATER],
                                   output_path=str(tmp / "o" / "h.jpg"), plan=plan))
    assert res.image_trace["underwater_intensity"] == pytest.approx(0.2)
    plan.target_l_std = 1000.0  # the background already has less contrast: no haze
    res = asyncio.run(comp.compose(bg, [obj], effects=[EffectType.UNDERWATER],
                                   output_path=str(tmp / "o" / "h2.jpg"), plan=plan))
    assert res.image_trace["underwater_intensity"] == 0.0


def test_compose_tiny_crop_is_skipped_not_crashing(real_composer):
    comp, bg, crop, tmp = real_composer
    tiny = tmp / "tiny.png"
    cv2.imwrite(str(tiny), rgba_crop(6, 4))
    plan = make_plan(tmp, min_object_px=16)
    obj = ObjectPlacement(image_path=str(tiny), class_name="Bottle", size_quantile=0.5)
    res = asyncio.run(comp.compose(bg, [obj], effects=[EffectType.COLOR_CORRECTION],
                                   output_path=str(tmp / "o" / "t.jpg"), plan=plan))
    assert res.objects_placed in (0, 1)  # no exception; the upscaling limit may discard it
