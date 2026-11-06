"""Unit tests of the v10 synthesis profile (no GPU or services)."""

import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from app import crop_augment, faithful_scale, output_format
from app.models.schemas import AnnotationBox, ObjectPlacement, TRACE_FIELDS
from app.object_planner import plan_objects
from app.real_stats import DEFAULT_STATS_PATH, RealStats, load_stats
from app.synth_context import CropUsage
from app.synth_profile import SynthProfile, V10_PROFILE, derive_image_seed, get_profile
from tests.synth_helpers import (fake_annotation, make_context, make_objects, make_stats_file,
                                 rgba_crop, seeded_rng)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------- perfiles
def test_legacy_profile_is_inactive():
    assert get_profile(None) is None
    assert get_profile("legacy") is None
    assert SynthProfile().active is False


def test_v10_profile_enables_everything():
    p = get_profile("v10")
    assert p.faithful_scale and p.faithful_count and p.paste_augment
    assert p.max_crop_reuse == 15 and p.max_objects_cap == 20
    assert p.trace and p.per_image_seed and p.output_resolution and p.jpeg_recompress
    assert p.background_round_robin and p.haze_from_stats and any(p.seabed_fraction.values())
    assert V10_PROFILE.effects_overrides["use_binary_alpha"] is False


def test_overrides_and_unknown_profile():
    p = get_profile("v10", {"max_crop_reuse": 5})
    assert p.max_crop_reuse == 5 and p.name == "v10"
    assert get_profile(None, {"max_crop_reuse": 3}).max_crop_reuse == 3
    with pytest.raises(ValueError):
        get_profile("v99")
    with pytest.raises(ValueError):
        get_profile("v10", {"max_objects_cap": 0})


def test_image_seed_is_deterministic_and_varies():
    assert derive_image_seed(5, 3) == derive_image_seed(5, 3)
    assert derive_image_seed(5, 3) != derive_image_seed(5, 4)
    assert derive_image_seed(5, 3) != derive_image_seed(6, 3)


# ---------------------------------------------------------------- estadisticas reales
def test_bundled_stats_are_train_only_and_complete():
    stats = load_stats(None)
    assert "training split" in stats.source
    for cls in ["Bottle", "Can", "Fishing_Net", "Glove", "Mask", "Metal_Debris", "Plastic_Debris", "Tire"]:
        assert stats.has_class(cls)
    assert stats.image_sizes[(416, 416)] > 9000
    assert os.path.exists(DEFAULT_STATS_PATH)


def test_stats_script_refuses_val_and_test(tmp_path):
    script = os.path.join(REPO, "scripts", "build_real_stats.py")
    if not os.path.exists(script):
        pytest.skip("scripts/ is not in the container image")
    for name in ("val.json", "test.json"):
        res = subprocess.run([sys.executable, script, "--train", str(tmp_path / name),
                              "--out", str(tmp_path / "o.json")], capture_output=True, text=True)
        assert res.returncode != 0 and "TRAIN" in (res.stderr + res.stdout)


def test_count_distribution_truncated(tmp_path):
    stats = RealStats(json.load(open(make_stats_file(tmp_path))))
    values, weights = stats.count_distribution(5)
    assert values == [1, 2] and weights == [70, 20]
    rng = seeded_rng(0)
    draws = [stats.sample_count(rng, 5) for _ in range(2000)]
    assert set(draws) == {1, 2}
    assert 0.74 < draws.count(1) / len(draws) < 0.82  # 70/90 conditioned


def test_truncated_size_stays_in_range_and_follows_distribution(tmp_path):
    stats = RealStats(json.load(open(make_stats_file(tmp_path))))
    sizes = [stats.truncated_size("Bottle", u / 100, 0.3, 0.5)[0] for u in range(100)]
    assert min(sizes) >= 0.3 and max(sizes) <= 0.5
    assert abs(np.mean(sizes) - 0.4) < 0.02  # conditioned uniform, not piled up at the limit
    assert stats.truncated_size("Can", 0.5, 0.5, 0.9) == (0.5, True)  # class out of range
    assert stats.truncated_size("Bottle", 0.5, 0.0, 2.0)[1] is False


# ---------------------------------------------------------------- faithful scale
def _scale_plan(tmp_path, q, **kw):
    stats = RealStats(json.load(open(make_stats_file(tmp_path))))
    return faithful_scale.FaithfulScale(stats, "Bottle", q, SynthProfile(faithful_scale=True, **kw))


def test_scale_hits_target_when_reachable(tmp_path):
    plan = _scale_plan(tmp_path, 0.5)
    scale = plan.resolve((300, 300), (300, 300), (640, 640), floor_px=16)
    assert plan.target == pytest.approx(0.5, abs=1e-6)
    assert 300 * scale / 640 == pytest.approx(0.5, abs=1e-3)


def test_scale_respects_upscale_and_fit_limits(tmp_path):
    big = _scale_plan(tmp_path, 1.0, max_crop_upscale=2.0)
    scale = big.resolve((100, 100), (100, 100), (640, 640), floor_px=16)
    assert scale <= 2.0 + 1e-9 and big.truncated is True
    assert 100 * scale / 640 <= 0.3125 + 1e-6  # 2x of a 100 px crop in 640
    fit = _scale_plan(tmp_path, 1.0, max_crop_upscale=50.0)
    scale = fit.resolve((300, 200), (300, 200), (640, 480), floor_px=16)
    assert 300 * scale <= 0.98 * 640 + 1e-6 and 200 * scale <= 0.98 * 480 + 1e-6


def test_scale_respects_min_pixels(tmp_path):
    plan = _scale_plan(tmp_path, 0.0, min_object_px=32)
    scale = plan.resolve((400, 400), (400, 400), (640, 640), floor_px=16)
    assert 400 * scale >= 32 - 1e-6


def test_scale_degenerate_crop_returns_max_scale(tmp_path):
    plan = _scale_plan(tmp_path, 0.5, max_crop_upscale=1.0)
    scale = plan.resolve((4, 4), (4, 4), (640, 640), floor_px=16)
    assert scale == 1.0 and plan.truncated


def test_alpha_bbox_ignores_transparent_margin():
    img = np.zeros((50, 80, 4), np.uint8)
    img[10:30, 20:60, 3] = 255
    assert faithful_scale.alpha_bbox_wh(img) == (40, 20)
    assert faithful_scale.alpha_bbox_wh(img[:, :, :3]) == (80, 50)


def test_make_faithful_only_when_enabled_and_class_known(tmp_path):
    stats = RealStats(json.load(open(make_stats_file(tmp_path))))
    on = SynthProfile(faithful_scale=True)
    assert faithful_scale.make_faithful(None, stats, "Bottle", 0.5) is None
    assert faithful_scale.make_faithful(SynthProfile(), stats, "Bottle", 0.5) is None
    assert faithful_scale.make_faithful(on, stats, "Tire", 0.5) is None
    assert faithful_scale.make_faithful(on, stats, "bottle", 0.5) is not None


# ---------------------------------------------------------------- aumentos
def test_augment_params_within_ranges():
    profile = SynthProfile(paste_augment=True, aug_rotation_deg=10, aug_perspective=0.05,
                           aug_brightness=0.1, aug_blur_prob=1.0, aug_blur_sigma_max=0.8)
    rng = seeded_rng(3)
    for _ in range(200):
        spec = crop_augment.sample_augment(rng, profile)
        assert -10 <= spec["rotation"] <= 10
        assert all(-0.05 <= v <= 0.05 for v in spec["perspective"])
        assert 0.9 <= spec["brightness"] <= 1.1
        assert 0.3 <= spec["blur_sigma"] <= 0.8
        json.dumps(spec)  # serializable


def test_augment_identity_spec_keeps_pixels():
    img = rgba_crop()
    spec = {"hflip": False, "perspective": None, "brightness": 1.0, "contrast": 1.0,
            "saturation": 1.0, "blur_sigma": None}
    assert np.array_equal(crop_augment.apply_augment(img, spec), img)


def test_augment_flip_keeps_alpha_and_mirrors():
    img = rgba_crop()
    img[:, :5, :3] = 255
    out = crop_augment.apply_augment(img, {"hflip": True})
    assert out.shape == img.shape and np.array_equal(out[:, ::-1, 3], img[:, :, 3])
    assert (out[:, -5:, :3] > 200).all()


def test_augment_perspective_and_color_change_image():
    img = rgba_crop()
    spec = {"hflip": False, "perspective": [0.05, 0, 0, 0.05, -0.05, 0, 0, -0.05],
            "brightness": 1.1, "contrast": 1.1, "saturation": 0.9, "blur_sigma": 0.6}
    out = crop_augment.apply_augment(img, spec)
    assert out.shape[2] == 4 and out.dtype == np.uint8
    assert out[:, :, 3].max() == 255 and out[:, :, 3].min() == 0
    assert out.shape[:2] != img.shape[:2] or not np.array_equal(out, img)


def test_augment_works_without_alpha():
    img = rgba_crop()[:, :, :3].copy()
    out = crop_augment.apply_augment(img, {"hflip": True, "brightness": 1.2, "blur_sigma": 0.5})
    assert out.shape == img.shape


# ---------------------------------------------------------------- tope de reutilizacion
def test_usage_cap_and_least_used_first():
    usage = CropUsage(make_objects(["A"], crops=3), max_reuse=2)
    rng = seeded_rng(0)
    picks = [usage.acquire("A", rng) for _ in range(6)]
    assert all(p is not None for p in picks)
    assert sorted(usage.used.values()) == [2, 2, 2]  # even spread
    assert usage.acquire("A", rng) is None and not usage.has_free("A")


def test_usage_first_three_picks_are_distinct():
    usage = CropUsage(make_objects(["A"], crops=3), max_reuse=5)
    rng = seeded_rng(1)
    assert len({usage.acquire("A", rng) for _ in range(3)}) == 3


def test_usage_settle_returns_unused_reservations():
    usage = CropUsage(make_objects(["A"], crops=2), max_reuse=1)
    rng = seeded_rng(0)
    a, b = usage.acquire("A", rng), usage.acquire("A", rng)
    assert usage.pending == 2 and not usage.has_free("A")
    usage.settle([a, b], [a])  # b no se uso
    assert usage.pending == 0 and usage.used[a] == 1 and usage.used[b] == 0
    assert usage.has_free("A")


def test_usage_without_cap_is_unlimited_and_preload():
    usage = CropUsage(make_objects(["A"], crops=1), max_reuse=None)
    assert all(usage.acquire("A", seeded_rng()) for _ in range(50))
    capped = CropUsage(make_objects(["A"], crops=1), max_reuse=2)
    capped.preload_from_coco({"annotations": [{"source_object": "/objs/A/crop_0.png"}] * 2})
    assert not capped.has_free("A")


def test_exhausted_report(tmp_path):
    ctx = make_context(tmp_path, max_crop_reuse=1, objects=make_objects(crops=2))
    rng = seeded_rng()
    for cls in ("Bottle", "Bottle"):
        ctx.usage.acquire(cls, rng)
    ctx.usage.pending = 0
    report = ctx.exhausted_report({"Bottle": 2, "Can": 0}, {"Bottle": 50, "Can": 10})
    assert set(report) == {"Bottle"}
    assert report["Bottle"]["achieved"] == 2 and report["Bottle"]["target"] == 50
    assert report["Bottle"]["crops"] == 2 and report["Bottle"]["max_crop_reuse"] == 1


# ---------------------------------------------------------------- planificador
def test_plan_counts_follow_truncated_distribution(tmp_path):
    ctx = make_context(tmp_path, faithful_count=True, max_objects_cap=2)
    objs = make_objects()
    sizes = []
    rng = seeded_rng(5)
    for _ in range(300):
        planned, _, requested = plan_objects(ctx, rng, ["Bottle", "Can"], {"Bottle": 99, "Can": 99}, objs, 5)
        sizes.append(len(planned))
        assert requested in (1, 2)
    assert set(sizes) == {1, 2}


def test_plan_allows_repeated_classes_and_caps_by_remaining(tmp_path):
    stats = make_stats_file(tmp_path, counts={"6": 1})
    ctx = make_context(tmp_path, faithful_count=True, max_objects_cap=20, stats_path=stats)
    planned, _, requested = plan_objects(ctx, seeded_rng(0), ["Bottle", "Can"],
                                         {"Bottle": 4, "Can": 1}, make_objects(), 5)
    names = [o.class_name for o in planned]
    assert requested == 6 and len(names) == 5  # only 4 + 1 fit up to the target
    assert names.count("Bottle") == 4 and names.count("Can") == 1


def test_plan_respects_cap_20_and_max_objects(tmp_path):
    stats = make_stats_file(tmp_path, counts={"47": 1, "3": 1})
    ctx = make_context(tmp_path, faithful_count=True, max_objects_cap=20, stats_path=stats)
    rng = seeded_rng(2)
    for _ in range(50):
        planned, _, requested = plan_objects(ctx, rng, ["Bottle"], {"Bottle": 999}, make_objects(), 5)
        assert requested == 3 and len(planned) == 3


def test_plan_legacy_count_when_not_faithful(tmp_path):
    ctx = make_context(tmp_path, paste_augment=True)
    planned, _, requested = plan_objects(ctx, seeded_rng(0), ["Bottle", "Can"], {"Bottle": 9, "Can": 9},
                                         make_objects(), 2)
    classes = [o.class_name for o in planned]
    assert 1 <= len(classes) <= 2 and len(set(classes)) == len(classes)  # without replacement


def test_plan_is_deterministic_per_seed_and_sorted_big_first(tmp_path):
    ctx = make_context(tmp_path, faithful_count=True, faithful_scale=True, paste_augment=True,
                       max_objects_cap=20, stats_path=make_stats_file(tmp_path, counts={"6": 1}))
    args = (["Bottle", "Can"], {"Bottle": 50, "Can": 50}, make_objects(), 5)
    a = plan_objects(ctx, seeded_rng(11), *args)[0]
    b = plan_objects(ctx, seeded_rng(11), *args)[0]
    assert [o.model_dump() for o in a] == [o.model_dump() for o in b]
    hints = [ctx.stats.quantile(o.class_name, o.size_quantile) for o in a]
    assert hints == sorted(hints, reverse=True)
    assert all(o.augment and o.rotation == o.augment["rotation"] for o in a)


def test_plan_reserves_crops_and_skips_exhausted_class(tmp_path):
    ctx = make_context(tmp_path, max_crop_reuse=1, objects=make_objects(crops=1))
    rng = seeded_rng(0)
    planned, reserved, _ = plan_objects(ctx, rng, ["Bottle", "Can"], {"Bottle": 5, "Can": 5}, make_objects(crops=1), 2)
    assert len(reserved) == len(planned) >= 1
    ctx.usage.settle(reserved, [])  # devuelve
    ctx.usage.acquire("Bottle", rng), ctx.usage.acquire("Can", rng)
    again, _, requested = plan_objects(ctx, rng, ["Bottle", "Can"], {"Bottle": 5, "Can": 5}, make_objects(crops=1), 2)
    assert again == [] and requested == 0


def test_seabed_fraction_forces_sink_only_for_floating_classes(tmp_path):
    ctx = make_context(tmp_path, seabed_fraction=1.0, objects=make_objects(["Glove", "Tire"]))
    planned, _, _ = plan_objects(ctx, seeded_rng(0), ["Glove", "Tire"], {"Glove": 9, "Tire": 9},
                                 make_objects(["Glove", "Tire"]), 2)
    by_class = {o.class_name: o.behavior_override for o in planned}
    assert by_class.get("Glove", "sink") == "sink"   # glove is 'float' by default
    assert by_class.get("Tire") is None              # tire is already 'sink'
    none = make_context(tmp_path, seabed_fraction=0.0, objects=make_objects(["Glove"]))
    p2, _, _ = plan_objects(none, seeded_rng(0), ["Glove"], {"Glove": 9}, make_objects(["Glove"]), 1)
    assert p2[0].behavior_override is None


# ---------------------------------------------------------------- fondos, semillas, plan
def test_background_round_robin_is_balanced(tmp_path):
    ctx = make_context(tmp_path, background_round_robin=True)
    n = len(ctx.bg_files)
    picks = [ctx.pick_background(a, seeded_rng()) for a in range(n * 7 + 2)]
    counts = {bg: picks.count(bg) for bg in ctx.bg_files}
    assert max(counts.values()) - min(counts.values()) <= 1
    assert picks[:n] != sorted(picks[:n]) or picks[n:2 * n] != picks[:n]  # shuffled per epoch
    assert ctx.pick_background(3, seeded_rng()) == ctx.pick_background(3, seeded_rng())


def test_background_random_without_round_robin(tmp_path):
    ctx = make_context(tmp_path, bgs=["/bgs/a.jpg", "/bgs/b.jpg"])
    assert ctx.pick_background(0, random.Random(1)) in ctx.bg_files
    assert ctx.background_name("/bgs/sub/a.jpg") == "sub/a.jpg"


def test_make_rng_is_global_without_per_image_seed(tmp_path):
    assert make_context(tmp_path).make_rng(1) is random
    ctx = make_context(tmp_path, per_image_seed=True)
    assert ctx.make_rng(1).random() == ctx.make_rng(1).random()
    assert ctx.make_rng(1).random() != ctx.make_rng(2).random()


# ---------------------------------------------------------------- resolution, jpeg, haze
def test_output_size_comes_from_real_sizes_and_is_bounded(tmp_path):
    ctx = make_context(tmp_path, output_resolution=True, max_output_side=800, max_bg_upscale=2.0)
    rng = seeded_rng(0)
    seen = set()
    for _ in range(300):
        w, h = output_format.sample_output_size(rng, ctx.stats, (640, 480), ctx.profile)
        seen.add((w, h))
        assert max(w, h) <= 800 and w <= 2 * 640 and h <= 2 * 480
    assert (416, 416) in seen and len(seen) > 1
    ratios = {round(w / h, 2) for w, h in seen}
    assert 1.0 in ratios and any(abs(r - 16 / 9) < 0.02 for r in ratios)  # the 16:9 aspect is preserved


def test_output_size_none_without_stats():
    empty = RealStats({"classes": {}})
    assert output_format.sample_output_size(seeded_rng(), empty, (640, 480), SynthProfile()) is None


def test_jpeg_quality_sampling(tmp_path):
    stats = RealStats(json.load(open(make_stats_file(tmp_path, jpeg=[70.0] * 101))))
    assert output_format.sample_jpeg_quality(seeded_rng(), stats, SynthProfile()) == 70
    fixed = SynthProfile(jpeg_quality_fixed=88)
    assert output_format.sample_jpeg_quality(seeded_rng(), stats, fixed) == 88
    nostats = RealStats(json.load(open(make_stats_file(tmp_path))))
    assert output_format.sample_jpeg_quality(seeded_rng(), nostats, SynthProfile()) is None
    assert 60 <= (load_stats(None).jpeg_quality(0.5) or 0) <= 100  # estimated on the real training split


def test_resize_to_updates_background_and_depth():
    bg = np.random.randint(0, 255, (120, 160, 3), np.uint8)
    depth = np.random.rand(120, 160).astype(np.float32)
    out_bg, out_depth = output_format.resize_to(bg, depth, (80, 80))
    assert out_bg.shape == (80, 80, 3) and out_depth.shape == (80, 80)
    same_bg, same_depth = output_format.resize_to(bg, depth, (160, 120))
    assert same_bg is bg and same_depth is depth


def test_haze_matches_real_contrast_and_is_capped():
    img = np.random.RandomState(0).randint(0, 255, (100, 100, 3)).astype(np.uint8)
    cur = output_format.luminance_std(img)
    t = output_format.matched_haze_intensity(img, cur * 0.9, 0.5, 0.3)
    assert t == pytest.approx(0.1, abs=1e-3)
    assert output_format.matched_haze_intensity(img, cur * 0.5, 0.12, 0.3) == pytest.approx(0.12)
    assert output_format.matched_haze_intensity(img, cur * 2, 0.12, 0.3) == 0.0  # already has less contrast
    assert output_format.matched_haze_intensity(img, None, 0.12, 0.3) == pytest.approx(0.12)


# ---------------------------------------------------------------- traceability: schema
def test_legacy_annotation_has_no_trace_values():
    ann = AnnotationBox(x=1, y=2, width=3, height=4, class_name="Can")
    assert all(getattr(ann, k) is None for k in TRACE_FIELDS)
    ann2 = AnnotationBox(x=1, y=2, width=3, height=4, class_name="Can", scale=1.2,
                         target_rel_size=0.3, augmentations={"hflip": True})
    assert ann2.model_dump()["augmentations"] == {"hflip": True}


def test_object_placement_new_fields_default_to_none():
    obj = ObjectPlacement(image_path="a.png", class_name="Can")
    assert obj.size_quantile is None and obj.augment is None and obj.behavior_override is None
    assert fake_annotation("Can").source_object is None


def test_sample_position_behavior_override_rests_on_seabed():
    from app import placement
    region = np.ones((100, 100), np.uint8)  # all open water (1)
    region[80:, :] = 2                       # seabed (2) at the bottom
    rng = random.Random(0)
    assert placement.get_behavior("Glove") == "float"
    for _ in range(30):
        x, y = placement.sample_position(region, "Glove", (10, 10), rng, behavior="sink")
        assert y + 10 >= 80  # the object's base rests on the seabed
    ys = [placement.sample_position(region, "Glove", (10, 10), random.Random(i))[1] for i in range(30)]
    assert min(ys) < 70      # without override it floats in the water column
