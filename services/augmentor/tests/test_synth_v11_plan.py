"""v11 variants l-n: crop scale within a range, position faithful to the data with an edge margin, and visibility by percentile."""

import importlib.util
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import crop_scale, placement, position_stats, visibility
from app.composer import ImageComposer
from app.object_planner import plan_objects
from app.synth_profile import (V10_PROFILE, V11_POSITION_OVERRIDES, V11_PROFILE, V11_SCALE_OVERRIDES,
                               V11_SHARP_OVERRIDES, V11_VIS15_OVERRIDES, V11_VISIBILITY_P5, V11_VISIBLE_OVERRIDES,
                               SynthProfile, get_profile)
from tests.synth_helpers import make_context


# ------------------------------------------------------------------ profiles: everything off by default
NEW_FLAGS = {"single_resample": False, "match_sharpness": False, "crop_scale_range": None,
             "position_from_stats": False, "visibility_percentile": None, "border_touch_fraction": None}


def test_new_features_are_off_in_legacy_v10_and_v11_profiles():
    for prof in (SynthProfile(), V10_PROFILE, V11_PROFILE):
        assert {k: getattr(prof, k) for k in NEW_FLAGS} == NEW_FLAGS
    assert get_profile(None) is None and get_profile("v10") == V10_PROFILE


def test_variant_overrides_are_chained_and_valid():
    chain = [V11_VISIBLE_OVERRIDES, V11_SHARP_OVERRIDES, V11_SCALE_OVERRIDES, V11_POSITION_OVERRIDES,
             V11_VIS15_OVERRIDES]
    for smaller, bigger in zip(chain, chain[1:]):
        assert smaller.items() <= bigger.items()
        assert len(bigger) > len(smaller)
    prof = get_profile("v11", {"copy_paste_real": True, **V11_VIS15_OVERRIDES})
    assert prof.single_resample and prof.match_sharpness and prof.position_from_stats
    assert prof.crop_scale_range == (0.6, 1.5) and prof.visibility_percentile == 15 and prof.visibility_filter


def test_profile_validators():
    with pytest.raises(ValueError):
        SynthProfile(crop_scale_range=(1.5, 0.6))
    with pytest.raises(ValueError):
        SynthProfile(crop_scale_range=(0.0, 1.0))
    with pytest.raises(ValueError):
        SynthProfile(sharpness_quantile_range=(0.8, 0.2))
    assert SynthProfile(crop_scale_range=(0.6, 1.5)).crop_scale_range == (0.6, 1.5)


def test_pilot_variants_k_to_n_are_chained_on_j(monkeypatch):
    monkeypatch.setenv("AUGMENTOR_APP_ROOT", str(Path(__file__).resolve().parents[1]))
    path = Path(__file__).resolve().parents[3] / "tools" / "run_variants.py"
    if not path.exists():  # the service image does not contain tools/
        pytest.skip("tools/run_variants.py is not available")
    spec = importlib.util.spec_from_file_location("run_variants_under_test", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    names = ["j_v11_visible", "k_sharp", "l_scale", "m_pos", "n_vis15"]
    assert all(n in pilot.VARIANTS for n in names)
    ov = [pilot.VARIANTS[n]["overrides"] for n in names]
    for smaller, bigger in zip(ov, ov[1:]):
        assert smaller.items() <= bigger.items() and len(bigger) > len(smaller)
    assert pilot.VARIANTS["j_v11_visible"]["overrides"] == {"copy_paste_real": True, "harmonize": True,
                                                            **V11_VISIBLE_OVERRIDES}
    for n in names:
        assert pilot.build_profile(pilot.VARIANTS[n], n).copy_paste_real
    assert pilot.build_profile(pilot.VARIANTS["n_vis15"], "n").visibility_percentile == 15
    assert pilot.build_profile(pilot.VARIANTS["j_v11_visible"], "j").visibility_percentile is None


# ------------------------------------------------------------------ visibility by percentile
def test_percentile_table_matches_the_p5_constant_and_is_increasing():
    table = visibility.load_percentile_table()
    for cls, p5 in V11_VISIBILITY_P5.items():
        assert table[cls]["p5"] == pytest.approx(p5, abs=0.01)
        assert table[cls]["p5"] < table[cls]["p15"] < table[cls]["p25"] < table[cls]["p50"]


def test_threshold_for_uses_the_percentile_and_takes_precedence():
    p5 = SynthProfile(visibility_filter=True, visibility_percentile=5, visibility_min_contrast=99.0)
    p15 = SynthProfile(visibility_filter=True, visibility_percentile=15)
    p25 = SynthProfile(visibility_filter=True, visibility_percentile=25)
    for cls in V11_VISIBILITY_P5:
        assert visibility.threshold_for(p5, cls) == pytest.approx(V11_VISIBILITY_P5[cls], abs=0.01)
        assert visibility.threshold_for(p5, cls) < visibility.threshold_for(p15, cls) < visibility.threshold_for(p25, cls)
    assert visibility.threshold_for(p15, "Unknown") is None
    assert visibility.threshold_for(SynthProfile(visibility_percentile=15), "Bottle") is None   # filter off


def test_unknown_percentile_raises_with_the_available_ones():
    prof = SynthProfile(visibility_filter=True, visibility_percentile=33)
    with pytest.raises(ValueError, match="p25"):
        visibility.threshold_for(prof, "Bottle")


def test_percentile_none_keeps_the_dict_and_float_thresholds():
    assert visibility.threshold_for(SynthProfile(visibility_filter=True, visibility_min_contrast=12.5), "Can") == 12.5
    d = SynthProfile(visibility_filter=True, visibility_min_contrast={"Can": 7.0})
    assert visibility.threshold_for(d, "Can") == 7.0 and visibility.threshold_for(d, "Tire") is None


# ------------------------------------------------------------------ crop scale
def _write_png(path, side):
    cv2.imwrite(str(path), np.full((side, side, 4), 255, np.uint8))


@pytest.fixture
def crops(tmp_path):
    out = {}
    for side in (30, 60, 120, 240, 480):
        for i in range(3):
            p = tmp_path / f"c{side}_{i}.png"
            _write_png(p, side)
            out[str(p)] = side
    return out


def test_crop_sizes_read_the_png_header_and_cache(crops, tmp_path):
    sizes = crop_scale.CropSizes()
    path = next(iter(crops))
    assert sizes.side(path) == pytest.approx(crops[path])
    assert np.isnan(sizes.side(str(tmp_path / "no_existe.png")))
    assert len(sizes.sides(list(crops))) == len(crops)


def test_needed_scale_and_log_distance():
    scales = crop_scale.needed_scales(np.array([50.0, 100.0]), 100.0)
    assert scales[0] == pytest.approx(2.0 / crop_scale.ROT_BBOX_FACTOR)
    d = crop_scale.log_distance(np.array([0.3, 0.6, 1.0, 1.5, 3.0]), (0.6, 1.5))
    assert d[0] > 0 and d[1] == d[2] == d[3] == 0 and d[4] == pytest.approx(np.log(2.0))


def test_pick_candidates_keeps_those_in_range(crops):
    sizes = crop_scale.CropSizes()
    kept = crop_scale.pick_candidates(list(crops), sizes, 120.0, (0.6, 1.5))
    sides = {crops[p] for p in kept}
    assert sides == {120}                       # 120 px gives scale ~ 0.9; 60 px ~ 1.8; 240 px ~ 0.45
    for p in kept:
        assert 0.6 <= 120.0 / (crops[p] * crop_scale.ROT_BBOX_FACTOR) <= 1.5


def test_pick_candidates_takes_the_closest_when_none_fits(crops):
    sizes = crop_scale.CropSizes()
    assert {crops[p] for p in crop_scale.pick_candidates(list(crops), sizes, 2000.0, (0.6, 1.5))} == {480}   # huge target: the largest crop
    assert {crops[p] for p in crop_scale.pick_candidates(list(crops), sizes, 2.0, (0.6, 1.5))} == {30}       # minimal target: the smallest


def test_pick_candidates_without_readable_sizes_returns_everything(tmp_path):
    ghosts = [str(tmp_path / f"x{i}.png") for i in range(3)]
    assert crop_scale.pick_candidates(ghosts, crop_scale.CropSizes(), 100.0, (0.6, 1.5)) == ghosts


def _scale_ctx(tmp_path, crops, **kw):
    objects = {"Bottle": [{"path": p} for p in crops]}
    return make_context(tmp_path, objects=objects, faithful_scale=True, **kw)


def _scales(ctx, objs, bg_wh):
    out = []
    for o in objs:
        side = ctx.crop_sizes.side(o.image_path)
        target = ctx.stats.quantile("Bottle", o.size_quantile) * (bg_wh[0] * bg_wh[1]) ** 0.5
        out.append(target / (side * crop_scale.ROT_BBOX_FACTOR))
    return np.array(out)


def test_plan_objects_picks_crops_whose_scale_is_in_range(tmp_path, crops):
    bg = (640, 640)
    base = _scale_ctx(tmp_path, crops, max_crop_reuse=1000)
    ranged = _scale_ctx(tmp_path, crops, max_crop_reuse=1000, crop_scale_range=(0.6, 1.5))
    objects = {"Bottle": [{"path": p} for p in crops]}
    chosen = {}
    for name, ctx in (("base", base), ("ranged", ranged)):
        rng = random.Random(3)
        objs = []
        for _ in range(60):
            o, reserved, _ = plan_objects(ctx, rng, ["Bottle"], {"Bottle": 10}, objects, 1, bg_wh=bg)
            objs += o
        chosen[name] = _scales(ctx, objs, bg)
    inside = lambda s: float(((s >= 0.6) & (s <= 1.5)).mean())   # noqa: E731
    assert inside(chosen["ranged"]) > inside(chosen["base"]) + 0.2 and inside(chosen["ranged"]) > 0.6


def test_plan_objects_keeps_the_faithful_quantile_and_the_reuse_cap(tmp_path, crops):
    ctx = _scale_ctx(tmp_path, crops, max_crop_reuse=2, crop_scale_range=(0.6, 1.5))
    rng = random.Random(1)
    objects = {"Bottle": [{"path": p} for p in crops]}
    seen = []
    for _ in range(40):
        objs, reserved, _ = plan_objects(ctx, rng, ["Bottle"], {"Bottle": 10}, objects, 1, bg_wh=(640, 640))
        ctx.usage.settle(reserved, [o.image_path for o in objs])
        seen += objs
    assert all(0.0 <= o.size_quantile <= 1.0 for o in seen)
    assert max(ctx.usage.used.values()) <= 2 and len(seen) == 15 * 2    # 15 recortes x 2 usos; despues se agotan


def test_plan_objects_without_range_or_bg_size_is_unchanged(tmp_path, crops):
    objects = {"Bottle": [{"path": p} for p in crops]}
    plans = []
    for kw, bg in (({}, None), ({}, (640, 640)), ({"crop_scale_range": (0.6, 1.5)}, None)):
        ctx = _scale_ctx(tmp_path, crops, max_crop_reuse=5, **kw)
        rng = random.Random(9)
        plans.append([(o.image_path, o.size_quantile) for _ in range(10)
                      for o in plan_objects(ctx, rng, ["Bottle"], {"Bottle": 10}, objects, 1, bg_wh=bg)[0]])
    assert plans[0] == plans[1] == plans[2]


def test_scale_filter_is_none_without_range_or_class_stats(tmp_path, crops):
    assert _scale_ctx(tmp_path, crops).scale_filter("Bottle", 0.5, (640, 640)) is None
    ctx = _scale_ctx(tmp_path, crops, crop_scale_range=(0.6, 1.5))
    assert ctx.scale_filter("Tire", 0.5, (640, 640)) is None          # no statistics for the class in the test JSON
    assert ctx.scale_filter("Bottle", 0.5, (640, 640)) is not None


# ------------------------------------------------------------------ position with an edge margin
def test_position_json_is_train_only_and_consistent():
    with open(position_stats.DEFAULT_POSITION_PATH, encoding="utf8") as f:
        data = json.load(f)
    assert "train" in data["source"].lower() and "val" not in data["source"].lower()
    assert set(data["classes"]) >= {"ALL", "Bottle", "Can", "Fishing_Net", "Glove", "Mask", "Metal_Debris",
                                    "Plastic_Debris", "Tire"}
    for cls, v in data["classes"].items():
        assert np.asarray(v["center_hist"]).sum() == pytest.approx(1.0, abs=1e-3), cls
        assert sum(v["touch_sides"].values()) == pytest.approx(1.0, abs=2e-3), cls
        assert 0.0 <= v["touch_fraction"] <= 1.0
    assert 0.2 < data["classes"]["ALL"]["touch_fraction"] < 0.4


def _stats():
    return position_stats.load_position_stats()


def _prof(**kw):
    return SynthProfile(name="t", position_from_stats=True, **kw)


def _place(profile, cls="Bottle", box=(0, 0, 80, 60), bg=(640, 480), n=1500, resample=False, seed=0):
    rng = random.Random(seed)
    stats = _stats()
    return [position_stats.place_box(stats, profile, cls, box, bg, rng, resample_center=resample) for _ in range(n)]


def _touch_rate(profile, box=(100, 100, 80, 60), bg=(640, 480), **kw):
    pos = _place(profile, box=box, bg=bg, **kw)
    boxes = [(x, y, x + box[2], y + box[3]) for x, y in pos]
    assert all(0 <= b[0] and b[2] <= bg[0] and 0 <= b[1] and b[3] <= bg[1] for b in boxes)   # never outside
    return float(np.mean([position_stats.touches_border(b, bg) for b in boxes]))


def test_touch_fraction_follows_the_real_class_value():
    stats = _stats()
    for cls in ("Bottle", "Glove", "Fishing_Net"):
        rate = _touch_rate(_prof(), cls=cls, n=2500)
        assert rate == pytest.approx(stats.touch_fraction(cls), abs=0.04), cls


def test_touch_fraction_override_zero_and_one():
    assert _touch_rate(_prof(border_touch_fraction=0.0)) == 0.0
    assert _touch_rate(_prof(border_touch_fraction=1.0)) == 1.0
    assert _touch_rate(_prof(border_touch_fraction=0.28), n=3000) == pytest.approx(0.28, abs=0.04)


def test_boxes_that_do_not_touch_keep_the_margin():
    bg, box = (640, 480), (0, 0, 80, 60)
    margin = position_stats.margin_px(_prof(position_margin=0.1), bg)
    assert margin == round(0.1 * 480)
    assert position_stats.margin_px(_prof(position_margin=0.0), bg) == position_stats.MIN_MARGIN_PX == 36
    assert position_stats.margin_px(_prof(), bg) == max(36, round(0.03 * 480))
    for x, y in _place(_prof(border_touch_fraction=0.0, position_margin=0.1), box=box, bg=bg, n=400):
        assert margin <= x <= bg[0] - 80 - margin and margin <= y <= bg[1] - 60 - margin


def test_box_too_big_for_the_margins_is_centered():
    bg, box = (200, 100), (0, 0, 195, 96)
    for x, y in _place(_prof(border_touch_fraction=0.0), box=box, bg=bg, n=20):
        assert (x, y) == ((200 - 195) // 2, (100 - 96) // 2)


def test_touching_boxes_are_flush_with_a_side():
    bg, box = (640, 480), (200, 200, 80, 60)
    for x, y in _place(_prof(border_touch_fraction=1.0), box=box, bg=bg, n=200):
        assert x == 0 or x == 640 - 80 or y == 0 or y == 480 - 60


def test_resampled_centers_follow_the_real_histogram():
    stats = _stats()
    rng = random.Random(4)
    pts = np.array([stats.sample_center("Glove", rng) for _ in range(4000)])
    h = stats.hist("Glove")
    assert 0 <= pts.min() and pts.max() <= 1
    emp, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=h.shape[0], range=[[0, 1], [0, 1]])
    assert np.abs(emp / emp.sum() - h).max() < 0.03


def test_density_is_positive_everywhere_and_higher_where_the_real_mass_is():
    stats = _stats()
    hist = stats.hist("Bottle")
    i, j = np.unravel_index(hist.argmax(), hist.shape)
    cx, cy = (j + 0.5) / hist.shape[1], (i + 0.5) / hist.shape[0]
    assert stats.density("Bottle", np.array([cx]), np.array([cy]))[0] > stats.density("Bottle", np.array([0.0]),
                                                                                         np.array([0.0]))[0]
    assert (stats.density("Bottle", np.array([0.0, 1.0]), np.array([1.0, 0.0])) > 0).all()


def test_unknown_class_falls_back_to_all():
    stats = _stats()
    assert stats.touch_fraction("NoExiste") == stats.touch_fraction("ALL")


def test_sample_position_center_weight_pulls_anchors_towards_the_dense_side():
    region = np.ones((200, 400), np.uint8)           # all open water
    left = lambda cx, cy: np.where(np.asarray(cx) < 100, 50.0, 0.5)   # noqa: E731
    xs_w, xs_u = [], []
    for seed in range(150):
        pw = placement.sample_position(region, "Glove", (20, 20), random.Random(seed), center_weight=left)
        pu = placement.sample_position(region, "Glove", (20, 20), random.Random(seed))
        xs_w.append(pw[0] + 10)
        xs_u.append(pu[0] + 10)
    assert np.mean(np.array(xs_w) < 100) > 0.8 and np.mean(np.array(xs_u) < 100) < 0.35


def test_sample_position_without_weight_is_unchanged():
    region = np.ones((100, 100), np.uint8)
    a = placement.sample_position(region, "Glove", (10, 10), random.Random(5))
    b = placement.sample_position(region, "Glove", (10, 10), random.Random(5), center_weight=None)
    assert a == b


def test_touches_border_definition():
    assert position_stats.touches_border((0, 50, 40, 90), (100, 100))
    assert position_stats.touches_border((10, 50, 97, 90), (100, 100))
    assert not position_stats.touches_border((10, 10, 90, 90), (100, 100), tol=3)


# ------------------------------------------------------------------ _stats_position (alpha box, not the canvas)
COMP = ImageComposer(use_gpu=False)

def _canvas(w=120, h=120, ellipse=(40, 30)):
    """RGBA canvas with a centred ellipse: the alpha box lies inside the canvas."""
    img = np.zeros((h, w, 4), np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), ellipse, 0, 0, 360, (200, 100, 50, 255), -1)
    return img, cv2.boundingRect((img[:, :, 3] > 10).astype(np.uint8))


def test_stats_position_places_the_alpha_box_not_the_canvas():
    canvas, (bx, by, bw, bh) = _canvas()
    bg = (640, 480)
    prof = _prof(border_touch_fraction=1.0)
    for seed in range(60):
        random_state = random.getstate()
        random.seed(seed)
        x, y = COMP._stats_position(prof, "Bottle", canvas, (200, 200), bg, semantic=True)
        random.setstate(random_state)
        box = (x + bx, y + by, x + bx + bw, y + by + bh)
        assert position_stats.touches_border(box, bg, tol=0) and box[0] >= 0 and box[2] <= bg[0]


def test_stats_position_keeps_the_margin_for_non_touching_boxes():
    canvas, (bx, by, bw, bh) = _canvas()
    bg = (640, 480)
    prof = _prof(border_touch_fraction=0.0, position_margin=0.1)
    m = position_stats.margin_px(prof, bg)
    random.seed(1)
    for start in ((0, 0), (520, 360), (300, 200), (-30, 400)):
        x, y = COMP._stats_position(prof, "Bottle", canvas, start, bg, semantic=True)
        assert x + bx >= m and y + by >= m and x + bx + bw <= bg[0] - m and y + by + bh <= bg[1] - m


def test_stats_position_resamples_the_center_without_semantic_placement():
    canvas, (bx, by, bw, bh) = _canvas()
    bg = (640, 480)
    prof = _prof(border_touch_fraction=0.0)
    random.seed(2)
    xs = [COMP._stats_position(prof, "Glove", canvas, (10, 10), bg, semantic=False)[0] for _ in range(80)]
    assert np.std(xs) > 50       # with semantic=False the centre comes from the histogram, not from the input position


def test_center_weight_is_none_unless_the_profile_asks():
    assert ImageComposer._center_weight(None, "Bottle", (640, 480)) is None
    assert ImageComposer._center_weight(SynthProfile(), "Bottle", (640, 480)) is None
    fn = ImageComposer._center_weight(_prof(), "Bottle", (640, 480))
    assert (np.asarray(fn(np.array([320.0, 10.0]), np.array([240.0, 10.0]))) > 0).all()


# ------------------------------------------------------------------ per-object edge decision and collision avoidance
def test_decide_touch_follows_the_class_fraction_and_the_override():
    stats, rng = _stats(), random.Random(0)
    rate = np.mean([position_stats.decide_touch(stats, _prof(), "Fishing_Net", rng) for _ in range(3000)])
    assert rate == pytest.approx(stats.touch_fraction("Fishing_Net"), abs=0.04)
    assert not any(position_stats.decide_touch(stats, _prof(border_touch_fraction=0.0), "Fishing_Net", rng)
                   for _ in range(200))
    assert all(position_stats.decide_touch(stats, _prof(border_touch_fraction=1.0), "Can", rng) for _ in range(200))


def test_place_box_honours_a_decision_already_taken():
    stats, rng, bg = _stats(), random.Random(1), (640, 480)
    prof = _prof(border_touch_fraction=1.0)             # the profile would say "touch", but touch=False wins
    for _ in range(100):
        x, y = position_stats.place_box(stats, prof, "Bottle", (200, 200, 80, 60), bg, rng, touch=False)
        assert not position_stats.touches_border((x, y, x + 80, y + 60), bg)


def test_search_position_avoids_busy_boxes_and_gives_up_with_the_last_candidate():
    stats, rng, bg = _stats(), random.Random(2), (640, 480)
    busy = (200, 120, 440, 360)                                         # busy central area
    avoid = lambda b: b[0] < busy[2] and b[2] > busy[0] and b[1] < busy[3] and b[3] > busy[1]   # noqa: E731
    prof = _prof(border_touch_fraction=0.0)
    free = [position_stats.search_position(stats, prof, "Glove", (0, 0), (60, 60), bg, rng, touch=False, avoid=avoid)
            for _ in range(200)]
    assert np.mean([not avoid((x, y, x + 60, y + 60)) for x, y in free]) > 0.9
    always = [position_stats.search_position(stats, prof, "Glove", (0, 0), (60, 60), bg, rng, touch=False,
                                             avoid=lambda b: True) for _ in range(5)]
    assert len(always) == 5 and all(isinstance(v, int) for p in always for v in p)


def test_fixed_touch_decision_keeps_the_border_fraction_under_rejection():
    """Collision rejection discards the centre (where the real objects are): with the decision fixed per object and the
    free-slot search, surviving objects keep touching the edge with the real fraction, no more."""
    stats, rng, bg = _stats(), random.Random(3), (640, 480)
    busy = (140, 100, 500, 380)
    avoid = lambda b: b[0] < busy[2] and b[2] > busy[0] and b[1] < busy[3] and b[3] > busy[1]   # noqa: E731
    prof, touches, n_ok = _prof(), 0, 0
    for _ in range(1500):
        touch = position_stats.decide_touch(stats, prof, "Bottle", rng)
        x, y = position_stats.search_position(stats, prof, "Bottle", (0, 0), (60, 60), bg, rng, touch=touch, avoid=avoid)
        if not avoid((x, y, x + 60, y + 60)):
            n_ok += 1
            touches += position_stats.touches_border((x, y, x + 60, y + 60), bg)
    # real Bottle: 15.2 %; deciding on every attempt (and repeating until a slot is found) would give ~ 48 %
    assert n_ok > 1400 and touches / n_ok == pytest.approx(stats.touch_fraction("Bottle"), abs=0.04)


def test_position_avoider_checks_real_boxes_and_existing_objects():
    prof = SynthProfile(name="t", copy_paste_real=True, position_from_stats=True, real_overlap_max=0.1)

    class _Plan:
        real_boxes = [(100, 100, 200, 200)]
    comp = ImageComposer(use_gpu=False)
    avoid = comp._position_avoider(prof, _Plan(), [(300, 300, 400, 400)])
    assert avoid((110, 110, 190, 190)) and avoid((310, 310, 390, 390)) and not avoid((220, 20, 260, 60))
    no_real = comp._position_avoider(SynthProfile(name="t", position_from_stats=True), _Plan(), [])
    assert not no_real((110, 110, 190, 190))


def test_place_object_keeps_non_touching_boxes_off_the_border_even_with_shadows(tmp_path):
    """Shadows widen the annotated box: with the default margin no 'non-touching' box ends up touching."""
    import asyncio
    from app.models.schemas import EffectsConfig, EffectType, ObjectPlacement
    from app.synth_context import ImagePlan
    img = np.zeros((90, 120, 4), np.uint8)
    cv2.ellipse(img, (60, 45), (55, 40), 0, 0, 360, (40, 90, 200, 255), -1)
    path = tmp_path / "o.png"
    cv2.imwrite(str(path), img)
    prof = SynthProfile(name="t", position_from_stats=True, border_touch_fraction=0.0, copy_paste_real=True)
    bg = np.full((300, 400, 3), 120, np.uint8)
    comp = ImageComposer(use_gpu=False, try_count=4)
    random.seed(0)
    for _ in range(25):
        res = asyncio.run(comp._place_object(
            composite=bg.copy(), obj_placement=ObjectPlacement(image_path=str(path), class_name="Bottle", scale=1.0,
                                                               rotation=0.0),
            depth_map=None, lighting_info=None, effects=[EffectType.SHADOWS], effects_config=EffectsConfig(),
            placement_mask=np.zeros(bg.shape[:2], np.uint8), existing_bboxes=[],
            plan=ImagePlan(profile=prof, stats=None)))
        assert res is not None and not position_stats.touches_border(res[1], (400, 300))
