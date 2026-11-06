"""v11 variants q: crop-to-background pairing by the colour of the source image (`background_match`) and local ring."""

import importlib.util
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import background_match as bm
from app import position_stats
from app.object_planner import plan_objects
from app.synth_profile import (V10_PROFILE, V11_PROFILE, V11_Q_BASE_OVERRIDES, V11_Q_LAB_LOCAL_OVERRIDES,
                               V11_Q_LAB_OVERRIDES, V11_Q_LAB_SAME_OVERRIDES, SynthProfile, get_profile)
from tests.synth_helpers import make_context

MATCH_FLAGS = {"background_match": None, "background_match_exclude_same_group": False,
               "background_match_allow_same_image": False, "background_match_hist_weight": 0.0,
               "local_patch_match": False}


# ------------------------------------------------------------------ tabla sintetica
def make_table(tmp_path, n_img=12, per_img=2, classes=("Bottle", "Can")):
    """Image i: mean LAB (10*i, 0, 0), group i // 3. Crop `<class>/<i>_<j>.png` from image i
    (its ring in the source: gray with L = 25 i). The neighbourhood is of CROPS: each image contributes `per_img` per class."""
    lab = np.zeros((n_img, 6), np.float32)
    lab[:, 0] = 10.0 * np.arange(n_img)
    keys, img, ring = [], [], []
    for c in classes:
        for i in range(n_img):
            for j in range(per_img):
                keys.append(f"{c}/{i}_{j}.png")
                img.append(i)
                ring.append([25.0 * i, 128, 128, 0, 0, 0])
    path = tmp_path / "match.npz"
    np.savez_compressed(
        path, files=np.array([f"bg_{i}.jpg" for i in range(n_img)]), image_id=np.arange(n_img, dtype=np.int32),
        group=(np.arange(n_img) // 3).astype(np.int32), lab=lab, hist=np.zeros((n_img, 64), np.uint8),
        crop_keys=np.array(keys), crop_img=np.array(img, np.int32), crop_ring=np.array(ring, np.float32),
        ring_frac=np.float32(0.25))
    return str(path)


def objects_for(n_img=12, per_img=2, classes=("Bottle", "Can")):
    return {c: [{"path": f"/objs/{c}/{i}_{j}.png"} for i in range(n_img) for j in range(per_img)] for c in classes}


def match_ctx(tmp_path, bgs=None, **kw):
    kw.setdefault("background_match", "lab_knn")
    kw.setdefault("background_match_path", make_table(tmp_path))
    bgs = bgs or [f"/bgs/bg_{i}.jpg" for i in range(12)]
    return make_context(tmp_path, objects=objects_for(), bgs=bgs, position_from_stats=False, **kw)


def src_of(path: str) -> int:
    return int(Path(path).stem.split("_")[0])


# ------------------------------------------------------------------ profiles: everything off by default
def test_flags_are_off_in_legacy_v10_and_v11_profiles():
    for prof in (SynthProfile(), V10_PROFILE, V11_PROFILE):
        assert {k: getattr(prof, k) for k in MATCH_FLAGS} == MATCH_FLAGS
    assert get_profile(None) is None and get_profile("v10") == V10_PROFILE


def test_variant_overrides_q():
    base = V11_Q_BASE_OVERRIDES
    assert base["tight_bbox"] and base["position_from_stats"] and base["single_resample"]
    assert base["match_sharpness"] is False and base["sharpness_grain"] is False
    assert V11_Q_LAB_OVERRIDES["background_match"] == "lab_knn" and V11_Q_LAB_OVERRIDES["background_match_exclude_same_group"]
    assert V11_Q_LAB_SAME_OVERRIDES["background_match"] == "lab_knn"
    assert "background_match_exclude_same_group" not in V11_Q_LAB_SAME_OVERRIDES
    assert V11_Q_LAB_LOCAL_OVERRIDES == {**V11_Q_LAB_SAME_OVERRIDES, "local_patch_match": True}
    for ov in (base, V11_Q_LAB_OVERRIDES, V11_Q_LAB_SAME_OVERRIDES, V11_Q_LAB_LOCAL_OVERRIDES):
        assert get_profile("v11", {"copy_paste_real": True, "harmonize": True, **ov}) is not None


def test_pilot_q_variants_are_based_on_p_pos_tight_without_sharpness():
    path = Path(__file__).resolve().parents[3] / "tools" / "run_variants.py"
    if not path.exists():  # the service image does not contain tools/
        pytest.skip("tools/run_variants.py is not available")
    spec = importlib.util.spec_from_file_location("run_variants_q_under_test", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    names = ("q_base", "q_lab", "q_lab_same", "q_lab_local")
    assert all(n in pilot.VARIANTS and pilot.VARIANTS[n]["real"] for n in names)
    profs = {n: pilot.build_profile(pilot.VARIANTS[n], n, save_masks=True) for n in names}
    p_ref = pilot.build_profile(pilot.VARIANTS["p_pos_tight"], "p")
    for n, prof in profs.items():
        assert prof.tight_bbox and not prof.match_sharpness and not prof.sharpness_grain and prof.save_masks
        assert prof.position_from_stats and prof.copy_paste_real and prof.harmonize
        assert prof.crop_scale_range == p_ref.crop_scale_range and prof.visibility_filter == p_ref.visibility_filter
    assert profs["q_base"].background_match is None and not profs["q_base"].local_patch_match
    assert profs["q_lab"].background_match == "lab_knn" and profs["q_lab"].background_match_exclude_same_group
    assert profs["q_lab_same"].background_match == "lab_knn" and not profs["q_lab_same"].background_match_exclude_same_group
    assert profs["q_lab_local"].local_patch_match and not profs["q_lab_local"].background_match_exclude_same_group
    assert not profs["q_lab_local"].background_match_allow_same_image


def test_profile_validators():
    with pytest.raises(ValueError):
        SynthProfile(background_match="rgb_knn")
    with pytest.raises(ValueError):
        SynthProfile(local_patch_match=True)                      # necesita position_from_stats
    with pytest.raises(ValueError):
        SynthProfile(background_match_k=0)
    assert SynthProfile(local_patch_match=True, position_from_stats=True).local_patch_match


# ------------------------------------------------------------------ descriptores
def test_lab_descriptor_and_histogram():
    img = np.full((90, 120, 3), (200, 100, 50), np.uint8)
    d = bm.lab_descriptor(img)
    assert d.shape == (6,) and np.allclose(d[3:], 0, atol=1e-4) and d[0] > 0
    h = bm.lab_histogram(img)
    assert h.shape == (64,) and h.sum() == pytest.approx(1.0) and (h > 0).sum() == 1
    assert bm.descriptor_distance(d, d) == 0
    other = bm.lab_descriptor(np.full((90, 120, 3), (20, 20, 20), np.uint8))
    assert bm.descriptor_distance(d, other) > 20


def test_ring_meter_matches_a_brute_force_ring():
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, (120, 160, 3)).astype(np.uint8)
    box = (50, 40, 90, 80)
    got = bm.RingMeter(img).stats(box, 0.25)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    r = 0.25 * 40
    mask = np.zeros(lab.shape[:2], bool)
    mask[int(np.floor(40 - r)):int(np.ceil(80 + r)), int(np.floor(50 - r)):int(np.ceil(90 + r))] = True
    mask[40:80, 50:90] = False
    px = lab[mask]
    assert np.allclose(got[:3], px.mean(axis=0), atol=1e-3) and np.allclose(got[3:], px.std(axis=0), atol=1e-3)


def test_ring_meter_clips_to_the_image_and_reports_unmeasurable_rings():
    img = np.full((60, 80, 3), 90, np.uint8)
    m = bm.RingMeter(img)
    assert m.stats((0, 0, 20, 20), 0.5) is not None                 # ring clipped by the border
    assert m.stats((0, 0, 80, 60), 0.5) is None                     # the box covers the whole image: no ring


def test_ring_meter_downscales_large_images_consistently():
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[:, :960] = (180, 180, 180)
    meter = bm.RingMeter(img)
    assert max(meter.w, meter.h) == bm.RING_SIDE
    left, right = meter.stats((100, 400, 300, 600), 0.25), meter.stats((1500, 400, 1700, 600), 0.25)
    assert left[0] > right[0] + 50 and left[3] < 1 and right[3] < 1


def test_crop_key_distinguishes_libraries():
    assert bm.crop_key("/app/datasets/Extracted_objects_train/Bottle/12.png") == "Bottle/12.png"
    assert bm.crop_key("x\\crops_poly\\Bottle\\poly_12.png") == "Bottle/poly_12.png"


def test_knn_weights():
    d = np.array([10.0, 12.0, 20.0])
    assert bm.knn_weights(d, 0).tolist() == [1.0, 0.0, 0.0]
    w = bm.knn_weights(d, 1.0)
    assert w[0] == 1.0 and w[0] > w[1] > w[2] and w[2] == pytest.approx(np.exp(-1))
    assert np.allclose(bm.knn_weights(d, 1e6), 1.0, atol=1e-4)
    assert bm.knn_weights(np.array([5.0, 5.0]), 1.0).tolist() == [1.0, 1.0]


# ------------------------------------------------------------------ eleccion de recortes
def make_matcher(tmp_path, **kw):
    prof = SynthProfile(name="t", background_match="lab_knn", background_match_path=make_table(tmp_path), **kw)
    return bm.Matcher(prof, "/bgs")


def fake_bg(m, idx):
    m._bg["/bgs/x.jpg"] = (idx, m.table.lab[idx], m.table.hist[idx]) if idx >= 0 else (-1, np.array([55, 0, 0, 0, 0, 0], np.float32), np.zeros(64))
    return "/bgs/x.jpg"


def all_paths():
    return [o["path"] for o in objects_for()["Bottle"]]


def test_pick_is_among_the_k_nearest_sources_and_never_the_same_image(tmp_path):
    m = make_matcher(tmp_path, background_match_k=8, background_match_temperature=1.0)
    bg = fake_bg(m, 5)
    rng = random.Random(0)
    picks = [src_of(m.pick(all_paths(), bg, rng)[0]) for _ in range(300)]
    assert 5 not in picks
    assert set(picks) <= {3, 4, 6, 7}                       # LAB L = 10 i: the 8 crops closest to 5, excluding 5
    assert set(picks) == {3, 4, 6, 7}                       # y se muestrean todos
    assert picks.count(4) + picks.count(6) > picks.count(3) + picks.count(7)    # the closest ones weigh more
    assert m.stats["matched"] == 300


def test_pick_with_zero_temperature_takes_the_nearest(tmp_path):
    m = make_matcher(tmp_path, background_match_k=5, background_match_temperature=0.0)
    bg = fake_bg(m, 5)
    assert {src_of(m.pick(all_paths(), bg, random.Random(i))[0]) for i in range(20)} <= {4, 6}


def test_exclude_same_group_removes_the_sequence_but_it_is_allowed_by_default(tmp_path):
    # image 4 = group 1 (3, 4, 5): excluding the group, neighbours 3 and 5 disappear
    excl = make_matcher(tmp_path, background_match_k=4, background_match_exclude_same_group=True)
    bg = fake_bg(excl, 4)
    picks = {src_of(excl.pick(all_paths(), bg, random.Random(i))[0]) for i in range(100)}
    assert picks == {2, 6}
    same = make_matcher(tmp_path, background_match_k=4)
    bg = fake_bg(same, 4)
    assert {src_of(same.pick(all_paths(), bg, random.Random(i))[0]) for i in range(100)} == {3, 5}


def test_allow_same_image_is_opt_in(tmp_path):
    m = make_matcher(tmp_path, background_match_k=1, background_match_allow_same_image=True)
    bg = fake_bg(m, 5)
    assert src_of(m.pick(all_paths(), bg, random.Random(0))[0]) == 5


def test_unknown_sources_and_empty_pools_fall_back_without_breaking(tmp_path):
    m = make_matcher(tmp_path)
    bg = fake_bg(m, 5)
    unknown = ["/objs/Bottle/zz.png", "/objs/Bottle/yy.png"]
    assert m.pick(unknown, bg, random.Random(0)) == unknown and m.stats["unmatched_no_source"] == 1
    assert src_of(m.pick(unknown + ["/objs/Bottle/3_0.png"], bg, random.Random(0))[0]) == 3   # the unknown one is discarded
    only_same = ["/objs/Bottle/5_0.png", "/objs/Bottle/5_1.png"]                              # only the background's own image
    assert m.pick(only_same, bg, random.Random(0)) == only_same and m.stats["unmatched_no_eligible"] == 1


def test_background_outside_the_table_is_measured_from_the_file(tmp_path):
    img = np.full((40, 40, 3), (60, 60, 60), np.uint8)
    f = tmp_path / "synthetic_bg.jpg"
    cv2.imwrite(str(f), img)
    m = make_matcher(tmp_path, background_match_k=1, background_match_temperature=0.0)
    idx, desc, _ = m.background(str(f))
    assert idx == -1 and desc[0] == pytest.approx(bm.lab_descriptor(cv2.imread(str(f)))[0])
    pick = src_of(m.pick(all_paths(), str(f), random.Random(0))[0])
    near = int(np.argmin(np.abs(m.table.lab[:, 0] - desc[0])))
    assert pick == near
    assert m.background(str(tmp_path / "missing.jpg")) is None


def test_hist_weight_adds_to_the_distance(tmp_path):
    m = make_matcher(tmp_path, background_match_hist_weight=1.0)
    m.table.hist[3, 0] = 1.0                     # image 3's histogram differs from the background's (zeros)
    bg = fake_bg(m, 5)
    d1 = m.distances(m._bg[bg], np.array([3, 4]))
    m.profile = m.profile.model_copy(update={"background_match_hist_weight": 0.0})
    d0 = m.distances(m._bg[bg], np.array([3, 4]))
    assert d1[0] - d0[0] == pytest.approx(bm.HIST_SCALE) and d1[1] == d0[1]


# ------------------------------------------------------------------ planner integration
def plan(ctx, bg, seed=0, n_img=1):
    rng, out = random.Random(seed), []
    for _ in range(n_img):
        objs, reserved, _ = plan_objects(ctx, rng, ["Bottle", "Can"], {"Bottle": 50, "Can": 50},
                                         objects_for(), 8, bg_wh=(640, 480), bg_file=bg)
        out += [o.image_path for o in objs]
        if ctx.usage is not None:
            ctx.usage.settle(reserved, [o.image_path for o in objs])
    return out


def test_planner_picks_crops_by_background_and_matches_the_flag_off_behaviour(tmp_path):
    bg = "/bgs/bg_8.jpg"
    ctx = match_ctx(tmp_path, background_match_k=3, background_match_temperature=0.0, faithful_count=True)
    picks = plan(ctx, bg, n_img=20)
    assert picks and {src_of(p) for p in picks} <= {7, 9, 6, 10}
    assert 8 not in {src_of(p) for p in picks} and ctx.info()["background_match"]["matched"] == len(picks)
    # off: passing bg_file changes nothing and consumes no randomness
    off = make_context(tmp_path, objects=objects_for(), position_from_stats=False, faithful_count=True)
    off2 = make_context(tmp_path, objects=objects_for(), position_from_stats=False, faithful_count=True)
    assert off.matcher is None and "background_match" not in off.info()
    a, b = plan(off, bg, seed=3, n_img=5), plan(off2, None, seed=3, n_img=5)
    assert a == b


def test_planner_respects_the_crop_reuse_cap_and_chains_with_the_scale_filter(tmp_path):
    ctx = match_ctx(tmp_path, background_match_k=1, background_match_temperature=0.0, max_crop_reuse=2, faithful_count=True)
    picks = plan(ctx, "/bgs/bg_8.jpg", n_img=40)
    counts = {p: picks.count(p) for p in set(picks)}
    assert max(counts.values()) <= 2                         # the cap is applied BEFORE choosing by similarity
    assert len(counts) > 4                                   # and when a crop is exhausted, the next closest one is used


# ------------------------------------------------------------------ local ring
def test_search_position_with_score_picks_the_best_free_candidate():
    stats, bg = position_stats.load_position_stats(), (640, 480)
    prof = SynthProfile(name="t", position_from_stats=True, border_touch_fraction=0.0)
    score = lambda b: float(b[0])                                                    # noqa: E731  (lower x = better)
    plain = [position_stats.search_position(stats, prof, "Glove", (0, 0), (60, 60), bg, random.Random(i), touch=False)
             for i in range(60)]
    best = [position_stats.search_position(stats, prof, "Glove", (0, 0), (60, 60), bg, random.Random(i), touch=False,
                                           score=score, n_candidates=8) for i in range(60)]
    assert np.mean([b[0] for b in best]) < np.mean([p[0] for p in plain])
    seen = []
    position_stats.search_position(stats, prof, "Glove", (0, 0), (60, 60), bg, random.Random(1), touch=False,
                                   score=lambda b: seen.append(b) or 0.0, n_candidates=5)
    assert len(seen) == 5


def test_search_position_with_score_respects_avoid_and_falls_back():
    stats, bg = position_stats.load_position_stats(), (640, 480)
    prof = SynthProfile(name="t", position_from_stats=True, border_touch_fraction=0.0)
    called = []
    pos = position_stats.search_position(stats, prof, "Glove", (0, 0), (60, 60), bg, random.Random(0), touch=False,
                                         avoid=lambda b: True, score=lambda b: called.append(b) or 0.0)
    assert not called and len(pos) == 2                                              # none free: as without score
    scored = []
    pos = position_stats.search_position(stats, prof, "Glove", (0, 0), (60, 60), bg, random.Random(0), touch=False,
                                         avoid=lambda b: b[1] < 200, score=lambda b: scored.append(b) or 0.0)
    assert scored and all(b[1] >= 200 for b in scored)                               # only the free ones are scored


def test_patch_scorer_prefers_the_ring_that_looks_like_the_source_ring(tmp_path):
    table = make_table(tmp_path)
    prof = SynthProfile(name="t", position_from_stats=True, local_patch_match=True, background_match_path=table)
    dark_left = np.zeros((300, 400, 3), np.uint8)
    dark_left[:, 200:] = (230, 230, 230)
    crop_bright = "/objs/Bottle/10_0.png"            # its ring in the source has L = 250: bright
    crop_dark = "/objs/Bottle/1_0.png"               # L = 25: dark
    bright_box, dark_box = (260, 100, 320, 160), (60, 100, 120, 160)
    s_b = bm.patch_scorer(prof, crop_bright, dark_left)
    s_d = bm.patch_scorer(prof, crop_dark, dark_left)
    assert s_b(bright_box) < s_b(dark_box) and s_d(dark_box) < s_d(bright_box)
    assert bm.patch_scorer(prof, "/objs/Bottle/unknown.png", dark_left) is None
    assert bm.patch_scorer(SynthProfile(name="t", background_match_path=table), crop_bright, dark_left) is None
    assert s_b((0, 0, 400, 300)) == float("inf")      # no measurable ring


# ------------------------------------------------------------------ tabla real incluida
def test_bundled_table_is_consistent():
    t = bm.load_match_table(None)
    n = len(t.files)
    assert n > 10000 and t.lab.shape == (n, 6) and t.hist.shape == (n, 64) and np.isfinite(t.lab).all()
    assert (t.group >= 0).all() and t.ring_frac == pytest.approx(0.25)
    assert t.crop_img.min() >= 0 and t.crop_img.max() < n and len(t.crop_keys) == len(t.crop_img) > 11000
    assert any(k.startswith("Bottle/poly_") for k in t.crop_keys) and "Bottle/1.png" in t.crop_keys
    assert t.hist.sum(axis=1).min() > 0.9
    assert t.source_image("/app/datasets/Extracted_objects_train/Bottle/1.png") == t.index["00000_HPD1684HDTV0829.jpg"]
    assert t.ring_of("/x/Bottle/1.png") is not None and t.ring_of("/x/Bottle/none.png") is None
    assert Path(bm.DEFAULT_MATCH_PATH).stat().st_size < 5_000_000


REAL_TRAIN = Path(os.environ.get("GEN_INPUTS", "gen_inputs")) / "real_train"


@pytest.mark.skipif(not (REAL_TRAIN / "00000_HPD1684HDTV0829.jpg").exists(),
                    reason="training images not available (set GEN_INPUTS)")
def test_bundled_descriptor_equals_the_one_measured_from_the_image():
    t = bm.load_match_table(None)
    name = "00000_HPD1684HDTV0829.jpg"
    img = cv2.imread(str(REAL_TRAIN / name))
    assert np.allclose(t.lab[t.index[name]], bm.lab_descriptor(img), atol=1e-3)


# ------------------------------------------------------------------ composer wiring
def test_place_object_with_local_patch_match_moves_the_object_towards_the_similar_ring(tmp_path):
    import asyncio
    from app.composer import ImageComposer
    from app.models.schemas import EffectsConfig, EffectType, ObjectPlacement
    from app.synth_context import ImagePlan
    img = np.zeros((40, 50, 4), np.uint8)
    cv2.ellipse(img, (25, 20), (22, 17), 0, 0, 360, (40, 90, 200, 255), -1)
    (tmp_path / "Bottle").mkdir()
    path = tmp_path / "Bottle" / "10_0.png"            # its ring in the source is bright (L = 250)
    cv2.imwrite(str(path), img)
    bg = np.zeros((300, 400, 3), np.uint8)
    bg[:, 200:] = (230, 230, 230)
    table = make_table(tmp_path)

    def run(local: bool) -> float:
        prof = SynthProfile(name="t", position_from_stats=True, border_touch_fraction=0.0, copy_paste_real=True,
                            local_patch_match=local, background_match_path=table)
        comp, right = ImageComposer(use_gpu=False, try_count=4), []
        random.seed(5)
        for _ in range(30):
            res = asyncio.run(comp._place_object(
                composite=bg.copy(), obj_placement=ObjectPlacement(image_path=str(path), class_name="Bottle", scale=1.0,
                                                                   rotation=0.0),
                depth_map=None, lighting_info=None, effects=[EffectType.SHADOWS], effects_config=EffectsConfig(),
                placement_mask=np.zeros(bg.shape[:2], np.uint8), existing_bboxes=[],
                plan=ImagePlan(profile=prof, stats=None)))
            assert res is not None
            right.append((res[1][0] + res[1][2]) / 2 > 200)
        return float(np.mean(right))

    assert run(True) > run(False) + 0.15 and run(True) > 0.75
