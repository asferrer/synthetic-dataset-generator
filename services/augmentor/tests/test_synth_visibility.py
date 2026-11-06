"""Visible copy-paste (v11): local-contrast visibility filter and haze and blur caps."""

import asyncio
import json

import cv2
import numpy as np
import pytest

from app import copy_paste, visibility
from app.composer import ComposeResult, ImageComposer
from app.models.schemas import AnnotationBox, EffectsConfig, ObjectPlacement
from app.synth_context import ImagePlan
from app.synth_profile import (V10_PROFILE, V11_PROFILE, V11_VISIBILITY_P5, V11_VISIBLE_OVERRIDES, SynthProfile,
                               get_profile)
from tests.test_synth_v11 import _prof, _run, batch_dirs  # noqa: F401  (batch_dirs is a fixture)


# ------------------------------------------------------------------ local contrast C
def _noisy_bg(h=120, w=160, base=(140, 110, 60), amp=6, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(np.array(base, np.int16) + rng.integers(-amp, amp + 1, (h, w, 3)), 0, 255).astype(np.uint8)


def _with_patch(delta, box=(60, 40, 100, 80)):
    img = _noisy_bg()
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = np.clip(img[y1:y2, x1:x2].astype(np.int16) + delta, 0, 255).astype(np.uint8)
    return img, box


def test_contrast_is_small_for_background_and_grows_with_the_patch():
    flat, box = _with_patch(0)
    values = [visibility.local_contrast(_with_patch(d)[0], box) for d in (0, 15, 40, 80)]
    assert values[0] < 15 and values == sorted(values) and values[-1] > 3 * values[0]
    assert visibility.local_contrast(flat, box) == pytest.approx(values[0])


def test_contrast_is_none_without_ring_or_box_pixels():
    img = _noisy_bg(10, 10)
    assert visibility.local_contrast(img, (0, 0, 10, 10)) is None            # the box covers the whole image: no ring
    assert visibility.local_contrast(img, (5, 5, 5, 9)) is None              # empty box
    assert visibility.local_contrast(img, (50, 50, 60, 60)) is None          # outside the image


def test_contrast_clips_the_box_and_downsamples_large_windows():
    img, box = _with_patch(60)
    clipped = visibility.local_contrast(img, (box[0], box[1], box[2] + 500, box[3] + 500))
    assert clipped is not None
    big = np.kron(img, np.ones((6, 6, 1), np.uint8))                          # 960 x 720: se reduce a 256 px
    small_c = visibility.local_contrast(img, box)
    big_c = visibility.local_contrast(big, tuple(v * 6 for v in box))
    assert big_c == pytest.approx(small_c, rel=0.15)


def test_luminance_std_inside_the_box():
    img, box = _with_patch(0)
    flat = img.copy()
    flat[40:80, 60:100] = 128
    assert visibility.box_luminance_std(flat, box) < 1.0
    assert visibility.box_luminance_std(img, box) > 1.0
    assert visibility.box_luminance_std(img, (0, 0, 2, 2)) is None            # < 16 px


def test_is_visible_threshold_scale_and_unmeasurable():
    img, box = _with_patch(50)
    c = visibility.local_contrast(img, box)
    assert visibility.is_visible(img, box, c - 0.1)[0] and not visibility.is_visible(img, box, c + 0.1)[0]
    assert not visibility.is_visible(img, box, c - 0.1, scale=0.5)[0]         # haze subtracts contrast
    ok, none_c = visibility.is_visible(_noisy_bg(10, 10), (0, 0, 10, 10), 1e9)
    assert ok and none_c is None                                              # no medible: no se descarta a ciegas


def test_an_invisible_copy_of_the_background_falls_below_the_real_p5():
    img, box = _with_patch(0)
    assert not visibility.is_visible(img, box, V11_VISIBILITY_P5["Bottle"])[0]
    assert visibility.is_visible(_with_patch(70)[0], box, V11_VISIBILITY_P5["Bottle"])[0]


# ------------------------------------------------------------------ per-class threshold and report
def test_threshold_for_float_dict_and_off():
    off = SynthProfile(name="t")
    assert visibility.threshold_for(off, "Can") is None
    on = SynthProfile(name="t", visibility_filter=True, visibility_min_contrast=12.5)
    assert visibility.threshold_for(on, "Can") == 12.5
    per = SynthProfile(name="t", visibility_filter=True, visibility_min_contrast={"Can": 9.0})
    assert visibility.threshold_for(per, "Can") == 9.0 and visibility.threshold_for(per, "Tire") is None
    with pytest.raises(ValueError):
        SynthProfile(name="t", visibility_min_contrast=-1.0)


def test_build_report_sums_rejections_and_drops_per_class():
    entries = [{"synth": {"visibility": {"rejected": {"Can": 3, "Tire": 1}, "dropped": {"Can": 1}}}},
               {"synth": {"visibility": {"rejected": {"Can": 2}, "dropped": {}}}},
               {"synth": {}}, {}]
    rep = visibility.build_report(entries, {"Can": 7, "Tire": 4, "Mask": 2})
    assert rep["Can"] == {"rejected_attempts": 5, "dropped_objects": 1, "pasted": 7}
    assert rep["Tire"] == {"rejected_attempts": 1, "dropped_objects": 0, "pasted": 4}
    assert rep["Mask"] == {"rejected_attempts": 0, "dropped_objects": 0, "pasted": 2}


def test_percentile_table_has_per_class_and_global_rows():
    table = visibility.percentile_table({"A": list(range(101)), "B": [10.0, 20.0]})
    assert table["A"]["p5"] == 5.0 and table["A"]["n"] == 101 and table["ALL"]["n"] == 103


# ------------------------------------------------------------------ haze and blur caps
def test_cap_kernel_keeps_odd_kernels_and_disables_below_three():
    assert copy_paste.cap_kernel(5, None) == 5 and copy_paste.cap_kernel(3, 5) == 3
    assert copy_paste.cap_kernel(5, 3) == 3 and copy_paste.cap_kernel(5, 4) == 3
    assert copy_paste.cap_kernel(3, 2) == 0 and copy_paste.cap_kernel(5, 0) == 0


def test_haze_and_blur_caps_only_act_with_copy_paste():
    base = SynthProfile(name="t", haze_max_intensity=0.12, copy_paste_haze_max=0.03, copy_paste_blur_max=0)
    assert copy_paste.haze_cap(base) == 0.12 and copy_paste.blur_kernel_cap(base) is None
    cp = base.model_copy(update={"copy_paste_real": True})
    assert copy_paste.haze_cap(cp) == 0.03 and copy_paste.blur_kernel_cap(cp) == 0
    loose = cp.model_copy(update={"copy_paste_haze_max": 0.5})
    assert copy_paste.haze_cap(loose) == 0.12                                  # never raises the global cap
    assert copy_paste.haze_cap(cp.model_copy(update={"copy_paste_haze_max": None})) == 0.12


def _blur_scene():
    rng = np.random.default_rng(1)
    obj = (rng.random((40, 40, 3)) * 255).astype(np.uint8)                     # muy nitido
    roi = cv2.GaussianBlur((rng.random((40, 40, 3)) * 255).astype(np.uint8), (15, 15), 0)
    roi = np.clip(roi.astype(np.int16) * 3 - 200, 0, 255).astype(np.uint8)     # background with some texture (> 10)
    return obj, roi, np.full((40, 40), 255, np.uint8)


def test_blur_matching_cap_disables_or_limits_the_kernel():
    comp = ImageComposer(use_gpu=False)
    obj, roi, mask = _blur_scene()
    plain = comp._apply_blur_matching(obj, roi, mask, 0.65)
    assert not np.array_equal(plain, obj), "the test case must trigger the blur"
    assert np.array_equal(comp._apply_blur_matching(obj, roi, mask, 0.65, blur_max=0), obj)
    assert np.array_equal(comp._apply_blur_matching(obj, roi, mask, 0.65, blur_max=2), obj)
    assert np.array_equal(comp._apply_blur_matching(obj, roi, mask, 0.65, blur_max=3), plain)   # the kernel was already 3


# ------------------------------------------------------------------ profiles: off by default
def test_visibility_and_caps_are_off_in_legacy_v10_v11_profiles():
    for prof in (SynthProfile(), V10_PROFILE, V11_PROFILE):
        assert not prof.visibility_filter and prof.copy_paste_haze_max is None and prof.copy_paste_blur_max is None
    assert get_profile(None) is None and get_profile("legacy") is None


def test_visible_overrides_activate_filter_and_caps():
    prof = get_profile("v11", {"copy_paste_real": True, "harmonize": True, **V11_VISIBLE_OVERRIDES})
    assert prof.visibility_filter and prof.copy_paste_haze_max == 0.0 and prof.copy_paste_blur_max == 3
    assert prof.visibility_min_contrast == V11_VISIBILITY_P5 and set(V11_VISIBILITY_P5) >= {"Bottle", "Fishing_Net"}


# ------------------------------------------------------------------ filter inside _place_object
def _object_png(tmp_path, bgr, size=30):
    """RGBA crop: opaque single-colour disc."""
    img = np.zeros((size, size, 4), np.uint8)
    img[:, :, :3] = bgr
    alpha = np.zeros((size, size), np.uint8)
    cv2.circle(alpha, (size // 2, size // 2), size // 2 - 1, 255, -1)
    img[:, :, 3] = alpha
    path = tmp_path / f"obj_{bgr[0]}.png"
    cv2.imwrite(str(path), img)
    return str(path)


def _place(tmp_path, bgr, threshold, haze_cap=0.0):
    prof = SynthProfile(name="t", visibility_filter=True, visibility_min_contrast=threshold,
                        haze_max_intensity=haze_cap if haze_cap else 0.12, copy_paste_real=True,
                        copy_paste_haze_max=haze_cap)
    plan = ImagePlan(profile=prof, stats=None)
    comp = ImageComposer(use_gpu=False, try_count=3)
    bg = _noisy_bg()
    trace = {}
    res = asyncio.run(comp._place_object(
        composite=bg, obj_placement=ObjectPlacement(image_path=_object_png(tmp_path, bgr), class_name="Bottle",
                                                    position=(60, 40), scale=1.0, rotation=0.0),
        depth_map=None, lighting_info=None, effects=[], effects_config=EffectsConfig(),
        placement_mask=np.zeros(bg.shape[:2], np.uint8), existing_bboxes=[], plan=plan, trace_out=trace))
    return res, trace


def test_place_object_drops_an_object_that_blends_with_the_background(tmp_path):
    res, trace = _place(tmp_path, (140, 110, 60), threshold=15.0)             # same colour as the background
    assert res is None and trace["visibility_rejects"] == 3                   # one rejection per attempt


def test_place_object_keeps_a_contrasted_object_and_threshold_zero_keeps_all(tmp_path):
    res, trace = _place(tmp_path, (20, 20, 220), threshold=15.0)
    assert res is not None and "visibility_rejects" not in trace
    res, trace = _place(tmp_path, (140, 110, 60), threshold=0.0)
    assert res is not None and "visibility_rejects" not in trace


def test_haze_cap_makes_the_filter_stricter_for_the_same_object(tmp_path):
    comp_bgr = (110, 90, 60)
    probe, _ = _place(tmp_path, comp_bgr, threshold=0.0)
    c = visibility.local_contrast(probe[0], probe[1])
    ok, _ = _place(tmp_path, comp_bgr, threshold=c * 0.99, haze_cap=0.0)
    assert ok is not None
    dropped, trace = _place(tmp_path, comp_bgr, threshold=c * 0.99, haze_cap=0.2)    # c * 0.8 < umbral
    assert dropped is None and trace["visibility_rejects"] == 3


def test_log_visibility_counts_rejections_and_unplaced_objects():
    log = {"rejected": {}, "dropped": {}}
    ImageComposer._log_visibility(log, "Can", {"visibility_rejects": 2}, unplaced=False)
    ImageComposer._log_visibility(log, "Can", {"visibility_rejects": 3}, unplaced=True)
    ImageComposer._log_visibility(log, "Tire", {}, unplaced=True)             # lost for another cause: does not count
    assert log == {"rejected": {"Can": 5}, "dropped": {"Can": 1}}


# ------------------------------------------------------------------ batch report
def _composer_with_visibility(monkeypatch):
    comp = ImageComposer(use_gpu=False)

    async def fake_depth(bg_image, cache_key=None):
        return np.zeros((4, 4), dtype=np.float32), "fallback_gradient"

    async def fake_compose(objects, **kwargs):
        anns = [AnnotationBox(x=50, y=50, width=4, height=4, class_name=o.class_name, area=16,
                              source_object=o.image_path) for o in objects]
        trace = {"visibility": {"rejected": {"Tire": 2}, "dropped": {"Tire": 1}}}
        return ComposeResult(annotations=anns, objects_placed=len(anns), image_size=(100, 80), image_trace=trace)

    async def fake_validate(*args, **kwargs):
        return True, None

    monkeypatch.setattr(comp, "_get_depth_map", fake_depth)
    monkeypatch.setattr(comp, "compose", fake_compose)
    monkeypatch.setattr(comp, "_validate_composed", fake_validate)
    return comp


def test_batch_writes_the_visibility_report_per_class(monkeypatch, batch_dirs, tmp_path):
    prof = _prof(tmp_path, visibility_filter=True)
    res = _run(_composer_with_visibility(monkeypatch), batch_dirs, prof)
    coco = json.load(open(res.output_coco_path, encoding="utf8"))
    rep = coco["info"]["synth_generator"]["visibility_filter"]
    n_img = len(coco["images"])
    assert rep["Tire"]["rejected_attempts"] == 2 * n_img and rep["Tire"]["dropped_objects"] == n_img
    assert rep["Tire"]["pasted"] == res.synthetic_counts["Tire"]
    out = json.load(open(res.output_coco_path.replace("synthetic_dataset.json", "visibility_report.json")))
    assert out == rep
    assert all(im["visibility"] == {"rejected": {"Tire": 2}, "dropped": {"Tire": 1}} for im in coco["images"])


def test_batch_without_the_filter_has_no_visibility_block(monkeypatch, batch_dirs, tmp_path):
    res = _run(_composer_with_visibility(monkeypatch), batch_dirs, _prof(tmp_path))
    coco = json.load(open(res.output_coco_path, encoding="utf8"))
    assert "visibility_filter" not in coco["info"]["synth_generator"]
