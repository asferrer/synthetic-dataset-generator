"""v11 variant k: single resample (`single_resample`) and sharpness/grain matching (`match_sharpness`)."""

import asyncio
import random

import cv2
import numpy as np
import pytest

from app import resample, sharpness
from app.composer import ImageComposer
from app.models.schemas import EffectsConfig, ObjectPlacement
from app.synth_context import ImagePlan
from app.synth_profile import SynthProfile


# ------------------------------------------------------------------ utilities
def _texture(h=90, w=90, seed=0, smooth=1.0):
    """Detailed texture (smoothed noise) in BGR."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3)).astype(np.float32)
    if smooth > 0:
        img = cv2.GaussianBlur(img, (0, 0), smooth)
    return np.clip(img, 0, 255).astype(np.uint8)


def _rgba(h=90, w=90, seed=0, outside=(0, 0, 255)):
    """RGBA crop: texture inside an ellipse; outside the alpha the RGB is `outside` (the crop's original background)."""
    rgb = _texture(h, w, seed)
    alpha = np.zeros((h, w), np.uint8)
    cv2.ellipse(alpha, (w // 2, h // 2), (w // 2 - 3, h // 2 - 3), 0, 0, 360, 255, -1)
    rgb[alpha == 0] = outside
    return np.dstack([rgb, alpha])


def _lapvar(img_bgra):
    alpha = img_bgra[:, :, 3] > 250
    inner = cv2.erode(alpha.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    lap = cv2.Laplacian(cv2.cvtColor(img_bgra[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F)
    return float(lap[inner].var())


# ------------------------------------------------------------------ remuestra unica
def test_rotation_matrix_matches_the_canvas_of_the_old_path():
    for w, h, ang in ((90, 60, 0.0), (90, 60, 30.0), (40, 100, -45.0)):
        m, (new_w, new_h) = resample.rotation_matrix(w, h, ang)
        cos, sin = abs(np.cos(np.radians(ang))), abs(np.sin(np.radians(ang)))
        assert abs(new_w - (h * sin + w * cos)) <= 1.0 and abs(new_h - (h * cos + w * sin)) <= 1.0
        assert m.shape == (2, 3)


def test_rotated_geometry_gives_canvas_and_alpha_box_without_rotating_the_rgb():
    img = _rgba(60, 100)
    canvas0, box0 = resample.rotated_geometry(img, 0.0)
    assert canvas0 == (100, 60) and box0[0] <= 100 and box0[1] <= 60
    canvas, box = resample.rotated_geometry(img, 40.0)
    assert canvas[0] > 100 and box[0] <= canvas[0] and box[1] <= canvas[1]
    assert resample.rotated_geometry(np.zeros((30, 30, 3), np.uint8), 20.0)[1] == resample.rotated_geometry(
        np.zeros((30, 30, 3), np.uint8), 20.0)[0]


@pytest.mark.parametrize("scale", [0.3, 0.55, 1.0, 1.7])
def test_warp_single_output_size_and_alpha(scale):
    img = _rgba(80, 120)
    out = resample.warp_single(img, 25.0, scale)
    (cw, ch), _ = resample.rotated_geometry(img, 25.0)
    assert out.shape[2] == 4 and abs(out.shape[1] - cw * scale) <= 3 and abs(out.shape[0] - ch * scale) <= 3
    assert out[:, :, 3].max() == 255 and out[:, :, 3].min() == 0


def test_warp_single_does_not_bleed_the_color_outside_the_alpha():
    img = _rgba(80, 80, outside=(0, 0, 255))     # the RGB outside the alpha is pure red
    for scale in (0.4, 1.0, 1.6):
        out = resample.warp_single(img, 20.0, scale)
        solid = out[:, :, 3] > 250
        assert solid.sum() > 50
        red_bleed = (out[:, :, 2].astype(int) - out[:, :, 0].astype(int))[solid]
        assert np.median(red_bleed) < 40    # the texture is grey-blue on average: no red is dragged in


def test_warp_single_handles_bgr_without_alpha_and_empty_results():
    out = resample.warp_single(_texture(), 10.0, 0.5)
    assert out.shape[2] == 3
    assert resample.warp_single(_rgba(8, 8), 0.0, 0.01) is None


def test_single_resample_keeps_more_detail_than_the_old_path():
    comp = ImageComposer(use_gpu=False)
    img = _rgba(90, 90)     # shorter side < 100: the legacy path applies a pre-blur
    old = comp._transform_object(img, (480, 640), None, (10, 10), 0.45, 20.0)[0]
    new = comp._transform_object(img, (480, 640), None, (10, 10), 0.45, 20.0, single_resample=True)[0]
    assert abs(old.shape[0] - new.shape[0]) <= 2 and abs(old.shape[1] - new.shape[1]) <= 2
    assert _lapvar(new) > 1.3 * _lapvar(old)


def test_single_resample_uses_the_same_random_calls_as_the_old_path():
    comp = ImageComposer(use_gpu=False)
    img = _rgba(90, 90)
    after = []
    for single in (False, True):
        random.seed(5)
        out = comp._transform_object(img, (480, 640), None, None, None, None, single_resample=single)
        assert out[0] is not None
        after.append(random.random())
    assert after[0] == after[1]


def test_single_resample_off_is_the_default_path():
    comp = ImageComposer(use_gpu=False)
    img = _rgba(90, 90)
    a = comp._transform_object(img, (480, 640), None, (5, 5), 0.5, 15.0)[0]
    b = comp._transform_object(img, (480, 640), None, (5, 5), 0.5, 15.0, single_resample=False)[0]
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ sharpness and noise measures
def test_hf_ratio_orders_sharp_above_blurred():
    sharp = _texture(64, 64, smooth=0.0)[:, :, 0].astype(np.float32)
    soft = cv2.GaussianBlur(sharp, (0, 0), 2.0)
    assert sharpness.hf_ratio(sharp) > 5 * sharpness.hf_ratio(soft)


def test_object_hf_is_none_for_tiny_or_empty_masks():
    img = _texture(40, 40)
    assert sharpness.object_hf(img, np.zeros((40, 40), np.uint8)) is None
    tiny = np.zeros((40, 40), np.uint8)
    tiny[5:9, 5:9] = 255
    assert sharpness.object_hf(img, tiny) is None


def test_flat_noise_recovers_the_noise_level_and_skips_blocks():
    rng = np.random.default_rng(1)
    gray = (128 + rng.normal(0, 5, (96, 96))).astype(np.float32)
    est = sharpness.flat_noise(gray)
    assert 3.0 < est < 5.2
    skip = np.zeros_like(gray, dtype=bool)
    skip[:, :] = True
    assert sharpness.flat_noise(gray, skip=skip) is None


def test_ring_noise_ignores_the_box():
    rng = np.random.default_rng(2)
    bg = np.clip(128 + rng.normal(0, 4, (200, 200, 3)), 0, 255).astype(np.uint8)
    bg[80:120, 80:120] = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)    # very noisy box
    est = sharpness.ring_noise(bg, (80, 80, 40, 40))
    assert est is not None and est < 5.0


def test_grain_sigma_is_quadrature_and_capped():
    assert sharpness.grain_sigma(3.0, 5.0, 8.0) == pytest.approx(4.0)
    assert sharpness.grain_sigma(6.0, 5.0, 8.0) == 0.0
    assert sharpness.grain_sigma(None, 5.0, 8.0) == 0.0 and sharpness.grain_sigma(1.0, None, 8.0) == 0.0
    assert sharpness.grain_sigma(0.0, 20.0, 8.0) == 8.0


def test_add_grain_has_the_requested_level_and_zero_is_identity():
    flat = np.full((80, 80, 3), 128, np.uint8)
    noisy = sharpness.add_grain(flat, 4.0, np.random.default_rng(0))
    gray_std = float(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY).astype(np.float32).std())
    assert 3.0 < gray_std < 4.6
    assert sharpness.add_grain(flat, 0.0, np.random.default_rng(0)) is flat


# ------------------------------------------------------------------ corrections
def _blurred_object():
    sharp = _texture(120, 120, smooth=0.0)
    mask = np.zeros((120, 120), np.uint8)
    cv2.ellipse(mask, (60, 60), (55, 50), 0, 0, 360, 255, -1)
    return sharp, cv2.GaussianBlur(sharp, (0, 0), 2.5), mask


def test_sharpen_to_raises_hf_ratio_to_the_target():
    sharp, _, mask = _blurred_object()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 1.0)
    before = sharpness.object_hf(blurred, mask)
    target = before * 1.7
    out = sharpness.sharpen_to(blurred, mask, target, max_amount=4.0)
    after = sharpness.object_hf(out, mask)
    assert before < target and after > before and after == pytest.approx(target, rel=0.35)
    assert out.dtype == np.uint8 and out.shape == blurred.shape


def test_sharpen_to_is_capped_by_max_amount():
    sharp, _, mask = _blurred_object()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 1.0)
    weak = sharpness.sharpen_to(blurred, mask, 10.0, max_amount=0.5)
    strong = sharpness.sharpen_to(blurred, mask, 10.0, max_amount=3.0)
    assert sharpness.object_hf(strong, mask) > sharpness.object_hf(weak, mask)


def test_soften_to_lowers_hf_ratio():
    sharp, _, mask = _blurred_object()
    hf = sharpness.object_hf(sharp, mask)
    out = sharpness.soften_to(sharp, mask, hf * 0.4)
    assert sharpness.object_hf(out, mask) < hf * 0.7


def test_normalized_blur_does_not_bleed_color_from_outside_the_mask():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, :] = (0, 0, 255)
    img[10:30, 10:30] = (255, 0, 0)
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 255
    out = sharpness.normalized_blur(img, mask, 3.0)
    assert out[15:25, 15:25, 2].max() < 5      # the outside red does not enter the blue object


def _stats_profile(**kw):
    return SynthProfile(name="t", match_sharpness=True, single_resample=True, **kw)


def test_match_sharpness_sharpens_a_blurred_object_and_adds_grain():
    sharp, blurred, mask = _blurred_object()
    rng = np.random.default_rng(3)
    comp_bg = np.clip(128 + rng.normal(0, 5, (300, 300, 3)), 0, 255).astype(np.uint8)
    smooth = cv2.GaussianBlur(blurred, (0, 0), 3.0)
    out, info = sharpness.match_sharpness(smooth, mask, comp_bg, (90, 90, 120, 120), "Bottle", _stats_profile(),
                                          random.Random(1))
    assert info["hf_after"] > info["hf_before"] and info["grain"] > 0
    assert sharpness.object_noise(out, mask) > (sharpness.object_noise(smooth, mask) or 0.0)


def test_match_sharpness_without_grain_and_target_gain():
    _, blurred, mask = _blurred_object()
    bg = np.full((300, 300, 3), 120, np.uint8)
    base = _stats_profile(sharpness_grain=False)
    _, info = sharpness.match_sharpness(blurred, mask, bg, (90, 90, 120, 120), "Bottle", base, random.Random(1))
    assert info["grain"] == 0.0
    hi = _stats_profile(sharpness_grain=False, sharpness_gain=2.0)
    _, info2 = sharpness.match_sharpness(blurred, mask, bg, (90, 90, 120, 120), "Bottle", hi, random.Random(1))
    assert info2["target_hf"] == pytest.approx(2 * info["target_hf"], rel=1e-3)


def test_match_sharpness_consumes_the_same_random_numbers_whatever_the_object():
    sharp, blurred, mask = _blurred_object()
    bg = np.full((300, 300, 3), 120, np.uint8)
    states = []
    for obj in (sharp, blurred, np.zeros_like(sharp)):
        r = random.Random(7)
        sharpness.match_sharpness(obj, mask, bg, (90, 90, 120, 120), "Can", _stats_profile(), r)
        states.append(r.random())
    assert states[0] == states[1] == states[2]


def test_match_sharpness_unknown_class_uses_the_global_quantiles():
    stats = sharpness.load_sharpness_stats()
    assert stats.target_hf("NoExiste", 0.5) == pytest.approx(stats.target_hf("all", 0.5))
    assert stats.target_hf("Fishing_Net", 0.5) > 2 * stats.target_hf("Tire", 0.5)


def test_real_sharpness_json_is_train_only_and_monotone():
    import json
    with open(sharpness.DEFAULT_SHARPNESS_PATH, encoding="utf8") as f:
        data = json.load(f)
    assert "train" in data["source"].lower() and "val" not in data["source"].lower()
    for cls, v in data["classes"].items():
        q = v["hf_ratio_quantiles"]
        assert len(q) == 11 and q == sorted(q) and v["n"] > 50, cls
    assert set(data["classes"]) >= {"all", "Bottle", "Can", "Fishing_Net", "Glove", "Mask", "Metal_Debris",
                                    "Plastic_Debris", "Tire"}


# ------------------------------------------------------------------ inside _place_object
def _place(tmp_path, **profile_kw):
    img = _rgba(70, 70, outside=(0, 0, 0))
    path = tmp_path / "obj.png"
    cv2.imwrite(str(path), img)
    prof = SynthProfile(name="t", copy_paste_real=True, **profile_kw)
    rng = np.random.default_rng(0)
    bg = np.clip(110 + rng.normal(0, 4, (240, 320, 3)), 0, 255).astype(np.uint8)
    comp = ImageComposer(use_gpu=False, try_count=2)
    res = asyncio.run(comp._place_object(
        composite=bg, obj_placement=ObjectPlacement(image_path=str(path), class_name="Bottle", position=(100, 80),
                                                    scale=1.0, rotation=0.0),
        depth_map=None, lighting_info=None, effects=[], effects_config=EffectsConfig(),
        placement_mask=np.zeros(bg.shape[:2], np.uint8), existing_bboxes=[], plan=ImagePlan(profile=prof, stats=None)))
    return res, bg


def test_place_object_with_match_sharpness_runs_and_changes_the_pasted_pixels(tmp_path):
    plain, bg = _place(tmp_path)
    sharp, _ = _place(tmp_path, match_sharpness=True, single_resample=True)
    assert plain is not None and sharp is not None
    x1, y1, x2, y2 = sharp[1]
    assert not np.array_equal(plain[0][y1:y2, x1:x2], sharp[0][y1:y2, x1:x2])
    outside = np.ones(bg.shape[:2], bool)
    outside[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = False
    assert np.array_equal(sharp[0][outside], bg[outside])
