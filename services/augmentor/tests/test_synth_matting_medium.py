"""Profile v11: `medium_transfer` and `edge_matting` flags (off by default) and their wiring in the composer."""

import asyncio

import cv2
import numpy as np
import pytest

from app.composer import ImageComposer, refine_object_mask
from app.models.schemas import EffectsConfig, ObjectPlacement
from app.synth_context import ImagePlan
from app.synth_profile import SynthProfile, V10_PROFILE, V11_PROFILE, V11_Q_LAB_SAME_OVERRIDES, get_profile

OBJ, SRC_BG, DST_BG = (40, 90, 200), (190, 150, 60), (60, 140, 90)    # BGR


# ------------------------------------------------------------------ profile
def test_new_flags_are_off_in_every_named_profile():
    for p in (SynthProfile(), V10_PROFILE, V11_PROFILE, SynthProfile(**{**V11_Q_LAB_SAME_OVERRIDES, "name": "q"})):
        assert p.medium_transfer is None and not p.edge_matting and not p.medium_dark_floor
    assert get_profile("v11", {"edge_matting": True}).edge_matting
    assert get_profile("v11", {"medium_transfer": "ring_hist", "medium_lambda": 0.5}).medium_lambda == 0.5
    assert get_profile("v11") == V11_PROFILE


def test_edge_matting_defaults_are_the_best_measured_in_gap_ladder():
    p = SynthProfile(name="t", edge_matting=True)
    assert (p.edge_matting_alpha, p.edge_matting_foreground, p.edge_matting_a0) == ("closed_form", "none", 0.0)
    assert p.edge_matting_edge_smoothing and p.medium_source == "crop"


@pytest.mark.parametrize("kw", [{"medium_transfer": "x"}, {"medium_lambda": 0.0}, {"medium_lambda": 1.5},
                                {"edge_matting_alpha": "x"}, {"edge_matting_foreground": "x"}, {"edge_matting_a0": 0.6},
                                {"medium_source": "tabla"}])
def test_invalid_values_are_rejected(kw):
    with pytest.raises(ValueError):
        SynthProfile(**kw)


# ------------------------------------------------------------------ matting alpha without smoothing
def test_refine_object_mask_keeps_fractional_alpha_when_asked():
    a = np.zeros((30, 30), np.uint8)
    a[8:22, 8:22] = 255
    a[7, 8:22] = a[22, 8:22] = 120
    a[10, 10] = 3                                      # <0.05: zeroed as usual
    a[23, 8:22] = np.linspace(30, 90, 14).astype(np.uint8)   # >5 levels: the already-gradual alpha path
    mask = (a > 10).astype(np.uint8) * 255
    raw, _ = refine_object_mask(a, mask, smooth_gradual=False)
    blurred, _ = refine_object_mask(a, mask)
    assert raw[7, 10] == pytest.approx(120 / 255.0) and raw[10, 10] == 0 and not np.allclose(raw, blurred)
    assert np.array_equal(blurred, refine_object_mask(a, mask, smooth_gradual=True)[0])


# ------------------------------------------------------------------ cableado en _place_object
def _crop(path, soft=2.0, size=(60, 80), r=20):
    """OBJ disc over SRC_BG with a soft edge; binary alpha. Returns the true alpha."""
    m = np.zeros(size, np.float32)
    cv2.circle(m, (size[1] // 2, size[0] // 2), r, 1.0, -1)
    alpha = cv2.GaussianBlur(m, (0, 0), soft)
    img = alpha[..., None] * np.array(OBJ, np.float64) + (1 - alpha[..., None]) * np.array(SRC_BG, np.float64)
    cv2.imwrite(str(path), np.dstack([np.rint(img).astype(np.uint8), ((alpha > 0.5) * 255).astype(np.uint8)]))
    return alpha


def _place(path, prof, pos, bg=None, cls="Bottle"):
    bg = np.full((200, 240, 3), DST_BG, np.uint8) if bg is None else bg
    res = asyncio.run(ImageComposer(use_gpu=False)._place_object(
        composite=bg.copy(), obj_placement=ObjectPlacement(image_path=str(path), class_name=cls, scale=1.0, rotation=0.0,
                                                           position=pos),
        depth_map=None, lighting_info=None, effects=[], effects_config=EffectsConfig(),
        placement_mask=np.zeros(bg.shape[:2], np.uint8), existing_bboxes=[], plan=ImagePlan(profile=prof, stats=None)))
    assert res is not None
    return res[0]


def test_edge_matting_removes_the_source_halo_from_the_composite(tmp_path):
    pytest.importorskip("pymatting")
    path = tmp_path / "Bottle" / "1.png"
    path.parent.mkdir()
    alpha = _crop(path)
    pos = (70, 60)
    base = _place(path, SynthProfile(name="t", edge_matting=False, harmonize=False), pos)
    prof = SynthProfile(name="t", edge_matting=True, edge_matting_foreground="germer", edge_matting_edge_smoothing=False)
    matte = _place(path, prof, pos)
    ideal = np.full_like(base, DST_BG, dtype=np.float64)
    h, w = alpha.shape
    ideal[pos[1]:pos[1] + h, pos[0]:pos[0] + w] = (alpha[..., None] * np.array(OBJ) + (1 - alpha[..., None]) * np.array(DST_BG))
    band = np.zeros(base.shape[:2], bool)
    band[pos[1]:pos[1] + h, pos[0]:pos[0] + w] = (alpha > 0.05) & (alpha < 0.95)
    err_base = np.abs(base.astype(float) - ideal)[band].mean()
    err_matte = np.abs(matte.astype(float) - ideal)[band].mean()
    assert err_matte < 0.85 * err_base


def test_edge_matting_caches_the_matte_per_crop(tmp_path):
    pytest.importorskip("pymatting")
    path = tmp_path / "Bottle" / "1.png"
    path.parent.mkdir()
    _crop(path)
    comp = ImageComposer(use_gpu=False)
    prof = SynthProfile(name="t", edge_matting=True)
    bg = np.full((200, 240, 3), DST_BG, np.uint8)
    for _ in range(2):
        asyncio.run(comp._place_object(
            composite=bg.copy(), obj_placement=ObjectPlacement(image_path=str(path), class_name="Bottle", scale=1.0,
                                                               rotation=0.0, position=(70, 60)),
            depth_map=None, lighting_info=None, effects=[], effects_config=EffectsConfig(),
            placement_mask=np.zeros(bg.shape[:2], np.uint8), existing_bboxes=[], plan=ImagePlan(profile=prof, stats=None)))
    assert len(comp._matte_cache) == 1


def test_medium_transfer_moves_the_object_towards_the_destination_medium(tmp_path):
    path = tmp_path / "Bottle" / "1.png"
    path.parent.mkdir()
    _crop(path, soft=0.5)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    pos = (70, 60)
    sel = (img[:, :, 3] > 200)
    sel = cv2.erode(sel.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

    def core_color(prof):
        out = _place(path, prof, pos)
        return out[pos[1]:pos[1] + sel.shape[0], pos[0]:pos[0] + sel.shape[1]][sel].mean(axis=0)

    off = core_color(SynthProfile(name="t"))
    on = core_color(SynthProfile(name="t", medium_transfer="ring_gain"))
    half = core_color(SynthProfile(name="t", medium_transfer="ring_gain", medium_lambda=0.5))
    hist = core_color(SynthProfile(name="t", medium_transfer="ring_hist"))
    assert np.allclose(off, OBJ, atol=3)
    # the source was blue (190) and the destination is redder (90 vs 60): blue goes down, red goes up (gains clipped to [0.5, 2])
    assert on[0] < off[0] - 8 and on[2] > off[2] + 20
    assert np.all(np.minimum(on, off) - 2 <= half) and np.all(half <= np.maximum(on, off) + 2)
    assert hist[0] < off[0] - 5


def test_medium_transfer_is_identity_without_a_measurable_source_ring(tmp_path):
    path = tmp_path / "Bottle" / "1.png"
    path.parent.mkdir()
    img = np.zeros((40, 40, 4), np.uint8)
    img[:, :, :3] = OBJ
    img[:, :, 3] = 255                                  # the crop fills its box: no source ring
    cv2.imwrite(str(path), img)
    off = _place(path, SynthProfile(name="t"), (70, 60))
    on = _place(path, SynthProfile(name="t", medium_transfer="ring_gain", medium_dark_floor=True), (70, 60))
    assert np.array_equal(off, on)


def test_flags_off_leave_the_composite_bit_identical(tmp_path):
    path = tmp_path / "Bottle" / "1.png"
    path.parent.mkdir()
    _crop(path)
    a = _place(path, SynthProfile(name="t", harmonize=True), (70, 60))
    b = _place(path, SynthProfile(name="t", harmonize=True, medium_transfer=None, edge_matting=False), (70, 60))
    assert np.array_equal(a, b)
