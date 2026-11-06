"""Trimap matting and foreground decontamination (M2; app/edge_matting.py). Needs `pymatting`."""

import cv2
import numpy as np
import pytest

from app import edge_matting as em

pm = pytest.importorskip("pymatting")

OBJ, BG = np.array([40, 90, 200], np.float64), np.array([190, 150, 60], np.float64)   # BGR


def _scene(size=(80, 100), r=26, soft=2.0):
    """OBJ-coloured disc over BG with a soft edge (true alpha) -> (BGRA with binary alpha, true alpha)."""
    m = np.zeros(size, np.float32)
    cv2.circle(m, (size[1] // 2, size[0] // 2), r, 1.0, -1)
    alpha = cv2.GaussianBlur(m, (0, 0), soft)
    img = alpha[..., None] * OBJ + (1 - alpha[..., None]) * BG
    bgra = np.dstack([np.rint(img).astype(np.uint8), ((alpha > 0.5) * 255).astype(np.uint8)])
    return bgra, alpha


def test_band_width_and_pad():
    assert em.band_width(10) == 2 and em.band_width(2000) == 3 and em.band_width(10 ** 6) == 8
    assert em.pad_for("Bottle", 2000) == 4 and em.pad_for("Fishing_Net", 2000) == 7


def test_trimap_bands_for_solids_and_transparents():
    m = np.zeros((60, 60), np.uint8)
    cv2.circle(m, (30, 30), 20, 1, -1)
    t = em.make_trimap(m, "Bottle")
    e = em.band_width(float(m.sum()))
    d_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    assert (t[d_in > e + 1] == 1).all() and (t[d_out > e + 1] == 0).all()
    assert (t == 0.5).sum() > 0 and set(np.unique(t)) == {0.0, 0.5, 1.0}
    tn = em.make_trimap(m, "Fishing_Net")
    assert (tn == 0.5).sum() > (t == 0.5).sum() and (tn[d_out > em.TRANSPARENT_DILATE + 1] == 0).all()
    assert (tn[d_in > 2.5] == 1).all()


def test_trimap_of_a_thin_object_keeps_some_foreground():
    m = np.zeros((40, 80), np.uint8)
    m[18:22, 5:75] = 1                       # 4 px thick: the erosion by e eats everything
    t = em.make_trimap(m, "Can")
    assert (t == 1).any() and (t == 0).any()


def test_matte_keeps_the_crop_size_by_default_and_pads_on_request():
    bgra, _ = _scene()
    out, pad = em.matte_bgra(bgra, "Bottle")
    assert pad == 0 and out.shape == bgra.shape and out.dtype == np.uint8
    p = em.pad_for("Bottle", float((bgra[:, :, 3] > 10).sum()))
    outp, padp = em.matte_bgra(bgra, "Bottle", pad=p)
    assert padp == p and outp.shape == (bgra.shape[0] + 2 * p, bgra.shape[1] + 2 * p, 4)
    with pytest.raises(ValueError, match="alpha"):
        em.estimate_alpha(np.zeros((10, 10, 3)), np.zeros((10, 10), np.float32), "x")
    with pytest.raises(ValueError, match="foreground"):
        em.estimate_fg(np.zeros((10, 10, 3)), np.zeros((10, 10)), "x")


def test_matte_recovers_fractional_alpha_better_than_the_binary_mask():
    bgra, alpha_true = _scene()
    out, pad = em.matte_bgra(bgra, "Bottle", fg_method="none")
    a = out[:, :, 3] / 255.0
    band = (alpha_true > 0.05) & (alpha_true < 0.95)
    assert band.sum() > 100
    err_matte = np.abs(a - alpha_true)[band].mean()
    err_binary = np.abs((bgra[:, :, 3] > 10) - alpha_true)[band].mean()
    assert err_matte < 0.5 * err_binary
    assert ((a > 0.02) & (a < 0.98)).sum() > 50                  # alpha is no longer binary


def test_foreground_estimation_removes_the_background_halo_of_the_edge():
    bgra, alpha_true = _scene()
    out, pad = em.matte_bgra(bgra, "Bottle", fg_method="germer")
    f = out[:, :, :3].astype(np.float64)
    band = (alpha_true > 0.3) & (alpha_true < 0.9)
    err_f = np.abs(f - OBJ)[band].mean()
    err_orig = np.abs(bgra[:, :, :3].astype(np.float64) - OBJ)[band].mean()
    assert err_f < 0.5 * err_orig                              # F is much closer to the object's colour
    out_cf, _ = em.matte_bgra(bgra, "Bottle", fg_method="closed_form")
    assert np.abs(out_cf[:, :, :3].astype(np.float64) - OBJ)[band].mean() < err_orig


def test_foreground_none_keeps_the_original_pixels():
    bgra, _ = _scene()
    out, pad = em.matte_bgra(bgra, "Bottle", fg_method="none")
    assert np.array_equal(out[:, :, :3], bgra[:, :, :3])


def test_a0_clips_the_weak_halo():
    bgra, _ = _scene()
    out0, _ = em.matte_bgra(bgra, "Bottle", a0=0.0)
    out1, _ = em.matte_bgra(bgra, "Bottle", a0=0.2)
    a0, a1 = out0[:, :, 3].astype(float), out1[:, :, 3].astype(float)
    assert (a1 > 0).sum() < (a0 > 0).sum() and a1.max() == 255
    weak = (a0 > 0) & (a0 / 255.0 <= 0.2)
    assert (a1[weak] <= 1).all()


def test_tiny_masks_are_returned_with_the_padding_only():
    bgra = np.zeros((10, 10, 4), np.uint8)
    bgra[4:6, 4:6] = (10, 20, 30, 255)
    out, pad = em.matte_bgra(bgra, "Bottle", pad=3)
    assert out.shape[:2] == (16, 16) and set(np.unique(out[:, :, 3])) <= {0, 255}


def test_alpha_methods_run_and_stay_in_range():
    bgra, _ = _scene()
    for method in em.ALPHA_METHODS:
        out, pad = em.matte_bgra(bgra, "Bottle", alpha_method=method)
        assert out[:, :, 3].max() >= 250 and out[:, :, 3].min() == 0


def test_missing_pymatting_raises_a_clear_error(monkeypatch):
    def boom():
        raise ImportError("edge_matting needs the `pymatting` package")
    monkeypatch.setattr(em, "_pm", boom)
    bgra, _ = _scene()
    with pytest.raises(ImportError, match="pymatting"):
        em.matte_bgra(bgra, "Bottle")


def test_replicated_padding_does_not_open_alpha_far_outside_the_crop():
    bgra, _ = _scene()
    out, pad = em.matte_bgra(bgra, "Bottle", pad=5)
    assert pad == 5 and out[:pad, :, 3].max() <= 255 * 0.5   # outside the crop the alpha can only be the matte's soft edge
    assert np.array_equal(out[pad:-pad, pad:-pad].shape[:2], bgra.shape[:2])


def test_default_matte_never_leaves_the_crop_box():
    bgra, _ = _scene()
    bgra[:, :, 3] = 0
    cv2.circle(bgra, (50, 40), 26, 255, -1)
    cv2.rectangle(bgra, (0, 0), (30, 79), (0, 0, 0, 0), -1)           # one side of the box without object
    out, pad = em.matte_bgra(bgra, "Bottle")
    assert pad == 0 and out.shape == bgra.shape
