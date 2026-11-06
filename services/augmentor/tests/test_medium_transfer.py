"""Ring-to-ring medium transfer in linear RGB (M1; app/medium_transfer.py)."""

import cv2
import numpy as np
import pytest

from app import medium_transfer as mt


def _scene(bgr, size=(120, 160), noise=0.0, seed=0):
    img = np.zeros(size + (3,), np.float32)
    img[:] = bgr
    if noise:
        img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _disc_mask(shape=(120, 160), center=(80, 60), r=18):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, center, r, 1, -1)
    return m


def test_srgb_linear_round_trip_is_within_one_level():
    u = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    assert np.abs(mt.to_srgb_u8(mt.to_linear(u)).astype(int) - u.astype(int)).max() <= 1
    assert mt.to_linear(np.array([[[0, 255, 128]]], np.uint8)).dtype == np.float32


def test_ring_region_skips_the_halo_and_the_excluded_boxes():
    m = _disc_mask()
    ring = mt.ring_region(m, r_in=3, r_out=10)
    assert not (ring & (m > 0)).any()
    d = cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, 3)
    assert d[ring].min() > 3 and d[ring].max() <= 10
    ex = np.zeros(m.shape, bool)
    ex[:, 85:] = True
    assert not (mt.ring_region(m, ex, 3, 10) & ex).any()
    assert mt.ring_region(m, ex, 3, 10).sum() < ring.sum()


def test_ring_outer_radius_is_clipped():
    assert mt.ring_outer_px(4) == 12.0 and mt.ring_outer_px(10 ** 6) == 40.0
    assert mt.ring_outer_px(2000) == pytest.approx(0.5 * np.sqrt(2000))


def test_medium_stats_none_for_tiny_ring_and_empty_mask():
    img = _scene((100, 120, 90))
    assert mt.medium_stats(img, np.zeros(img.shape[:2], np.uint8)) is None
    full = np.ones(img.shape[:2], np.uint8)
    assert mt.medium_stats(img, full) is None          # no ring left outside the mask
    st = mt.medium_stats(img, _disc_mask())
    assert st is not None and st.pixels.shape[1] == 3 and len(st.pixels) >= mt.MIN_RING_PIXELS


def test_dark_floor_is_the_low_percentile_in_linear():
    img = _scene((100, 100, 100))
    img[:2, :] = (10, 20, 30)
    f = mt.dark_floor(img, percentile=1.0)
    assert np.allclose(f, mt.to_linear(np.array([[[10, 20, 30]]], np.uint8))[0, 0], atol=0.002)


def test_identity_when_origin_and_destination_have_the_same_medium():
    img = _scene((140, 110, 60), noise=3)
    m = _disc_mask()
    st = mt.medium_stats(img, m)
    obj = _scene((50, 200, 120), size=(40, 40), noise=4)
    for mode, floor in (("ring_gain", False), ("ring_gain", True), ("ring_hist", False)):
        out = mt.transfer(obj, st, st, mode, 1.0, floor)
        assert np.abs(out.astype(int) - obj.astype(int)).max() <= 2, (mode, floor)


def test_ring_gain_matches_the_median_ratio_per_channel():
    src = mt.medium_stats(_scene((120, 80, 40)), _disc_mask())
    dst = mt.medium_stats(_scene((60, 100, 80)), _disc_mask())
    obj = np.full((20, 20, 3), 128, np.uint8)
    out = mt.transfer(obj, src, dst, "ring_gain", 1.0, False)
    g, _, _ = mt.channel_gains(src, dst, False)
    expect = mt.to_srgb_u8(g * mt.to_linear(obj))
    assert np.abs(out.astype(int) - expect.astype(int)).max() <= 1
    assert out[0, 0, 0] < 128 and out[0, 0, 1] > 128          # blue goes down, green goes up


def test_gain_is_clipped_and_dead_channels_use_one():
    lo = mt.MediumStats(np.tile([0.5, 0.5, 0.5], (50, 1)).astype(np.float32), np.zeros(3, np.float32))
    hi = mt.MediumStats(np.tile([0.1, 0.9, 0.5], (50, 1)).astype(np.float32), np.zeros(3, np.float32))
    g, _, _ = mt.channel_gains(lo, hi, False)
    assert np.allclose(g, [0.5, 1.8, 1.0], atol=1e-5)
    dead = mt.MediumStats(np.tile([0.001, 0.5, 0.5], (50, 1)).astype(np.float32), np.zeros(3, np.float32))
    assert mt.channel_gains(dead, hi, False)[0][0] == 1.0
    small = mt.MediumStats(np.tile([0.2, 0.5, 0.5], (50, 1)).astype(np.float32), np.zeros(3, np.float32))
    big = mt.MediumStats(np.tile([0.9, 0.5, 0.5], (50, 1)).astype(np.float32), np.zeros(3, np.float32))
    assert mt.channel_gains(small, big, False)[0][0] == 2.0


def test_dark_floor_shifts_the_black_level():
    f_s, f_d = np.array([0.02, 0.02, 0.02], np.float32), np.array([0.10, 0.10, 0.10], np.float32)
    src = mt.MediumStats(np.tile([0.3, 0.3, 0.3], (50, 1)).astype(np.float32), f_s)
    dst = mt.MediumStats(np.tile([0.3, 0.3, 0.3], (50, 1)).astype(np.float32), f_d)
    black = np.zeros((10, 10, 3), np.uint8)
    out = mt.transfer(black, src, dst, "ring_gain", 1.0, True)
    assert out.mean() > mt.transfer(black, src, dst, "ring_gain", 1.0, False).mean() + 5   # without a floor, black stays black
    assert mt.transfer(black, src, dst, "ring_gain", 1.0, False).max() == 0


def test_lambda_mixes_linearly_and_zero_is_identity():
    src = mt.medium_stats(_scene((120, 80, 40)), _disc_mask())
    dst = mt.medium_stats(_scene((60, 100, 80)), _disc_mask())
    obj = np.full((8, 8, 3), 150, np.uint8)
    full = mt.to_linear(mt.transfer(obj, src, dst, "ring_gain", 1.0, False))
    half = mt.to_linear(mt.transfer(obj, src, dst, "ring_gain", 0.5, False))
    assert np.allclose(half, 0.5 * (mt.to_linear(obj) + full), atol=0.01)
    assert np.array_equal(mt.transfer(obj, src, dst, "ring_gain", 0.0, False), obj)


def test_missing_stats_leave_the_object_untouched_and_unknown_mode_fails():
    st = mt.medium_stats(_scene((120, 80, 40)), _disc_mask())
    obj = np.full((8, 8, 3), 99, np.uint8)
    assert np.array_equal(mt.transfer(obj, None, st), obj) and np.array_equal(mt.transfer(obj, st, None), obj)
    with pytest.raises(ValueError, match="mode"):
        mt.transfer(obj, st, st, "other")


def test_ring_hist_maps_the_source_ring_distribution_onto_the_destination_one():
    rng = np.random.default_rng(1)
    src_px = np.clip(rng.normal(0.2, 0.03, (4000, 3)), 0.01, 1).astype(np.float32)
    dst_px = np.clip(rng.normal(0.5, 0.08, (4000, 3)), 0.01, 1).astype(np.float32)
    src, dst = mt.MediumStats(src_px, np.zeros(3, np.float32)), mt.MediumStats(dst_px, np.zeros(3, np.float32))
    obj = mt.to_srgb_u8(src_px[:1024].reshape(32, 32, 3))
    out = mt.to_linear(mt.transfer(obj, src, dst, "ring_hist", 1.0))
    assert abs(out.mean() - dst_px.mean()) < 0.03 and abs(out.std() - dst_px.std()) < 0.03
    # the map is monotone per channel: pixel order is preserved
    a, b = mt.to_linear(obj)[..., 0].ravel(), out[..., 0].ravel()
    order = np.argsort(a, kind="stable")
    assert np.all(np.diff(b[order]) >= -0.01)


def test_hist_map_handles_a_constant_source():
    qs, qd = mt.hist_map(np.full(100, 0.3, np.float32), np.linspace(0.2, 0.6, 100).astype(np.float32))
    out = mt.apply_hist_map(np.array([0.1, 0.3, 0.9]), qs, qd)
    assert np.all(np.isfinite(out)) and out[1] == pytest.approx(0.3 + (np.median(qd) - 0.3), abs=0.05)


def test_ring_excludes_other_boxes_in_stats():
    img = _scene((100, 100, 100))
    img[:, 95:] = (250, 250, 250)                 # another object on the right
    m = _disc_mask()
    ex = np.zeros(m.shape, bool)
    ex[:, 95:] = True
    with_ex, without = mt.medium_stats(img, m, ex), mt.medium_stats(img, m)
    assert with_ex.pixels.max() < without.pixels.max()
