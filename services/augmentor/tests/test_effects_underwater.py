"""Unit tests for physically-based underwater effects."""

import numpy as np
import pytest

from app import effects_underwater as eu


@pytest.fixture
def gray_image():
    return np.full((60, 80, 3), 200, dtype=np.uint8)


@pytest.fixture
def gradient_depth():
    """Vertical gradient: 0 (near) at top, 1 (far) at bottom."""
    d = np.linspace(0, 1, 60, dtype=np.float32)[:, np.newaxis]
    return np.repeat(d, 80, axis=1)


class TestBeerLambert:
    def test_zero_depth_keeps_original(self, gray_image):
        depth = np.zeros((60, 80), dtype=np.float32)
        out = eu.apply_beer_lambert(gray_image, depth, strength=1.0)
        np.testing.assert_allclose(out, gray_image, atol=1)

    def test_far_field_approaches_water_color(self, gray_image):
        depth = np.ones((60, 80), dtype=np.float32)
        water = (180, 120, 30)
        out = eu.apply_beer_lambert(
            gray_image, depth, water_color=water, water_clarity="very_murky", strength=1.0
        )
        # Red channel attenuates hardest: should be closest to veil value
        mean_bgr = out.reshape(-1, 3).mean(axis=0)
        assert abs(mean_bgr[2] - water[2]) < abs(mean_bgr[0] - water[0]) + 60

    def test_red_attenuates_faster_than_blue(self, gray_image, gradient_depth):
        out = eu.apply_beer_lambert(
            gray_image, gradient_depth, water_color=(0, 0, 0), strength=1.0
        )
        far_row = out[-1].astype(int)
        # With black veil, remaining signal is pure transmission: R < G < B
        assert far_row[:, 2].mean() < far_row[:, 1].mean() < far_row[:, 0].mean()

    def test_monotonic_attenuation_with_depth(self, gray_image, gradient_depth):
        out = eu.apply_beer_lambert(
            gray_image, gradient_depth, water_color=(0, 0, 0), strength=1.0
        ).astype(int)
        near = out[5].mean()
        mid = out[30].mean()
        far = out[-5].mean()
        assert near > mid > far

    def test_strength_zero_is_identity(self, gray_image, gradient_depth):
        out = eu.apply_beer_lambert(gray_image, gradient_depth, strength=0.0)
        np.testing.assert_array_equal(out, gray_image)

    def test_depth_resized_when_resolution_differs(self, gray_image):
        small_depth = np.ones((6, 8), dtype=np.float32) * 0.5
        out = eu.apply_beer_lambert(gray_image, small_depth)
        assert out.shape == gray_image.shape


class TestCaustics:
    def test_seed_reproducible(self):
        p1 = eu.generate_caustics_pattern((40, 50), seed=123)
        p2 = eu.generate_caustics_pattern((40, 50), seed=123)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        p1 = eu.generate_caustics_pattern((40, 50), seed=1)
        p2 = eu.generate_caustics_pattern((40, 50), seed=2)
        assert not np.array_equal(p1, p2)

    def test_pattern_normalized(self):
        p = eu.generate_caustics_pattern((40, 50), seed=7)
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_apply_brightens_only(self, gray_image):
        out = eu.apply_caustics(gray_image, intensity=0.2, seed=3)
        assert (out.astype(int) >= gray_image.astype(int) - 1).all()

    def test_depth_aware_fades_far_field(self, gray_image, gradient_depth):
        out = eu.apply_caustics(
            gray_image, intensity=0.5, seed=3, depth_map=gradient_depth
        ).astype(int)
        near_boost = out[:10].mean() - 200
        far_boost = out[-10:].mean() - 200
        assert near_boost > far_boost

    def test_zero_intensity_is_identity(self, gray_image):
        out = eu.apply_caustics(gray_image, intensity=0.0, seed=3)
        np.testing.assert_allclose(out, gray_image, atol=1)


class TestOcclusion:
    def _mask_alpha(self, h=20, w=20):
        mask = np.full((h, w), 255, dtype=np.uint8)
        alpha = np.ones((h, w), dtype=np.float32)
        return mask, alpha

    def test_no_occluders_full_visibility(self):
        mask, alpha = self._mask_alpha()
        depth = np.ones((100, 100), dtype=np.float32)  # everything far
        m, a, vis = eu.occlude_object(mask, alpha, depth, (40, 40))
        assert vis == 1.0
        assert m.sum() == 255 * 400

    def test_closer_background_occludes(self):
        mask, alpha = self._mask_alpha()
        depth = np.full((100, 100), 0.8, dtype=np.float32)
        # Left half of the patch area is much closer than the object
        depth[:, :50] = 0.1
        m, a, vis = eu.occlude_object(mask, alpha, depth, (40, 40))
        assert 0.0 < vis < 1.0
        # Occluded pixels zeroed in both mask and alpha
        assert m[:, :10].sum() == 0
        assert a[:, :10].sum() == 0

    def test_visibility_ratio_matches_mask(self):
        mask, alpha = self._mask_alpha()
        depth = np.full((100, 100), 0.8, dtype=np.float32)
        depth[:, :50] = 0.1
        m, _, vis = eu.occlude_object(mask, alpha, depth, (40, 40))
        assert vis == pytest.approx((m > 0).sum() / 400)

    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        alpha = np.zeros((10, 10), dtype=np.float32)
        depth = np.ones((50, 50), dtype=np.float32)
        _, _, vis = eu.occlude_object(mask, alpha, depth, (5, 5))
        assert vis == 0.0

    def test_patch_outside_background(self):
        mask, alpha = self._mask_alpha()
        depth = np.ones((50, 50), dtype=np.float32)
        _, _, vis = eu.occlude_object(mask, alpha, depth, (200, 200))
        assert vis == 1.0
