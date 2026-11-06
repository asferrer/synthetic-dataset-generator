"""Import and pure-function tests for the app.composition package (no GPU, no services)."""

import importlib

import numpy as np
import pytest

MODULES = [
    "results", "geometry", "validation", "single_image", "batch", "batch_parallel", "output",
    "debug_viz", "object_placement", "transform", "scene_analysis", "effects", "core",
]
MIXINS = {
    "validation": "ValidationMixin", "single_image": "SingleImageMixin", "batch": "BatchMixin",
    "batch_parallel": "ParallelBatchMixin", "output": "OutputMixin", "debug_viz": "DebugVizMixin",
    "object_placement": "PlacementMixin", "transform": "TransformMixin",
    "scene_analysis": "SceneMixin", "effects": "EffectsMixin",
}


@pytest.mark.parametrize("name", MODULES)
def test_each_module_imports(name):
    assert importlib.import_module(f"app.composition.{name}") is not None


def test_mixins_do_not_overlap_and_are_inherited():
    from app.composition.core import ImageComposer
    seen = {}
    for name, cls in MIXINS.items():
        mixin = getattr(importlib.import_module(f"app.composition.{name}"), cls)
        assert issubclass(ImageComposer, mixin)
        for attr in vars(mixin):
            if attr.startswith("__"):
                continue
            assert attr not in seen, f"{attr} defined in both {seen[attr]} and {cls}"
            seen[attr] = cls
            assert hasattr(ImageComposer, attr)


def test_facade_keeps_public_api():
    from app import composer
    from app.composition import core, geometry, results
    assert composer.ImageComposer is core.ImageComposer
    assert composer.ComposeResult is results.ComposeResult
    assert composer.BatchResult is results.BatchResult
    assert composer.refine_object_mask is geometry.refine_object_mask
    assert composer.cv2.imread  # tests patch cv2 through the facade


def test_composer_instance_state_and_public_methods():
    from app.composer import ImageComposer
    comp = ImageComposer(use_gpu=False)
    for cache in ("_depth_cache", "_scene_cache", "_object_cache", "_matte_cache", "_medium_src_cache"):
        assert hasattr(comp, cache)
    for method in ("compose", "compose_batch", "compose_batch_parallel", "estimate_lighting"):
        assert callable(getattr(comp, method))


def test_compute_iou():
    from app.composition.geometry import compute_iou
    assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert compute_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert compute_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0


def test_check_mask_overlap():
    from app.composition.geometry import check_mask_overlap
    bg = np.zeros((20, 20), np.uint8)
    bg[:10, :10] = 255
    cand = np.full((5, 5), 255, np.uint8)
    assert check_mask_overlap(bg, cand, (2, 2)) is True
    assert check_mask_overlap(bg, cand, (12, 12)) is False
    assert check_mask_overlap(None, cand, (0, 0)) is False


def test_real_world_size_lookup_uses_the_given_config():
    from app.composition.geometry import REAL_WORLD_SIZES, get_real_world_size

    class Cfg:
        def get_size(self, name):
            return 1.5 if name == "x" else 0.1

    assert get_real_world_size("x", Cfg()) == 1.5
    assert REAL_WORLD_SIZES["default"] > 0


def test_blur_budget_scales_with_object_size_and_clamps():
    from app.composition.results import BlurBudget
    small = BlurBudget(max_blur_budget=10.0, object_size=10)
    big = BlurBudget(max_blur_budget=10.0, object_size=200)
    assert small._get_scaled_budget() == 5.0  # clamped at 0.5
    assert big._get_scaled_budget() == 20.0
    assert small.apply(8.0) == 5.0
    assert small.apply(1.0) == 0
    assert not small.can_apply(0.1)


def test_center_weight_is_none_without_stats():
    from app.composition.object_placement import PlacementMixin
    assert PlacementMixin._center_weight(None, "Bottle", (100, 100)) is None
