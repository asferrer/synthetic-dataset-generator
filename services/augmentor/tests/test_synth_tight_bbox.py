"""Annotated box fitted to the mask (profile v11, `tight_bbox`) versus the recomputation from shadows and global effects."""
import asyncio

import cv2
import numpy as np
import pytest

from app.models.schemas import EffectsConfig, EffectType, ObjectPlacement
from app.synth_profile import V10_PROFILE, V11_PROFILE, V11_TIGHT_OVERRIDES, V11_VIS15_OVERRIDES, SynthProfile, get_profile
from tests.test_synth_batch import make_plan, real_composer  # noqa: F401  (real_composer es un fixture)


def _mask_rect(mask):
    ys, xs = np.nonzero(mask > 127)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _compose(comp, bg, crop, tmp, name, tight, effects, config=None):
    """Composes an object at a fixed position and returns (annotated box x1y1x2y2, mask box)."""
    plan = make_plan(tmp, tight_bbox=tight)
    captured = []
    orig = comp._place_object

    async def place(*args, **kwargs):
        res = await orig(*args, **kwargs)
        if res is not None:
            captured.append(_mask_rect(res[2]))
        return res

    comp._place_object = place
    obj = ObjectPlacement(image_path=crop, class_name="Bottle", scale=1.0, rotation=0, position=(60, 40))
    res = asyncio.run(comp.compose(bg, [obj], effects=effects, effects_config=config or EffectsConfig(),
                                   output_path=str(tmp / "o" / f"{name}.jpg"), plan=plan))
    a = res.annotations[0]
    return (a.x, a.y, a.x + a.width, a.y + a.height), captured[0]


# ------------------------------------------------------------------ perfiles
def test_tight_bbox_is_off_in_legacy_v10_v11_and_on_only_in_the_tight_overrides():
    for prof in (SynthProfile(), V10_PROFILE, V11_PROFILE, get_profile("v11", V11_VIS15_OVERRIDES)):
        assert prof.tight_bbox is False
    assert get_profile("v11", V11_TIGHT_OVERRIDES).tight_bbox is True
    assert {k: v for k, v in V11_TIGHT_OVERRIDES.items() if k != "tight_bbox"} == V11_VIS15_OVERRIDES


# ------------------------------------------------------------------ compositor
def test_global_colour_change_inflates_the_box_20px_per_side_and_tight_bbox_does_not(real_composer):  # noqa: F811
    comp, bg, crop, tmp = real_composer
    cfg = EffectsConfig(underwater_intensity=0.6)
    old, mask = _compose(comp, bg, crop, tmp, "old", False, [EffectType.UNDERWATER], cfg)
    new, mask2 = _compose(comp, bg, crop, tmp, "new", True, [EffectType.UNDERWATER], cfg)
    assert mask == mask2
    assert new == mask                                           # box = mask
    grew = [mask[0] - old[0], mask[1] - old[1], old[2] - mask[2], old[3] - mask[3]]
    assert all(g > 0 for g in grew) and max(grew) == 20          # the legacy behaviour is preserved


def test_tight_bbox_ignores_shadows_and_keeps_the_mask_box(real_composer):  # noqa: F811
    comp, bg, crop, tmp = real_composer
    cfg = EffectsConfig(shadow_opacity=0.5, shadow_blur=41)
    old, mask = _compose(comp, bg, crop, tmp, "s_old", False, [EffectType.SHADOWS], cfg)
    new, _ = _compose(comp, bg, crop, tmp, "s_new", True, [EffectType.SHADOWS], cfg)
    assert new == mask
    assert old != mask                                           # without the flag the shadow widens the box


def test_tight_bbox_without_effects_matches_the_legacy_box(real_composer):  # noqa: F811
    comp, bg, crop, tmp = real_composer
    old, mask = _compose(comp, bg, crop, tmp, "n_old", False, [EffectType.COLOR_CORRECTION])
    new, _ = _compose(comp, bg, crop, tmp, "n_new", True, [EffectType.COLOR_CORRECTION])
    assert old == new == mask
