"""`save_masks` (profile v11): the pasted object's mask goes into the COCO as RLE, off by default and altering nothing else."""
import asyncio

import numpy as np

from app.mask_rle import decode_rle, encode_rle
from app.models.schemas import EffectType, ObjectPlacement
from app.synth_profile import V10_PROFILE, V11_PROFILE, V11_TIGHT_OVERRIDES, SynthProfile, get_profile
from tests.test_synth_batch import make_plan, real_composer  # noqa: F401  (real_composer is a fixture)


def test_rle_roundtrip_including_first_pixel_set_and_empty():
    rng = np.random.RandomState(3)
    for m in (rng.rand(7, 11) > 0.5, np.zeros((4, 5), bool), np.ones((3, 6), bool)):
        rle = encode_rle(m.astype(np.uint8) * 255)
        assert rle["size"] == list(m.shape) and sum(rle["counts"]) == m.size
        assert np.array_equal(decode_rle(rle), m.astype(np.uint8))


def test_rle_is_compatible_with_pycocotools_when_available():
    mask_utils = __import__("pytest").importorskip("pycocotools.mask")
    m = (np.random.RandomState(5).rand(9, 13) > 0.4).astype(np.uint8)
    rle = mask_utils.frPyObjects(encode_rle(m), 9, 13)
    assert np.array_equal(mask_utils.decode(rle), m)


def test_save_masks_is_off_in_every_profile():
    for prof in (SynthProfile(), V10_PROFILE, V11_PROFILE, get_profile("v11", V11_TIGHT_OVERRIDES)):
        assert prof.save_masks is False


def _compose(comp, bg, crop, tmp, name, **kw):
    plan = make_plan(tmp, **kw)
    obj = ObjectPlacement(image_path=crop, class_name="Bottle", scale=1.0, rotation=0, position=(60, 40))
    return asyncio.run(comp.compose(bg, [obj], effects=[EffectType.COLOR_CORRECTION],
                                    output_path=str(tmp / "o" / f"{name}.jpg"), plan=plan))


def test_save_masks_adds_only_the_mask_and_it_matches_the_tight_box(real_composer):  # noqa: F811
    comp, bg, crop, tmp = real_composer
    off = _compose(comp, bg, crop, tmp, "off", tight_bbox=True)
    on = _compose(comp, bg, crop, tmp, "on", tight_bbox=True, save_masks=True)
    a0, a1 = off.annotations[0], on.annotations[0]
    assert a0.mask_rle is None
    assert a0.model_dump(exclude={"mask_rle"}) == a1.model_dump(exclude={"mask_rle"})   # mismas cajas y trazas
    m = decode_rle(a1.mask_rle)
    ys, xs = np.nonzero(m)
    assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == (a1.x, a1.y, a1.x + a1.width, a1.y + a1.height)
    assert (tmp / "o" / "off.jpg").read_bytes() == (tmp / "o" / "on.jpg").read_bytes()  # identical image
