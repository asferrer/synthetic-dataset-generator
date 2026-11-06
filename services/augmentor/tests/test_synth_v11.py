"""Profile v11: rectangular-crop filter, copy-paste onto real backgrounds and local harmonisation."""

import asyncio
import json

import cv2
import numpy as np
import pytest

from app import composer as composer_mod
from app import copy_paste, crop_filter, harmonize
from app.composer import ComposeResult, ImageComposer
from app.models.schemas import AnnotationBox
from app.synth_context import SynthContext
from app.synth_profile import SynthProfile, V10_PROFILE, V11_PROFILE, get_profile
from tests.synth_helpers import fake_annotation, make_stats_file


# ------------------------------------------------------------------ perfiles
def test_v11_is_v10_without_paste_augment_and_with_rect_filter():
    assert V11_PROFILE.name == "v11" and not V11_PROFILE.paste_augment and V11_PROFILE.crop_rect_filter
    assert not V11_PROFILE.copy_paste_real and not V11_PROFILE.harmonize
    same = {k: v for k, v in V11_PROFILE.model_dump().items()
            if k not in ("name", "paste_augment", "crop_rect_filter")}
    assert same == {k: v for k, v in V10_PROFILE.model_dump().items()
                    if k not in ("name", "paste_augment", "crop_rect_filter")}


def test_v10_and_legacy_do_not_activate_v11_features():
    assert V10_PROFILE.paste_augment and not V10_PROFILE.crop_rect_filter
    assert not V10_PROFILE.copy_paste_real and not V10_PROFILE.harmonize
    assert get_profile(None) is None and get_profile("v10") == V10_PROFILE
    assert get_profile("v11", {"copy_paste_real": True}).copy_paste_real


# ------------------------------------------------------------------ rectangular filter
def _ellipse_alpha(w=60, h=40):
    a = np.zeros((h, w), np.uint8)
    cv2.ellipse(a, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, 255, -1)
    return a


def test_rect_stats_separate_ellipse_from_rectangle():
    fill_e, sides_e = crop_filter.rect_stats(_ellipse_alpha())
    assert 0.6 < fill_e < 0.85 and sides_e == 0
    assert crop_filter.rect_stats(np.full((40, 60), 255, np.uint8)) == (1.0, 4)
    three = np.zeros((40, 60), np.uint8)
    three[:, :] = 255
    three[-1, :] = 0   # transparent bottom side
    fill, sides = crop_filter.rect_stats(three)
    assert sides == 3 and fill > 0.9


def test_is_rectangular_thresholds_and_sides_can_be_disabled():
    assert crop_filter.is_rectangular(0.95, 0, 0.9, 3)
    assert not crop_filter.is_rectangular(0.85, 2, 0.9, 3)
    assert crop_filter.is_rectangular(0.5, 3, 0.9, 3)
    assert not crop_filter.is_rectangular(0.5, 4, 0.9, None)


def _write_crop(path, alpha):
    img = np.zeros(alpha.shape + (4,), np.uint8)
    img[:, :, :3] = (40, 100, 160)
    img[:, :, 3] = alpha
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)


def _library(tmp_path):
    objs = {"Fishing_Net": [], "Tire": [], "Can": []}
    spec = {"Fishing_Net": ["rect", "ell", "ell"], "Tire": ["ell", "ell"], "Can": ["rect"]}
    for cls, kinds in spec.items():
        for i, kind in enumerate(kinds):
            p = tmp_path / cls / f"{i}.png"
            _write_crop(p, np.full((30, 50), 255, np.uint8) if kind == "rect" else _ellipse_alpha(50, 30))
            objs[cls].append({"path": str(p), "name": p.name})
    return objs


def test_filter_removes_rectangles_and_reports_per_class(tmp_path):
    objs = _library(tmp_path)
    rep = crop_filter.filter_rectangular(objs, 0.9, 3)
    assert rep["removed_by_class"] == {"Can": 1, "Fishing_Net": 1}
    assert rep["kept_by_class"] == {"Fishing_Net": 2, "Tire": 2}
    assert rep["total_removed"] == 2 and rep["total_kept"] == 4
    assert "Can" not in objs and len(objs["Fishing_Net"]) == 2   # an empty class disappears


def test_filter_keeps_unreadable_crops_and_is_inactive_by_default(tmp_path):
    objs = {"Tire": [{"path": str(tmp_path / "missing.png")}]}
    assert crop_filter.apply_library_filters(objs, SynthProfile()) is None
    assert crop_filter.apply_library_filters(objs, None) is None
    rep = crop_filter.apply_library_filters(objs, SynthProfile(crop_rect_filter=True))
    assert rep["unreadable_kept"] == 1 and len(objs["Tire"]) == 1


def test_threshold_is_configurable(tmp_path):
    objs = _library(tmp_path)
    rep = crop_filter.apply_library_filters(objs, SynthProfile(crop_rect_filter=True, crop_rect_max_fill=0.6,
                                                               crop_rect_min_sides=None))
    assert rep["removed_by_class"] == {"Can": 1, "Fishing_Net": 3, "Tire": 2}
    assert objs == {}


# ------------------------------------------------------------------ copy-paste: annotations
def _coco(tmp_path, boxes):
    coco = {"images": [{"id": 1, "file_name": "bg_0.jpg", "width": 100, "height": 80},
                       {"id": 2, "file_name": "bg_1.jpg", "width": 100, "height": 80}],
            "categories": [{"id": 0, "name": "Bottle"}, {"id": 1, "name": "Can"}],
            "annotations": [{"id": i, "image_id": 1, "category_id": c, "bbox": b, "iscrowd": 0}
                            for i, (c, b) in enumerate(boxes)]}
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(coco), encoding="utf8")
    return str(path)


def test_real_annotations_load_and_clip(tmp_path):
    real = copy_paste.RealAnnotations.load(_coco(tmp_path, [(0, [10, 10, 20, 20]), (1, [90, 70, 30, 30])]))
    assert real.get("bg_1.jpg") == [] and real.get("missing.jpg") == []
    boxes = copy_paste.real_annotation_boxes(real.get("bg_0.jpg"), (100, 80))
    assert [(b.class_name, b.x, b.y, b.width, b.height, b.origin) for b in boxes] == [
        ("Bottle", 10, 10, 20, 20, "real"), ("Can", 90, 70, 10, 10, "real")]
    assert copy_paste.real_annotation_boxes([{"x": 120, "y": 0, "w": 5, "h": 5, "cls": "Can"}], (100, 80)) == []
    with pytest.raises(FileNotFoundError):
        copy_paste.RealAnnotations.load(str(tmp_path / "nope.json"))


def test_merge_keeps_real_boxes_and_tags_pasted():
    real = [AnnotationBox(x=1, y=2, width=3, height=4, class_name="Can", area=12, origin="real")]
    pasted = [AnnotationBox(x=10, y=10, width=5, height=5, class_name="Bottle", area=25, source_object="/o/b.png")]
    merged = copy_paste.merge_annotations(real, pasted)
    assert [a.origin for a in merged] == ["real", "pasted"]
    assert merged[0] == real[0] and merged[1].source_object == "/o/b.png"
    assert pasted[0].origin is None   # the input is not modified


def test_overlap_limit_uses_the_smaller_box():
    real = [(0, 0, 100, 100)]
    assert copy_paste.overlap_with_real((200, 200, 220, 220), real) == 0.0
    assert copy_paste.overlap_with_real((10, 10, 30, 30), real) == 1.0       # small object inside the real box
    assert copy_paste.overlap_with_real((90, 0, 190, 100), real) == pytest.approx(0.1)
    assert not copy_paste.exceeds_real_overlap((90, 0, 190, 100), real, 0.1)
    assert copy_paste.exceeds_real_overlap((80, 0, 180, 100), real, 0.1)
    assert not copy_paste.exceeds_real_overlap((10, 10, 30, 30), [], 0.0)


def test_plan_keeps_resolution_and_original_jpeg_quality(tmp_path):
    bg = tmp_path / "bg_0.jpg"
    img = (np.random.default_rng(0).random((80, 100, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(bg), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    prof = SynthProfile(name="t", copy_paste_real=True, output_resolution=True, jpeg_recompress=True,
                        stats_path=make_stats_file(tmp_path, jpeg=[75.0] * 101))
    _coco(tmp_path, [(0, [10, 10, 20, 20])])
    ctx = SynthContext(prof, {"Bottle": [{"path": "/o/b.png"}]}, str(tmp_path), [str(bg)], base_seed=1)
    plan = ctx.make_plan(0, str(bg), (100, 80), ctx.make_rng(0))
    assert plan.output_size is None and 78 <= plan.jpeg_quality <= 82
    assert plan.real_boxes == [(10, 10, 30, 30)]


# ------------------------------------------------------------------ copy-paste: full batch (simulated compose)
@pytest.fixture
def batch_dirs(tmp_path):
    bg = tmp_path / "real"
    bg.mkdir()
    for i in range(2):
        cv2.imwrite(str(bg / f"bg_{i}.jpg"), np.zeros((80, 100, 3), np.uint8))
    _coco(bg, [(0, [10, 10, 20, 20])])
    objs = _library(tmp_path / "objs")
    return str(bg), str(tmp_path / "objs"), str(tmp_path / "out"), objs


def _composer(monkeypatch):
    comp = ImageComposer(use_gpu=False)

    async def fake_depth(bg_image, cache_key=None):
        return np.zeros((4, 4), dtype=np.float32), "fallback_gradient"

    async def fake_compose(objects, **kwargs):
        plan = kwargs["plan"]
        anns = [fake_annotation(o.class_name, o.image_path) for o in objects]
        anns = [AnnotationBox(x=50, y=50, width=4, height=4, class_name=a.class_name, area=16,
                              source_object=a.source_object) for a in anns]
        real = copy_paste.real_annotation_boxes(plan.real_annotations, (100, 80)) \
            if plan.profile.copy_paste_real else []
        return ComposeResult(annotations=anns, objects_placed=len(anns), image_size=(100, 80), real_annotations=real)

    async def fake_validate(*args, **kwargs):
        return True, None

    monkeypatch.setattr(comp, "_get_depth_map", fake_depth)
    monkeypatch.setattr(comp, "compose", fake_compose)
    monkeypatch.setattr(comp, "_validate_composed", fake_validate)
    return comp


def _run(comp, dirs, prof, **kw):
    bg, obj, out, _ = dirs
    return asyncio.run(comp.compose_batch(
        num_images=3, backgrounds_dir=bg, objects_dir=obj, output_dir=out, targets_per_class={"Tire": 3, "Fishing_Net": 3},
        max_objects_per_image=2, synth_profile=prof, seed=5, **kw))


def _prof(tmp_path, **kw):
    return SynthProfile(name="t", per_image_seed=True, background_round_robin=True, trace=True,
                        stats_path=make_stats_file(tmp_path), **kw)


def test_batch_copy_paste_writes_real_and_pasted_boxes(monkeypatch, batch_dirs, tmp_path):
    comp = _composer(monkeypatch)
    res = _run(comp, batch_dirs, _prof(tmp_path, copy_paste_real=True))
    coco = json.load(open(res.output_coco_path, encoding="utf8"))
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for im in coco["images"]:
        anns = by_img[im["id"]]
        origins = [a["origin"] for a in anns]
        has_real = im["background"] == "bg_0.jpg"
        assert origins.count("real") == (1 if has_real else 0) and origins.count("pasted") >= 1
        assert all(("source_object" in a) == (a["origin"] == "pasted") for a in anns)
    assert sum(res.synthetic_counts.values()) == sum(a["origin"] == "pasted" for a in coco["annotations"])


def test_batch_without_copy_paste_has_no_origin_field(monkeypatch, batch_dirs, tmp_path):
    res = _run(_composer(monkeypatch), batch_dirs, _prof(tmp_path))
    coco = json.load(open(res.output_coco_path, encoding="utf8"))
    assert coco["annotations"] and all("origin" not in a for a in coco["annotations"])


def test_batch_rect_filter_report_in_coco_and_file(monkeypatch, batch_dirs, tmp_path):
    res = _run(_composer(monkeypatch), batch_dirs, _prof(tmp_path, crop_rect_filter=True))
    coco = json.load(open(res.output_coco_path, encoding="utf8"))
    rep = coco["info"]["synth_generator"]["rect_filter"]
    assert rep["removed_by_class"] == {"Can": 1, "Fishing_Net": 1}
    out = json.load(open(res.output_coco_path.replace("synthetic_dataset.json", "rect_filter_report.json")))
    assert out == rep
    assert all(a["source_object"].count("Can") == 0 for a in coco["annotations"])


def test_parallel_batch_rejects_copy_paste(batch_dirs, tmp_path):
    bg, obj, out, _ = batch_dirs
    with pytest.raises(ValueError, match="sequential"):
        asyncio.run(ImageComposer(use_gpu=False).compose_batch_parallel(
            num_images=1, backgrounds_dir=bg, objects_dir=obj, output_dir=out,
            synth_profile=_prof(tmp_path, copy_paste_real=True)))


# ------------------------------------------------------------------ harmonisation
def _scene(env_bgr=(160, 110, 40)):
    comp = np.zeros((120, 160, 3), np.uint8)
    comp[:] = env_bgr
    rng = np.random.default_rng(0)
    comp = np.clip(comp.astype(np.int16) + rng.integers(-12, 12, comp.shape), 0, 255).astype(np.uint8)
    return comp


def _lab_mean(img, mask=None):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[mask] .mean(axis=0) if mask is not None else lab.reshape(-1, 3).mean(axis=0)


def test_harmonize_moves_object_towards_ring_statistics():
    comp = _scene()
    box = (60, 40, 30, 30)
    obj = np.full((30, 30, 3), (30, 60, 200), np.uint8)   # reddish object in blue water
    mask = np.full((30, 30), 255, np.uint8)
    ring = harmonize.ring_mask(comp.shape[:2], box, 0.5)
    target = _lab_mean(comp, ring)
    before = np.abs(_lab_mean(obj) - target)
    out = harmonize.harmonize_object(obj, mask, comp, box, strength=0.8)
    after = np.abs(_lab_mean(out) - target)
    assert after[1] < before[1] and after[2] < before[2]
    assert out.shape == obj.shape and out.dtype == np.uint8


def test_harmonize_strength_zero_and_missing_ring_are_noops():
    comp = _scene()
    obj = np.full((30, 30, 3), (30, 60, 200), np.uint8)
    mask = np.full((30, 30), 255, np.uint8)
    assert harmonize.harmonize_object(obj, mask, comp, (60, 40, 30, 30), strength=0.0) is obj
    tiny = np.zeros((6, 6, 3), np.uint8)
    full_box = (0, 0, 6, 6)   # the box covers the whole background: no ring
    assert harmonize.harmonize_object(obj, mask, tiny, full_box, strength=1.0) is obj


def test_harmonize_intensity_is_monotonic():
    comp = _scene()
    box = (60, 40, 30, 30)
    obj = np.full((30, 30, 3), (30, 60, 200), np.uint8)
    mask = np.full((30, 30), 255, np.uint8)
    target = _lab_mean(comp, harmonize.ring_mask(comp.shape[:2], box, 0.5))
    dist = [np.abs(_lab_mean(harmonize.harmonize_object(obj, mask, comp, box, s)) - target)[1:].sum()
            for s in (0.0, 0.3, 0.6, 1.0)]
    assert dist[0] > dist[1] > dist[2] > dist[3]


def test_ring_excludes_the_box_and_is_clipped_to_the_image():
    ring = harmonize.ring_mask((50, 50), (0, 0, 20, 20), 0.5)
    assert not ring[:20, :20].any() and ring[:30, :30].sum() == 30 * 30 - 20 * 20
    assert ring.shape == (50, 50)


def test_composer_module_exposes_new_helpers():
    assert composer_mod.copy_paste is copy_paste and composer_mod.harmonize is harmonize
