"""Batch loops must stop once the per-class instance targets are met.

Regression: compose_batch_parallel kept iterating towards num_images (counted
in images) after every class had its instances, requesting a depth map per
iteration until the 5 x num_images safety limit.
"""

import asyncio
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from app import composer as composer_mod
from app.composer import ComposeResult, ImageComposer

CLASSES = ["Bottle", "Can", "Tire"]


def fake_annotation(class_name):
    return SimpleNamespace(class_name=class_name, x=0, y=0, width=4, height=4,
                           area=16, source_object=None)


@pytest.fixture
def dirs(tmp_path):
    bg_dir = tmp_path / "bg"
    bg_dir.mkdir()
    for i in range(3):
        (bg_dir / f"bg_{i}.jpg").write_bytes(b"x")
    obj_dir = tmp_path / "obj"
    for cls in CLASSES:
        (obj_dir / cls).mkdir(parents=True)
        (obj_dir / cls / "crop_0.png").write_bytes(b"x")
    out_dir = tmp_path / "out"
    return str(bg_dir), str(obj_dir), str(out_dir)


@pytest.fixture
def composer(monkeypatch):
    comp = ImageComposer(use_gpu=False)
    calls = {"depth": 0, "compose": 0}

    async def fake_depth(bg_image, cache_key=None):
        calls["depth"] += 1
        return np.zeros((4, 4), dtype=np.float32), "fallback_gradient"

    async def fake_compose(objects, **kwargs):
        calls["compose"] += 1
        if "on_compose" in calls:
            calls["on_compose"](calls["compose"])
        anns = [fake_annotation(o.class_name) for o in objects]
        return ComposeResult(annotations=anns, objects_placed=len(anns))

    async def fake_validate(*args, **kwargs):
        return True, None

    monkeypatch.setattr(comp, "_get_depth_map", fake_depth)
    monkeypatch.setattr(comp, "compose", fake_compose)
    monkeypatch.setattr(comp, "_validate_composed", fake_validate)
    monkeypatch.setattr(composer_mod.cv2, "imread", lambda p: np.zeros((8, 8, 3), np.uint8))
    comp.calls = calls
    return comp


def run_parallel(comp, dirs, targets, num_images=1000, progress=None):
    bg_dir, obj_dir, out_dir = dirs
    return asyncio.run(comp.compose_batch_parallel(
        num_images=num_images,
        backgrounds_dir=bg_dir,
        objects_dir=obj_dir,
        output_dir=out_dir,
        concurrent_limit=2,
        targets_per_class=targets,
        max_objects_per_image=3,
        progress_callback=progress,
    ))


def test_parallel_stops_when_class_targets_met(composer, dirs):
    targets = {cls: 20 for cls in CLASSES}
    result = run_parallel(composer, dirs, targets)

    for cls in CLASSES:
        assert result.synthetic_counts[cls] >= 20
    # Several instances per image, so far fewer images than num_images
    assert result.images_generated < 1000
    # No idle iterations: at most the in-flight tasks finish after the stop
    assert composer.calls["depth"] <= composer.calls["compose"] + 4


def test_parallel_ignores_classes_without_target(composer, dirs):
    # Tire has crops but no target: it must neither fail nor be generated
    result = run_parallel(composer, dirs, {"Bottle": 10, "Can": 10})

    assert result.images_rejected == 0
    assert result.synthetic_counts.get("Tire", 0) == 0
    assert result.synthetic_counts["Bottle"] >= 10


def test_parallel_final_progress_has_nothing_pending(composer, dirs):
    reports = []
    result = run_parallel(composer, dirs, {cls: 15 for cls in CLASSES}, progress=reports.append)

    last = reports[-1]
    assert last["pending"] == 0
    assert last["total_target"] == result.images_generated
    assert last["next_image_index"] >= result.images_generated


def read_coco(out_dir):
    path = os.path.join(out_dir, "synthetic_dataset.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def test_parallel_flushes_coco_during_batch(composer, dirs):
    # The COCO is the commit point: it must exist on disk well before the end
    seen = []
    composer.calls["on_compose"] = lambda n: seen.append(read_coco(dirs[2])) if n == 150 else None
    result = run_parallel(composer, dirs, {cls: 150 for cls in CLASSES})

    assert seen and seen[0] is not None
    assert len(seen[0]["images"]) >= 50
    final = read_coco(dirs[2])
    assert len(final["images"]) == result.images_generated
    # No duplicated entries from repeated flushes
    assert len({img["file_name"] for img in final["images"]}) == len(final["images"])
    assert len({ann["id"] for ann in final["annotations"]}) == len(final["annotations"])


def test_sequential_stops_when_class_targets_met(composer, dirs):
    bg_dir, obj_dir, out_dir = dirs
    result = asyncio.run(composer.compose_batch(
        num_images=1000,
        backgrounds_dir=bg_dir,
        objects_dir=obj_dir,
        output_dir=out_dir,
        targets_per_class={cls: 20 for cls in CLASSES},
        max_objects_per_image=3,
    ))

    assert all(result.synthetic_counts[cls] >= 20 for cls in CLASSES)
    assert composer.calls["depth"] == composer.calls["compose"]


def test_sequential_flushes_coco_during_batch(composer, dirs):
    bg_dir, obj_dir, out_dir = dirs
    seen = []
    composer.calls["on_compose"] = lambda n: seen.append(read_coco(out_dir)) if n == 120 else None
    result = asyncio.run(composer.compose_batch(
        num_images=1000,
        backgrounds_dir=bg_dir,
        objects_dir=obj_dir,
        output_dir=out_dir,
        targets_per_class={cls: 150 for cls in CLASSES},
        max_objects_per_image=3,
    ))

    assert seen and seen[0] is not None and len(seen[0]["images"]) >= 50
    assert len(read_coco(out_dir)["images"]) == result.images_generated
