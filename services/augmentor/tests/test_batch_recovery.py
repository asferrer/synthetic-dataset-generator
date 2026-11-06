"""Recovery of an interrupted batch from its committed COCO (app.batch_recovery)."""

import json
import os

import pytest

from app.batch_recovery import (
    COCO_NAME,
    UNFLUSHED_DIR,
    class_counts,
    image_index,
    recover_batch_state,
    write_json_atomic,
)
from app.composer import ImageComposer

CATEGORIES = [{"id": 0, "name": "Bottle"}, {"id": 1, "name": "Can"}]


def make_image(images_dir, idx, with_annotation=True, with_image=True):
    base = os.path.join(images_dir, f"synthetic_{idx:05d}")
    if with_image:
        with open(base + ".jpg", "wb") as f:
            f.write(b"x")
    if with_annotation:
        with open(base + "_annotations.json", "w") as f:
            json.dump([{"class_name": "Bottle"}], f)


def make_coco(out_dir, indices, category_ids):
    coco = {
        "images": [{"id": i, "file_name": f"synthetic_{idx:05d}.jpg", "width": 8, "height": 8}
                   for i, idx in enumerate(indices)],
        "annotations": [{"id": n + 1, "image_id": 0, "category_id": c, "bbox": [0, 0, 1, 1]}
                        for n, c in enumerate(category_ids)],
        "categories": CATEGORIES,
    }
    write_json_atomic(os.path.join(out_dir, COCO_NAME), coco)


@pytest.fixture
def batch_dir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    return str(tmp_path), str(images_dir)


def test_write_json_atomic_replaces_without_leftovers(tmp_path):
    path = str(tmp_path / "a.json")
    write_json_atomic(path, {"v": 1})
    write_json_atomic(path, {"v": 2})
    with open(path) as f:
        assert json.load(f) == {"v": 2}
    assert os.listdir(tmp_path) == ["a.json"]


def test_image_index_parses_images_and_annotations():
    assert image_index("synthetic_00042.jpg") == 42
    assert image_index("synthetic_10983_annotations.json") == 10983
    assert image_index("progress.json") is None


def test_recover_sets_aside_uncommitted_files(batch_dir):
    out_dir, images_dir = batch_dir
    for idx in range(5):
        make_image(images_dir, idx)
    # Annotation left behind by a rejected attempt (image already removed)
    make_image(images_dir, 7, with_image=False)
    make_coco(out_dir, indices=[0, 1, 2], category_ids=[0, 0, 1])

    state = recover_batch_state(out_dir)

    assert state["generated"] == 3
    assert state["counts"] == {"Bottle": 2, "Can": 1}
    assert state["next_image_index"] == 8  # never reuse an index seen on disk
    assert state["moved"] == 5  # images 3 and 4 (jpg + json) and the orphan json
    kept = sorted(os.listdir(images_dir))
    assert kept == sorted(f"synthetic_{i:05d}{s}" for i in range(3) for s in (".jpg", "_annotations.json"))
    assert len(os.listdir(os.path.join(out_dir, UNFLUSHED_DIR))) == 5


def test_recover_is_idempotent(batch_dir):
    out_dir, images_dir = batch_dir
    for idx in range(4):
        make_image(images_dir, idx)
    make_coco(out_dir, indices=[0, 1], category_ids=[0])

    first = recover_batch_state(out_dir)
    second = recover_batch_state(out_dir)

    assert second["moved"] == 0
    assert second["next_image_index"] == first["next_image_index"] == 4


def test_recover_without_coco_commits_nothing(batch_dir):
    out_dir, images_dir = batch_dir
    for idx in range(3):
        make_image(images_dir, idx)

    state = recover_batch_state(out_dir)

    assert state["generated"] == 0
    assert state["counts"] == {}
    assert state["next_image_index"] == 3
    assert os.listdir(images_dir) == []


def test_recover_with_corrupt_coco_commits_nothing(batch_dir):
    out_dir, images_dir = batch_dir
    make_image(images_dir, 0)
    with open(os.path.join(out_dir, COCO_NAME), "w") as f:
        f.write('{"images": [')

    state = recover_batch_state(out_dir)

    assert state["generated"] == 0
    assert state["next_image_index"] == 1


def test_class_counts_handles_missing_coco():
    assert class_counts(None) == {}


def test_generate_coco_json_does_not_mutate_existing(tmp_path):
    comp = ImageComposer(use_gpu=False)
    existing = {"images": [{"id": 0, "file_name": "synthetic_00000.jpg", "width": 8, "height": 8}],
                "annotations": [], "categories": CATEGORIES}
    path = str(tmp_path / COCO_NAME)
    new = [{"image_name": "synthetic_00001.jpg", "width": 8, "height": 8, "annotations": []}]

    comp._generate_coco_json(new, ["Bottle", "Can"], path, existing_coco=existing)
    comp._generate_coco_json(new, ["Bottle", "Can"], path, existing_coco=existing)

    assert len(existing["images"]) == 1
    with open(path) as f:
        assert len(json.load(f)["images"]) == 2


def test_discard_output_removes_image_and_annotation(tmp_path):
    images_dir = str(tmp_path)
    make_image(images_dir, 3)
    ImageComposer._discard_output(os.path.join(images_dir, "synthetic_00003.jpg"))
    assert os.listdir(images_dir) == []
