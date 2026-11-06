"""Tests for the scripts that prepare the real-background inputs (no GPU, no services)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_real_train as brt  # noqa: E402
import measure_real_visibility as mrv  # noqa: E402


def _coco(root: Path, n=3):
    images, anns = [], []
    for i in range(n):
        name = f"images/set/img_{i}.jpg"
        (root / "images" / "set").mkdir(parents=True, exist_ok=True)
        arr = np.full((120, 160, 3), 90, np.uint8)
        arr[40:80, 60:100] = (200, 60, 60)
        Image.fromarray(arr).save(root / name)
        images.append({"id": 10 + i, "file_name": name, "width": 160, "height": 120, "source": "set"})
        anns.append({"id": i, "image_id": 10 + i, "category_id": 1, "bbox": [60, 40, 40, 40], "area": 1600,
                     "segmentation": [[1, 2, 3, 4, 5, 6]]})
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "Can"}]}


def test_new_name_uses_id_and_basename():
    assert brt.new_name({"id": 7, "file_name": "a/b/c.jpg"}) == "00007_c.jpg"


def test_build_writes_boxes_only_and_flat_names(tmp_path):
    train = _coco(tmp_path)
    out = tmp_path / "real_train"
    copied, present, missing, bad_size, by_source = brt.copy_images(train, tmp_path, out)
    assert (copied, present, missing, bad_size) == (3, 0, [], [])
    assert by_source["set"] == 3
    coco = brt.write_annotations(train, out)
    assert all(a["segmentation"] == [] for a in coco["annotations"])
    assert sorted(p.name for p in out.glob("*.jpg")) == ["00010_img_0.jpg", "00011_img_1.jpg", "00012_img_2.jpg"]
    # second run copies nothing
    assert brt.copy_images(train, tmp_path, out)[:2] == (0, 3)


def test_subset_by_ids_requires_existing_ids(tmp_path):
    train = _coco(tmp_path)
    sub = brt.subset_by_ids(train, {"images": [{"id": 11}]})
    assert [im["id"] for im in sub["images"]] == [11] and len(sub["annotations"]) == 1
    with pytest.raises(SystemExit):
        brt.subset_by_ids(train, {"images": [{"id": 99}]})


def test_refuses_validation_or_test_coco(tmp_path):
    with pytest.raises(SystemExit):
        brt.load_train(tmp_path / "val.json")
    with pytest.raises(SystemExit):
        mrv.load_boxes(tmp_path / "test_set" / "annotations.json")


def test_visibility_percentiles_from_a_built_folder(tmp_path):
    train = _coco(tmp_path)
    out = tmp_path / "real_train"
    brt.copy_images(train, tmp_path, out)
    brt.write_annotations(train, out)
    contrast, lstd = mrv.measure(out / "annotations.json")
    assert len(contrast["Can"]) == 3 and all(c > 10 for c in contrast["Can"])
