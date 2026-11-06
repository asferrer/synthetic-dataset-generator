"""tools/refit_coco_boxes.py: refits the boxes of a COCO without touching the original."""
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "refit_coco_boxes", Path(__file__).resolve().parents[3] / "tools" / "refit_coco_boxes.py")
refit_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(refit_mod)


def _coco(*anns):
    return {"info": {}, "images": [{"id": 1, "width": 200, "height": 150}], "categories": [],
            "annotations": [{"id": i, "image_id": 1, "category_id": 0, "iscrowd": 0, "segmentation": [], **a}
                            for i, a in enumerate(anns)]}


def test_undo_margin_shrinks_free_sides_and_keeps_border_sides():
    assert refit_mod.undo_margin([40, 30, 100, 80], 200, 150, 20) == [60, 50, 60, 40]
    # left and top sides on the border (clipped): kept; right and bottom sides free: -20
    assert refit_mod.undo_margin([0, 0, 120, 100], 200, 150, 20) == [0, 0, 100, 80]
    # right side on the image border
    assert refit_mod.undo_margin([60, 30, 140, 80], 200, 150, 20) == [80, 50, 120, 40]


def test_undo_margin_leaves_boxes_too_small_for_the_margin():
    assert refit_mod.undo_margin([50, 50, 30, 30], 200, 150, 20) == [50, 50, 30, 30]


def test_box_from_segmentation():
    assert refit_mod.box_from_segmentation([[10, 20, 50, 20, 50, 60, 10, 60]]) == [10, 20, 40, 40]
    assert refit_mod.box_from_segmentation([]) is None and refit_mod.box_from_segmentation(None) is None


def test_refit_skips_real_boxes_keeps_original_and_reports_counts():
    coco = _coco({"bbox": [40, 30, 100, 80], "area": 8000, "origin": "pasted"},
                 {"bbox": [40, 30, 100, 80], "area": 8000, "origin": "real"})
    changed, skipped = refit_mod.refit(coco, "undo-margin", 20)
    pasted, real = coco["annotations"]
    assert (changed, skipped) == (1, 0)
    assert pasted["bbox"] == [60, 50, 60, 40] and pasted["area"] == 2400 and pasted["bbox_original"] == [40, 30, 100, 80]
    assert real["bbox"] == [40, 30, 100, 80] and "bbox_original" not in real


def test_refit_segmentation_without_masks_skips(tmp_path):
    coco = _coco({"bbox": [1, 2, 3, 4], "area": 12})
    assert refit_mod.refit(coco, "segmentation", 20) == (0, 1)


def test_cli_writes_a_new_file_and_refuses_to_overwrite(tmp_path, monkeypatch):
    src, dst = tmp_path / "in.json", tmp_path / "out.json"
    src.write_text(json.dumps(_coco({"bbox": [40, 30, 100, 80], "area": 8000})))
    monkeypatch.setattr("sys.argv", ["x", str(src), str(dst), "--method", "undo-margin"])
    refit_mod.main()
    assert json.loads(dst.read_text())["annotations"][0]["bbox"] == [60, 50, 60, 40]
    assert json.loads(src.read_text())["annotations"][0]["bbox"] == [40, 30, 100, 80]
    monkeypatch.setattr("sys.argv", ["x", str(src), str(src), "--method", "undo-margin"])
    with pytest.raises(SystemExit):
        refit_mod.main()
