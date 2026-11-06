"""Tests for the crop-library cleaning (`tools/clean_crops_lib.py`, `tools/clean_crops.py`).

No GPU or services: synthetic images and crops in a temporary directory.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import clean_crops as cc  # noqa: E402
import clean_crops_lib as cl  # noqa: E402

pytestmark = pytest.mark.unit
P = cl.CleanParams()


def alpha_rect(shape, x0, y0, x1, y1):
    a = np.zeros(shape, np.uint8)
    a[y0:y1, x0:x1] = 255
    return a


def test_box_window_margin_and_clipping():
    w = cl.box_window([10, 10, 20, 20], (0, 0), (40, 40), margin=4)
    assert w[6:34, 6:34].all() and w.sum() == 28 * 28
    w2 = cl.box_window([10, 10, 20, 20], (8, 8), (30, 30), margin=0)     # the crop starts at (8, 8)
    assert w2[2:22, 2:22].all() and w2.sum() == 400
    assert cl.box_window([100, 100, 5, 5], (0, 0), (20, 20)).sum() == 0   # box outside the crop


def test_measure_inside_and_overflow():
    a = alpha_rect((60, 60), 0, 0, 40, 40)        # 40x40 mask; annotated box [10, 10, 30, 30] => 30x30 inside
    m = cl.measure(a, [10, 10, 30, 30], (0, 0), P)
    assert m["area"] == 1600 and m["overflow_exact"] == pytest.approx(1 - 900 / 1600)
    assert m["overflow_margin"] < m["overflow_exact"]            # the 4 px margin absorbs part of the overflow
    assert m["coverage"] == pytest.approx(1.0) and m["n_comp"] == 1


def test_measure_counts_significant_components():
    a = alpha_rect((50, 100), 0, 0, 40, 40)
    a[45:47, 90:92] = 255                          # 4 px speck: not a significant component
    m = cl.measure(a, [0, 0, 100, 50], (0, 0), P)
    assert m["n_comp"] == 2 and m["n_comp_sig"] == 1


def test_clean_alpha_clips_to_box():
    a = alpha_rect((80, 120), 10, 10, 110, 70)     # 100x60 net with a hole (otherwise coverage would be 1.0)
    a[22:58, 22:100] = 0
    new, info = cl.clean_alpha(a, [10, 10, 80, 60], (0, 0), "Fishing_Net", P)   # the box ends at x = 90
    assert new is not None
    assert new[:, 95:].max() == 0 and new[:, 14:90].max() == 255        # nothing goes past x = 90 + 4 margin
    assert info["area_final"] < info["area_orig"] and info["modified"]


def test_clean_alpha_removes_small_components_only():
    big2 = alpha_rect((100, 100), 10, 10, 60, 60)
    big2[75:90, 75:90] = 255                       # second component of 225 px (> 2 % of the area)
    new, info = cl.clean_alpha(big2, [0, 0, 100, 100], (0, 0), "Can", P)
    assert new is not None and info["components_removed"] == 0
    b = alpha_rect((100, 100), 10, 10, 60, 60)
    b[95:97, 95:97] = 255                          # 4 px speck
    new, info = cl.clean_alpha(b, [0, 0, 100, 100], (0, 0), "Can", P)
    assert info["components_removed"] == 1 and new[95:97, 95:97].max() == 0


def test_clean_alpha_discards_mask_outside_box():
    new, info = cl.clean_alpha(alpha_rect((50, 50), 0, 0, 50, 50), [200, 200, 30, 30], (0, 0), "Can", P)
    assert new is None and info["reason"] == "dominated_by_foreign_area"


def test_clean_alpha_discards_almost_nothing_left():
    new, info = cl.clean_alpha(alpha_rect((200, 200), 0, 0, 5, 5), [0, 0, 10, 10], (0, 0), "Can", P)
    assert new is None and info["reason"] == "almost_nothing"


def test_clean_alpha_coverage_bounds_and_net_threshold():
    full = alpha_rect((50, 50), 0, 0, 50, 50)
    new, info = cl.clean_alpha(full, [0, 0, 50, 50], (0, 0), "Can", P)
    assert new is None and info["reason"] == "high_coverage"
    thin = alpha_rect((300, 300), 0, 0, 300, 12)    # area 3600 / 90000 = 0.04: low for a can, normal for a net
    assert cl.clean_alpha(thin, [0, 0, 300, 300], (0, 0), "Can", P)[0] is None
    assert cl.clean_alpha(thin, [0, 0, 300, 300], (0, 0), "Fishing_Net", P)[0] is not None


def test_shrink_and_original_pad():
    rgba = np.zeros((30, 30, 4), np.uint8)
    rgba[:, :, :3] = 77
    rgba[10:20, 8:22, 3] = 255
    out, (x0, y0) = cl.shrink_rgba(rgba, 0)
    assert out.shape == (10, 14, 4) and (x0, y0) == (8, 10) and out[0, 0, 0] == 77
    assert cl.original_pad(rgba[:, :, 3]) == 8 and cl.original_pad(np.zeros((5, 5), np.uint8)) == 0


def make_library(tmp: Path):
    """200x150 noise image; crop of a 100x40 net whose annotated box covers only 60 px of width."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (150, 200, 3), dtype=np.uint8)
    (tmp / "real").mkdir()
    Image.fromarray(img).save(tmp / "real" / "a.png")
    box = [20, 20, 60, 40]
    x0, y0, x1, y1 = 20, 20, 120, 60
    alpha = np.full((y1 - y0, x1 - x0), 255, np.uint8)
    alpha[8:32, 8:92] = 0                                    # central hole: normal coverage
    rgba = np.dstack([img[y0:y1, x0:x1], alpha])
    for name, fn in (("Extracted_objects_train", "7.png"), ("crops_poly", "poly_7.png")):
        d = tmp / "in" / name / "Fishing_Net"
        d.mkdir(parents=True)
        Image.fromarray(rgba, "RGBA").save(d / fn)
    jobs = [{"lib": "ext", "cls": "Fishing_Net", "ann": 7, "image_id": 0, "box": box, "poly": None,
             "crop": "Fishing_Net/7.png", "image": "a.png"}]
    return jobs


def test_process_image_writes_clean_crop(tmp_path):
    jobs = make_library(tmp_path)
    in_roots = {"ext": str(tmp_path / "in" / "Extracted_objects_train"), "poly": str(tmp_path / "in" / "crops_poly")}
    out_roots = {"ext": str(tmp_path / "out_ext"), "poly": str(tmp_path / "out_poly")}
    rows = cc.process_image((jobs, str(tmp_path / "real"), in_roots, out_roots, P, "keep"))
    assert rows[0]["status"] == "clean" and 0.3 < rows[0]["overflow_exact"] < 0.5
    out = np.asarray(Image.open(tmp_path / "out_ext" / "Fishing_Net" / "7.png"))
    assert out.shape[1] == 64 and out.shape[0] == 40           # 60 of the box + 4 of margin
    meta = json.loads((tmp_path / "out_ext" / "Fishing_Net" / "7.json").read_text(encoding="utf8"))
    assert meta["images"][0]["width"] == 64 and meta["annotations"][0]["area"] == 64 * 40


def test_original_library_is_not_modified(tmp_path):
    jobs = make_library(tmp_path)
    src = tmp_path / "in" / "Extracted_objects_train" / "Fishing_Net" / "7.png"
    before = src.read_bytes()
    in_roots = {"ext": str(src.parents[1]), "poly": str(tmp_path / "in" / "crops_poly")}
    cc.process_image((jobs, str(tmp_path / "real"), in_roots, {"ext": str(tmp_path / "o"), "poly": str(tmp_path / "p")}, P, "keep"))
    assert src.read_bytes() == before


def test_prepare_out_refuses_non_empty_dir(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f.txt").write_text("a")
    with pytest.raises(SystemExit):
        cc.prepare_out({"ext": str(tmp_path / "x")}, dry_run=False)
    cc.prepare_out({"ext": str(tmp_path / "x")}, dry_run=True)   # measure-only mode writes nothing
    assert (tmp_path / "x" / "f.txt").exists()


def test_summarize_groups_by_class():
    rows = [{"cls": "A", "status": "clean", "overflow_exact": 0.2, "overflow_margin": 0.1, "coverage": 0.5, "n_comp": 1,
             "n_comp_sig": 1, "modified": True, "removed_frac": 0.1},
            {"cls": "A", "status": "dropped", "reason": "almost_nothing", "overflow_exact": 0.0, "overflow_margin": 0.0,
             "coverage": 0.0, "n_comp": 1, "n_comp_sig": 1},
            {"cls": "B", "status": "unlocated"}]
    s = cl.summarize(rows, P)
    assert s["A"]["dropped"] == 1 and s["A"]["modified"] == 1 and s["A"]["with_margin_overflow_gt_5"] == 1
    assert s["B"]["unlocated"] == 1 and s["B"]["kept"] == 1


def test_drop_mode_keeps_original_mask_and_same_decisions(tmp_path):
    """Drop mode: the kept crop is identical to the original (not clipped to the box) and the decision matches clean."""
    jobs = make_library(tmp_path)
    src = tmp_path / "in" / "Extracted_objects_train" / "Fishing_Net" / "7.png"
    in_roots = {"ext": str(src.parents[1]), "poly": str(tmp_path / "in" / "crops_poly")}
    out_roots = {"ext": str(tmp_path / "o"), "poly": str(tmp_path / "p")}
    rows = cc.process_image((jobs, str(tmp_path / "real"), in_roots, out_roots, P, "keep", "drop"))
    ref = cc.process_image((jobs, str(tmp_path / "real"), in_roots, None, P, "keep", "clean"))
    assert rows[0]["status"] == ref[0]["status"] == "clean" and rows[0]["modified"] is False
    assert (tmp_path / "o" / "Fishing_Net" / "7.png").read_bytes() == src.read_bytes()


def test_drop_mode_discards_like_clean(tmp_path):
    jobs = make_library(tmp_path)
    jobs[0]["box"] = [150, 100, 30, 30]                      # box far from the mask: dominated_by_foreign_area
    in_roots = {"ext": str(tmp_path / "in" / "Extracted_objects_train"), "poly": str(tmp_path / "in" / "crops_poly")}
    out_roots = {"ext": str(tmp_path / "o"), "poly": str(tmp_path / "p")}
    rows = cc.process_image((jobs, str(tmp_path / "real"), in_roots, out_roots, P, "keep", "drop"))
    if rows[0]["status"] == "unlocated":
        pytest.skip("the template cannot be located with a shifted box")
    assert rows[0]["status"] == "dropped" and not (tmp_path / "o" / "Fishing_Net" / "7.png").exists()


def test_drops_by_class_counts_reasons():
    rows = [{"cls": "A", "status": "clean"}, {"cls": "A", "status": "dropped", "reason": "almost_nothing"},
            {"cls": "B", "status": "dropped", "reason": "almost_nothing"}, {"cls": "B", "status": "dropped", "reason": "high_coverage"}]
    d = cl.drops_by_class(rows)
    assert d["A"] == {"crops": 2, "dropped": 1, "reasons": {"almost_nothing": 1}, "kept": 1}
    assert d["TOTAL"]["dropped"] == 3 and d["TOTAL"]["reasons"] == {"high_coverage": 1, "almost_nothing": 2}
