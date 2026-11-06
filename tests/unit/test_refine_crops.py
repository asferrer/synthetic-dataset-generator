"""Tests for the SAM3 crop refinement (`tools/refine_crops_lib.py`, `tools/refine_crops_sam3.py`, `tools/sam3_candidates.py`).

No GPU, torch or services needed: SAM3 is replaced by synthetic masks.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import refine_crops_lib as rl  # noqa: E402
import refine_crops_sam3 as rc  # noqa: E402
import sam3_candidates as sc  # noqa: E402

pytestmark = pytest.mark.unit
SHAPE = (120, 160)


def disk(cx, cy, r, shape=SHAPE):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def cand(mode, score, mask):
    return rl.Cand(mode, score, mask, int(mask.sum()))


def test_mask_iou_and_box():
    a, b = disk(50, 50, 20), disk(50, 50, 20)
    assert rl.mask_iou(a, b) == 1.0
    assert rl.mask_iou(a, disk(120, 50, 10)) == 0.0
    assert rl.mask_bbox(np.zeros(SHAPE, bool)) is None
    assert rl.mask_bbox(disk(50, 50, 10)) == (40, 40, 21, 21)
    assert rl.box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_rings_mask_rasterizes_square():
    m = rl.rings_mask([[10, 10, 30, 10, 30, 30, 10, 30]], SHAPE)
    assert 400 <= m.sum() <= 480
    assert m[20, 20] and not m[5, 5]
    assert rl.rings_mask([[1, 2]], SHAPE).sum() == 0


def textured(h=120, w=160, seed=3):
    return np.random.RandomState(seed).randint(0, 255, (h, w, 3)).astype(np.uint8)


def test_locate_crop_and_original_mask_ext():
    rgb = textured()
    obj = disk(80, 60, 20)
    ys, xs = np.where(obj)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgba = np.dstack([rgb[y0:y1, x0:x1], obj[y0:y1, x0:x1].astype(np.uint8) * 255])
    (x, y), rmse = rl.locate_crop(rgb, rgba, [55, 35, 50, 50])
    assert (x, y) == (x0, y0) and rmse < 0.5
    full, _ = rl.original_mask_ext(rgb, rgba, [55, 35, 50, 50])
    assert rl.mask_iou(full, obj) == 1.0
    other = textured(seed=9)
    assert rl.original_mask_ext(other, rgba, [55, 35, 50, 50]) == (None, None)     # the crop is not from this image


def test_select_best_prefers_polygon_agreement():
    poly = disk(60, 60, 25)
    good = cand("boxtext", 0.8, disk(60, 60, 24))
    bad = cand("box", 0.99, disk(110, 60, 12))          # high score but another object
    text = cand("text", 0.9, disk(60, 60, 26))
    best, rows = rl.select_best([bad, good, text], poly, [30, 30, 60, 60], "Can", rl.Params())
    assert best == 1
    assert all(r["mode"] != "text" for r in rows)


def test_weak_reference_rejects_sparse_mask_that_spans_the_box():
    box = [30, 30, 60, 60]
    full = cand("boxtext", 0.9, disk(60, 60, 28))
    sparse = np.zeros(SHAPE, bool)
    sparse[30:90:10, 30:90] = True
    sparse[30:90, 30:90:10] = True                       # grid: the box matches the annotated one but there is almost no mask
    thin = cand("boxtext", 0.95, sparse)
    p = rl.Params(cov_min=0.0, cov_min_net=0.0)
    best_no_weak, _ = rl.select_best([thin, full], None, box, "Fishing_Net", p)
    best_weak, rows = rl.select_best([thin, full], None, box, "Fishing_Net", p, weak=disk(60, 60, 27))
    assert best_no_weak == 0                              # without a weak reference the grid wins
    assert best_weak == 1                                 # with the original mask the full mask wins


def test_overflow_and_coverage_filters():
    box = [40, 40, 40, 40]
    big = cand("box", 0.9, disk(60, 60, 55))             # overflows the box a lot
    tiny = cand("box", 0.9, disk(60, 60, 2))             # low coverage and minimum area
    union = np.ones(SHAPE, bool)
    for c, reason in ((big, "box_overflow"), (tiny, "low_coverage")):
        met = rl.candidate_metrics(c, box, SHAPE, union)
        assert reason in rl.rejection_reasons(c, met, "Bottle", rl.Params())
    best, rows = rl.select_best([big, tiny], None, box, "Bottle", rl.Params())
    assert best is None and rl.main_reason(rows) in {"box_overflow", "low_coverage", "min_area", "low_ref"}


def test_class_membership_uses_text_concept():
    box = [30, 30, 60, 60]
    obj = cand("box", 0.9, disk(60, 60, 25))
    far_text = cand("text", 0.9, disk(140, 100, 10))     # the concept is elsewhere
    met = rl.candidate_metrics(obj, box, SHAPE, rl.concept_union([obj, far_text], 0.4))
    assert "off_class" in rl.rejection_reasons(obj, met, "Glove", rl.Params())
    ok = cand("boxtext", 0.7, disk(60, 60, 25))          # concept + box with a good score is enough
    met2 = rl.candidate_metrics(ok, box, SHAPE, rl.concept_union([ok, far_text], 0.4))
    assert "off_class" not in rl.rejection_reasons(ok, met2, "Glove", rl.Params())


def test_net_gets_low_coverage_allowed():
    c = cand("box", 0.9, disk(60, 60, 10))
    met = rl.candidate_metrics(c, [20, 20, 120, 80], SHAPE, np.ones(SHAPE, bool))
    assert "low_coverage" in rl.rejection_reasons(c, met, "Bottle", rl.Params())
    assert "low_coverage" not in rl.rejection_reasons(c, met, rl.NET, rl.Params())


def test_clean_mask_removes_specks_and_fills_holes():
    m = disk(60, 60, 25)
    m[58:62, 58:62] = False                               # internal hole
    m[5:7, 5:7] = True                                    # loose speck
    out = rl.clean_mask(m, rl.Params())
    assert not out[5, 5] and out[60, 60]
    assert rl.clean_mask(m, rl.Params(), fill_holes=False)[60, 60] == False  # noqa: E712  (nets: the hole is kept)


def synthetic_net(h=120, w=160, seed=0):
    """Blue water with noise and a mesh of red strands inside a rectangle."""
    rng = np.random.RandomState(seed)
    img = np.clip(np.array([20, 80, 140]) + rng.randn(h, w, 3) * 2, 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), bool)
    mask[30:90, 40:120] = True
    img[30:90:12, 40:120] = [200, 60, 50]
    img[30:90, 40:120:12] = [200, 60, 50]
    return img, mask


def test_net_alpha_removes_water_between_strands():
    img, mask = synthetic_net()
    alpha = rl.net_alpha(img, mask)
    assert alpha is not None and alpha.max() <= 1.0 and alpha[~mask].sum() == 0
    opaque_frac = float((alpha > 0.5).sum() / mask.sum())
    assert 0.05 < opaque_frac < 0.6                       # only the strands, not the whole rectangle
    bin_bg = rl.background_fraction(img, mask.astype(np.float32), mask)
    fine_bg = rl.background_fraction(img, alpha, mask)
    assert bin_bg > 0.5 > fine_bg                         # the background inside the mask drops with the fine alpha


def test_background_fraction_none_without_ring():
    img = np.zeros((20, 20, 3), np.uint8)
    full = np.ones((20, 20), np.float32)
    assert rl.background_fraction(img, full) is None


def test_rect_fill_and_make_rgba():
    alpha = np.zeros((50, 50), np.float32)
    alpha[10:30, 10:30] = 1.0
    assert rl.rect_fill(alpha) == 1.0
    rgb = np.zeros((50, 50, 3), np.uint8)
    rgba = rl.make_rgba(rgb, alpha, pad=5)
    assert rgba.shape == (30, 30, 4) and rgba[5, 5, 3] == 255 and rgba[0, 0, 3] == 0
    assert rl.make_rgba(rgb, np.zeros((50, 50), np.float32)) is None
    u8 = (disk(25, 25, 10, (50, 50)) * 255).astype(np.uint8)
    assert 0.7 < rl.rect_fill(u8) < 0.85                  # a disc fills pi/4 of its box


def test_mask_to_polygons_roundtrip():
    polys = rl.mask_to_polygons(disk(60, 60, 25))
    assert len(polys) == 1
    back = rl.rings_mask(polys, SHAPE)
    assert rl.mask_iou(back, disk(60, 60, 25)) > 0.93


def test_candidate_pack_and_load_roundtrip(tmp_path):
    shape = (200, 300)
    region = sc.region_for([100, 50, 60, 40], (300, 200), 0.25)
    m1, m2 = disk(130, 70, 18, shape), disk(60, 150, 10, shape)
    np.savez_compressed(tmp_path / "7.npz", **sc.pack_candidates([("box", 0.9, m1), ("text", 0.5, m2)], region))
    cands = rl.load_candidates(tmp_path / "7.npz", shape)
    assert [c.mode for c in cands] == ["box", "text"]
    assert (cands[0].mask == m1).sum() >= m1.sum() - 5 and cands[0].area_full == m1.sum()
    assert cands[1].mask.sum() == 0 and cands[1].area_full == m2.sum()      # outside the region: visible in area_full


def test_region_for_clamps_to_image():
    assert sc.region_for([0, 0, 100, 100], (150, 120), 0.25) == (0, 0, 126, 120)


def test_stratified_sample_is_seeded_and_prefers_polygons():
    cands = [{"lib": "ext", "cls": "Can", "ann": i, "poly": [[0, 0, 1, 0, 1, 1]] if i % 4 == 0 else None} for i in range(40)]
    a = rc.stratified_sample(cands, {"Can": 6}, seed=21)
    assert a == rc.stratified_sample(cands, {"Can": 6}, seed=21)
    assert a != rc.stratified_sample(cands, {"Can": 6}, seed=22)
    assert len(a) == 6 and sum(1 for c in a if c["poly"]) == 3


def test_train_only_guard(tmp_path):
    for name in ("val.json", "split_test.json"):
        (tmp_path / name).write_text("{}")
        with pytest.raises(SystemExit):
            rc.read_train_only(tmp_path / name)
    (tmp_path / "train.json").write_text("{}")
    assert rc.read_train_only(tmp_path / "train.json") == {}


def test_refine_one_end_to_end(tmp_path):
    """One object with a good candidate and another with an overflowing SAM3 mask, no services."""
    from PIL import Image
    h, w = 120, 160
    rgb = np.full((h, w, 3), (20, 80, 140), np.uint8)
    obj = disk(80, 60, 20)
    rgb[obj] = (220, 200, 60)
    real = tmp_path / "real"
    real.mkdir()
    Image.fromarray(rgb).save(real / "a.jpg", quality=100)
    out = tmp_path / "out"
    (out / "cand").mkdir(parents=True)
    region = sc.region_for([55, 35, 50, 50], (w, h), 0.25)
    np.savez_compressed(out / "cand" / "1.npz", **sc.pack_candidates(
        [("boxtext", 0.9, obj), ("text", 0.8, obj), ("box", 0.95, disk(80, 60, 70))], region))
    job = {"lib": "poly", "cls": "Can", "ann": 1, "image_id": 0, "box": [55.0, 35.0, 50.0, 50.0],
           "poly": [[60, 40, 100, 40, 100, 80, 60, 80]], "image": "a.jpg", "crop": "Can/poly_1.png"}

    class A:
        pass
    a = A()
    a.out, a.real_root, a.crops_root = str(out), str(real), str(tmp_path)
    row, alpha, mask = rc.refine_one(job, a, rl.Params(), out / "library")
    assert row["status"] == "accepted" and row["mode"] == "boxtext"
    assert (out / "library" / "Can" / "sam3r_1.png").exists() and (out / "library" / "Can" / "sam3r_1.json").exists()
    assert rl.mask_iou(alpha > 0.5, obj) > 0.95
    job2 = {**job, "ann": 2}
    row2, a2, _ = rc.refine_one(job2, a, rl.Params(), out / "library")
    assert row2["status"] == "no_sam3_candidates" and a2 is None


def test_refine_one_ext_uses_located_original(tmp_path):
    """Extracted_objects_train library: the original crop is located in the image and compared with the refined mask."""
    from PIL import Image
    rgb = textured()
    obj = disk(80, 60, 20)
    orig = disk(80, 60, 17)                                  # original mask smaller than the SAM3 one
    ys, xs = np.where(orig)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    crops = tmp_path / "Extracted_objects_train" / "Can"
    crops.mkdir(parents=True)
    Image.fromarray(np.dstack([rgb[sl], orig[sl].astype(np.uint8) * 255]), "RGBA").save(crops / "1.png")
    real = tmp_path / "real"
    real.mkdir()
    Image.fromarray(rgb).save(real / "a.png")
    out = tmp_path / "out"
    (out / "cand").mkdir(parents=True)
    region = sc.region_for([55, 35, 50, 50], (160, 120), 0.25)
    np.savez_compressed(out / "cand" / "1.npz", **sc.pack_candidates([("boxtext", 0.9, obj), ("text", 0.8, obj)], region))
    job = {"lib": "ext", "cls": "Can", "ann": 1, "image_id": 0, "box": [55.0, 35.0, 50.0, 50.0], "poly": None,
           "image": "a.png", "crop": "Can/1.png"}

    class A:
        pass
    a = A()
    a.out, a.real_root, a.crops_root = str(out), str(real), str(tmp_path)
    row, alpha, _ = rc.refine_one(job, a, rl.Params(), out / "library")
    assert row["status"] == "accepted" and row["orig_rmse"] < 0.5
    assert row["cov_ref"] > row["cov_orig"] and 0.6 < row["iou_orig_ref"] < 0.8
