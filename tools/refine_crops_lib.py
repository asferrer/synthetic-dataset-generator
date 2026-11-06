"""Pure pieces of the SAM3 crop refinement (no torch, no services; only NumPy, OpenCV and PIL).

Used by `refine_crops_sam3.py` (plan / build) and the tests. Conventions: masks are `bool (H, W)` in full-image
coordinates; boxes are `[x, y, w, h]` as in COCO.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

ALPHA_THR = 10      # alpha threshold used by the compositor
CROP_PAD = 5        # padding around the mask in the new crops (as in `crop_with_mask`)
NET = "Fishing_Net"


@dataclass
class Params:
    """Quality-filter thresholds (stored in the manifest)."""
    w_ref: float = 0.7              # weight of the reference agreement in the combined score (rest: SAM3 score)
    min_ref_poly: float = 0.5       # minimum mask IoU with the polygon
    min_ref_box: float = 0.6        # minimum IoU of the mask box with the annotated box (no polygon, no original mask)
    min_ref_weak: float = 0.4       # minimum agreement with the weak reference (no polygon, with original mask)
    overflow_max: float = 0.15      # maximum fraction of the mask outside the grown annotated box
    box_dilate: float = 0.05        # growth of the annotated box when measuring the overflow
    cov_min: float = 0.10           # minimum coverage (mask area / box area)
    cov_min_net: float = 0.03
    cov_max: float = 0.97
    rect_fill_max: float = 0.90     # rectangular crops: opaque fraction of the fitted box (as in `crop_filter`)
    text_thr: float = 0.4           # minimum score of the text-concept instances
    class_cov_min: float = 0.5      # fraction of the mask covered by concept instances
    class_score_min: float = 0.5    # alternative: score of the concept + box candidate
    min_area: int = 100
    min_comp_frac: float = 0.02     # components smaller than this fraction of the area are dropped
    net_alpha_min: float = 0.35     # minimum opaque fraction of a net fine alpha (if less remains, the ring held net pixels)
    extra: Dict = field(default_factory=dict)


# ------------------------------------------------------------------ geometry
def rings_mask(rings, shape_hw) -> np.ndarray:
    """Bool mask of the union of flat COCO rings [x1, y1, x2, y2, ...]."""
    h, w = shape_hw
    im = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(im)
    for r in rings or []:
        if isinstance(r, list) and len(r) >= 6:
            draw.polygon(list(zip(r[0::2], r[1::2])), fill=255)
    return np.asarray(im) > 0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def mask_bbox(m: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Box [x, y, w, h] of the true pixels (None if empty)."""
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def box_mask(box, shape_hw, dilate: float = 0.0) -> np.ndarray:
    """Rectangular mask of the box (grown `dilate` per side relative to its size), clipped to the image."""
    h, w = shape_hw
    x, y, bw, bh = box
    x0, x1 = max(0, int(x - dilate * bw)), min(w, int(np.ceil(x + bw * (1 + dilate))))
    y0, y1 = max(0, int(y - dilate * bh)), min(h, int(np.ceil(y + bh * (1 + dilate))))
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def _match(window: np.ndarray, rgba: np.ndarray):
    """(rmse, (x, y)) of the best fit of the crop opaque pixels in `window` (both RGB)."""
    mk = (rgba[:, :, 3] > ALPHA_THR).astype(np.uint8) * 255
    ch, cw = rgba.shape[:2]
    if window.shape[0] < ch or window.shape[1] < cw or not mk.any():
        return float("inf"), (0, 0)
    res = cv2.matchTemplate(window, np.ascontiguousarray(rgba[:, :, :3]), cv2.TM_SQDIFF, mask=cv2.merge([mk] * 3))
    mn, _, loc, _ = cv2.minMaxLoc(res)
    return float(np.sqrt(max(mn, 0.0) / (3.0 * float((mk > 0).sum())))), loc


def locate_crop(rgb: np.ndarray, rgba: np.ndarray, box, pad: int = 60, max_rmse: float = 3.0):
    """Position (x, y) of a library crop in its image and rmse; None if it does not fit (rmse > max_rmse).

    SAM3 crops are fitted to their mask box (not the annotated one), so they are located by template matching with
    the alpha mask, first near the annotated box and, if that fails, over the whole image."""
    x, y, w, h = [int(round(v)) for v in box]
    h_img, w_img = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w_img, x + w + pad), min(h_img, y + h + pad)
    rmse, (dx, dy) = _match(rgb[y0:y1, x0:x1], rgba)
    pos = (x0 + dx, y0 + dy)
    if rmse >= 1.5:
        rmse2, (ex, ey) = _match(rgb, rgba)
        if rmse2 < rmse:
            rmse, pos = rmse2, (ex, ey)
    return (pos, rmse) if rmse <= max_rmse else None


def original_mask_ext(rgb: np.ndarray, rgba: np.ndarray, box):
    """(original crop mask placed in the image, fit rmse) or (None, None) if it cannot be located."""
    found = locate_crop(rgb, rgba, box)
    if found is None:
        return None, None
    (x, y), rmse = found
    h_img, w_img = rgb.shape[:2]
    ch, cw = rgba.shape[:2]
    full = np.zeros((h_img, w_img), bool)
    full[y:y + ch, x:x + cw] = rgba[:, :, 3] > ALPHA_THR
    return full, rmse


# ------------------------------------------------------------------ candidates
@dataclass
class Cand:
    mode: str
    score: float
    mask: np.ndarray        # bool (H, W) of the full image (empty outside the stored region)
    area_full: int          # pixels of the full mask, also outside the region


def load_candidates(path, shape_hw) -> List[Cand]:
    """Read a `sam3_candidates.py` .npz and restore the masks in image coordinates."""
    z = np.load(path)
    x0, y0, x1, y1 = [int(v) for v in z["region"]]
    rh, rw = [int(v) for v in z["shape"]]
    out = []
    for i in range(len(z["scores"])):
        reg = np.unpackbits(z["bits"][i])[: rh * rw].reshape(rh, rw).astype(bool)
        full = np.zeros(shape_hw, bool)
        full[y0:y0 + rh, x0:x0 + rw] = reg
        out.append(Cand(str(z["modes"][i]), float(z["scores"][i]), full, int(z["area_full"][i])))
    return out


def concept_union(cands: List[Cand], thr: float) -> np.ndarray:
    """Union of the text-only prompt instances with score >= thr (empty if none)."""
    shape = cands[0].mask.shape if cands else (1, 1)
    texts = [c.mask for c in cands if c.mode == "text" and c.score >= thr]
    return np.logical_or.reduce(texts) if texts else np.zeros(shape, bool)


# ------------------------------------------------------------------ filters and selection
def candidate_metrics(c: Cand, box, shape_hw, union: np.ndarray, dilate: float = 0.05) -> Dict[str, float]:
    """Coverage, overflow and concept membership of a candidate."""
    area = max(1, c.area_full)
    inside = np.logical_and(c.mask, box_mask(box, shape_hw, 0.0)).sum()
    near = np.logical_and(c.mask, box_mask(box, shape_hw, dilate)).sum()
    return {"coverage": float(inside / max(1.0, box[2] * box[3])),
            "overflow_frac": 1.0 - float(near / area),
            "concept_frac": float(np.logical_and(c.mask, union).sum() / max(1, c.mask.sum())),
            "area": float(c.mask.sum())}


def rejection_reasons(c: Cand, met: Dict[str, float], cls: str, p: Params) -> List[str]:
    """Reasons a candidate fails the quality filters (empty list = passes)."""
    why = []
    if met["area"] < p.min_area:
        why.append("min_area")
    if met["overflow_frac"] > p.overflow_max:
        why.append("box_overflow")
    cmin = p.cov_min_net if cls == NET else p.cov_min
    if met["coverage"] < cmin:
        why.append("low_coverage")
    if met["coverage"] > p.cov_max:
        why.append("high_coverage")
    if met["concept_frac"] < p.class_cov_min and not (c.mode == "boxtext" and c.score >= p.class_score_min):
        why.append("off_class")
    return why


def ref_agreement(c: Cand, ref_mask: Optional[np.ndarray], box, weak: Optional[np.ndarray] = None,
                  dilate: float = 0.05) -> float:
    """Agreement with the reference.

    - With a polygon: mask IoU with the polygon.
    - Without a polygon but with an original mask (`weak`): mean of the box IoU (mask vs annotated) and the IoU with the
      original mask clipped to the grown annotated box. Box IoU alone would reward sparse masks that touch the box
      edges.
    - With neither: box IoU."""
    if ref_mask is not None:
        return mask_iou(c.mask, ref_mask)
    bb = mask_bbox(c.mask)
    biou = box_iou(bb, box) if bb else 0.0
    if weak is None:
        return biou
    return 0.5 * biou + 0.5 * mask_iou(c.mask, np.logical_and(weak, box_mask(box, c.mask.shape, dilate)))


def select_best(cands: List[Cand], ref_mask: Optional[np.ndarray], box, cls: str, p: Params,
                weak: Optional[np.ndarray] = None):
    """(chosen index or None, diagnostic rows per candidate). Combines reference agreement and SAM3 score.

    Candidates of the `text` prompt are never chosen (they only define the class concept)."""
    shape = cands[0].mask.shape if cands else (1, 1)
    union = concept_union(cands, p.text_thr)
    min_ref = p.min_ref_poly if ref_mask is not None else (p.min_ref_weak if weak is not None else p.min_ref_box)
    rows, best, best_val = [], None, -1.0
    for i, c in enumerate(cands):
        if c.mode == "text":
            continue
        met = candidate_metrics(c, box, shape, union, p.box_dilate)
        ref = ref_agreement(c, ref_mask, box, weak, p.box_dilate)
        why = rejection_reasons(c, met, cls, p) + (["low_ref"] if ref < min_ref else [])
        comb = p.w_ref * ref + (1 - p.w_ref) * c.score
        rows.append({"i": i, "mode": c.mode, "score": round(c.score, 3), "ref": round(ref, 3), "comb": round(comb, 3),
                     **{k: round(v, 3) for k, v in met.items()}, "reasons": why})
        if not why and comb > best_val:
            best, best_val = i, comb
    return best, rows


def main_reason(rows) -> str:
    """Rejection reason of an object: that of the candidate with the fewest reasons, ties broken by combined score."""
    if not rows:
        return "no_candidates"
    top = min(rows, key=lambda r: (len(r["reasons"]), -r["comb"]))
    return top["reasons"][0] if top["reasons"] else "none"


# ------------------------------------------------------------------ cleaning and fine alpha
def clean_mask(mask: np.ndarray, p: Params, fill_holes: bool = True) -> np.ndarray:
    """Remove small components (< min_comp_frac of the area) and, if asked, fill internal holes."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas[1:] >= p.min_comp_frac * areas[1:].sum()
    out = keep[lab]
    if fill_holes:
        k, lab2, _, _ = cv2.connectedComponentsWithStats((~out).astype(np.uint8), connectivity=4)
        border = set(np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])).tolist())
        for j in range(1, k):
            if j not in border:
                out |= lab2 == j
    return out


def ring_mask(mask: np.ndarray, width: int = 12) -> np.ndarray:
    """Outer ring `width` px wide around the mask (excluding the 2 px next to the edge)."""
    m8 = mask.astype(np.uint8)
    outer = cv2.dilate(m8, np.ones((2 * width + 1,) * 2, np.uint8)) > 0
    inner = cv2.dilate(m8, np.ones((5, 5), np.uint8)) > 0
    return outer & ~inner


def background_model(rgb: np.ndarray, mask: np.ndarray, width: int = 12):
    """(per-pixel distance to the outer water, noise threshold t0) or None if the ring is too small.

    Distance = Lab delta-E to the median of the outer ring, smoothed with a 0.7 px Gaussian; t0 = 90th percentile of
    that distance within the ring itself (minimum 6)."""
    ring = ring_mask(mask, width)
    if ring.sum() < 50:
        return None
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    med = np.median(lab[ring], axis=0)
    dist = cv2.GaussianBlur(np.linalg.norm(lab - med, axis=2), (0, 0), 0.7)
    return dist, max(6.0, float(np.percentile(dist[ring], 90)))


def net_alpha(rgb: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Fine alpha [0, 1] of a net: inside the mask, remove what is indistinguishable from the surrounding water.

    Linear ramp of the water distance between t0 and 2*t0 (see `background_model`)."""
    bg = background_model(rgb, mask)
    if bg is None:
        return None
    dist, t0 = bg
    return (np.clip((dist - t0) / t0, 0.0, 1.0) * mask).astype(np.float32)


def background_fraction(rgb: np.ndarray, alpha: np.ndarray, ref_mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Fraction of opaque pixels (alpha > 0.5) indistinguishable from the outer-ring water (distance <= t0).

    The ring is computed on `ref_mask` (by default the mask itself) so that two masks of the same object are measured
    with the same background model. It is a colour probe, not ground truth: for nets with a fine alpha it is low by
    construction."""
    m = alpha > 0.5
    bg = background_model(rgb, (alpha > 0) if ref_mask is None else ref_mask)
    if bg is None or not m.any():
        return None
    dist, t0 = bg
    return float((dist[m] <= t0).mean())


def rect_fill(alpha: np.ndarray) -> float:
    """Opaque fraction of the mask-fitted box (`crop_filter` criterion); alpha uint8 0-255 or float 0-1."""
    thr = ALPHA_THR if alpha.dtype == np.uint8 else ALPHA_THR / 255.0
    ys, xs = np.where(alpha > thr)
    if xs.size == 0:
        return 0.0
    return float(xs.size / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)))


# ------------------------------------------------------------------ output
def make_rgba(rgb: np.ndarray, alpha01: np.ndarray, pad: int = 0) -> Optional[np.ndarray]:
    """RGBA uint8 fitted to the box of alpha > 0 plus `pad` px per side (None if there are no pixels).

    As in the Extracted_objects_train library, RGB is kept outside the mask (alpha 0): matting and the generator
    medium computation need it."""
    bb = mask_bbox(alpha01 > 0)
    if bb is None:
        return None
    x, y, w, h = bb
    h_img, w_img = alpha01.shape
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w_img, x + w + pad), min(h_img, y + h + pad)
    a = np.round(alpha01[y0:y1, x0:x1] * 255).astype(np.uint8)
    return np.dstack([rgb[y0:y1, x0:x1], a])


def mask_to_polygons(mask: np.ndarray, min_area: int = 20) -> List[List[float]]:
    """External contours of the mask as flat COCO rings (simplified to 1 px)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) >= 3:
            out.append([float(v) for v in c.ravel()])
    return out
