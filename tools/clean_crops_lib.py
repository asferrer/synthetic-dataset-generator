"""Pure pieces of the crop-library cleaning (no torch, no services; only NumPy and OpenCV).

Problem: some crop masks (mostly Fishing_Net) extend past the annotated box of the object and carry water or
background pixels that are later pasted as part of the object. This module measures the overflow and cleans the
crop alpha channel with simple rules. No SAM and no network: only the geometry of the annotated box.

Conventions: alpha is `uint8 (H, W)` in crop coordinates; the annotated box is `[x, y, w, h]` in original-image
coordinates; `pos = (x, y)` is the crop position in that image. The `CleanParams` thresholds were fixed up front and
are not tuned against downstream detection metrics.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ALPHA_THR = 10          # same threshold as the compositor and `crop_filter`
NET = "Fishing_Net"


@dataclass
class CleanParams:
    """Cleaning rules (stored in the report)."""
    margin_px: int = 4              # tolerated margin around the annotated box (px of the original image)
    min_comp_frac: float = 0.02     # after clipping, components < 2 % of the remaining area are removed
    min_area: int = 100             # minimum area after clipping (px^2)
    cov_min: float = 0.10           # minimum coverage after clipping (area / annotated box area)
    cov_min_net: float = 0.03       # same for nets (thin and sparse)
    cov_max: float = 0.97           # maximum coverage
    max_overflow: float = 0.50      # if over half of the mask lies outside box + margin, the crop is not the object
    overflow_report_thr: float = 0.05   # "overflow" threshold used in the reports


def box_window(box, pos, shape_hw, margin: int = 0) -> np.ndarray:
    """Bool mask (H, W) in crop coordinates of the annotated box grown by `margin` px on each side."""
    h, w = shape_hw
    x, y, bw, bh = box
    x0 = int(np.floor(x - pos[0] - margin))
    y0 = int(np.floor(y - pos[1] - margin))
    x1 = int(np.ceil(x + bw - pos[0] + margin))
    y1 = int(np.ceil(y + bh - pos[1] + margin))
    m = np.zeros((h, w), bool)
    m[max(0, y0):max(0, min(h, y1)), max(0, x0):max(0, min(w, x1))] = True
    return m


def components(mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """(number of components, labels, area per label with background at index 0); 8-connectivity."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n - 1, lab, stats[:, cv2.CC_STAT_AREA]


def measure(alpha: np.ndarray, box, pos, p: CleanParams) -> Dict[str, float]:
    """Measurements of a crop against its annotated box.

    - `overflow_exact`: fraction of the mask outside the annotated box (no margin).
    - `overflow_margin`: same with the box grown by `margin_px`.
    - `coverage`: mask area inside the box / box area.
    - `n_comp`: connected components of the mask; `n_comp_sig`: those with at least `min_comp_frac` of the area."""
    mask = alpha > ALPHA_THR
    area = int(mask.sum())
    barea = max(1.0, float(box[2]) * float(box[3]))
    if area == 0:
        return {"area": 0, "overflow_exact": 0.0, "overflow_margin": 0.0, "coverage": 0.0, "n_comp": 0, "n_comp_sig": 0}
    inside = int((mask & box_window(box, pos, mask.shape, 0)).sum())
    near = int((mask & box_window(box, pos, mask.shape, p.margin_px)).sum())
    n, _, areas = components(mask)
    return {"area": area, "overflow_exact": 1.0 - inside / area, "overflow_margin": 1.0 - near / area,
            "coverage": inside / barea, "n_comp": n,
            "n_comp_sig": int((areas[1:] >= p.min_comp_frac * area).sum())}


def clean_alpha(alpha: np.ndarray, box, pos, cls: str, p: CleanParams) -> Tuple[Optional[np.ndarray], Dict]:
    """(clean alpha or None if dropped, info).

    1. Clip to the annotated box + `margin_px` (outside becomes alpha 0; RGB is untouched).
    2. Remove components smaller than `min_comp_frac` of the remaining area.
    3. Drop if the overflow was > `max_overflow`, if less than `min_area` remains, or if the coverage falls outside
       [cov_min, cov_max] (cov_min_net for nets)."""
    mask = alpha > ALPHA_THR
    area0 = int(mask.sum())
    win = box_window(box, pos, mask.shape, p.margin_px)
    keep = mask & win
    area1 = int(keep.sum())
    info: Dict = {"area_orig": area0, "area_clipped": area1, "clipped_frac": 0.0 if area0 == 0 else 1.0 - area1 / area0}
    if area0 == 0 or info["clipped_frac"] > p.max_overflow:
        return None, {**info, "reason": "dominated_by_foreign_area" if area0 else "empty"}
    n, lab, areas = components(keep)
    ok = np.zeros(n + 1, bool)
    ok[1:] = areas[1:] >= p.min_comp_frac * area1
    keep = ok[lab] if n else keep
    area2 = int(keep.sum())
    info["area_final"] = area2
    info["components_removed"] = int(n - ok[1:].sum())
    info["modified"] = area2 < area0
    info["removed_frac"] = 1.0 - area2 / area0
    barea = max(1.0, float(box[2]) * float(box[3]))
    cov = area2 / barea
    info["coverage"] = cov
    if area2 < p.min_area:
        return None, {**info, "reason": "almost_nothing"}
    if cov < (p.cov_min_net if cls == NET else p.cov_min):
        return None, {**info, "reason": "low_coverage"}
    if cov > p.cov_max:
        return None, {**info, "reason": "high_coverage"}
    soft = (~mask) & (alpha > 0) & win             # soft alpha of 1 to 10 inside the window is kept
    return np.where(keep | soft, alpha, 0).astype(np.uint8), info


def shrink_rgba(rgba: np.ndarray, pad: int = 0) -> np.ndarray:
    """Crop the RGBA to the box of alpha > threshold plus `pad` px per side (RGB outside the mask is kept)."""
    ys, xs = np.where(rgba[:, :, 3] > ALPHA_THR)
    h, w = rgba.shape[:2]
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    return rgba[y0:y1, x0:x1], (int(x0), int(y0))


def original_pad(alpha: np.ndarray) -> int:
    """Padding of the original crop around its mask (0 if tight)."""
    ys, xs = np.where(alpha > ALPHA_THR)
    if xs.size == 0:
        return 0
    h, w = alpha.shape
    return int(min(xs.min(), ys.min(), w - 1 - xs.max(), h - 1 - ys.max()))


def quantiles(values: List[float]) -> Dict[str, Optional[float]]:
    """Median, p90 and max (None if there are no values)."""
    if not values:
        return {"median": None, "p90": None, "max": None}
    v = np.asarray(values, float)
    return {"median": round(float(np.median(v)), 4), "p90": round(float(np.percentile(v, 90)), 4), "max": round(float(v.max()), 4)}


def summarize(rows: List[Dict], p: CleanParams) -> Dict[str, Dict]:
    """Per class: overflow, coverage and component distributions, and what was clipped and dropped.

    Each row: cls, lib, status (clean / dropped / unlocated) and the `measure` values (if located)."""
    out: Dict[str, Dict] = {}
    for cls in sorted({r["cls"] for r in rows}):
        rs = [r for r in rows if r["cls"] == cls]
        loc = [r for r in rs if r["status"] != "unlocated"]
        thr = p.overflow_report_thr
        out[cls] = {
            "crops": len(rs), "located": len(loc), "unlocated": len(rs) - len(loc),
            "overflow_exact": quantiles([r["overflow_exact"] for r in loc]),
            "overflow_margin": quantiles([r["overflow_margin"] for r in loc]),
            "with_margin_overflow_gt_5": sum(r["overflow_margin"] > thr for r in loc),
            "with_margin_overflow_gt_15": sum(r["overflow_margin"] > 0.15 for r in loc),
            "with_margin_overflow_gt_30": sum(r["overflow_margin"] > 0.30 for r in loc),
            "coverage": quantiles([r["coverage"] for r in loc]),
            "components": quantiles([r["n_comp"] for r in loc]),
            "significant_components": quantiles([r["n_comp_sig"] for r in loc]),
            "modified": sum(r["status"] == "clean" and r.get("modified", False) for r in rs),
            "dropped": sum(r["status"] == "dropped" for r in rs),
            "drop_reasons": _count([r["reason"] for r in rs if r["status"] == "dropped"]),
            "kept": sum(r["status"] in ("clean", "unlocated") for r in rs),
            "with_area_removed_gt_5": sum(r["status"] == "clean" and r.get("removed_frac", 0) > 0.05 for r in rs),
            "with_area_removed_gt_15": sum(r["status"] == "clean" and r.get("removed_frac", 0) > 0.15 for r in rs),
            "median_area_removed_of_modified": quantiles(
                [r["removed_frac"] for r in rs if r["status"] == "clean" and r.get("modified", False)])["median"],
        }
    return out


def drops_by_class(rows: List[Dict]) -> Dict[str, Dict]:
    """Per class: crops, dropped (with reasons) and kept, plus a total. Summary for the drop-only mode."""
    out: Dict[str, Dict] = {}
    for cls in sorted({r["cls"] for r in rows}):
        rs = [r for r in rows if r["cls"] == cls]
        out[cls] = {"crops": len(rs), "dropped": sum(r["status"] == "dropped" for r in rs),
                    "reasons": _count([r["reason"] for r in rs if r["status"] == "dropped"])}
        out[cls]["kept"] = out[cls]["crops"] - out[cls]["dropped"]
    tot = {"crops": sum(v["crops"] for v in out.values()), "dropped": sum(v["dropped"] for v in out.values()),
           "kept": sum(v["kept"] for v in out.values()), "reasons": {}}
    for v in out.values():
        for k, n in v["reasons"].items():
            tot["reasons"][k] = tot["reasons"].get(k, 0) + n
    out["TOTAL"] = tot
    return out


def _count(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return dict(sorted(out.items()))
