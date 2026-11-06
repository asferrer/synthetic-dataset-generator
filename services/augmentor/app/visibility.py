"""Visibility of a pasted object: local contrast against the surrounding background ring (profile v11).

With copy-paste the background is a real image, so a very translucent, hazy or blurred object ends up almost invisible
and its box would be label noise. The measure is

    C = mean over the box pixels of || LAB(pixel) - LAB mean of the ring ||

(Euclidean distance in OpenCV LAB, 0-255). The ring is the box grown by `ring_frac` x its longest side (minimum 4 px)
minus the box itself, the same one used by harmonisation. An object that blends with its surroundings gets a C close
to that of the background texture. The same measure is computed on the REAL training objects
and the threshold is set at a low percentile of that distribution:
whatever is less visible than almost any real object is discarded.

Stateless and without random numbers.
"""

import json
import os
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.harmonize import MIN_RING_PIXELS

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2
MIN_BOX_PIXELS = 16
MAX_WINDOW_SIDE = 256  # the window (box + ring) is shrunk to this longest side to speed up the measure


def _window(box: Box, shape_hw: Tuple[int, int], ring_frac: float) -> Optional[Tuple[Box, Box]]:
    """(clipped window, box inside the window), both (x1, y1, x2, y2); None if the box is empty."""
    h, w = shape_hw
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    if x2 <= x1 or y2 <= y1:
        return None
    pad = max(4, int(round(ring_frac * max(x2 - x1, y2 - y1))))
    wx1, wy1, wx2, wy2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    return (wx1, wy1, wx2, wy2), (x1 - wx1, y1 - wy1, x2 - wx1, y2 - wy1)


def _shrink(win_bgr: np.ndarray, box: Box) -> Tuple[np.ndarray, Box]:
    side = max(win_bgr.shape[:2])
    if side <= MAX_WINDOW_SIDE:
        return win_bgr, box
    s = MAX_WINDOW_SIDE / side
    small = cv2.resize(win_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bx1, by1, bx2, by2 = (int(round(v * s)) for v in box)
    return small, (bx1, by1, max(bx2, bx1 + 1), max(by2, by1 + 1))


def local_contrast(image_bgr: np.ndarray, box: Box, ring_frac: float = 0.5) -> Optional[float]:
    """Local contrast C of the box against its ring; None if there are not enough pixels to measure it."""
    win = _window(box, image_bgr.shape[:2], ring_frac)
    if win is None:
        return None
    (wx1, wy1, wx2, wy2), inner = win
    small, (bx1, by1, bx2, by2) = _shrink(image_bgr[wy1:wy2, wx1:wx2], inner)
    inside = np.zeros(small.shape[:2], dtype=bool)
    inside[by1:by2, bx1:bx2] = True
    ring = ~inside  # the window is already the grown box: ring = window - box
    if ring.sum() < MIN_RING_PIXELS or inside.sum() < MIN_BOX_PIXELS:
        return None
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring_mean = lab[ring].mean(axis=0)
    return float(np.linalg.norm(lab[inside] - ring_mean, axis=1).mean())


def box_luminance_std(image_bgr: np.ndarray, box: Box) -> Optional[float]:
    """Standard deviation of L (LAB) inside the box."""
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(image_bgr.shape[1], box[2]), min(image_bgr.shape[0], box[3])
    if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < MIN_BOX_PIXELS:
        return None
    return float(cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)[:, :, 0].std())


def is_visible(image_bgr: np.ndarray, box: Box, threshold: float, ring_frac: float = 0.5,
               scale: float = 1.0) -> Tuple[bool, Optional[float]]:
    """(visible, C): visible if `C * scale >= threshold`. If it cannot be measured it is accepted (no blind discards).

    `scale` accounts for what a later global effect (haze) will subtract from the contrast measured before applying it.
    """
    c = local_contrast(image_bgr, box, ring_frac)
    if c is None:
        return True, None
    return c * scale >= threshold, c


def percentile_table(values_by_class: Dict[str, Sequence[float]], qs: Sequence[float] = (1, 5, 10, 50)) -> dict:
    """Percentiles of C per class and overall (to set the threshold and document it)."""
    out = {}
    pooled = [v for vals in values_by_class.values() for v in vals]
    for name, vals in list(values_by_class.items()) + [("ALL", pooled)]:
        if len(vals):
            arr = np.asarray(vals, dtype=np.float64)
            out[name] = {"n": int(arr.size), **{f"p{q:g}": round(float(np.percentile(arr, q)), 2) for q in qs}}
    return out


DEFAULT_PERCENTILES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                                        "visibility_percentiles_train.json")


@lru_cache(maxsize=2)
def load_percentile_table(path: str = DEFAULT_PERCENTILES_PATH) -> Dict[str, Dict[str, float]]:
    """{class: {"p5": C, "p15": C, ...}} of the REAL training objects (training split only)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["C"]


@lru_cache(maxsize=8)
def percentile_thresholds(percentile: int) -> Dict[str, float]:
    """Per-class C threshold for a percentile of the real objects (clear error if the table lacks it)."""
    key = f"p{percentile:g}"
    table = {c: v[key] for c, v in load_percentile_table().items() if c != "ALL" and key in v}
    if not table:
        available = sorted({k for v in load_percentile_table().values() for k in v if k.startswith("p")})
        raise ValueError(f"visibility_percentile={percentile}: not in the table (available: {available})")
    return table


def threshold_for(profile, class_name: str) -> Optional[float]:
    """C threshold of a class (None = no filter: filter off or class absent from the dict).

    With `visibility_percentile` the threshold is that percentile of the class's real training objects.
    """
    if not profile.visibility_filter:
        return None
    if profile.visibility_percentile is not None:
        return percentile_thresholds(profile.visibility_percentile).get(class_name)
    thr = profile.visibility_min_contrast
    return thr.get(class_name) if isinstance(thr, dict) else float(thr)


def build_report(entries: Sequence[dict], pasted: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    """Per-class filter summary: rejected attempts, dropped objects and pasted objects that remained.

    `entries`: per-image records of the batch; the filter writes into `entry["synth"]["visibility"]` the rejected
    attempts (`rejected`) and the objects left unplaced because of it (`dropped`), per class.
    """
    report: Dict[str, Dict[str, int]] = {cls: {"rejected_attempts": 0, "dropped_objects": 0, "pasted": n}
                                         for cls, n in pasted.items()}
    for e in entries:
        v = (e.get("synth") or {}).get("visibility") or {}
        for key, field in (("rejected", "rejected_attempts"), ("dropped", "dropped_objects")):
            for cls, n in v.get(key, {}).items():
                report.setdefault(cls, {"rejected_attempts": 0, "dropped_objects": 0, "pasted": 0})[field] += n
    return report
