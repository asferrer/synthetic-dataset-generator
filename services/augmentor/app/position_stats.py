"""Position faithful to the real training data with an edge margin (profile v11, `position_from_stats`).

Without this, over half of the pasted boxes touch the image edge (real: about a quarter), because the position is
clamped to the canvas and large objects or anchors near the edge end up stuck to it. Here:

  * Touching the edge is decided per object: with the `touch_fraction` probability of its class (the real training one,
    or the profile value) the object's box is snapped to a side chosen by the real frequency of each side; otherwise
    the box stays inside with a minimum margin (`position_margin`, fraction of the shorter side, minimum 36 px).
  * The centre is sampled from the real 2D histogram of its class: with semantic placement it weights the candidate
    anchors (`center_weight`); without it (no region map) the centre is sampled from the distribution.
Everything is done on the ALPHA BOX, not on the crop canvas (rotation leaves transparent corners).
The JSON comes from `scripts/build_position_stats.py` (training split only).
"""

import json
import os
from functools import lru_cache
from typing import Callable, Optional, Tuple

import numpy as np

DEFAULT_POSITION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "position_stats_train.json")
# After compositing the annotated box grows: the bbox is recomputed with shadows (up to 80 px) and, after the global
# effects (caustics/underwater), by up to 20 px per side. Too small a margin lets boxes that should not touch the edge
# touch it in the final output.
# 36 px = 20 (global effects) + 16 (shadows and blur).
MIN_MARGIN_PX = 36
POSITION_TRIES = 24   # candidates per placement attempt before giving up with the last one
SIDES = ("left", "right", "top", "bottom")
Box = Tuple[int, int, int, int]  # x, y, w, h


class PositionStats:
    """Per-class centre histogram, fraction of boxes touching the edge and side frequency."""

    def __init__(self, data: dict):
        self.source = data.get("source", "")
        self._classes = {c.lower(): v for c, v in data["classes"].items()}

    def _get(self, class_name: str) -> dict:
        return self._classes.get(class_name.lower(), self._classes["all"])

    def touch_fraction(self, class_name: str) -> float:
        return float(self._get(class_name)["touch_fraction"])

    def side_weights(self, class_name: str) -> list:
        sides = self._get(class_name)["touch_sides"]
        return [max(float(sides[s]), 1e-6) for s in SIDES]

    def hist(self, class_name: str) -> np.ndarray:
        return np.asarray(self._get(class_name)["center_hist"], dtype=np.float64)

    def density(self, class_name: str, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        """Real density (per cell) at normalised centres; smoothed with a floor so no cell is zeroed."""
        h = self.hist(class_name)
        n = h.shape[0]
        i = np.clip((np.asarray(cy) * n).astype(int), 0, n - 1)
        j = np.clip((np.asarray(cx) * n).astype(int), 0, n - 1)
        return h[i, j] + 0.1 / (n * n)

    def sample_center(self, class_name: str, rng) -> Tuple[float, float]:
        """Normalised centre (cx, cy) sampled from the real histogram (cell by weight, uniform inside)."""
        h = self.hist(class_name)
        n = h.shape[0]
        k = rng.choices(range(n * n), weights=(h.ravel() + 1e-9).tolist(), k=1)[0]
        return (k % n + rng.random()) / n, (k // n + rng.random()) / n


@lru_cache(maxsize=4)
def load_position_stats(path: Optional[str] = None) -> PositionStats:
    with open(path or DEFAULT_POSITION_PATH, "r", encoding="utf-8") as f:
        return PositionStats(json.load(f))


def center_weight_fn(stats: PositionStats, class_name: str, bg_wh: Tuple[int, int]) -> Callable:
    """Weight of a candidate anchor from the real density of the box centre (in image pixels)."""
    w, h = bg_wh
    return lambda cx_px, cy_px: stats.density(class_name, np.asarray(cx_px) / w, np.asarray(cy_px) / h)


def margin_px(profile, bg_wh: Tuple[int, int]) -> int:
    return max(MIN_MARGIN_PX, int(round(profile.position_margin * min(bg_wh))))


def _inside(lo: int, size: int, total: int, margin: int) -> int:
    """Origin of a segment of length `size` inside [margin, total - margin] (centred if it does not fit)."""
    if total - size < 2 * margin:
        return (total - size) // 2
    return max(margin, min(lo, total - size - margin))


def _snap(box: Box, side: str, bg_wh: Tuple[int, int]) -> Tuple[int, int]:
    x, y, w, h = box
    return {"left": (0, y), "right": (bg_wh[0] - w, y), "top": (x, 0), "bottom": (x, bg_wh[1] - h)}[side]


def decide_touch(stats: PositionStats, profile, class_name: str, rng) -> bool:
    """Whether this object's box touches the edge: Bernoulli with the real fraction of its class (or the profile's).

    Decided ONCE per object (not per attempt): otherwise rejection by overlap with the kept real objects, which
    discards the centre (where the real ones are) more than the edge, would bias survivors towards the edge.
    """
    p_touch = profile.border_touch_fraction
    return rng.random() < (stats.touch_fraction(class_name) if p_touch is None else p_touch)


def place_box(stats: PositionStats, profile, class_name: str, box: Box, bg_wh: Tuple[int, int], rng,
              resample_center: bool = False, touch: Optional[bool] = None) -> Tuple[int, int]:
    """New corner (x, y) of the alpha box `box` = (x, y, w, h) following the real distribution.

    `resample_center`: without semantic placement, the centre is drawn from the real histogram before applying the edge.
    `touch`: decision already taken (`decide_touch`); None takes it here.
    """
    x, y, w, h = box
    if resample_center:
        cx, cy = stats.sample_center(class_name, rng)
        x, y = int(cx * bg_wh[0] - w / 2), int(cy * bg_wh[1] - h / 2)
    if touch is None:
        touch = decide_touch(stats, profile, class_name, rng)
    if touch:
        side = rng.choices(SIDES, weights=stats.side_weights(class_name), k=1)[0]
        return _snap((max(0, min(x, bg_wh[0] - w)), max(0, min(y, bg_wh[1] - h)), w, h), side, bg_wh)
    m = margin_px(profile, bg_wh)
    return _inside(x, w, bg_wh[0], m), _inside(y, h, bg_wh[1], m)


def search_position(
    stats: PositionStats, profile, class_name: str, offset: Tuple[int, int], size: Tuple[int, int],
    bg_wh: Tuple[int, int], rng, first_base: Optional[Tuple[int, int]] = None, next_base: Optional[Callable] = None,
    touch: Optional[bool] = None, avoid: Optional[Callable] = None, score: Optional[Callable] = None,
    n_candidates: int = 8,
) -> Tuple[int, int]:
    """Canvas corner whose alpha box follows the real distribution and avoids the occupied boxes.

    `offset`/`size`: corner and size of the alpha box inside the canvas. `first_base`: canvas corner of the first
    attempt (semantic placement); None = the centre is drawn from the histogram. `next_base`: function giving another
    semantic corner for later attempts (None or a None result: histogram centre). `avoid`: f(box x1,y1,x2,y2) -> True
    on collision (with kept real boxes or other pasted ones); after POSITION_TRIES attempts the last one is kept.
    `score` (local_patch_match): f(box x1,y1,x2,y2) -> distance (lower = better); instead of keeping the first free
    position, up to `n_candidates` free ones are gathered (within the same POSITION_TRIES attempts) and the one with the
    lowest distance is chosen; with no free one it behaves as without `score`.
    """
    (bx, by), (w, h) = offset, size
    chosen = (0, 0)
    free = []   # (distance, canvas corner) of the free candidates
    for k in range(POSITION_TRIES):
        base = first_base if k == 0 else (next_base() if next_base is not None else None)
        px, py = base if base is not None else (0, 0)
        x, y = place_box(stats, profile, class_name, (px + bx, py + by, w, h), bg_wh, rng,
                         resample_center=base is None, touch=touch)
        chosen = (x - bx, y - by)
        if avoid is None or not avoid((x, y, x + w, y + h)):
            if score is None:
                break
            free.append((score((x, y, x + w, y + h)), chosen))
            if len(free) >= n_candidates:
                break
    if free:
        return min(free, key=lambda t: t[0])[1]
    return chosen


def touches_border(box_xyxy: Tuple[int, int, int, int], bg_wh: Tuple[int, int], tol: int = 3) -> bool:
    """True if the box (x1, y1, x2, y2) is within <= tol px of any image side."""
    return (box_xyxy[0] <= tol or box_xyxy[1] <= tol or bg_wh[0] - box_xyxy[2] <= tol
            or bg_wh[1] - box_xyxy[3] <= tol)
