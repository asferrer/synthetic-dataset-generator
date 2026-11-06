"""Crop choice by the scale it needs (profile v11, `crop_scale_range`).

Many pastes are shrunk below 0.5 (higher discriminator AUC than the rest) and strong enlargements lose detail and
grain. The scale is what the object needs to reach the target size of `faithful_scale` (square root of the relative
area sampled from the real distribution of its class), so it can be changed by CHOOSING ANOTHER CROP, without touching
the real-faithful size:

    needed scale of a crop = target size (px) / (native geometric side of the crop x ROT_BBOX_FACTOR)

Among the free crops (respecting `max_crop_reuse`) those leaving the scale inside `crop_scale_range` are preferred;
among them, the least used with random tie-breaking (as in `CropUsage.acquire`). If none fits, the closest to the
range is chosen (distance on a log scale). The native side is read from the PNG header (without decoding it) and cached.
"""

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Inflation of the alpha box after a uniform rotation in [-45, 45] degrees relative to the crop's native canvas
# (measured on 400 training crops: median 1.05, mean 1.07, p10-p90 0.96-1.21)
ROT_BBOX_FACTOR = 1.06
TIE_TOLERANCE = 0.05   # crops within this log distance of the best count as a tie


class CropSizes:
    """Native geometric side sqrt(width x height) of each crop, read on demand."""

    def __init__(self):
        self._side: Dict[str, float] = {}

    def side(self, path: str) -> float:
        if path not in self._side:
            self._side[path] = self._read(path)
        return self._side[path]

    @staticmethod
    def _read(path: str) -> float:
        if not os.path.isfile(path):
            return float("nan")
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        return math.sqrt(max(1, w) * max(1, h))

    def sides(self, paths: Sequence[str]) -> np.ndarray:
        return np.array([self.side(p) for p in paths], dtype=np.float64)


def needed_scales(sides: np.ndarray, target_px: float) -> np.ndarray:
    """Scale each crop would need to reach `target_px` (square root of the box area in the image)."""
    return target_px / (sides * ROT_BBOX_FACTOR)


def log_distance(scales: np.ndarray, scale_range: Tuple[float, float]) -> np.ndarray:
    """Log distance of each scale to the range; 0 inside it."""
    lo, hi = scale_range
    d = np.zeros_like(scales)
    d = np.where(scales < lo, np.log(lo / np.maximum(scales, 1e-9)), d)
    return np.where(scales > hi, np.log(np.maximum(scales, 1e-9) / hi), d)


def pick_candidates(candidates: List[str], sizes: CropSizes, target_px: float,
                    scale_range: Tuple[float, float]) -> List[str]:
    """Crops leaving the scale inside the range or, if none, the closest ones (tie = tolerance)."""
    sides = sizes.sides(candidates)
    known = np.isfinite(sides)
    if not known.any():
        return candidates
    dist = np.where(known, log_distance(needed_scales(np.where(known, sides, 1.0), target_px), scale_range), np.inf)
    best = float(dist.min())
    keep = [p for p, d in zip(candidates, dist) if d <= best + (TIE_TOLERANCE if best > 0 else 0.0)]
    return keep or candidates


def target_px_for(stats, class_name: str, quantile: float, bg_wh: Tuple[int, int]) -> Optional[float]:
    """Target size (px) of an object of the class at the given quantile; None if the class has no statistics."""
    if not stats.has_class(class_name):
        return None
    return stats.quantile(class_name, quantile) * math.sqrt(bg_wh[0] * bg_wh[1])
