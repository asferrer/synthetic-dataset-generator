"""Local colour harmonisation of the pasted object (profile v11).

Reinhard-style (2001) LAB statistics transfer restricted to a ring around the object's box: the mean and standard
deviation of each object channel (opaque pixels) move towards those of the background ring. It is a cheap alternative
to Harmonizer / PCT-Net. `strength` controls a, b; the L channel gets `strength * l_weight` so detail is not destroyed.
It is applied on top of the compositor's colour correction (which only uses the rectangle covered by the object).
"""

from typing import Tuple

import cv2
import numpy as np

MIN_RING_PIXELS = 64
RATIO_RANGE = (0.5, 2.0)  # bound on the std ratio (nearly flat objects blow up Reinhard)


def ring_mask(shape_hw: Tuple[int, int], box_xywh: Tuple[int, int, int, int], ring_frac: float) -> np.ndarray:
    """Boolean ring mask: box grown by `ring_frac * longest side` minus the box itself."""
    h, w = shape_hw
    x, y, bw, bh = box_xywh
    pad = max(4, int(round(ring_frac * max(bw, bh))))
    ring = np.zeros((h, w), dtype=bool)
    ring[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)] = True
    ring[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)] = False
    return ring


def _stats(lab: np.ndarray, where: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    px = lab[where]
    return px.mean(axis=0), px.std(axis=0) + 1e-6


def harmonize_object(
    obj_bgr: np.ndarray, obj_mask: np.ndarray, composite: np.ndarray,
    box_xywh: Tuple[int, int, int, int], strength: float, l_weight: float = 0.5, ring_frac: float = 0.5,
) -> np.ndarray:
    """Returns `obj_bgr` with LAB statistics moved towards those of the ring; no effect if there is not enough data."""
    if strength <= 0:
        return obj_bgr
    ring = ring_mask(composite.shape[:2], box_xywh, ring_frac)
    opaque = obj_mask > 0
    if ring.sum() < MIN_RING_PIXELS or opaque.sum() < MIN_RING_PIXELS // 4:
        return obj_bgr
    obj_lab = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    env_lab = cv2.cvtColor(composite, cv2.COLOR_BGR2LAB).astype(np.float32)
    mo, so = _stats(obj_lab, opaque)
    mr, sr = _stats(env_lab, ring)
    ratio = np.clip(sr / so, *RATIO_RANGE)
    transferred = (obj_lab - mo) * ratio + mr
    weight = np.array([strength * l_weight, strength, strength], dtype=np.float32)
    out = obj_lab * (1.0 - weight) + transferred * weight
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
