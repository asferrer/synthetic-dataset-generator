"""Sharpness and grain of the pasted object (profile v11, variants k-n).

After compositing, pasted objects come out blurrier and with less grain than real ones (lower high-frequency ratio,
lower Laplacian variance, less noise in flat areas). `match_sharpness` fixes both on the already harmonised object and
before the alpha:

  * Sharpness: `hf_ratio` (fraction of spectral energy above half Nyquist) is measured on the central 50 % of the
    object box resampled to 64 x 64, and driven to the value of a REAL object of its class (quantile sampled from
    `real_sharpness_train.json`, computed on the training split only). If below, an unsharp mask with an intensity
    found by bisection is applied; if well above, a slight blur is applied. The unsharp mask uses a blur normalised by
    the alpha mask (it does not drag in the RGB outside the object).
  * Grain: Gaussian noise on the object with the deviation it lacks to match the flat-area noise of the background ring
    (8 x 8 block estimator).

Stateless; randomness comes from a `random.Random` passed in.
"""

import json
import os
from functools import lru_cache
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

DEFAULT_SHARPNESS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_sharpness_train.json")
MEASURE_SIDE = 64
CENTRAL_FRACTION = 0.5     # the statistic is measured on the central 50 % of the box
FLAT_BLOCK = 8
FLAT_FRACTION = 0.2        # fraction of least textured blocks used to estimate the noise
MIN_MEASURE_PIXELS = 8
BLUR_TRIGGER = 1.5         # blur only if the object exceeds the target by this factor
BISECTION_STEPS = 6


class SharpnessStats:
    """Per-class quantiles (0..100 % in 10 % steps) of hf_ratio of the real training objects."""

    def __init__(self, data: dict):
        self.source = data.get("source", "")
        self._hf = {c.lower(): np.asarray(v["hf_ratio_quantiles"], dtype=np.float64) for c, v in data["classes"].items()}

    def target_hf(self, class_name: str, u: float) -> float:
        q = self._hf.get(class_name.lower(), self._hf["all"])
        return float(np.interp(u, np.linspace(0.0, 1.0, len(q)), q))


@lru_cache(maxsize=4)
def load_sharpness_stats(path: Optional[str] = None) -> SharpnessStats:
    with open(path or DEFAULT_SHARPNESS_PATH, "r", encoding="utf-8") as f:
        return SharpnessStats(json.load(f))


# ------------------------------------------------------------------ measures
def hf_ratio(g64: np.ndarray) -> float:
    """Fraction of spectral energy (without DC) above half Nyquist, in a 64 x 64 image."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(g64 - g64.mean()))) ** 2
    yy, xx = np.mgrid[-32:32, -32:32]
    return float(f[np.hypot(xx, yy) > 16].sum() / max(f.sum(), 1e-9))


def to_measure_size(gray: np.ndarray) -> np.ndarray:
    """Gray region resampled to 64 x 64 (AREA if larger, LINEAR otherwise)."""
    interp = cv2.INTER_AREA if min(gray.shape) >= MEASURE_SIDE else cv2.INTER_LINEAR
    return cv2.resize(gray, (MEASURE_SIDE, MEASURE_SIDE), interpolation=interp)


def central_region(gray: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Central 50 % of the box (x1, y1, x2, y2) of a gray image; None if too small."""
    x1, y1, x2, y2 = box
    m = (1.0 - CENTRAL_FRACTION) / 2.0
    cx1, cx2 = int(x1 + m * (x2 - x1)), int(x2 - m * (x2 - x1))
    cy1, cy2 = int(y1 + m * (y2 - y1)), int(y2 - m * (y2 - y1))
    cx2, cy2 = max(cx2, cx1 + 4), max(cy2, cy1 + 4)
    if cx2 - cx1 < MIN_MEASURE_PIXELS or cy2 - cy1 < MIN_MEASURE_PIXELS:
        return None
    return gray[cy1:cy2, cx1:cx2]


def mask_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < MIN_MEASURE_PIXELS:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def object_hf(bgr: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """hf_ratio of the central 50 % of the mask box (None if it cannot be measured)."""
    box = mask_box(mask)
    if box is None:
        return None
    region = central_region(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), box)
    return None if region is None else hf_ratio(to_measure_size(region))


def flat_noise(gray: np.ndarray, skip: Optional[np.ndarray] = None) -> Optional[float]:
    """Median standard deviation of the 20 % least textured 8 x 8 blocks (blocks where `skip` is True do not count)."""
    h, w = (gray.shape[0] // FLAT_BLOCK) * FLAT_BLOCK, (gray.shape[1] // FLAT_BLOCK) * FLAT_BLOCK
    if h < FLAT_BLOCK or w < FLAT_BLOCK:
        return None
    shape = (h // FLAT_BLOCK, FLAT_BLOCK, w // FLAT_BLOCK, FLAT_BLOCK)
    sd = gray[:h, :w].reshape(shape).std(axis=(1, 3))
    if skip is not None:
        sd = sd[~skip[:h, :w].reshape(shape).any(axis=(1, 3))]
    sd = np.sort(sd.ravel())
    return float(np.median(sd[: max(1, int(len(sd) * FLAT_FRACTION))])) if len(sd) >= 5 else None


def object_noise(bgr: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Flat-area noise of the object (only blocks inside the eroded mask)."""
    inner = cv2.erode((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return flat_noise(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), skip=~inner)


def ring_noise(composite: np.ndarray, box_xywh: Tuple[int, int, int, int], ring_frac: float = 0.5) -> Optional[float]:
    """Flat-area noise of the ring around the box (blocks touching the box do not count)."""
    h, w = composite.shape[:2]
    x, y, bw, bh = box_xywh
    pad = max(FLAT_BLOCK, int(round(ring_frac * max(bw, bh))))
    wx1, wy1, wx2, wy2 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
    gray = cv2.cvtColor(composite[wy1:wy2, wx1:wx2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    inside = np.zeros(gray.shape, dtype=bool)
    inside[max(0, y - wy1):y + bh - wy1, max(0, x - wx1):x + bw - wx1] = True
    return flat_noise(gray, skip=inside)


# ------------------------------------------------------------------ corrections
def normalized_blur(bgr: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Mask-normalised Gaussian blur: pixels outside the object contribute no colour."""
    m = (mask > 0).astype(np.float32)
    num = cv2.GaussianBlur(bgr.astype(np.float32) * m[:, :, None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)[:, :, None]


def _unsharp(bgr: np.ndarray, low: np.ndarray, amount: float) -> np.ndarray:
    full = bgr.astype(np.float32)
    return np.clip(full + amount * (full - low), 0, 255).astype(np.uint8)


def _box_side(mask: np.ndarray) -> float:
    box = mask_box(mask)
    return float(max(box[2] - box[0], box[3] - box[1])) if box else float(MEASURE_SIDE)


def sharpen_to(bgr: np.ndarray, mask: np.ndarray, target: float, max_amount: float) -> np.ndarray:
    """Unsharp mask whose intensity (bisection in [0, max_amount]) brings hf_ratio to the target."""
    low = normalized_blur(bgr, mask, max(0.6, 0.5 * _box_side(mask) / MEASURE_SIDE))   # sigma ~ 1 px at scale 64
    top = _unsharp(bgr, low, max_amount)
    if (object_hf(top, mask) or 0.0) < target:
        return top
    lo, hi = 0.0, max_amount
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        if (object_hf(_unsharp(bgr, low, mid), mask) or 0.0) < target:
            lo = mid
        else:
            hi = mid
    return _unsharp(bgr, low, hi)


def soften_to(bgr: np.ndarray, mask: np.ndarray, target: float) -> np.ndarray:
    """Gaussian blur (bisection on sigma <= 1.5 px at scale 64) that lowers hf_ratio towards the target."""
    lo, hi = 0.0, 1.5 * max(1.0, _box_side(mask) / MEASURE_SIDE)
    best = bgr
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        cand = np.clip(normalized_blur(bgr, mask, max(mid, 0.1)), 0, 255).astype(np.uint8)
        if (object_hf(cand, mask) or 0.0) > target:
            lo = mid
        else:
            hi, best = mid, cand
    return best


def add_grain(bgr: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise of deviation `sigma` (gray) with a small chromatic part."""
    if sigma <= 0:
        return bgr
    h, w = bgr.shape[:2]
    noise = rng.normal(0.0, sigma, (h, w, 1)) * 0.9 + rng.normal(0.0, sigma, (h, w, 3)) * 0.4
    return np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def grain_sigma(obj_noise: Optional[float], env_noise: Optional[float], cap: float) -> float:
    """Noise deviation the object lacks to match the ring (in quadrature), capped."""
    if obj_noise is None or env_noise is None:
        return 0.0
    return float(min(cap, np.sqrt(max(0.0, env_noise ** 2 - obj_noise ** 2))))


def match_sharpness(
    obj_bgr: np.ndarray, mask: np.ndarray, composite: np.ndarray, box_xywh: Tuple[int, int, int, int],
    class_name: str, profile, rnd, stats: Optional[SharpnessStats] = None,
) -> Tuple[np.ndarray, Dict[str, Optional[float]]]:
    """Object with the sharpness of a real one of its class and the background grain; (object, measurements for the trace).

    `rnd`: an already seeded `random.Random` (always consumes 2 numbers, whether or not a correction is applied).
    """
    stats = stats or load_sharpness_stats(profile.sharpness_stats_path)
    u = rnd.uniform(*profile.sharpness_quantile_range)
    grain_rng = np.random.default_rng(rnd.getrandbits(32))
    target = stats.target_hf(class_name, u) * profile.sharpness_gain
    before = object_hf(obj_bgr, mask)
    info: Dict[str, Optional[float]] = {"target_hf": round(target, 5), "grain": 0.0,
                                        "hf_before": None if before is None else round(before, 5)}
    out = obj_bgr
    if before is not None and before < target:
        out = sharpen_to(obj_bgr, mask, target, profile.sharpness_max_amount)
    elif before is not None and before > BLUR_TRIGGER * target:
        out = soften_to(obj_bgr, mask, target)
    if profile.sharpness_grain:
        sigma = grain_sigma(object_noise(out, mask), ring_noise(composite, box_xywh), profile.sharpness_grain_max)
        out, info["grain"] = add_grain(out, sigma, grain_rng), round(sigma, 3)
    after = object_hf(out, mask)
    info["hf_after"] = None if after is None else round(after, 5)
    return out, info
