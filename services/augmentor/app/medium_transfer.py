"""Ring-to-ring medium transfer in linear RGB (profile v11, improvement M1).

Idea: the crop carries the colour of ITS water. Instead of matching the object to the background (object-to-background
Reinhard, which erases the object's own colour), the medium change is estimated from the RING surrounding the object in
the source and in the destination, and that same map is applied to the object:

  * `ring_gain`: per-channel gain in linear space, g = (m_d - v_d) / (m_s - v_s), clipped to [0.5, 2];
    I' = g * (F - v_s) + v_d (with `dark_floor`; without it v = 0 and g = m_d / m_s).
    m = ring median, v = 1st percentile of the whole image (approximation of the backscatter floor).
  * `ring_hist`: monotone histogram map between the two rings (Pitie 2007), 64 bins per channel, smoothed.

The output is blended with the input with strength `lam`: F' = (1 - lam) * F + lam * I'. All computation is in linear
RGB (inverse sRGB) and returns to 8-bit sRGB. Pure functions (OpenCV and NumPy only); independent of the rest of the
generator. Correct order in the compositor: alpha + foreground F first, medium on F, composition
alpha * F' + (1 - alpha) * background.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

RING_IN = 3                  # px between the mask and the ring (excludes the edge halo)
RING_OUT_FRAC = 0.5          # r_out = clip(0,5 * sqrt(area), 12, 40) px
RING_OUT_RANGE = (12.0, 40.0)
MIN_RING_PIXELS = 24         # fewer ring pixels: not estimated (identity)
GAIN_RANGE = (0.5, 2.0)
MIN_SIGNAL = 0.02            # m - v below this (linear) -> g = 1
FLOOR_PERCENTILE = 1.0
FLOOR_MAX_SIDE = 400         # maximum side of the subsample used to measure the dark floor
HIST_BINS = 64
MODES = ("ring_gain", "ring_hist")


def _build_lut() -> np.ndarray:
    u = np.arange(256, dtype=np.float32) / 255.0
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4).astype(np.float32)


_SRGB_TO_LIN = _build_lut()


def to_linear(bgr_u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB -> linear float32 in [0, 1] (same shape)."""
    return _SRGB_TO_LIN[bgr_u8]


def to_srgb_u8(lin: np.ndarray) -> np.ndarray:
    """linear float -> uint8 sRGB (clipped to [0, 1])."""
    lin = np.clip(lin, 0.0, 1.0)
    s = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


def ring_outer_px(area: float) -> float:
    """Outer ring radius (px) for an object of `area` px."""
    return float(np.clip(RING_OUT_FRAC * math.sqrt(max(area, 0.0)), *RING_OUT_RANGE))


@dataclass
class MediumStats:
    """Medium statistics of an image around an object: ring pixels (linear, n x 3) and dark floor (3,)."""

    pixels: np.ndarray
    floor: np.ndarray

    @property
    def median(self) -> np.ndarray:
        return np.median(self.pixels, axis=0)


def dark_floor(img_bgr: np.ndarray, percentile: float = FLOOR_PERCENTILE) -> np.ndarray:
    """Per-channel `percentile` of the whole image in linear space (subsampled with a regular stride)."""
    step = max(1, max(img_bgr.shape[:2]) // FLOOR_MAX_SIDE)
    return np.percentile(to_linear(img_bgr[::step, ::step]).reshape(-1, 3), percentile, axis=0).astype(np.float32)


def ring_region(mask: np.ndarray, exclude: Optional[np.ndarray] = None, r_in: float = RING_IN,
                r_out: Optional[float] = None) -> np.ndarray:
    """Boolean ring outside `mask`: distance to the object in (r_in, r_out]; without the `exclude` pixels."""
    m = mask > 0
    if r_out is None:
        r_out = ring_outer_px(float(m.sum()))
    d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
    ring = (d > r_in) & (d <= r_out)
    if exclude is not None:
        ring &= ~(exclude > 0)
    return ring


def medium_stats(img_bgr: np.ndarray, mask: np.ndarray, exclude: Optional[np.ndarray] = None,
                 floor_img: Optional[np.ndarray] = None, floor: Optional[np.ndarray] = None) -> Optional[MediumStats]:
    """Ring statistics of object `mask` in `img_bgr` (same H x W shape). None if the ring is too small.

    Only a window around the object is examined. The dark floor comes from `floor_img` (the image itself by default),
    unless it is passed precomputed in `floor`."""
    m = mask > 0
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    r_out = ring_outer_px(float(m.sum()))
    pad = int(math.ceil(r_out)) + 2
    h, w = m.shape
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    ex = None if exclude is None else exclude[y0:y1, x0:x1]
    ring = ring_region(m[y0:y1, x0:x1], ex, RING_IN, r_out)
    if int(ring.sum()) < MIN_RING_PIXELS:
        return None
    if floor is None:
        floor = dark_floor(img_bgr if floor_img is None else floor_img)
    return MediumStats(to_linear(img_bgr[y0:y1, x0:x1][ring]), np.asarray(floor, np.float32))


def channel_gains(src: MediumStats, dst: MediumStats, use_floor: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(g, v_s, v_d) per channel. Without dark floor v = 0. Where the source signal is < MIN_SIGNAL, g = 1."""
    zero = np.zeros(3, np.float32)
    v_s, v_d = (src.floor, dst.floor) if use_floor else (zero, zero)
    sig_s, sig_d = src.median - v_s, dst.median - v_d
    g = np.where(sig_s >= MIN_SIGNAL, sig_d / np.maximum(sig_s, 1e-6), 1.0)
    return np.clip(g, *GAIN_RANGE).astype(np.float32), v_s, v_d


def _smooth(v: np.ndarray) -> np.ndarray:
    return np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")


def hist_map(src_px: np.ndarray, dst_px: np.ndarray, bins: int = HIST_BINS) -> Tuple[np.ndarray, np.ndarray]:
    """(source quantiles, destination quantiles) of a channel: points of a smoothed monotone source -> destination map."""
    p = (np.arange(bins) + 0.5) / bins
    qs, qd = np.quantile(src_px, p), np.quantile(dst_px, p)
    qs, qd = np.maximum.accumulate(_smooth(qs)), np.maximum.accumulate(_smooth(qd))
    return qs, qd


def apply_hist_map(x: np.ndarray, qs: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Interpolated map; outside the ring range it shifts like the end point (slope 1)."""
    if qs[-1] - qs[0] < 1e-6:
        return x + (np.median(qd) - np.median(qs))
    y = np.interp(x, qs, qd)
    y = np.where(x < qs[0], x + (qd[0] - qs[0]), y)
    return np.where(x > qs[-1], x + (qd[-1] - qs[-1]), y)


def transfer(obj_bgr: np.ndarray, src: Optional[MediumStats], dst: Optional[MediumStats], mode: str = "ring_gain",
             lam: float = 1.0, use_floor: bool = False) -> np.ndarray:
    """`obj_bgr` (uint8) with the medium of ring `src` swapped for that of `dst`; identity if either is missing or lam <= 0."""
    if mode not in MODES:
        raise ValueError(f"medium_transfer: unknown mode {mode!r} (available: {MODES})")
    if lam <= 0 or src is None or dst is None:
        return obj_bgr
    f = to_linear(obj_bgr)
    if mode == "ring_gain":
        g, v_s, v_d = channel_gains(src, dst, use_floor)
        out = g * (f - v_s) + v_d
    else:
        out = np.empty_like(f)
        for c in range(3):
            qs, qd = hist_map(src.pixels[:, c], dst.pixels[:, c])
            out[..., c] = apply_hist_map(f[..., c], qs, qd)
    lam = min(float(lam), 1.0)
    return to_srgb_u8((1.0 - lam) * f + lam * out)
