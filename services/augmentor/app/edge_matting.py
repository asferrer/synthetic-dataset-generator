"""Trimap matting and foreground decontamination of the crop (profile v11, improvement M2).

Idea: the library crop comes with a binary SAM mask and, inside its box, the original pixels (also those outside the
mask: the source background). Contour pixels are a mix of object + source background; pasted on another background
they leave a halo of the source water. Here:

  1. trimap from the mask: foreground = erosion by e px, background = outside the dilation by e px, unknown = the band
     (e = clip(round(0.06 * sqrt(area)), 2, 8)); transparent nets and plastics: unknown = 6 px dilation minus 1 px
     erosion (almost the whole footprint);
  2. fractional alpha by Closed-Form Matting (Levin 2008, `pymatting`, MIT, CPU) -- or KNN / Large Kernel;
  3. foreground F by Fast Multi-Level Foreground Estimation (Germer 2020, `estimate_foreground_ml`) -- or in closed
     form -- instead of the original pixel;
  4. weak-halo clipping: alpha' = clip((alpha - a0) / (1 - a0), 0, 1).

The output is a BGRA crop with fractional alpha' and F as colour, the same size as the crop (optionally enlarged by
`pad` px per side replicating the border; unusual: outside the box the colour would be made up). It is applied BEFORE
scaling/rotating the crop and before any medium change: medium on F, composition alpha' * F' + (1 - alpha') * background.
`pymatting` is imported lazily; if missing, a profile with `edge_matting` fails with a clear message (dependency in
requirements.txt).
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

ALPHA_THR = 10                 # threshold of the crop's binary mask (as in the compositor)
TRANSPARENT_CLASSES = ("Fishing_Net", "Plastic_Debris")
TRANSPARENT_DILATE = 6
TRANSPARENT_ERODE = 1
SOLID_FRAC = 0.06
SOLID_RANGE = (2, 8)
MIN_AREA = 40                  # smaller masks are not processed (the crop is returned as is, with pad)
ALPHA_METHODS = ("closed_form", "knn", "lkm")
FG_METHODS = ("germer", "closed_form", "none")


def available() -> bool:
    """True if `pymatting` can be imported."""
    try:
        import pymatting  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _pm():
    try:
        import pymatting
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError("edge_matting needs the `pymatting` package (MIT): pip install pymatting "
                          "(services/augmentor/requirements.txt)") from e
    return pymatting


def band_width(area: float) -> int:
    """e_f = e_b for solid objects: clip(round(0.06 * sqrt(area)), 2, 8) px."""
    return int(np.clip(round(SOLID_FRAC * math.sqrt(max(area, 0.0))), *SOLID_RANGE))


def pad_for(cls: Optional[str], area: float) -> int:
    """Margin (px per side) added to the crop so the outer unknown band fits."""
    return (TRANSPARENT_DILATE if cls in TRANSPARENT_CLASSES else band_width(area)) + 1


def _ellipse(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def make_trimap(mask: np.ndarray, cls: Optional[str] = None) -> np.ndarray:
    """float32 trimap (1 foreground, 0 background, 0.5 unknown) from the binary mask `mask` (>0 = object)."""
    m = (mask > 0).astype(np.uint8)
    area = float(m.sum())
    if cls in TRANSPARENT_CLASSES:
        e_f, e_b = TRANSPARENT_ERODE, TRANSPARENT_DILATE
    else:
        e_f = e_b = band_width(area)
    fg = cv2.erode(m, _ellipse(e_f))
    while not fg.any() and e_f > 1:       # narrow object: the erosion eats the whole foreground
        e_f -= 1
        fg = cv2.erode(m, _ellipse(e_f))
    if not fg.any():
        fg = cv2.erode(m, _ellipse(1)) if cv2.erode(m, _ellipse(1)).any() else m
    bg_zone = cv2.dilate(m, _ellipse(e_b)) == 0
    tri = np.full(m.shape, 0.5, np.float32)
    tri[bg_zone] = 0.0
    tri[fg > 0] = 1.0
    return tri


def estimate_alpha(rgb01: np.ndarray, trimap: np.ndarray, method: str = "closed_form") -> np.ndarray:
    """float32 alpha in [0, 1]. `rgb01` is float64 (h, w, 3) in [0, 1]."""
    pm = _pm()
    if method == "closed_form":
        a = pm.estimate_alpha_cf(rgb01, trimap.astype(np.float64))
    elif method == "knn":
        a = pm.estimate_alpha_knn(rgb01, trimap.astype(np.float64))
    elif method == "lkm":
        a = pm.estimate_alpha_lkm(rgb01, trimap.astype(np.float64))
    else:
        raise ValueError(f"edge_matting: unknown alpha method {method!r} (available: {ALPHA_METHODS})")
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def estimate_fg(rgb01: np.ndarray, alpha: np.ndarray, method: str = "germer") -> np.ndarray:
    """Foreground F as float in [0, 1] (h, w, 3); `none` returns the original image."""
    if method == "none":
        return rgb01
    pm = _pm()
    a = alpha.astype(np.float64)
    if method == "germer":
        f = pm.estimate_foreground_ml(rgb01, a)
    elif method == "closed_form":
        f = pm.estimate_foreground_cf(rgb01, a)
    else:
        raise ValueError(f"edge_matting: unknown foreground method {method!r} (available: {FG_METHODS})")
    return np.clip(f, 0.0, 1.0)


def matte_bgra(crop_bgra: np.ndarray, cls: Optional[str] = None, alpha_method: str = "closed_form",
               fg_method: str = "germer", a0: float = 0.0, pad: int = 0) -> Tuple[np.ndarray, int]:
    """(BGRA crop with fractional alpha' and F, margin added per side). Any failure returns the crop with the margin.

    By default `pad` = 0: the matte lives inside the crop box (the box keeps the original pixels, also those outside
    the mask) and there is nothing outside. With `pad` > 0 (e.g. `pad_for`) the crop is enlarged by replicating its
    border (the binary alpha outside is 0): alpha may spill out of the mask, but outside the box the colour is made up."""
    mask = (crop_bgra[:, :, 3] > ALPHA_THR).astype(np.uint8)
    area = float(mask.sum())
    padded = cv2.copyMakeBorder(crop_bgra, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else crop_bgra.copy()
    if pad:
        padded[:pad, :, 3] = 0
        padded[-pad:, :, 3] = 0
        padded[:, :pad, 3] = 0
        padded[:, -pad:, 3] = 0
    if area < MIN_AREA:
        return padded, pad
    pmask = (padded[:, :, 3] > ALPHA_THR).astype(np.uint8)
    rgb = padded[:, :, :3][:, :, ::-1].astype(np.float64) / 255.0
    try:
        alpha = estimate_alpha(rgb, make_trimap(pmask, cls), alpha_method)
        fg = estimate_fg(rgb, alpha, fg_method)
    except ImportError:
        raise
    except Exception:  # noqa: BLE001  (singular system, degenerate trimap...: the crop is kept)
        return padded, pad
    if a0 > 0:
        alpha = np.clip((alpha - a0) / (1.0 - a0), 0.0, 1.0)
    out = padded.copy()
    out[:, :, :3] = np.clip(np.rint(fg[:, :, ::-1] * 255.0), 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    return out, pad
