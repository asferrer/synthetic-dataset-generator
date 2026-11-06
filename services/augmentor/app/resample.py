"""Single crop resampling: rotation and scale in one warpAffine (profile v11, `single_resample`).

The legacy generator (`_transform_object`) resamples the crop three times: Gaussian pre-blur if the shorter side is
< 100 px, INTER_CUBIC rotation and then rescaling (AREA or LANCZOS4). After compositing, the high-frequency ratio and
Laplacian variance drop well below the real images even though the source crops are just as sharp.

Here the crop is premultiplied by alpha (RGB outside the object, which keeps the crop's original background, does not
bleed through the edge) and resampled ONCE:
  * scale >= 1: warpAffine with rotation and scale at once, INTER_LANCZOS4;
  * scale < 1: integer box decimation (INTER_AREA, factor k = floor(1/scale), exact and alias-free) and then a single
    INTER_CUBIC warpAffine with the residual scale (in (0.5, 1]) and the rotation. No pre-blur.
It consumes no random numbers. The target scale is decided with the canvas and alpha box ALREADY rotated
(`rotated_geometry`), as in the legacy path, so `faithful_scale` receives the same thing.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

ALPHA_THRESHOLD = 10
MIN_DECIMATION = 2


def rotation_matrix(w: int, h: int, angle: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """2x3 rotation matrix about the centre with the enlarged canvas (same as `_transform_object`)."""
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += (new_w - w) / 2
    m[1, 2] += (new_h - h) / 2
    return m, (new_w, new_h)


def rotated_geometry(img: np.ndarray, angle: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """((width, height) of the rotated canvas, (width, height) of the rotated alpha box) without rotating the RGB."""
    h, w = img.shape[:2]
    if angle == 0:
        canvas, alpha = (w, h), img[:, :, 3] if img.ndim == 3 and img.shape[2] == 4 else None
    else:
        m, canvas = rotation_matrix(w, h, angle)
        alpha = None
        if img.ndim == 3 and img.shape[2] == 4 and min(canvas) > 0:
            alpha = cv2.warpAffine(np.ascontiguousarray(img[:, :, 3]), m, canvas, flags=cv2.INTER_LINEAR)
    if alpha is None:
        return canvas, canvas
    ys, xs = np.nonzero(alpha > ALPHA_THRESHOLD)
    if len(xs) == 0:
        return canvas, canvas
    return canvas, (int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _premultiply(img: np.ndarray) -> np.ndarray:
    """float32 image BGR*alpha + alpha (0-1); without an alpha channel, float32 BGR with alpha 1."""
    f = img.astype(np.float32)
    if f.shape[2] == 4:
        alpha = f[:, :, 3:4] / 255.0
        return np.concatenate([f[:, :, :3] * alpha, alpha], axis=2)
    return np.concatenate([f, np.ones(f.shape[:2] + (1,), np.float32)], axis=2)


def _unpremultiply(p: np.ndarray, with_alpha: bool) -> np.ndarray:
    alpha = np.clip(p[:, :, 3:4], 0.0, 1.0)
    rgb = np.where(alpha > 1e-3, p[:, :, :3] / np.maximum(alpha, 1e-3), 0.0)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, np.round(alpha * 255).astype(np.uint8)], axis=2) if with_alpha else rgb


def _decimate(p: np.ndarray, scale: float) -> Tuple[np.ndarray, float]:
    """(image decimated by an integer factor with INTER_AREA, remaining scale to apply)."""
    k = int(1.0 / scale) if scale < 1.0 else 1
    if k < MIN_DECIMATION:
        return p, scale
    h, w = p.shape[:2]
    small = cv2.resize(p, (max(1, int(round(w / k))), max(1, int(round(h / k)))), interpolation=cv2.INTER_AREA)
    return small, scale * w / small.shape[1]


def warp_single(img: np.ndarray, angle: float, scale: float) -> Optional[np.ndarray]:
    """BGR/BGRA crop rotated `angle` degrees and scaled by `scale` with a single interpolation; None if it ends up empty."""
    with_alpha = img.ndim == 3 and img.shape[2] == 4
    p, residual = _decimate(_premultiply(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)), scale)
    h, w = p.shape[:2]
    m, (rw, rh) = rotation_matrix(w, h, angle)
    out_w, out_h = int(rw * residual), int(rh * residual)
    if out_w <= 0 or out_h <= 0:
        return None
    # scale about the pixel centre: x_dst = s * (x + 0.5) - 0.5
    shift = 0.5 * residual - 0.5
    full = np.vstack([m, [0.0, 0.0, 1.0]])
    scale_m = np.array([[residual, 0.0, shift], [0.0, residual, shift], [0.0, 0.0, 1.0]])
    interp = cv2.INTER_LANCZOS4 if residual > 1.0 else cv2.INTER_CUBIC
    warped = cv2.warpAffine(p, (scale_m @ full)[:2], (out_w, out_h), flags=interp, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    return _unpremultiply(warped, with_alpha)
