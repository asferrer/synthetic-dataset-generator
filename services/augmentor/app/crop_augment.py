"""Per-paste crop augmentations (profile v10).

`sample_augment` draws the parameters with an rng (reproducible per image) and returns them as a JSON-serialisable
dict; `apply_augment` applies them to a BGR/BGRA crop, so the COCO can record exactly what was done.

Order: horizontal flip, perspective, colour jitter (brightness, contrast, saturation) and mild blur. Rotation is not
applied here: it is passed as `ObjectPlacement.rotation` and applied by `_transform_object`.
"""

from typing import Any, Dict

import cv2
import numpy as np

from app.synth_profile import SynthProfile

_CORNERS = 4


def sample_augment(rng, profile: SynthProfile) -> Dict[str, Any]:
    """Augmentation parameters of one paste, within the profile ranges."""
    p = profile.aug_perspective
    return {
        "hflip": rng.random() < profile.aug_hflip_prob,
        "rotation": round(rng.uniform(-profile.aug_rotation_deg, profile.aug_rotation_deg), 2),
        "perspective": [round(rng.uniform(-p, p), 4) for _ in range(_CORNERS * 2)] if p > 0 else None,
        "brightness": round(1.0 + rng.uniform(-profile.aug_brightness, profile.aug_brightness), 3),
        "contrast": round(1.0 + rng.uniform(-profile.aug_contrast, profile.aug_contrast), 3),
        "saturation": round(1.0 + rng.uniform(-profile.aug_saturation, profile.aug_saturation), 3),
        "blur_sigma": (
            round(rng.uniform(0.3, profile.aug_blur_sigma_max), 2)
            if rng.random() < profile.aug_blur_prob else None
        ),
    }


def _split(img: np.ndarray):
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3], img[:, :, 3]
    return img, None


def _join(bgr: np.ndarray, alpha) -> np.ndarray:
    return bgr if alpha is None else np.dstack([bgr, alpha])


def apply_perspective(img: np.ndarray, offsets) -> np.ndarray:
    """Moves the 4 corners (as a fraction of the side) and crops to the new rectangle."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = np.asarray(offsets, dtype=np.float32).reshape(_CORNERS, 2) * np.float32([w, h])
    dst = src + jitter
    dst -= dst.min(axis=0)
    new_w, new_h = int(np.ceil(dst[:, 0].max())), int(np.ceil(dst[:, 1].max()))
    if new_w < 2 or new_h < 2:
        return img
    matrix = cv2.getPerspectiveTransform(src, dst)
    border = (0, 0, 0, 0) if img.shape[2] == 4 else (0, 0, 0)
    return cv2.warpPerspective(img, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def apply_color(bgr: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    """Brightness, contrast (around the mean) and saturation on a uint8 BGR image."""
    out = bgr.astype(np.float32)
    mean = out.mean()
    out = (out - mean) * contrast + mean * brightness
    if abs(saturation - 1.0) > 1e-6:
        gray = out.mean(axis=2, keepdims=True)
        out = gray + (out - gray) * saturation
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_augment(img: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    """Applies the augmentation described by `spec` (from sample_augment) to the crop."""
    out = img
    if spec.get("hflip"):
        out = cv2.flip(out, 1)
    if spec.get("perspective"):
        out = apply_perspective(out, spec["perspective"])
    bgr, alpha = _split(out)
    bgr = apply_color(bgr, spec.get("brightness", 1.0), spec.get("contrast", 1.0),
                      spec.get("saturation", 1.0))
    sigma = spec.get("blur_sigma")
    if sigma:
        bgr = cv2.GaussianBlur(bgr, (0, 0), sigmaX=float(sigma))
    return _join(np.ascontiguousarray(bgr), alpha)
