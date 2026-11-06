"""Resolution, JPEG compression and global haze faithful to the real data (profile v10).

- Output resolution/aspect: sampled from the real training distribution (`image_sizes`). The background is resized
  BEFORE pasting objects, so they are pasted at the final resolution (as in real data) and boxes need no rescaling.
  The factor is limited so the background is not enlarged by more than `max_bg_upscale` nor above `max_output_side`.
- JPEG quality: sampled from the qualities estimated on the real training images (`image_stats.jpeg_quality_quantiles`)
  or fixed if `jpeg_quality_fixed`.
- Global haze: the `underwater` tint/haze mixes a uniform colour with weight t, which multiplies luminance deviations
  by (1 - t). t is chosen per image so the final contrast (std of L) equals one sampled from the real images, without
  exceeding `haze_max_intensity`; if the background already has less contrast than the target, t = 0 (backgrounds are
  already underwater photos).
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from app.real_stats import RealStats
from app.synth_profile import SynthProfile


def sample_output_size(
    rng, stats: RealStats, bg_wh: Tuple[int, int], profile: SynthProfile
) -> Optional[Tuple[int, int]]:
    """Output (w, h) sampled from the real resolutions, or None if there is no data."""
    if not stats.image_sizes:
        return None
    sizes = sorted(stats.image_sizes)
    w, h = rng.choices(sizes, weights=[stats.image_sizes[s] for s in sizes], k=1)[0]
    bg_w, bg_h = bg_wh
    factor = min(
        1.0,
        profile.max_bg_upscale * bg_w / w,
        profile.max_bg_upscale * bg_h / h,
        profile.max_output_side / max(w, h),
    )
    return max(1, int(round(w * factor))), max(1, int(round(h * factor)))


def sample_jpeg_quality(rng, stats: RealStats, profile: SynthProfile) -> Optional[int]:
    """Output JPEG quality (None = OpenCV default)."""
    if profile.jpeg_quality_fixed is not None:
        return int(profile.jpeg_quality_fixed)
    quality = stats.jpeg_quality(rng.random())
    return None if quality is None else int(round(min(100.0, max(1.0, quality))))


# Standard luminance quantisation table (quality 50), natural order (same as scripts/build_image_stats.py)
_STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
             14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
             18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
             49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def estimate_jpeg_quality(path: str) -> Optional[float]:
    """JPEG quality (1-100) from a file's quantisation table, or None if not a JPEG."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            if im.format != "JPEG" or not getattr(im, "quantization", None):
                return None
            table = im.quantization[0]
    except OSError:
        return None
    scale = 100.0 * sum(table) / sum(_STD_LUMA)
    quality = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return float(min(100.0, max(1.0, quality)))


def resize_to(bg_image: np.ndarray, depth_map: Optional[np.ndarray], size: Tuple[int, int]):
    """Resizes the background (INTER_AREA/CUBIC) and the depth map to size (w, h)."""
    w, h = size
    if bg_image.shape[1] == w and bg_image.shape[0] == h:
        return bg_image, depth_map
    shrink = w * h < bg_image.shape[0] * bg_image.shape[1]
    bg_image = cv2.resize(bg_image, (w, h),
                          interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
    if depth_map is not None:
        depth_map = cv2.resize(depth_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return bg_image, depth_map


def luminance_std(image: np.ndarray) -> float:
    """Std of L (LAB, 0-255) of the image downscaled to 224 px (same measure as build_image_stats)."""
    small = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0].std())


def matched_haze_intensity(
    composite: np.ndarray, target_std: Optional[float], max_intensity: float, default: float
) -> float:
    """Haze strength t that brings the contrast of `composite` close to `target_std`."""
    if target_std is None:
        return min(default, max_intensity)
    current = luminance_std(composite)
    if current <= 1e-6:
        return 0.0
    return float(min(max(1.0 - target_std / current, 0.0), max_intensity))
