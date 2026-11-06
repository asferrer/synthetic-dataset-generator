"""Class-faithful scale (profile v10).

The target size of an object is the square root of the relative area of its box (sqrt(w*h / (W*H))), sampled from the
real distribution of its class in the training split. It is computed on the already rotated/augmented crop, using the
tight box of its alpha, and converted into a scale factor.

Interaction with depth: the depth-based scale (`_calculate_depth_aware_scale`) uses a per-class physical size and a
random triangular depth (with position None it does not look at the map), i.e. it does not depend on the scene. The
empirical distribution already integrates the real variation of distance, so with `faithful_scale` it replaces it; the
depth map is still used for blur, shadows, occlusion and haze. If the class is not in the statistics, the usual
depth-based scale is used.

Physical limits: the object canvas fits in the image (98 %), crop upscaling <= max_crop_upscale and minimum box side
>= `floor_px`. If these cut the reachable range, the real distribution conditioned on it is sampled
(RealStats.truncated_size).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from app.real_stats import RealStats
from app.synth_profile import SynthProfile

FIT_FRACTION = 0.98
ALPHA_THRESHOLD = 10


def alpha_bbox_wh(img: np.ndarray) -> Tuple[int, int]:
    """(width, height) of the tight alpha box (or of the image if there is no alpha)."""
    if img.ndim == 3 and img.shape[2] == 4:
        ys, xs = np.nonzero(img[:, :, 3] > ALPHA_THRESHOLD)
        if len(xs):
            return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    return img.shape[1], img.shape[0]


def scale_limits(
    canvas_wh: Tuple[int, int], bbox_wh: Tuple[int, int], bg_wh: Tuple[int, int],
    max_upscale: float, floor_px: int,
) -> Tuple[float, float]:
    """(s_min, s_max) of the allowed scale factor."""
    p_w, p_h = canvas_wh
    s_max = min(max_upscale, FIT_FRACTION * bg_wh[0] / p_w, FIT_FRACTION * bg_wh[1] / p_h)
    s_min = floor_px / max(1, min(bbox_wh))
    return s_min, s_max


@dataclass
class FaithfulScale:
    """Scale plan of one object; `target` and `truncated` are filled in on resolve."""
    stats: RealStats
    class_name: str
    quantile: float
    profile: SynthProfile
    target: Optional[float] = None
    truncated: bool = False

    def resolve(
        self, canvas_wh: Tuple[int, int], bbox_wh: Tuple[int, int], bg_wh: Tuple[int, int],
        floor_px: int,
    ) -> float:
        """Scale factor so the box reaches the sampled target size."""
        s_min, s_max = scale_limits(canvas_wh, bbox_wh, bg_wh, self.profile.max_crop_upscale,
                                    max(floor_px, self.profile.min_object_px))
        bg_root = float(np.sqrt(bg_wh[0] * bg_wh[1]))
        box_root = float(np.sqrt(bbox_wh[0] * bbox_wh[1]))
        if s_min >= s_max:
            self.target, self.truncated = s_max * box_root / bg_root, True
            return s_max
        lo, hi = s_min * box_root / bg_root, s_max * box_root / bg_root
        self.target, self.truncated = self.stats.truncated_size(self.class_name, self.quantile, lo, hi)
        return self.target * bg_root / box_root


def make_faithful(profile: Optional[SynthProfile], stats: Optional[RealStats],
                  class_name: str, quantile: Optional[float]) -> Optional[FaithfulScale]:
    """FaithfulScale if the profile asks for it and the class has statistics."""
    if profile is None or not profile.faithful_scale or quantile is None or stats is None:
        return None
    if not stats.has_class(class_name):
        return None
    return FaithfulScale(stats, class_name, quantile, profile)
