"""Filter for "rectangular" crops in the object library (profile v11).

Some crops (mostly Fishing_Net) keep their background: the alpha mask fills almost the whole box and pasting them
shows a rectangle of foreign water. Crops whose opaque-pixel fraction in the box exceeds `crop_rect_max_fill`, or that
are opaque on `crop_rect_min_sides` or more box sides, are discarded when the library is loaded. (A filled ellipse
covers pi/4 = 0.785 of its box.) The per-class count of discarded crops is returned and stored in the output (COCO
`info.synth_generator.rect_filter` and `rect_filter_report.json`).
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

ALPHA_THRESHOLD = 10  # same threshold the compositor uses for the binary mask


def rect_stats(alpha: np.ndarray, side_coverage: float = 0.9) -> Tuple[float, int]:
    """(opaque-pixel fraction of the box, number of sides whose opaque fraction >= side_coverage)."""
    opaque = alpha > ALPHA_THRESHOLD
    if opaque.size == 0:
        return 0.0, 0
    sides = (opaque[0], opaque[-1], opaque[:, 0], opaque[:, -1])
    return float(opaque.mean()), sum(float(s.mean()) >= side_coverage for s in sides)


def is_rectangular(fill: float, sides: int, max_fill: float, min_sides: Optional[int]) -> bool:
    """True if the crop fills too much of its box or has too many opaque sides."""
    return fill > max_fill or (min_sides is not None and sides >= min_sides)


def measure_crop(path: str, side_coverage: float = 0.9) -> Optional[Tuple[float, int]]:
    """(fill, sides) of a crop; None if unreadable. No alpha channel = fully opaque box."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        return rect_stats(img[:, :, 3], side_coverage)
    return 1.0, 4


def filter_rectangular(
    objects_by_class: Dict[str, List[dict]], max_fill: float, min_sides: Optional[int],
    side_coverage: float = 0.9, workers: int = 8,
) -> Dict[str, object]:
    """Removes rectangular crops in place and returns the report (discarded and kept per class)."""
    items = [(cls, obj) for cls, objs in objects_by_class.items() for obj in objs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        measures = list(pool.map(lambda it: measure_crop(it[1]["path"], side_coverage), items))
    removed: Dict[str, int] = {}
    kept: Dict[str, int] = {}
    unreadable = 0
    drop = set()
    for i, ((cls, _), m) in enumerate(zip(items, measures)):
        unreadable += m is None
        if m is not None and is_rectangular(m[0], m[1], max_fill, min_sides):
            drop.add(i)
            removed[cls] = removed.get(cls, 0) + 1
    for i, (cls, _) in enumerate(items):
        if i not in drop:
            kept[cls] = kept.get(cls, 0) + 1
    for cls in list(objects_by_class):
        survivors = [o for o in objects_by_class[cls] if not any(o is items[i][1] for i in drop if items[i][0] == cls)]
        objects_by_class[cls] = survivors
        if not survivors:
            del objects_by_class[cls]
            logger.warning(f"Rectangular filter: class {cls} is left without crops")
    return {"max_fill": max_fill, "min_opaque_sides": min_sides, "side_coverage": side_coverage,
            "removed_by_class": dict(sorted(removed.items())), "kept_by_class": dict(sorted(kept.items())),
            "total_removed": sum(removed.values()), "total_kept": sum(kept.values()), "unreadable_kept": unreadable}


def apply_library_filters(objects_by_class: Dict[str, List[dict]], profile) -> Optional[Dict[str, object]]:
    """Library filters of the profile. Returns the rectangular-filter report (None if inactive)."""
    if profile is None or not profile.crop_rect_filter:
        return None
    report = filter_rectangular(objects_by_class, profile.crop_rect_max_fill, profile.crop_rect_min_sides,
                                profile.crop_rect_side_coverage)
    logger.info(f"Rectangular crop filter: removed {report['total_removed']} "
                f"({report['removed_by_class']}), {report['total_kept']} kept")
    return report
