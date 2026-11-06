"""
Semantic, physics-aware object placement.

Maps each object class's buoyancy behavior (sink/float/hover, defined in the
domain config physics_properties) to the scene regions where the object
plausibly appears, and samples positions from the region map produced by
scene analysis. This replaces uniform random placement so that e.g. tires
land on the seafloor and plastic bags drift in open water.

Region ids (segmentation service):
0=unknown, 1=open_water, 2=seafloor, 3=surface, 4=vegetation, 5=rocky,
6=sandy, 7=murky
"""

import json
import logging
import os
import random
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Scene regions where each buoyancy behavior plausibly places an object
BEHAVIOR_REGIONS: Dict[str, Tuple[int, ...]] = {
    "float": (3, 1),    # surface, open water
    "hover": (1, 4),    # open water (mid column), vegetation
    "sink": (2, 5, 6),  # seafloor, rocky, sandy
}

# Fallback when the domain config is not available in the container
_DEFAULT_BEHAVIORS: Dict[str, str] = {
    "bottle": "sink",
    "can": "sink",
    "fishing_net": "hover",
    "glove": "float",
    "mask": "float",
    "metal_debris": "sink",
    "plastic_bag": "float",
    "plastic_debris": "float",
    "tire": "sink",
}


@lru_cache(maxsize=1)
def load_class_behaviors() -> Dict[str, str]:
    """Map class_name/keyword (lowercase) -> behavior from domain configs.

    Reads every JSON under DOMAINS_PATH (default /app/config/domains) and
    merges physics_properties.behavior per object class, on top of built-in
    defaults for the standard marine litter classes.
    """
    behaviors = dict(_DEFAULT_BEHAVIORS)
    domains_path = os.environ.get("DOMAINS_PATH", "/app/config/domains")
    try:
        for fname in sorted(os.listdir(domains_path)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(domains_path, fname), "r", encoding="utf-8") as f:
                    domain = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Skipping domain config {fname}: {e}")
                continue
            for obj in domain.get("objects", []):
                behavior = (obj.get("physics_properties") or {}).get("behavior")
                class_name = obj.get("class_name")
                if not behavior or not class_name:
                    continue
                behaviors[class_name.lower()] = behavior
                for keyword in obj.get("keywords", []):
                    behaviors.setdefault(keyword.lower(), behavior)
    except FileNotFoundError:
        logger.info(
            f"Domain config dir not found ({domains_path}), using built-in behaviors"
        )
    return behaviors


def get_behavior(class_name: str) -> Optional[str]:
    """Buoyancy behavior for a class, or None if unknown."""
    return load_class_behaviors().get(class_name.lower())


def region_map_at(
    region_map: Optional[np.ndarray], size: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Resize a region map to (width, height) with nearest-neighbour."""
    if region_map is None:
        return None
    w, h = size
    if region_map.shape[:2] == (h, w):
        return region_map
    return cv2.resize(region_map, (w, h), interpolation=cv2.INTER_NEAREST)


ANCHOR_CANDIDATES = 128  # candidate anchors when weighting by the real density (center_weight)


def _weighted_anchor(xs, ys, obj_h: int, behavior: str, rng, center_weight: Callable) -> int:
    """Index of an anchor chosen among random candidates, weighted by the real density of the box centre."""
    cand = [rng.randrange(len(xs)) for _ in range(ANCHOR_CANDIDATES)]
    cx = np.asarray([xs[i] for i in cand], dtype=np.float64)
    cy = np.asarray([ys[i] for i in cand], dtype=np.float64) - (obj_h / 2 if behavior == "sink" else 0.0)
    weights = np.asarray(center_weight(cx, cy), dtype=np.float64) + 1e-12
    return rng.choices(cand, weights=weights.tolist(), k=1)[0]


def sample_position(
    region_map: np.ndarray,
    class_name: str,
    obj_size: Tuple[int, int],
    rng=random,
    behavior: Optional[str] = None,
    center_weight: Optional[Callable] = None,
) -> Optional[Tuple[int, int]]:
    """Sample a top-left position whose anchor lies in a compatible region.

    The anchor is the bottom-center for sinking objects (they rest on the
    floor) and the center for floating/hovering ones.

    Args:
        region_map: HxW uint8 region ids at background resolution
        class_name: object class (behavior looked up in domain config)
        obj_size: (width, height) of the transformed object
        rng: random source (module or random.Random instance)
        behavior: forces the buoyancy behavior (e.g. "sink" to rest a normally
            floating object on the seabed); None uses the class behavior
        center_weight: optional f(cx_px, cy_px) -> weights; anchors are then drawn
            with the real density of the box center (profile position_from_stats)

    Returns:
        (x, y) top-left position, or None if the behavior is unknown or no
        compatible region exists (caller should fall back to random).
    """
    behavior = behavior or get_behavior(class_name)
    if behavior not in BEHAVIOR_REGIONS:
        return None

    h, w = region_map.shape[:2]
    obj_w, obj_h = obj_size
    if obj_w <= 0 or obj_h <= 0 or obj_w >= w or obj_h >= h:
        return None

    allowed = np.isin(region_map, BEHAVIOR_REGIONS[behavior])
    ys, xs = np.nonzero(allowed)
    if len(xs) == 0:
        return None

    idx = (rng.randrange(len(xs)) if center_weight is None
           else _weighted_anchor(xs, ys, obj_h, behavior, rng, center_weight))
    anchor_x, anchor_y = int(xs[idx]), int(ys[idx])

    x = anchor_x - obj_w // 2
    if behavior == "sink":
        y = anchor_y - obj_h  # bottom of object touches the anchor
    else:
        y = anchor_y - obj_h // 2

    # Clamp inside the image
    x = max(0, min(x, w - obj_w))
    y = max(0, min(y, h - obj_h))
    return x, y
