"""
Image Composer Module
=====================
Thin facade kept for backward compatibility.

The implementation lives in the :mod:`app.composition` package, split by
responsibility (placement, transforms, effects, validation, batch planning,
output I/O, debug visualizations).  ``from app.composer import ImageComposer``
and the other public names below keep working unchanged.

Migrated from SyntheticDataAugmentor for microservice architecture.
"""

# Module attributes that callers/tests historically reached through ``app.composer``
import cv2  # noqa: F401
import numpy as np  # noqa: F401
from app import (  # noqa: F401
    background_match, copy_paste, crop_augment, edge_matting, effects_underwater,
    faithful_scale, harmonize, medium_transfer, output_format, placement,
    position_stats, resample, sharpness, visibility,
)

from app.composition.results import BatchResult, BlurBudget, ComposeResult, VRAMMonitor
from app.composition.geometry import (
    REAL_WORLD_SIZES,
    REFERENCE_CAPTURE_DISTANCE,
    UNDERWATER_VISIBILITY,
    check_mask_overlap,
    compute_iou,
    get_real_world_size,
    refine_object_mask,
)
from app.composition.core import ImageComposer

__all__ = [
    "ImageComposer",
    "ComposeResult",
    "BatchResult",
    "BlurBudget",
    "VRAMMonitor",
    "REAL_WORLD_SIZES",
    "REFERENCE_CAPTURE_DISTANCE",
    "UNDERWATER_VISIBILITY",
    "get_real_world_size",
    "compute_iou",
    "check_mask_overlap",
    "refine_object_mask",
]
