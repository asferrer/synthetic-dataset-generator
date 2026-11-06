"""Result and bookkeeping types shared by the composition modules (no GPU needed)."""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
from loguru import logger

from app.models.schemas import AnnotationBox


@dataclass
class ComposeResult:
    """Result from single image composition"""
    annotations: List[AnnotationBox] = field(default_factory=list)
    objects_placed: int = 0
    depth_used: bool = False
    depth_source: Optional[str] = None  # service | fallback_gradient | precomputed
    effects_applied: List[str] = field(default_factory=list)
    image_size: Optional[Tuple[int, int]] = None  # (w, h) output size (profile with its own resolution)
    image_trace: Dict[str, Any] = field(default_factory=dict)  # per-image decisions (profile v10)
    real_annotations: List[AnnotationBox] = field(default_factory=list)  # copy-paste: kept real boxes


@dataclass
class BlurBudget:
    """
    Tracks cumulative blur to prevent over-processing small objects.

    Manages a "blur budget" that scales with object size. Smaller objects have
    tighter budgets to prevent them from becoming unrecognizable blobs.
    """
    max_blur_budget: float  # Max total blur in pixels
    object_size: int  # min(width, height) of object
    used_blur: float = 0.0

    def _get_scaled_budget(self) -> float:
        """
        Calculate scaled budget based on object size.

        Smaller objects get proportionally smaller budgets:
        - 100px object: 100% of budget
        - 50px object: 50% of budget
        - 25px object: 50% of budget (clamped at 0.5)
        """
        return self.max_blur_budget * max(self.object_size / 100, 0.5)

    def can_apply(self, blur_amount: float) -> bool:
        """Check if blur can be applied without exceeding budget."""
        return (self.used_blur + blur_amount) <= self._get_scaled_budget()

    def apply(self, blur_amount: float) -> float:
        """
        Apply blur and return actual amount (may be reduced).

        Returns:
            Actual blur amount applied (0 if budget exceeded)
        """
        scaled_budget = self._get_scaled_budget()
        available = scaled_budget - self.used_blur
        actual = min(blur_amount, max(available, 0))
        self.used_blur += actual
        return actual


class VRAMMonitor:
    """Monitors VRAM usage and triggers cleanup when needed."""

    def __init__(self, threshold: float = 0.5, device: int = 0):
        """
        Initialize VRAM monitor.

        Args:
            threshold: VRAM usage fraction to trigger cleanup (default 0.5 = 50%)
            device: CUDA device ID to monitor (default 0)
        """
        self.threshold = threshold
        self.device = device
        self.last_check = 0
        self.check_interval = 2  # Check every 2 iterations

    def get_vram_usage(self) -> float:
        """
        Returns VRAM usage as fraction (0.0-1.0).

        Returns:
            VRAM usage fraction, or 0.0 if CUDA not available
        """
        if not torch.cuda.is_available():
            return 0.0

        try:
            allocated = torch.cuda.memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            return allocated / total
        except Exception as e:
            logger.warning(f"Failed to get VRAM usage: {e}")
            return 0.0

    def get_vram_stats(self) -> Dict[str, float]:
        """
        Get detailed VRAM statistics.

        Returns:
            Dict with allocated_gb, reserved_gb, total_gb, usage_fraction
        """
        if not torch.cuda.is_available():
            return {}

        try:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            usage = allocated / (total or 1)

            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'usage_fraction': usage,
            }
        except Exception as e:
            logger.warning(f"Failed to get VRAM stats: {e}")
            return {}

    def should_cleanup(self) -> bool:
        """
        Check if cleanup is needed (VRAM > threshold).

        Returns:
            True if VRAM usage exceeds threshold
        """
        self.last_check += 1
        if self.last_check % self.check_interval == 0:
            usage = self.get_vram_usage()
            if usage > self.threshold:
                stats = self.get_vram_stats()
                logger.warning(
                    f"VRAM at {usage*100:.1f}% "
                    f"({stats.get('allocated_gb', 0):.2f}GB / {stats.get('total_gb', 32):.0f}GB), "
                    f"triggering cleanup"
                )
                return True
        return False


@dataclass
class BatchResult:
    """Result from batch composition"""
    images_generated: int = 0
    images_rejected: int = 0
    synthetic_counts: Dict[str, int] = field(default_factory=dict)
    output_coco_path: Optional[str] = None
    exhausted_classes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # classes exhausted by max_crop_reuse
    synth_profile: Optional[str] = None
