"""Rejection checks: composed-image validation, visibility gate and object-identity check."""

from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np
from loguru import logger

from app import copy_paste
from app import visibility
from app.models.schemas import AnnotationBox, EffectsConfig


class ValidationMixin:
    """Rejection checks: composed-image validation, visibility gate and object-identity check."""

    async def _validate_composed(
        self,
        image_path: str,
        annotations: List[AnnotationBox],
        image_height: int,
        validate_quality: bool,
        validate_physics: bool,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Run optional quality/physics validation on a composed image.

        Returns (is_valid, quality_info). quality_info is a JSON-ready dict
        stored in the COCO image entry; None when no validation ran.
        """
        if not (validate_quality or validate_physics):
            return True, None

        is_valid = True
        info: Dict[str, Any] = {}

        if validate_quality and self.quality_validator:
            try:
                score = await self.quality_validator.validate(
                    image_path=image_path,
                    annotations=annotations,
                )
                info.update({
                    "perceptual_quality": round(score.perceptual_quality, 4),
                    "anomaly_score": round(score.anomaly_score, 4),
                    "composition_score": round(score.composition_score, 4),
                    "overall_score": round(score.overall_score, 4),
                    "overall_pass": score.overall_pass,
                })
                if not score.overall_pass:
                    is_valid = False
            except Exception as e:
                logger.warning(f"Quality validation failed (image kept): {e}")

        if validate_physics and self.physics_validator:
            try:
                violations = await self.physics_validator.validate(
                    annotations=annotations,
                    image_height=image_height,
                )
                info["physics_violations"] = [
                    {
                        "type": v.violation_type,
                        "object_class": v.object_class,
                        "severity": v.severity,
                    }
                    for v in violations
                ]
                if any(v.severity == "high" for v in violations):
                    is_valid = False
            except Exception as e:
                logger.warning(f"Physics validation failed (image kept): {e}")

        return is_valid, info or None

    @staticmethod
    def _fails_visibility(profile, class_name: str, composite: np.ndarray, bbox: Tuple) -> bool:
        """True if the just-pasted object falls below its class's local-contrast threshold.

        It is measured before the global effects: the contrast is discounted by the maximum possible haze.
        """
        thr = visibility.threshold_for(profile, class_name) if profile is not None else None
        if thr is None:
            return False
        visible, _ = visibility.is_visible(composite, bbox, thr, profile.visibility_ring,
                                           scale=1.0 - copy_paste.haze_cap(profile))
        return not visible

    @staticmethod
    def _log_visibility(log: Dict[str, Dict[str, int]], class_name: str, trace_out: Dict[str, Any],
                        unplaced: bool) -> None:
        """Adds to the image record the attempts rejected for visibility and the object lost because of it."""
        n = trace_out.get("visibility_rejects", 0)
        if n:
            log["rejected"][class_name] = log["rejected"].get(class_name, 0) + n
            if unplaced:
                log["dropped"][class_name] = log["dropped"].get(class_name, 0) + 1

    def _validate_object_identity(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        mask: np.ndarray,
        effects_config: EffectsConfig
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate that object remains recognizable after effects.

        Checks:
        1. Color shift in LAB < 30 (JND threshold)
        2. Sharpness ratio > 0.3 (preserve 30% of original sharpness)
        3. Contrast ratio > 0.5 (preserve 50% of original contrast)

        Args:
            original: Original object image (before effects)
            processed: Processed object image (after effects)
            mask: Binary mask of object
            effects_config: Configuration with validation thresholds

        Returns:
            (is_valid, metrics_dict)
        """
        mask_bool = mask > 127
        if not mask_bool.any():
            return True, {}

        try:
            # 1. Color shift in LAB space
            orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
            proc_lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            color_shift = np.linalg.norm(
                orig_lab[mask_bool].mean(axis=0) - proc_lab[mask_bool].mean(axis=0)
            )

            # 2. Sharpness preservation
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            orig_sharp = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
            proc_sharp = cv2.Laplacian(proc_gray, cv2.CV_64F).var()
            sharpness_ratio = proc_sharp / max(orig_sharp, 1e-6)

            # 3. Contrast preservation
            orig_roi = original[mask_bool]
            proc_roi = processed[mask_bool]
            orig_contrast = (orig_roi.max() - orig_roi.min()) / (orig_roi.max() + orig_roi.min() + 1e-6)
            proc_contrast = (proc_roi.max() - proc_roi.min()) / (proc_roi.max() + proc_roi.min() + 1e-6)
            contrast_ratio = proc_contrast / max(orig_contrast, 1e-6)

            metrics = {
                'color_shift': float(color_shift),
                'sharpness_ratio': float(sharpness_ratio),
                'contrast_ratio': float(contrast_ratio),
            }

            is_valid = (
                color_shift < effects_config.max_color_shift and
                sharpness_ratio > effects_config.min_sharpness_ratio and
                contrast_ratio > effects_config.min_contrast_ratio
            )

            return is_valid, metrics

        except Exception as e:
            logger.warning(f"Identity validation failed with error: {e}")
            return True, {}  # Don't reject on validation error
