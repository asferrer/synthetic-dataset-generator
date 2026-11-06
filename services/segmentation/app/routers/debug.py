"""
Rutas de depuración y explicabilidad.
"""

import logging
import os
import time
from pathlib import Path

import cv2
from fastapi import APIRouter

from app.models.schemas import (
    DebugAnalyzeRequest,
    DebugAnalyzeResponse,
    DebugCompatibilityRequest,
    DebugCompatibilityResponse,
)
from app.service_state import state
from app.utils.region_encoding import encode_region_map_base64

logger = logging.getLogger(__name__)


router = APIRouter()


@router.post("/debug/analyze", response_model=DebugAnalyzeResponse, tags=["Debug"])
async def debug_analyze_scene(request: DebugAnalyzeRequest):
    """
    Analyze scene with full debug information for explainability.

    Returns detailed information about how SAM3/heuristics made decisions,
    including region masks, confidence scores, and decision logs.
    """
    try:
        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        # Use pre-initialized debug analyzer to avoid re-creation overhead
        debug_output_dir = "/shared/segmentation/debug"

        # Check if debug_scene_analyzer supports debug mode
        if state.debug_scene_analyzer is not None and hasattr(state.debug_scene_analyzer, 'analyze_scene_with_debug'):
            image_id = request.image_id or f"debug_{int(time.time())}"
            analysis, debug_info = state.debug_scene_analyzer.analyze_scene_with_debug(
                image,
                save_visualization=request.save_visualization,
                image_id=image_id,
            )

            # Get masks directory
            masks_dir = os.path.join(debug_output_dir, f"{image_id}_masks")

            # Encode region_map for transfer
            region_map_b64 = None
            if hasattr(analysis, 'region_map') and analysis.region_map is not None:
                region_map_b64 = encode_region_map_base64(analysis.region_map)

            return DebugAnalyzeResponse(
                success=True,
                dominant_region=analysis.dominant_region.value if hasattr(analysis.dominant_region, 'value') else str(analysis.dominant_region),
                region_scores=analysis.region_scores,
                scene_brightness=analysis.scene_brightness,
                water_clarity=analysis.water_clarity,
                color_temperature=analysis.color_temperature,
                analysis_method=debug_info.analysis_method,
                processing_time_ms=debug_info.processing_time_ms,
                sam3_prompts_used=debug_info.sam3_prompts_used,
                region_confidences=debug_info.region_confidences,
                decision_log=debug_info.decision_log,
                region_map_base64=region_map_b64,
                visualization_path=debug_info.visualization_path,
                masks_directory=masks_dir if os.path.exists(masks_dir) else None,
            )
        else:
            # Fallback: basic analysis without debug
            result = state.scene_analyzer.analyze_scene(image)
            return DebugAnalyzeResponse(
                success=True,
                dominant_region=result.get("dominant_region", "unknown") if isinstance(result, dict) else (result.dominant_region.value if hasattr(result.dominant_region, 'value') else str(result.dominant_region)),
                region_scores=result.get("region_scores", {}) if isinstance(result, dict) else result.region_scores,
                scene_brightness=result.get("scene_brightness", 0.5) if isinstance(result, dict) else result.scene_brightness,
                water_clarity=result.get("water_clarity", "unknown") if isinstance(result, dict) else result.water_clarity,
                color_temperature=result.get("color_temperature", "neutral") if isinstance(result, dict) else result.color_temperature,
                analysis_method="heuristic",
                processing_time_ms=0.0,
                sam3_prompts_used=[],
                region_confidences={},
                decision_log=["Debug mode not fully supported with current analyzer"],
            )

    except Exception as e:
        logger.error(f"Debug analysis failed: {e}")
        return DebugAnalyzeResponse(
            success=False,
            dominant_region="unknown",
            region_scores={},
            scene_brightness=0.0,
            water_clarity="unknown",
            color_temperature="unknown",
            analysis_method="error",
            processing_time_ms=0.0,
            sam3_prompts_used=[],
            region_confidences={},
            decision_log=[f"Error: {str(e)}"],
            error=str(e),
        )


@router.post("/debug/compatibility", response_model=DebugCompatibilityResponse, tags=["Debug"])
async def debug_check_compatibility(request: DebugCompatibilityRequest):
    """
    Check object-scene compatibility with detailed debug information.

    Returns the compatibility decision along with alternative positions
    and explanations for the decision.
    """
    try:
        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        h, w = image.shape[:2]

        # Use pre-initialized debug analyzer to avoid re-creation overhead
        if state.debug_scene_analyzer is not None and hasattr(state.debug_scene_analyzer, 'check_object_scene_compatibility_with_debug'):
            # First analyze the scene
            analysis = state.debug_scene_analyzer.analyze_scene(image)

            # Then check compatibility with debug
            position = (request.position_x, request.position_y)
            score, reason, decision = state.debug_scene_analyzer.check_object_scene_compatibility_with_debug(
                request.object_class,
                position,
                analysis,
                (h, w),
            )

            # Convert alternatives to list format
            alternatives = [
                [float(x), float(y), float(s)]
                for x, y, s in decision.alternative_positions[:5]
            ] if decision.alternative_positions else []

            return DebugCompatibilityResponse(
                success=True,
                is_compatible=score >= 0.4,
                score=float(score),
                reason=reason,
                decision=decision.decision,
                region_at_position=decision.region_at_position,
                alternatives=alternatives,
            )
        else:
            # Fallback
            analysis = state.scene_analyzer.analyze_scene(image)
            position = (request.position_x, request.position_y)

            if hasattr(state.scene_analyzer, 'check_object_scene_compatibility'):
                score, reason = state.scene_analyzer.check_object_scene_compatibility(
                    request.object_class, position, analysis, (h, w)
                )
            else:
                score, reason = 0.6, "Default compatibility"

            return DebugCompatibilityResponse(
                success=True,
                is_compatible=score >= 0.4,
                score=float(score),
                reason=reason,
                decision="accepted" if score >= 0.4 else "rejected",
                region_at_position="unknown",
                alternatives=[],
            )

    except Exception as e:
        logger.error(f"Debug compatibility check failed: {e}")
        return DebugCompatibilityResponse(
            success=False,
            is_compatible=False,
            score=0.0,
            reason="Error during check",
            decision="error",
            region_at_position="unknown",
            alternatives=[],
            error=str(e),
        )
