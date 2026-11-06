"""
Rutas de análisis de escena, compatibilidad, colocación y segmentación por texto.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter

from app.models.schemas import (
    AnalyzeSceneRequest,
    AnalyzeSceneResponse,
    CompatibilityCheckRequest,
    CompatibilityCheckResponse,
    SegmentTextRequest,
    SegmentTextResponse,
    SuggestPlacementRequest,
    SuggestPlacementResponse,
)
from app.routers.health import wait_for_sam3
from app.service_state import state
from app.utils.region_encoding import encode_region_map_base64

logger = logging.getLogger(__name__)


router = APIRouter()


@router.post("/analyze", response_model=AnalyzeSceneResponse, tags=["Analysis"])
async def analyze_scene(request: AnalyzeSceneRequest):
    """Analyze scene regions in an image"""
    start_time = time.time()

    try:
        # Wait for SAM3 if still loading (with 30s timeout for this endpoint)
        if state.sam3_loading and state.scene_analyzer is None:
            logger.info("Waiting for SAM3 to load before analyzing scene...")
            await wait_for_sam3(timeout=30.0)

        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        # Analyze scene
        if hasattr(state.scene_analyzer, 'analyze_scene'):
            result = state.scene_analyzer.analyze_scene(image)

            # Handle both object and dict returns
            if hasattr(result, 'dominant_region'):
                # Object with attributes - encode region_map for transfer
                region_map_b64 = None
                if hasattr(result, 'region_map') and result.region_map is not None:
                    region_map_b64 = encode_region_map_base64(result.region_map)

                return AnalyzeSceneResponse(
                    success=True,
                    dominant_region=result.dominant_region.value if hasattr(result.dominant_region, 'value') else str(result.dominant_region),
                    region_scores=result.region_scores,
                    depth_zones={k: list(v) for k, v in result.depth_zones.items()},
                    scene_brightness=result.scene_brightness,
                    water_clarity=result.water_clarity,
                    color_temperature=result.color_temperature,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    region_map_base64=region_map_b64,
                )
            else:
                # Dict return
                return AnalyzeSceneResponse(
                    success=True,
                    dominant_region=result.get("dominant_region", "unknown"),
                    region_scores=result.get("region_scores", {}),
                    depth_zones=result.get("depth_zones", {}),
                    scene_brightness=result.get("scene_brightness", 0.5),
                    water_clarity=result.get("water_clarity", "moderate"),
                    color_temperature=result.get("color_temperature", "neutral"),
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
        else:
            raise RuntimeError("Scene analyzer not initialized")

    except FileNotFoundError as e:
        return AnalyzeSceneResponse(
            success=False,
            dominant_region="unknown",
            region_scores={},
            depth_zones={},
            scene_brightness=0.0,
            water_clarity="unknown",
            color_temperature="unknown",
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )
    except Exception as e:
        logger.error(f"Scene analysis failed: {e}")
        return AnalyzeSceneResponse(
            success=False,
            dominant_region="unknown",
            region_scores={},
            depth_zones={},
            scene_brightness=0.0,
            water_clarity="unknown",
            color_temperature="unknown",
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )


@router.post("/check-compatibility", response_model=CompatibilityCheckResponse, tags=["Analysis"])
async def check_compatibility(request: CompatibilityCheckRequest):
    """Check if object placement is compatible with scene"""
    try:
        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        h, w = image.shape[:2]

        # Analyze scene first
        if hasattr(state.scene_analyzer, 'analyze_scene'):
            scene_result = state.scene_analyzer.analyze_scene(image)
        else:
            raise RuntimeError("Scene analyzer not initialized")

        # Check compatibility
        position = (request.position_x, request.position_y)

        if hasattr(state.scene_analyzer, 'check_object_scene_compatibility'):
            score, reason = state.scene_analyzer.check_object_scene_compatibility(
                request.object_class,
                position,
                scene_result,
                (h, w),
            )
        else:
            score, reason = 0.6, "Default compatibility"

        # Get best region suggestion
        suggested_region = None
        if hasattr(state.scene_analyzer, 'get_best_placement_region'):
            best_region = state.scene_analyzer.get_best_placement_region(
                request.object_class,
                scene_result,
            )
            if best_region:
                suggested_region = best_region.value if hasattr(best_region, 'value') else str(best_region)

        return CompatibilityCheckResponse(
            success=True,
            is_compatible=score >= 0.4,
            score=score,
            reason=reason,
            suggested_region=suggested_region,
        )

    except Exception as e:
        logger.error(f"Compatibility check failed: {e}")
        return CompatibilityCheckResponse(
            success=False,
            is_compatible=False,
            score=0.0,
            reason="Error during check",
            error=str(e),
        )


@router.post("/suggest-placement", response_model=SuggestPlacementResponse, tags=["Analysis"])
async def suggest_placement(request: SuggestPlacementRequest):
    """Suggest best placement position for an object"""
    try:
        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        h, w = image.shape[:2]

        # Analyze scene
        if hasattr(state.scene_analyzer, 'analyze_scene'):
            scene_result = state.scene_analyzer.analyze_scene(image)
        else:
            raise RuntimeError("Scene analyzer not initialized")

        # Convert existing positions
        existing = [(p[0], p[1]) for p in request.existing_positions]

        # Get suggestion
        if hasattr(state.scene_analyzer, 'suggest_placement_position'):
            position = state.scene_analyzer.suggest_placement_position(
                request.object_class,
                (request.object_width, request.object_height),
                scene_result,
                (h, w),
                existing,
                request.min_distance,
            )
        else:
            # Simple fallback
            import random
            margin = 50
            position = (
                random.randint(margin, w - request.object_width - margin),
                random.randint(margin, h - request.object_height - margin),
            )

        if position is None:
            return SuggestPlacementResponse(
                success=False,
                error="No valid placement position found",
            )

        # Get best region
        best_region = None
        if hasattr(state.scene_analyzer, 'get_best_placement_region'):
            region = state.scene_analyzer.get_best_placement_region(
                request.object_class,
                scene_result,
            )
            if region:
                best_region = region.value if hasattr(region, 'value') else str(region)

        # Get compatibility score
        score = 0.7
        if hasattr(state.scene_analyzer, 'check_object_scene_compatibility'):
            score, _ = state.scene_analyzer.check_object_scene_compatibility(
                request.object_class,
                position,
                scene_result,
                (h, w),
            )

        return SuggestPlacementResponse(
            success=True,
            position_x=position[0],
            position_y=position[1],
            best_region=best_region,
            compatibility_score=score,
        )

    except Exception as e:
        logger.error(f"Placement suggestion failed: {e}")
        return SuggestPlacementResponse(
            success=False,
            error=str(e),
        )


@router.post("/segment-text", response_model=SegmentTextResponse, tags=["Segmentation"])
async def segment_text(request: SegmentTextRequest):
    """
    Text-driven segmentation using SAM3 (Segment Anything Model 3).

    SAM3 uses Promptable Concept Segmentation (PCS) to segment all instances
    in an image that match a given text concept.
    """
    start_time = time.time()

    # Wait for SAM3 if still loading
    if state.sam3_loading:
        logger.info("Waiting for SAM3 to load for text segmentation...")
        await wait_for_sam3(timeout=60.0)

    if not state.sam3_available:
        error_msg = state.sam3_load_error or "SAM3 not available. Install: pip install transformers>=4.45.0"
        return SegmentTextResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=error_msg,
        )

    try:
        import torch
        from PIL import Image as PILImage

        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        h, w = image.shape[:2]

        # Convert BGR to RGB PIL Image (required by SAM3)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        # Process with SAM3 text-prompted segmentation (Promptable Concept Segmentation)
        # Following official HuggingFace documentation: https://huggingface.co/facebook/sam3
        inputs = state.sam3_processor(
            images=pil_image,
            text=request.text_prompt,
            return_tensors="pt"
        ).to(state.device)

        with torch.no_grad():
            outputs = state.sam3_model(**inputs)

        # Post-process to get instance segmentation results
        # Use original_sizes from processor for accurate mask resizing
        target_sizes = inputs.get("original_sizes")
        if target_sizes is not None:
            target_sizes = target_sizes.tolist()
        else:
            target_sizes = [(h, w)]

        results = state.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=request.threshold,
            mask_threshold=request.threshold,
            target_sizes=target_sizes
        )[0]

        # Combine all masks
        combined_mask = np.zeros((h, w), dtype=np.float32)
        max_confidence = 0.0

        if 'masks' in results and len(results['masks']) > 0:
            for mask, score in zip(results['masks'], results['scores']):
                mask_np = mask.cpu().numpy().astype(np.float32)
                score_val = score.cpu().item()

                # Resize if needed
                if mask_np.shape != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h))

                combined_mask = np.maximum(combined_mask, mask_np)
                max_confidence = max(max_confidence, score_val)

        # Create binary mask
        mask_binary = (combined_mask > 0.5).astype(np.uint8) * 255

        # Save mask
        output_dir = Path("/shared/segmentation")
        output_dir.mkdir(parents=True, exist_ok=True)

        mask_filename = f"mask_{Path(request.image_path).stem}_{int(time.time())}.png"
        mask_path = output_dir / mask_filename
        cv2.imwrite(str(mask_path), mask_binary)

        # Calculate coverage
        coverage = float((mask_binary > 0).sum()) / (h * w)

        return SegmentTextResponse(
            success=True,
            mask_path=str(mask_path),
            mask_coverage=coverage,
            confidence=max_confidence,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        logger.error(f"Text segmentation failed: {e}")
        return SegmentTextResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )
