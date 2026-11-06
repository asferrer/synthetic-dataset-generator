"""
Rutas de salud y estado del modelo.
"""

import asyncio
import time

from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.service_state import state


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health

    Returns healthy even while SAM3 is loading in background.
    Check sam3_loading and sam3_load_progress for detailed status.
    """
    # Service is healthy as long as it's running
    # SAM3 loading happens in background and doesn't affect basic health
    return HealthResponse(
        status="healthy",
        sam3_available=state.sam3_available,
        sam3_loading=state.sam3_loading,
        sam3_load_progress=state.sam3_load_progress,
        sam3_load_error=state.sam3_load_error,
        gpu_available=state.gpu_available,
        model_loaded=state.scene_analyzer is not None,
        version="1.0.0",
    )


async def wait_for_sam3(timeout: float = 300.0) -> bool:
    """Wait for SAM3 to finish loading.

    Args:
        timeout: Maximum time to wait in seconds (default 5 minutes)

    Returns:
        True if SAM3 is available, False if loading failed or timed out
    """
    if state.sam3_available:
        return True

    if not state.sam3_loading:
        # Not loading and not available = loading failed
        return False

    # Wait for loading to complete
    start = time.time()
    while state.sam3_loading and (time.time() - start) < timeout:
        await asyncio.sleep(0.5)

    return state.sam3_available


@router.get("/model-status", tags=["Health"])
async def model_status():
    """Get detailed model loading status"""
    return {
        "sam3": {
            "available": state.sam3_available,
            "loading": state.sam3_loading,
            "progress": state.sam3_load_progress,
            "error": state.sam3_load_error,
        },
        "gpu": {
            "available": state.gpu_available,
            "device": state.device,
        },
        "scene_analyzer": {
            "initialized": state.scene_analyzer is not None,
        },
        "object_extractor": {
            "initialized": state.object_extractor is not None,
        },
    }
