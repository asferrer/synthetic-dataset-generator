"""
Inicialización del servicio: GPU, modelo SAM 3, analizadores y ciclo de vida.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.object_extractor import ObjectExtractor
from app.scene_analyzer import SemanticSceneAnalyzer
from app.service_state import JOB_DATABASE_AVAILABLE, get_job_db, state, thread_pool

logger = logging.getLogger(__name__)


def init_scene_analyzer():
    """Initialize scene analyzers with SAM3 model if available"""
    # Normal analyzer for regular requests
    state.scene_analyzer = SemanticSceneAnalyzer(
        use_sam3=state.sam3_available,
        device="cuda" if state.gpu_available else "cpu",
        sam3_model=state.sam3_model,
        sam3_processor=state.sam3_processor,
    )

    # Debug analyzer (pre-initialized to avoid creation on each debug request)
    debug_output_dir = "/shared/segmentation/debug"
    os.makedirs(debug_output_dir, exist_ok=True)
    state.debug_scene_analyzer = SemanticSceneAnalyzer(
        use_sam3=state.sam3_available,
        device="cuda" if state.gpu_available else "cpu",
        sam3_model=state.sam3_model,
        sam3_processor=state.sam3_processor,
        debug=True,
        debug_output_dir=debug_output_dir,
    )

    logger.info(f"Scene analyzers initialized (SAM3: {state.sam3_available})")


def init_object_extractor():
    """Initialize object extractor with shared SAM3 model"""
    state.object_extractor = ObjectExtractor(
        sam3_model=state.sam3_model,
        sam3_processor=state.sam3_processor,
        device=state.device
    )
    logger.info(f"ObjectExtractor initialized (SAM3: {state.sam3_available})")


def init_gpu():
    """Initialize GPU detection (fast, non-blocking)"""
    import torch

    # Check GPU availability
    state.gpu_available = torch.cuda.is_available()
    state.device = "cuda" if state.gpu_available else "cpu"

    if state.gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU available: {gpu_name} ({gpu_mem:.1f}GB VRAM)")
    else:
        logger.warning("No GPU available, using CPU")


def _load_sam3_sync():
    """Load SAM3 model synchronously (called from background thread)"""
    import torch

    state.sam3_load_progress = "Starting SAM3 load..."

    # Try to load SAM3 (Segment Anything Model 3)
    # Released 2025-11-19, uses Promptable Concept Segmentation (PCS)
    try:
        from transformers import Sam3Processor, Sam3Model

        # Official model: facebook/sam3 (848M parameters)
        model_id = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
        hf_token = os.environ.get("HF_TOKEN")

        logger.info(f"[Background] Loading SAM3 model: {model_id}")
        state.sam3_load_progress = f"Loading processor from {model_id}..."
        load_start = time.time()

        # Load processor (fast)
        state.sam3_processor = Sam3Processor.from_pretrained(
            model_id,
            token=hf_token,
        )
        proc_time = time.time() - load_start
        logger.info(f"[Background] Processor loaded in {proc_time:.1f}s")
        state.sam3_load_progress = "Loading model weights..."

        # Optimized model loading:
        # - torch_dtype=float16: Uses half precision (50% less VRAM, faster loading)
        # - low_cpu_mem_usage=True: Reduces peak RAM usage during loading
        model_start = time.time()
        use_fp16 = state.gpu_available and os.environ.get("SAM3_FP32", "").lower() != "true"

        if use_fp16:
            logger.info("[Background] Loading model in FP16 (half precision)...")
            state.sam3_load_progress = "Loading model in FP16 mode..."
            state.sam3_model = Sam3Model.from_pretrained(
                model_id,
                token=hf_token,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(state.device)
        else:
            logger.info("[Background] Loading model in FP32 (full precision)...")
            state.sam3_load_progress = "Loading model in FP32 mode..."
            state.sam3_model = Sam3Model.from_pretrained(
                model_id,
                token=hf_token,
                low_cpu_mem_usage=True,
            ).to(state.device)

        state.sam3_model.eval()
        model_time = time.time() - model_start
        logger.info(f"[Background] Model loaded in {model_time:.1f}s")

        # Mark SAM3 as available BEFORE initializing dependent components
        # so they are created with SAM3 support enabled
        state.sam3_available = True

        # Initialize dependent components
        state.sam3_load_progress = "Initializing scene analyzer..."
        init_scene_analyzer()
        init_object_extractor()

        state.sam3_load_progress = "Ready"
        total_time = time.time() - load_start
        logger.info(f"[Background] SAM3 fully initialized in {total_time:.1f}s total")

    except ImportError as e:
        error_msg = f"SAM3 not available (transformers may need update): {e}"
        logger.warning(f"[Background] {error_msg}")
        logger.info("Install transformers from main: pip install git+https://github.com/huggingface/transformers.git")
        state.sam3_load_error = error_msg
        state.sam3_available = False
        # Still initialize analyzers with heuristic-based mode
        init_scene_analyzer()
        init_object_extractor()
    except Exception as e:
        error_msg = f"SAM3 loading failed: {e}"
        logger.warning(f"[Background] {error_msg}")
        state.sam3_load_error = error_msg
        state.sam3_available = False
        # Still initialize analyzers with heuristic-based mode
        init_scene_analyzer()
        init_object_extractor()
    finally:
        state.sam3_loading = False


async def init_sam3_background():
    """Initialize SAM3 model in background (non-blocking)"""
    state.sam3_loading = True
    state.sam3_load_error = None
    state.sam3_load_progress = "Queued for loading..."

    logger.info("Starting SAM3 background loading...")

    # Run the sync loading function in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(thread_pool, _load_sam3_sync)


def init_sam3():
    """Initialize SAM3 model synchronously (legacy, blocking mode)

    Use init_sam3_background() for non-blocking startup.
    """
    init_gpu()
    _load_sam3_sync()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic with fast non-blocking startup"""
    logger.info("Starting Segmentation Service...")
    startup_start = time.time()

    # Initialize database for job persistence
    if JOB_DATABASE_AVAILABLE and get_job_db is not None:
        try:
            state.db = get_job_db()
            logger.info("Job database initialized for persistence")

            # Mark any jobs left in running/queued state as interrupted
            # (they were abandoned when the service was last shut down)
            orphaned = state.db.mark_orphaned_jobs("segmentation")
            if orphaned:
                logger.warning(f"Marked {orphaned} orphaned segmentation jobs as interrupted")
        except Exception as e:
            logger.warning(f"Failed to initialize job database: {e}")
            state.db = None
    else:
        logger.warning("Job database not available - jobs will not persist across restarts")

    # Fast startup: Initialize GPU detection only (non-blocking)
    init_gpu()

    # Check if we should use blocking or background loading
    # Set SAM3_LAZY_LOAD=false to use blocking mode (useful for debugging)
    use_lazy_load = os.environ.get("SAM3_LAZY_LOAD", "true").lower() != "false"

    if use_lazy_load:
        # Start SAM3 loading in background - service is immediately available
        logger.info("Service starting with lazy SAM3 loading (non-blocking)")
        state._loading_task = asyncio.create_task(init_sam3_background())
    else:
        # Blocking mode: Wait for SAM3 to load before accepting requests
        logger.info("Service starting with blocking SAM3 loading")
        _load_sam3_sync()

    startup_time = time.time() - startup_start
    logger.info(f"Segmentation Service ready in {startup_time:.1f}s (SAM3 loading in background: {use_lazy_load})")
    yield

    # Cleanup
    logger.info("Shutting down Segmentation Service...")

    # Cancel background loading if still running
    if state._loading_task and not state._loading_task.done():
        state._loading_task.cancel()
        try:
            await state._loading_task
        except asyncio.CancelledError:
            pass

    state.sam3_model = None
    state.scene_analyzer = None
