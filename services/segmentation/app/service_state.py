"""
Estado compartido del servicio de segmentación: estado del modelo, registros de
trabajos en memoria, pool de hilos y control de concurrencia del etiquetado.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from app.object_extractor import ObjectExtractor

# Shared utilities
try:
    from shared.vram_monitor import VRAMMonitor
    VRAM_MONITOR_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.insert(0, '/app')
        from shared.vram_monitor import VRAMMonitor
        VRAM_MONITOR_AVAILABLE = True
    except ImportError:
        VRAM_MONITOR_AVAILABLE = False
        VRAMMonitor = None

# Job database for persistence
try:
    from shared.job_database import JobDatabase, get_job_db
    JOB_DATABASE_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.insert(0, '/app')
        from shared.job_database import JobDatabase, get_job_db
        JOB_DATABASE_AVAILABLE = True
    except ImportError:
        JOB_DATABASE_AVAILABLE = False
        JobDatabase = None
        get_job_db = None


# Global state
class ServiceState:
    """Global service state"""
    scene_analyzer = None
    debug_scene_analyzer = None  # Pre-initialized debug analyzer to avoid re-creation on each request
    sam3_model = None
    sam3_processor = None
    device: str = "cpu"
    sam3_available: bool = False
    sam3_loading: bool = False  # True while SAM3 is being loaded in background
    sam3_load_error: Optional[str] = None  # Error message if loading failed
    sam3_load_progress: str = ""  # Progress message during loading
    gpu_available: bool = False
    object_extractor: Optional[ObjectExtractor] = None
    db: Optional["JobDatabase"] = None  # Database for job persistence
    _loading_task: Optional[asyncio.Task] = None  # Background loading task


state = ServiceState()

# Job tracking for async operations
extraction_jobs: Dict[str, Dict[str, Any]] = {}
sam3_conversion_jobs: Dict[str, Dict[str, Any]] = {}
labeling_jobs: Dict[str, Dict[str, Any]] = {}

# Thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=2)

# Concurrency control for labeling jobs
# Note: SAM3 uses ~4-6GB VRAM per job. Adjust based on available GPU memory.
# Conservative settings to prevent GPU overheating and leave memory headroom.
# Set to 1 to prevent resource saturation with large datasets (13k+ images)
MAX_CONCURRENT_LABELING_JOBS = 1
MAX_CONCURRENT_IMAGES_PER_JOB = 4
labeling_job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LABELING_JOBS)
