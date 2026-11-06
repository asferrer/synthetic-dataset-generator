"""ImageComposer: the public composer class, assembled from the composition mixins."""

import os
from typing import Optional

from loguru import logger

from app.caches import LRUCache

from app.composition.validation import ValidationMixin
from app.composition.single_image import SingleImageMixin
from app.composition.batch import BatchMixin
from app.composition.batch_parallel import ParallelBatchMixin
from app.composition.output import OutputMixin
from app.composition.debug_viz import DebugVizMixin
from app.composition.object_placement import PlacementMixin
from app.composition.transform import TransformMixin
from app.composition.scene_analysis import SceneMixin
from app.composition.effects import EffectsMixin

class ImageComposer(
    ValidationMixin,
    SingleImageMixin,
    BatchMixin,
    ParallelBatchMixin,
    OutputMixin,
    DebugVizMixin,
    PlacementMixin,
    TransformMixin,
    SceneMixin,
    EffectsMixin,
):
    """
    Handles synthetic image composition with realistic effects.

    Features:
    - Depth-aware object scaling and placement
    - Multi-light source shadow generation
    - 11+ realism effects (color correction, blur, caustics, etc.)
    - Physics-aware composition
    """

    def __init__(
        self,
        use_gpu: bool = True,
        depth_service_url: Optional[str] = None,
        max_upscale_ratio: float = 4.0,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.4,
        overlap_threshold: float = 0.1,
        try_count: int = 5,
    ):
        self.use_gpu = use_gpu
        self.depth_service_url = depth_service_url or os.environ.get("DEPTH_SERVICE_URL", "http://depth:8001")
        self.max_upscale_ratio = max_upscale_ratio
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.overlap_threshold = overlap_threshold
        self.try_count = try_count

        # Lazy-loaded components
        self._lighting_estimator = None
        self._segmentation_client = None

        # Per-background caches: batches reuse few backgrounds, so depth
        # maps, scene analyses and decoded objects are shared across images
        self._depth_cache = LRUCache(maxsize=32)
        self._scene_cache = LRUCache(maxsize=32)
        self._object_cache = LRUCache(maxsize=128)
        self._matte_cache = LRUCache(maxsize=128)       # matted crops (edge_matting)
        self._medium_src_cache = LRUCache(maxsize=512)  # source-ring statistics (medium_transfer)

        # Validators injected by the service after startup (see main.py)
        self.quality_validator = None
        self.physics_validator = None

        # Segmentation service URL
        self.segmentation_service_url = os.environ.get("SEGMENTATION_SERVICE_URL", "http://segmentation:8002")

        logger.info(f"ImageComposer initialized (GPU: {use_gpu})")

    @property
    def lighting_estimator(self):
        """Lazy load lighting estimator"""
        if self._lighting_estimator is None:
            try:
                from app.lighting_engine import AdvancedLightingEstimator
                self._lighting_estimator = AdvancedLightingEstimator(
                    max_light_sources=3,
                    intensity_threshold=0.6,
                    use_hdr_estimation=False
                )
                logger.info("AdvancedLightingEstimator loaded")
            except ImportError as e:
                logger.warning(f"LightingEstimator not available: {e}, using fallback")
        return self._lighting_estimator

    @property
    def segmentation_client(self):
        """Lazy load segmentation client for scene analysis via HTTP"""
        if self._segmentation_client is None:
            try:
                from app.segmentation_client import SegmentationClient
                self._segmentation_client = SegmentationClient(
                    service_url=self.segmentation_service_url,
                    timeout=30,
                    debug=True,
                )
                logger.info(f"SegmentationClient initialized: {self.segmentation_service_url}")
            except ImportError as e:
                logger.warning(f"SegmentationClient not available: {e}")
        return self._segmentation_client
