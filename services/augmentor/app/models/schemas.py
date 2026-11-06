"""
Pydantic schemas for Augmentor Service API
==========================================
Defines request/response models for all endpoints.
"""

from typing import List, Optional, Dict, Tuple, Any
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# Enums
# =============================================================================

class EffectType(str, Enum):
    """Available realism effects"""
    COLOR_CORRECTION = "color_correction"
    BLUR_MATCHING = "blur_matching"
    LIGHTING = "lighting"
    UNDERWATER = "underwater"
    MOTION_BLUR = "motion_blur"
    SHADOWS = "shadows"
    CAUSTICS = "caustics"
    POISSON_BLEND = "poisson_blend"
    EDGE_SMOOTHING = "edge_smoothing"


class LightType(str, Enum):
    """Types of light sources"""
    DIRECTIONAL = "directional"
    POINT = "point"
    AREA = "area"


class DepthZone(str, Enum):
    """Depth zones for object placement"""
    NEAR = "near"
    MID = "mid"
    FAR = "far"


class WaterClarity(str, Enum):
    """Water clarity levels for underwater attenuation"""
    CLEAR = "clear"
    MURKY = "murky"
    VERY_MURKY = "very_murky"


# =============================================================================
# Data Models
# =============================================================================

class DatasetInfo(BaseModel):
    """Metadata information for the generated dataset (COCO format info section)"""
    name: str = Field("Synthetic Dataset", description="Dataset name")
    description: str = Field("", description="Dataset description")
    version: str = Field("1.0", description="Dataset version")
    year: int = Field(default_factory=lambda: datetime.now().year, description="Year of creation")
    contributor: str = Field("", description="Dataset contributor/author")
    url: str = Field("", description="Dataset URL or project link")
    license_name: str = Field("", description="License name (e.g., CC BY 4.0, MIT)")
    license_url: str = Field("", description="License URL")
    date_created: Optional[str] = Field(None, description="Creation date (auto-filled if empty)")


class ObjectPlacement(BaseModel):
    """Object to be placed in the composition"""
    image_path: str = Field(..., description="Path to object image (RGBA)")
    class_name: str = Field(..., description="Object class name")
    position: Optional[Tuple[int, int]] = Field(None, description="(x, y) position. Auto if None")
    scale: Optional[float] = Field(None, ge=0.1, le=5.0, description="Scale factor. Depth-aware if None")
    rotation: Optional[float] = Field(None, ge=-180, le=180, description="Rotation degrees. Random if None")
    material: Optional[str] = Field("plastic", description="Material type for physics validation")
    # Synthesis profile (v10): decisions pre-sampled by the planner
    size_quantile: Optional[float] = Field(None, ge=0, le=1, description="Quantile of the class's real size (faithful scale)")
    augment: Optional[Dict[str, Any]] = Field(None, description="Per-paste augmentation parameters")
    behavior_override: Optional[str] = Field(None, description="Forces the buoyancy behaviour (e.g. 'sink')")


class AnnotationBox(BaseModel):
    """Bounding box annotation for generated object"""
    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    width: int = Field(..., description="Box width")
    height: int = Field(..., description="Box height")
    class_name: str = Field(..., description="Object class name")
    confidence: float = Field(1.0, ge=0, le=1, description="Confidence score")
    area: int = Field(0, description="Object area in pixels")
    source_object: Optional[str] = Field(
        None, description="Object crop used for this annotation (path inside objects_dir)"
    )
    # Profile v10 traceability (None under the legacy behaviour)
    scale: Optional[float] = Field(None, description="Scale factor applied to the crop")
    target_rel_size: Optional[float] = Field(None, description="Target square root of the box's relative area")
    augmentations: Optional[Dict[str, Any]] = Field(None, description="Augmentations applied to the crop")
    # Copy-paste onto real images (v11): "real" = kept original box, "pasted" = pasted object
    origin: Optional[str] = Field(None, description="real | pasted (only with copy_paste_real)")
    # Mask of the pasted object (uncompressed COCO RLE, column-major); None unless `save_masks` is set in the profile
    mask_rle: Optional[Dict[str, Any]] = Field(None, description="RLE {size:[h,w], counts:[...]} of the object mask")


TRACE_FIELDS = ("scale", "target_rel_size", "augmentations")


class EffectsConfig(BaseModel):
    """Configuration for realism effects.

    Defaults tuned for underwater scenes via systematic parameter studies:
    a grid search over color_intensity/blur_strength/caustics/shadow, and an underwater tint
    integration study (the underwater effect is part of the default pipeline; water_color is a
    balanced ocean blue-green).
    """
    # ===== Optimised defaults =====
    color_intensity: float = Field(0.25, ge=0, le=1, description="Color correction intensity. 0.22-0.28 optimal: LAB Reinhard transfer adapts tone without bleaching")
    blur_strength: float = Field(0.65, ge=0, le=2, description="Blur matching strength. 0.6-0.7 optimal: matches background DOF without losing object detail")
    underwater_intensity: float = Field(0.33, ge=0, le=1, description="Underwater tint intensity. 0.28-0.35 optimal: unifies color cast between object and background")
    caustics_intensity: float = Field(0.12, ge=0, le=0.5, description="Caustics effect intensity. 0.10-0.13 optimal: unifies illumination without artefacts")
    shadow_opacity: float = Field(0.12, ge=0, le=0.5, description="Shadow darkness. 0.11-0.13 optimal: anchors objects to background with diffuse underwater shadow")

    # Options
    lighting_type: str = Field("ambient", description="spotlight|gradient|ambient")
    lighting_intensity: float = Field(0.5, ge=0, le=1, description="Lighting effect intensity")
    motion_blur_probability: float = Field(0.2, ge=0, le=1, description="Probability of motion blur")
    motion_blur_kernel: int = Field(11, ge=3, le=50, description="Motion blur kernel size")
    shadow_blur: int = Field(28, ge=3, le=61, description="Shadow blur kernel size. 25-30 optimal: soft diffuse shadow natural for underwater")
    edge_feather: int = Field(12, ge=1, le=25, description="Edge feathering pixels for edge_smoothing effect. 10-14 optimal: smooth transition zone at object boundary. Used only when edge_smoothing is in effects list")

    # Water settings
    water_color: Tuple[int, int, int] = Field((180, 120, 30), description="BGR water tint color. (180,120,30) = balanced ocean blue-green.")
    water_clarity: WaterClarity = Field(WaterClarity.CLEAR, description="Water clarity level")

    # ===== PHYSICAL REALISM (default OFF to preserve parameter-study comparability) =====
    underwater_physical: bool = Field(False, description="Use Beer-Lambert depth-dependent attenuation instead of flat tint for the underwater effect")
    caustics_depth_aware: bool = Field(False, description="Fade caustics with distance using the depth map")
    depth_occlusion: bool = Field(False, description="Occlude object pixels behind closer background geometry; objects <25% visible are retried")

    # Identity validation
    # Note: Disabled by default as thresholds are difficult to calibrate for strong underwater effects
    validate_identity: bool = Field(False, description="Validate object remains recognizable after effects")
    max_color_shift: float = Field(50, ge=10, le=100, description="Max LAB color shift for identity validation (delta-E units)")
    min_sharpness_ratio: float = Field(0.15, ge=0.05, le=1.0, description="Min sharpness preservation ratio for identity validation")
    min_contrast_ratio: float = Field(0.2, ge=0.05, le=1.0, description="Min contrast preservation ratio for identity validation")

    # Alpha blending
    use_binary_alpha: bool = Field(True, description="Use binary alpha instead of gradients to prevent color bleeding")
    alpha_feather_radius: int = Field(2, ge=0, le=3, description="Alpha feathering pixels. 2 optimal: anti-aliased edge without colour bleed")

    # Blur budget
    max_blur_budget: float = Field(10.0, ge=3, le=20, description="Max cumulative blur in pixels for small objects")

    # Size constraints
    min_object_size_ratio: float = Field(0.025, ge=0.01, le=0.1, description="Min object size as ratio of image dimension (2.5% = 16px for 640x640)")
    absolute_min_size: int = Field(15, ge=10, le=30, description="Absolute minimum object size in pixels")

    # Caustics
    caustics_deterministic: bool = Field(True, description="DEPRECATED (no effect): caustics are now seeded per image for dataset diversity")

    # Bbox recalculation
    recalculate_bbox_after_global: bool = Field(True, description="Recalculate bbox after global effects (caustics, underwater)")


class LightSourceInfo(BaseModel):
    """Detected light source information"""
    light_type: LightType = Field(..., description="Type of light source")
    position: Tuple[float, float, float] = Field(..., description="(azimuth, elevation, distance)")
    intensity: float = Field(..., ge=0, le=1, description="Light intensity")
    color: Tuple[int, int, int] = Field(..., description="Light color (RGB)")
    shadow_softness: float = Field(0.5, ge=0, le=1, description="Shadow edge softness")


class QualityScoreInfo(BaseModel):
    """Quality validation scores"""
    perceptual_quality: float = Field(..., ge=0, le=1, description="LPIPS-based quality")
    distribution_match: float = Field(1.0, ge=0, le=1, description="FID-based distribution match")
    anomaly_score: float = Field(..., ge=0, le=1, description="Anomaly detection score")
    composition_score: float = Field(..., ge=0, le=1, description="Composition plausibility")
    overall_score: float = Field(..., ge=0, le=1, description="Weighted average score")
    overall_pass: bool = Field(..., description="Meets all thresholds")


class PhysicsViolationInfo(BaseModel):
    """Physics validation violation"""
    violation_type: str = Field(..., description="Type of violation")
    object_class: str = Field(..., description="Class of violating object")
    severity: str = Field(..., description="low|medium|high")
    description: str = Field(..., description="Human-readable description")
    suggested_fix: Optional[str] = Field(None, description="How to fix the violation")


class LightingInfo(BaseModel):
    """Complete lighting estimation result"""
    light_sources: List[LightSourceInfo] = Field(default_factory=list)
    dominant_direction: Tuple[float, float] = Field(..., description="(azimuth, elevation)")
    color_temperature: float = Field(..., ge=2000, le=12000, description="Color temperature in Kelvin")
    ambient_intensity: float = Field(..., ge=0, le=1, description="Ambient light intensity")


# =============================================================================
# Request Models
# =============================================================================

class ComposeRequest(BaseModel):
    """Request to compose a single synthetic image"""
    background_path: str = Field(..., description="Path to background image")
    objects: List[ObjectPlacement] = Field(..., min_length=1, description="Objects to place")

    # Pre-computed data (optional)
    depth_map_path: Optional[str] = Field(None, description="Path to depth map (.npy)")
    lighting_info: Optional[LightingInfo] = Field(None, description="Pre-computed lighting")

    # Effects
    effects: List[EffectType] = Field(
        default=[EffectType.COLOR_CORRECTION, EffectType.BLUR_MATCHING],
        description="Effects to apply"
    )
    effects_config: EffectsConfig = Field(default_factory=EffectsConfig)

    # Validation
    validate_quality: bool = Field(False, description="Run quality validation")
    validate_physics: bool = Field(False, description="Run physics validation")

    # Placement
    semantic_placement: bool = Field(
        True,
        description="Place objects in scene regions compatible with their buoyancy (sink/float/hover). Disable for random-placement ablation."
    )

    # Output
    output_path: str = Field(..., description="Path to save composed image")
    save_annotations: bool = Field(True, description="Save annotation JSON")


class ComposeBatchRequest(BaseModel):
    """Request to compose multiple synthetic images"""
    backgrounds_dir: str = Field(..., description="Directory with background images")
    objects_dir: str = Field(..., description="Directory with object images by class")
    output_dir: str = Field(..., description="Output directory")

    # Generation config
    num_images: int = Field(..., ge=1, le=100000, description="Number of images to generate")
    targets_per_class: Optional[Dict[str, int]] = Field(None, description="Target count per class")
    max_objects_per_image: int = Field(5, ge=1, le=20, description="Max objects per image")

    # Effects
    effects: List[EffectType] = Field(
        default=[
            EffectType.COLOR_CORRECTION,
            EffectType.BLUR_MATCHING,
            EffectType.UNDERWATER,
            EffectType.CAUSTICS,
            EffectType.SHADOWS,
            EffectType.EDGE_SMOOTHING,
        ],
        description="Effects to apply. Default tuned for underwater realism."
    )
    effects_config: EffectsConfig = Field(default_factory=EffectsConfig)

    # Depth
    depth_aware: bool = Field(True, description="Use depth-aware placement")
    depth_service_url: Optional[str] = Field(None, description="URL to depth service")

    # Placement
    semantic_placement: bool = Field(
        True,
        description="Place objects in scene regions compatible with their buoyancy (sink/float/hover). Disable for random-placement ablation."
    )

    # Reproducibility
    synth_profile: Optional[str] = Field(
        None, description="Synthesis profile ('v10', 'v11'); None = legacy behaviour (no synthesis profile)"
    )
    synth_overrides: Optional[Dict[str, Any]] = Field(
        None, description="SynthProfile fields that replace those of the profile"
    )
    seed: Optional[int] = Field(
        None,
        description="Seed for Python/NumPy RNGs at job start. Fully reproducible only in sequential mode "
                    "(parallel mode interleaves random draws across concurrent compositions)"
    )

    # Validation
    validate_quality: bool = Field(False, description="Run quality validation")
    validate_physics: bool = Field(False, description="Run physics validation")
    reject_invalid: bool = Field(True, description="Reject images that fail validation")

    # Debug/Documentation
    save_pipeline_debug: bool = Field(
        False,
        description="Save intermediate pipeline images for first iteration (for documentation/paper)"
    )

    # Parallel Processing
    parallel: bool = Field(
        True,
        description="Enable parallel batch processing (4x faster throughput)"
    )
    concurrent_limit: int = Field(
        4,
        ge=1,
        le=8,
        description="Max images to process concurrently (2-8, recommended: 4)"
    )
    vram_threshold: float = Field(
        0.7,
        ge=0.5,
        le=0.9,
        description="VRAM usage threshold to trigger cleanup (0.5-0.9, recommended: 0.7)"
    )

    # Dataset Metadata
    dataset_info: Optional[DatasetInfo] = Field(
        None,
        description="Dataset metadata (name, description, version, license, etc.)"
    )


class ValidateRequest(BaseModel):
    """Request to validate a composed image"""
    image_path: str = Field(..., description="Path to image to validate")
    annotations: List[AnnotationBox] = Field(default_factory=list, description="Object annotations")

    # Reference data
    reference_images: Optional[List[str]] = Field(None, description="Reference images for FID")
    depth_map_path: Optional[str] = Field(None, description="Depth map for physics validation")

    # Options
    check_quality: bool = Field(True, description="Run LPIPS quality check")
    check_anomalies: bool = Field(True, description="Run anomaly detection")
    check_physics: bool = Field(True, description="Run physics validation")

    # Thresholds
    min_perceptual_quality: float = Field(0.7, ge=0, le=1)
    min_anomaly_score: float = Field(0.6, ge=0, le=1)


class LightingRequest(BaseModel):
    """Request to estimate lighting from background"""
    image_path: str = Field(..., description="Path to background image")

    # Options
    max_light_sources: int = Field(3, ge=1, le=5, description="Max light sources to detect")
    intensity_threshold: float = Field(0.6, ge=0.3, le=0.9, description="Min intensity threshold")
    estimate_hdr: bool = Field(False, description="Estimate HDR environment map")

    # Underwater
    apply_water_attenuation: bool = Field(False, description="Apply underwater attenuation")
    depth_category: DepthZone = Field(DepthZone.MID, description="Water depth category")
    water_clarity: WaterClarity = Field(WaterClarity.CLEAR, description="Water clarity")


# =============================================================================
# Response Models
# =============================================================================

class ComposeResponse(BaseModel):
    """Response from compose endpoint"""
    success: bool
    output_path: str
    annotations: List[AnnotationBox] = Field(default_factory=list)
    objects_placed: int
    depth_used: bool
    depth_source: Optional[str] = Field(None, description="service | fallback_gradient | precomputed")
    effects_applied: List[str]

    # Validation results (if requested)
    quality_score: Optional[QualityScoreInfo] = None
    physics_violations: List[PhysicsViolationInfo] = Field(default_factory=list)
    is_valid: bool = True
    rejection_reason: Optional[str] = None

    processing_time_ms: float
    error: Optional[str] = None


class ComposeBatchResponse(BaseModel):
    """Response from batch compose endpoint"""
    success: bool
    job_id: str
    status: str = Field(..., description="queued|processing|completed|failed")

    # Progress (if processing or completed)
    images_generated: int = 0
    images_rejected: int = 0
    images_pending: int = 0
    total_items: int = 0  # Total target samples to generate

    # Results (if completed)
    synthetic_counts: Dict[str, int] = Field(default_factory=dict)
    output_coco_path: Optional[str] = None
    output_dir: Optional[str] = Field(None, description="Job-specific output directory")

    processing_time_ms: float = 0
    error: Optional[str] = None


class ValidateResponse(BaseModel):
    """Response from validate endpoint"""
    is_valid: bool
    quality_score: QualityScoreInfo
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    physics_violations: List[PhysicsViolationInfo] = Field(default_factory=list)

    processing_time_ms: float
    error: Optional[str] = None


class LightingResponse(BaseModel):
    """Response from lighting estimation endpoint"""
    success: bool
    lighting_info: Optional[LightingInfo] = None

    processing_time_ms: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="healthy|degraded|unhealthy")
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_used: Optional[str] = None
    validators_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class InfoResponse(BaseModel):
    """Service info response"""
    service: str = "augmentor"
    version: str
    description: str
    endpoints: List[str]
    capabilities: Dict[str, bool]
    effects_available: List[str]
