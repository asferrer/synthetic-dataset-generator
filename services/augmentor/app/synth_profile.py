"""Synthesis profiles.

The default profile (`legacy`) activates nothing: the generator behaves exactly as in its original mode. Profile
`v10` activates the fidelity and robustness improvements described in docs/SYNTHESIS_PROFILE.md; `v11` adjusts it
after the ablation study. Any field can be overridden from the request (`synth_overrides`).
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class SynthProfile(BaseModel):
    """Fidelity/robustness options; all disabled by default."""

    name: str = "legacy"

    # Class-faithful scale (empirical distribution of the real training split)
    faithful_scale: bool = False
    stats_path: Optional[str] = Field(None, description="real_stats JSON; None = the bundled training-split one")
    min_object_px: int = Field(16, ge=4, description="Minimum box side (px) of the pasted object")
    max_crop_upscale: float = Field(2.0, ge=1.0, description="Maximum upscaling of a crop")
    min_mask_area_ratio: Optional[float] = Field(None, description="Replaces the compositor's min_area_ratio")
    max_mask_area_ratio: Optional[float] = Field(None, description="Replaces the compositor's max_area_ratio")

    # Faithful objects per image
    faithful_count: bool = False
    max_objects_cap: int = Field(20, ge=1, le=100, description="Cap on the number of objects per image")
    placement_try_count: Optional[int] = Field(None, ge=1, description="Attempts per object (None = the compositor's)")

    # Per-paste augmentations
    paste_augment: bool = False
    aug_hflip_prob: float = Field(0.5, ge=0, le=1)
    aug_rotation_deg: float = Field(20.0, ge=0, le=180, description="Uniform rotation in [-r, r]")
    aug_perspective: float = Field(0.06, ge=0, le=0.3, description="Maximum corner displacement (fraction of the side)")
    aug_brightness: float = Field(0.15, ge=0, le=0.6)
    aug_contrast: float = Field(0.15, ge=0, le=0.6)
    aug_saturation: float = Field(0.15, ge=0, le=0.6)
    aug_blur_prob: float = Field(0.2, ge=0, le=1)
    aug_blur_sigma_max: float = Field(1.0, ge=0.1, le=3.0)

    # Crop reuse cap
    max_crop_reuse: Optional[int] = Field(None, ge=1, description="Maximum uses per crop (None = no cap)")

    # Traceability
    trace: bool = False
    per_image_seed: bool = False

    # Faithful resolution and compression
    output_resolution: bool = False
    max_output_side: int = Field(1280, ge=64, description="Maximum longest side of the output image")
    max_bg_upscale: float = Field(2.0, ge=1.0, description="Maximum background upscaling when changing resolution")
    jpeg_recompress: bool = False
    jpeg_quality_fixed: Optional[int] = Field(None, ge=1, le=100, description="Fixed JPEG quality; replaces the sampled one")

    # Background balance
    background_round_robin: bool = False

    # Harmonisation and edges
    haze_from_stats: bool = False
    haze_max_intensity: float = Field(0.12, ge=0, le=1, description="Cap on the global haze/tint intensity")
    effects_overrides: Dict[str, Any] = Field(default_factory=dict, description="EffectsConfig fields that replace those of the request")

    # Physical placement
    seabed_fraction: Union[float, Dict[str, float]] = Field(
        0.0, description="float: fraction of floating/suspended objects resting on the seabed (same for all classes, "
        "only affects non-'sink' ones). dict {class: p}: per-class probability of being on the seabed; "
        "see docs/SYNTHESIS_PROFILE.md")
    water_column_behavior: str = Field("float", description="Behaviour of a 'sink' object that (with per-class "
                                       "seabed_fraction) does not rest on the seabed")

    # Additional crop libraries (added to objects_dir without copying or merging folders)
    extra_objects_dirs: List[str] = Field(default_factory=list, description="Extra <class>/<crop>.png roots")

    # Rectangular-crop filter (v11): alpha mask that fills almost its whole box
    crop_rect_filter: bool = False
    crop_rect_max_fill: float = Field(0.90, gt=0, le=1, description="A crop is discarded if the fraction of opaque "
                                      "pixels in its box exceeds this value")
    crop_rect_min_sides: Optional[int] = Field(3, ge=1, le=4, description="Or if >= n box sides are opaque "
                                               "(None: only the fraction is used)")
    crop_rect_side_coverage: float = Field(0.9, gt=0, le=1, description="A side counts as opaque if this fraction "
                                           "of its pixels is opaque")

    # Copy-paste onto real training images (v11, Ghiasi 2021): real background with its boxes kept
    copy_paste_real: bool = False
    real_annotations_path: Optional[str] = Field(None, description="COCO of the real backgrounds; None = "
                                                 "<backgrounds_dir>/annotations.json")
    real_overlap_max: float = Field(0.1, ge=0, le=1, description="Maximum overlap fraction between a pasted box and a "
                                    "kept real box (relative to the area of the smaller of the two)")

    # Local colour harmonisation of the pasted object (v11, Reinhard 2001 in a ring around the box)
    harmonize: bool = False
    harmonize_strength: float = Field(0.6, ge=0, le=1, description="Strength of the colour transfer (a, b)")
    harmonize_l_weight: float = Field(0.5, ge=0, le=1, description="Fraction of the strength applied to the L channel")
    harmonize_ring: float = Field(0.5, gt=0, le=2, description="Ring width, as a fraction of the box's longest side")

    # Copy-paste: the background is already a photo with its water column, so the object's haze and blur are limited
    # (None = no dedicated cap; only effective with copy_paste_real)
    copy_paste_haze_max: Optional[float] = Field(None, ge=0, le=1, description="Cap on the global haze intensity "
                                                 "with copy_paste_real (replaces haze_max_intensity if lower)")
    copy_paste_blur_max: Optional[int] = Field(None, ge=0, description="Cap (px) on the blend blur kernel "
                                               "with copy_paste_real; < 3 = no blur")

    # Visibility filter (v11): an object whose local contrast C against the background ring, measured after
    # compositing, is below the threshold is discarded (placement is retried; otherwise it is neither pasted nor annotated)
    visibility_filter: bool = False
    visibility_min_contrast: Union[float, Dict[str, float]] = Field(
        17.85, description="Threshold on C: float (all classes) or {class: threshold} (absent class = no filter). "
        "Taken from a low percentile of C over the REAL training objects")
    visibility_ring: float = Field(0.5, gt=0, le=2, description="Ring width, as a fraction of the box's longest side")

    # Single resample (v11, variant k): rotation + scale in one warpAffine without pre-blur (app/resample.py), and
    # sharpness and grain matched to those of a real object of its class (app/sharpness.py)
    single_resample: bool = False
    match_sharpness: bool = False
    sharpness_stats_path: Optional[str] = Field(None, description="real_sharpness JSON; None = the bundled training-split one")
    sharpness_quantile_range: Tuple[float, float] = Field((0.25, 0.75), description="Target real hf_ratio quantile: "
                                                          "uniform in this range (0-1)")
    sharpness_gain: float = Field(1.0, gt=0, le=3, description="Factor on the target hf_ratio (compensates what soft "
                                  "edges and JPEG subtract afterwards)")
    sharpness_max_amount: float = Field(3.0, gt=0, le=10, description="Maximum strength of the unsharp mask")
    sharpness_grain: bool = Field(True, description="Re-inject grain matching the flat-area noise of the ring")
    sharpness_grain_max: float = Field(8.0, ge=0, description="Maximum deviation (gray levels) of the added grain")

    # Crop scale within a range, by choosing another crop (v11, variant l; app/crop_scale.py)
    crop_scale_range: Optional[Tuple[float, float]] = Field(None, description="(min, max) of the paste scale; "
                                                            "None = unrestricted")

    # Position faithful to the training split with an edge margin (v11, variant m; app/position_stats.py)
    position_from_stats: bool = False
    position_stats_path: Optional[str] = Field(None, description="position_stats JSON; None = the bundled training-split one")
    position_margin: float = Field(0.03, ge=0, le=0.2, description="Edge margin of boxes that do not touch it "
                                   "(fraction of the shorter side, minimum 36 px; covers what shadows, blur and global effects widen)")
    border_touch_fraction: Optional[float] = Field(None, ge=0, le=1, description="Fraction of boxes touching the "
                                                   "edge; None = the real one of their class in the training split")

    # Visibility threshold by percentile of the real training objects (v11, variant n); replaces visibility_min_contrast
    visibility_percentile: Optional[int] = Field(None, description="Percentile (1, 5, 10, 15, 25 or 50) of C per class of "
                                                 "the real training objects; None = use visibility_min_contrast")

    # Annotated box = box fitted to the object mask (v11, variant o). Off = legacy behaviour (no synthesis profile):
    # the box is joined to the region changed by shadows (+-80 px) and global effects (+-20 px), and with any global
    # colour change it grows ~20 px per side.
    tight_bbox: bool = False

    # Store each pasted object's mask in the COCO (`mask_rle`, uncompressed RLE). Off by default: it changes neither
    # the random draws nor the boxes nor the images; it only adds the field.
    save_masks: bool = False

    # Crop-to-background pairing by the colour of the crop's source image (v11, variants q; app/background_match.py).
    # Off = legacy behaviour (no synthesis profile; same random draws).
    background_match: Optional[str] = Field(None, description="None | 'lab_knn': the crop is chosen among the k whose "
                                            "source image has the global LAB closest to the background's")
    background_match_k: int = Field(20, ge=1, description="Neighbours to sample from")
    background_match_temperature: float = Field(1.0, ge=0, description="Sampling temperature among the k (0 = the "
                                                "closest; weights exp(-(d-d1)/(T*(dk-d1))))")
    background_match_exclude_same_group: bool = Field(False, description="Exclude crops from images of the same group "
                                                      "as the background (False: legitimate within a training-split batch)")
    background_match_allow_same_image: bool = Field(False, description="Allow crops from the background's own image "
                                                    "(always False in batches)")
    background_match_hist_weight: float = Field(0.0, ge=0, description="Weight of the coarse LAB histogram (L1 x 100) added "
                                                "to the mean/std distance; 0 = mean and std only")
    background_match_path: Optional[str] = Field(None, description="Descriptor NPZ; None = the bundled training-split one")
    local_patch_match: bool = Field(False, description="Among the candidate positions of position_from_stats, the one "
                                    "whose box ring is most similar (LAB) to the crop's ring in its source image")
    local_patch_candidates: int = Field(8, ge=2, le=24, description="Valid candidate positions compared")

    # Ring-to-ring medium transfer in linear RGB (v11, M1; app/medium_transfer.py).
    # Off = legacy behaviour. Replaces `harmonize` (they can be combined: it is applied BEFORE harmonisation).
    medium_transfer: Optional[str] = Field(None, description="None | 'ring_gain' (per-channel gain between the medians "
                                           "of the source and destination rings) | 'ring_hist' (histogram map)")
    medium_lambda: float = Field(1.0, gt=0, le=1, description="Strength: F' = (1 - lambda) * F + lambda * transferred")
    medium_dark_floor: bool = Field(False, description="ring_gain with dark floor (1st percentile of each image): "
                                    "I' = g * (F - v_s) + v_d")
    medium_source: str = Field("crop", description="Where the crop's SOURCE ring comes from: 'crop' = what lies "
                               "outside the mask inside the crop itself (no table)")

    # Trimap matting and foreground decontamination of the crop (v11, M2; app/edge_matting.py).
    # Applied to the crop before scaling/rotating: native fractional alpha + F as colour. Needs `pymatting`.
    edge_matting: bool = False
    edge_matting_alpha: str = Field("closed_form", description="closed_form | knn | lkm (pymatting, CPU)")
    edge_matting_foreground: str = Field("none", description="none (F = original pixel) | germer "
                                         "(estimate_foreground_ml) | closed_form (estimate_foreground_cf)")
    edge_matting_a0: float = Field(0.0, ge=0, lt=0.5, description="Weak-halo clipping: alpha' = clip((alpha - a0) / (1 - a0))")
    edge_matting_edge_smoothing: bool = Field(True, description="Keep the compositor's edge smoothing on the matting "
                                              "alpha (False: pure alpha' * F + (1 - alpha') * background composition)")

    @field_validator("medium_transfer")
    @classmethod
    def _check_medium_transfer(cls, v):
        if v is not None and v not in ("ring_gain", "ring_hist"):
            raise ValueError("medium_transfer must be None, 'ring_gain' or 'ring_hist'")
        return v

    @field_validator("medium_source")
    @classmethod
    def _check_medium_source(cls, v):
        if v not in ("crop",):
            raise ValueError("medium_source must be 'crop'")
        return v

    @field_validator("edge_matting_alpha")
    @classmethod
    def _check_matting_alpha(cls, v):
        if v not in ("closed_form", "knn", "lkm"):
            raise ValueError("edge_matting_alpha must be closed_form, knn or lkm")
        return v

    @field_validator("edge_matting_foreground")
    @classmethod
    def _check_matting_fg(cls, v):
        if v not in ("germer", "closed_form", "none"):
            raise ValueError("edge_matting_foreground must be germer, closed_form or none")
        return v

    @field_validator("background_match")
    @classmethod
    def _check_background_match(cls, v):
        if v is not None and v != "lab_knn":
            raise ValueError("background_match must be None or 'lab_knn'")
        return v

    @model_validator(mode="after")
    def _check_local_patch(self):
        if self.local_patch_match and not self.position_from_stats:
            raise ValueError("local_patch_match requires position_from_stats")
        return self

    @field_validator("seabed_fraction")
    @classmethod
    def _check_seabed(cls, v):
        values = list(v.values()) if isinstance(v, dict) else [v]
        if any(not 0.0 <= x <= 1.0 for x in values):
            raise ValueError("seabed_fraction must be in [0, 1]")
        return v

    @field_validator("visibility_min_contrast")
    @classmethod
    def _check_visibility(cls, v):
        if any(x < 0 for x in (v.values() if isinstance(v, dict) else [v])):
            raise ValueError("visibility_min_contrast cannot be negative")
        return v

    @field_validator("crop_scale_range")
    @classmethod
    def _check_scale_range(cls, v):
        if v is not None and not 0 < v[0] <= v[1]:
            raise ValueError("crop_scale_range must be (min, max) with 0 < min <= max")
        return v

    @field_validator("sharpness_quantile_range")
    @classmethod
    def _check_quantile_range(cls, v):
        if not 0.0 <= v[0] <= v[1] <= 1.0:
            raise ValueError("sharpness_quantile_range must be (a, b) with 0 <= a <= b <= 1")
        return v

    @property
    def active(self) -> bool:
        """True if any behaviour differs from the legacy generator."""
        return self != SynthProfile()


# Per-class probability of being on the seabed, estimated on the real training split.
# Glove and Mask cannot be estimated (>50 % undetermined); set to 0.0.
V10_SEABED: Dict[str, float] = {
    "Bottle": 0.5, "Can": 1.0, "Metal_Debris": 1.0, "Tire": 1.0,
    "Fishing_Net": 0.5, "Plastic_Debris": 0.35, "Glove": 0.0, "Mask": 0.0,
}

V10_PROFILE = SynthProfile(
    name="v10",
    faithful_scale=True,
    min_object_px=16,
    max_crop_upscale=2.0,
    min_mask_area_ratio=0.0002,
    max_mask_area_ratio=0.95,
    faithful_count=True,
    max_objects_cap=20,
    placement_try_count=12,
    paste_augment=True,
    max_crop_reuse=15,
    trace=True,
    per_image_seed=True,
    output_resolution=True,
    jpeg_recompress=True,
    background_round_robin=True,
    haze_from_stats=True,
    haze_max_intensity=0.12,
    effects_overrides={"use_binary_alpha": False, "alpha_feather_radius": 4},
    seabed_fraction=dict(V10_SEABED),
    extra_objects_dirs=[],
)

# v11: v10 without per-paste augmentations (they did not help in the ablation) and with the rectangular-crop filter.
# Copy-paste and harmonisation stay off: they are enabled separately (ablation, see docs/SYNTHESIS_PROFILE.md).
V11_PROFILE = V10_PROFILE.model_copy(update={
    "name": "v11", "paste_augment": False, "crop_rect_filter": True,
    "effects_overrides": dict(V10_PROFILE.effects_overrides), "seabed_fraction": dict(V10_SEABED),
})

# 5th percentile of the local contrast C of the REAL training objects per class (computed on the real training
# boxes; Metal_Debris has very few). Threshold of the visibility filter.
V11_VISIBILITY_P5: Dict[str, float] = {
    "Bottle": 16.30, "Can": 14.32, "Fishing_Net": 26.80, "Glove": 21.82,
    "Mask": 24.22, "Metal_Debris": 10.77, "Plastic_Debris": 17.04, "Tire": 15.22,
}

# "Visible" copy-paste overrides (variant j): no global haze (the real background already carries its medium and a
# uniform haze does not bring the object closer to the background, it only subtracts contrast), blend blur kernel
# <= 3 and a per-class visibility filter. Kept out of V11_PROFILE so variants f/g/h/i stay reproducible.
V11_VISIBLE_OVERRIDES: Dict[str, Any] = {
    "copy_paste_haze_max": 0.0, "copy_paste_blur_max": 3,
    "visibility_filter": True, "visibility_min_contrast": dict(V11_VISIBILITY_P5),
}

# Chained ablation variants k-n (each adds one improvement to the previous one).
# They stay off in v10/V11_PROFILE and in variants c-j.
# sharpness_gain 1.3: calibrated on CPU; the final hf_ratio (after soft edges, blending and JPEG) comes out ~20 % below the target
V11_SHARP_OVERRIDES: Dict[str, Any] = {**V11_VISIBLE_OVERRIDES, "single_resample": True, "match_sharpness": True,
                                       "sharpness_gain": 1.3}
V11_SCALE_OVERRIDES: Dict[str, Any] = {**V11_SHARP_OVERRIDES, "crop_scale_range": (0.6, 1.5)}
V11_POSITION_OVERRIDES: Dict[str, Any] = {**V11_SCALE_OVERRIDES, "position_from_stats": True}
V11_VIS15_OVERRIDES: Dict[str, Any] = {**V11_POSITION_OVERRIDES, "visibility_percentile": 15}

# Variant o: n + box fitted to the mask (no union with shadows nor with the global colour change).
V11_TIGHT_OVERRIDES: Dict[str, Any] = {**V11_VIS15_OVERRIDES, "tight_bbox": True}

# Variants q: base = p_pos_tight WITHOUT `match_sharpness` (nor its grain) and with tight_bbox. Sharpening
# over-corrected the interior noise without lowering the AUC; q_base isolates that effect.
V11_Q_BASE_OVERRIDES: Dict[str, Any] = {**V11_POSITION_OVERRIDES, "match_sharpness": False, "sharpness_grain": False,
                                        "tight_bbox": True}
V11_Q_LAB_OVERRIDES: Dict[str, Any] = {**V11_Q_BASE_OVERRIDES, "background_match": "lab_knn",
                                       "background_match_exclude_same_group": True}
V11_Q_LAB_SAME_OVERRIDES: Dict[str, Any] = {**V11_Q_BASE_OVERRIDES, "background_match": "lab_knn"}
V11_Q_LAB_LOCAL_OVERRIDES: Dict[str, Any] = {**V11_Q_LAB_SAME_OVERRIDES, "local_patch_match": True}

# Variants r: M1 (ring-to-ring medium) and M2 (matting) on top of the q_lab_same base. Off in every named profile.
# `harmonize` is set True by REAL_BASE in tools/run_variants.py: r_medium turns it off because M1 replaces it.
V11_R_MATTE_OVERRIDES: Dict[str, Any] = {**V11_Q_LAB_SAME_OVERRIDES, "edge_matting": True}
V11_R_MATTE_FG_OVERRIDES: Dict[str, Any] = {**V11_R_MATTE_OVERRIDES, "edge_matting_foreground": "germer"}
V11_R_MEDIUM_OVERRIDES: Dict[str, Any] = {**V11_Q_LAB_SAME_OVERRIDES, "harmonize": False, "medium_transfer": "ring_gain",
                                          "medium_lambda": 0.5}
V11_R_MEDIUM_HARM_OVERRIDES: Dict[str, Any] = {**V11_Q_LAB_SAME_OVERRIDES, "medium_transfer": "ring_gain", "medium_lambda": 1.0}

_PROFILES: Dict[str, SynthProfile] = {"legacy": SynthProfile(), "v10": V10_PROFILE, "v11": V11_PROFILE}


def get_profile(name: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> Optional[SynthProfile]:
    """Profile by name with overrides; None (or an unchanged 'legacy') = legacy behaviour."""
    if name is None and not overrides:
        return None
    key = name or "legacy"
    if key not in _PROFILES:
        raise ValueError(f"Unknown profile: {key!r} (available: {sorted(_PROFILES)})")
    profile = _PROFILES[key].model_copy(update=overrides or {})
    profile = SynthProfile.model_validate(profile.model_dump())
    return profile if profile.active else None


def derive_image_seed(base_seed: int, attempt: int) -> int:
    """Deterministic per-attempt image seed (32 bits) derived from the batch seed."""
    digest = hashlib.sha256(f"{base_seed}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:4], "big")
