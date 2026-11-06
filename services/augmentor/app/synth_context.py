"""State of a batch under an active profile: crops, backgrounds, seeds and per-image plan.

Everything is synchronous (no awaits), so under asyncio the operations are atomic across coroutines of the same loop.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app import background_match, copy_paste, crop_scale, output_format
from app.real_stats import RealStats, load_stats
from app.synth_profile import SynthProfile, derive_image_seed


def merge_extra_libraries(objects_by_class: Dict[str, List[dict]], profile: Optional[SynthProfile]) -> None:
    """Adds to `objects_by_class` the crops of `profile.extra_objects_dirs` (<root>/<class>/<crop>.png).

    Folders are neither copied nor merged: each crop keeps its full path (the root tells the origin apart in
    `source_object`). A profile without extra roots changes nothing. A missing root is an explicit error.
    """
    import os
    for root in (profile.extra_objects_dirs if profile else []):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"extra_objects_dirs: directory not found: {root}")
        for cls in sorted(os.listdir(root)):
            cdir = os.path.join(root, cls)
            if not os.path.isdir(cdir):
                continue
            for name in sorted(os.listdir(cdir)):
                if name.lower().endswith((".png", ".jpg", ".jpeg")):
                    objects_by_class.setdefault(cls, []).append({"path": os.path.join(cdir, name), "name": name})
        logger.info(f"Extra library {root} added")


class CropUsage:
    """Per-crop usage counter with cap `max_reuse` (None = no cap).

    `acquire` reserves a use (the least used crop, random tie-break);
    `settle` confirms or returns the reservations when the image finishes.
    """

    def __init__(self, objects_by_class: Dict[str, List[dict]], max_reuse: Optional[int]):
        self.max_reuse = max_reuse
        self.paths: Dict[str, List[str]] = {c: [o["path"] for o in objs] for c, objs in objects_by_class.items()}
        self.used: Dict[str, int] = defaultdict(int)
        self.pending = 0

    def has_free(self, cls: str) -> bool:
        if self.max_reuse is None:
            return bool(self.paths.get(cls))
        return any(self.used[p] < self.max_reuse for p in self.paths.get(cls, ()))

    def acquire(self, cls: str, rng, narrow=None) -> Optional[str]:
        """Reserves a use. `narrow` (optional): function that reduces the list of free crops (e.g. by scale)."""
        candidates = [p for p in self.paths.get(cls, ()) if self.max_reuse is None or self.used[p] < self.max_reuse]
        if not candidates:
            return None
        if narrow is not None:
            candidates = narrow(candidates)
        least = min(self.used[p] for p in candidates)
        path = rng.choice([p for p in candidates if self.used[p] == least])
        self.used[path] += 1
        self.pending += 1
        return path

    def settle(self, reserved: List[str], kept: List[Optional[str]]) -> None:
        """Closes the reservations of an image; those not used in `kept` are returned."""
        remaining = list(kept)
        for path in reserved:
            self.pending -= 1
            if path in remaining:
                remaining.remove(path)
            else:
                self.used[path] -= 1

    def preload_from_coco(self, coco: Optional[dict]) -> None:
        """On resume: counts the uses already present in the confirmed COCO."""
        for ann in (coco or {}).get("annotations", []):
            if ann.get("source_object"):
                self.used[ann["source_object"]] += 1

    def stats(self, cls: str) -> Dict[str, int]:
        return {"crops": len(self.paths.get(cls, ())), "uses": sum(self.used[p] for p in self.paths.get(cls, ()))}


@dataclass
class ImagePlan:
    """Per-image decisions applied inside compose()."""
    profile: SynthProfile
    seed: Optional[int] = None
    background: Optional[str] = None
    output_size: Optional[Tuple[int, int]] = None
    jpeg_quality: Optional[int] = None
    target_l_std: Optional[float] = None
    objects_requested: int = 0
    stats: Optional[RealStats] = None
    real_annotations: List[Any] = field(default_factory=list)   # copy-paste: kept real boxes (dict)

    @property
    def real_boxes(self) -> List[Tuple[int, int, int, int]]:
        """Kept real boxes as (x1, y1, x2, y2) at the output resolution."""
        return [(int(r["x"]), int(r["y"]), int(r["x"] + r["w"]), int(r["y"] + r["h"])) for r in self.real_annotations]


class SynthContext:
    """Profile + real statistics + crop usage + seeds of a batch."""

    def __init__(self, profile: SynthProfile, objects_by_class: Dict[str, List[dict]],
                 backgrounds_dir: str, bg_files: List[str], base_seed: Optional[int] = None):
        self.profile = profile
        self.stats: RealStats = load_stats(profile.stats_path)
        self.usage = CropUsage(objects_by_class, profile.max_crop_reuse) if profile.max_crop_reuse else None
        self.base_seed = base_seed if base_seed is not None else random.getrandbits(32)
        self.backgrounds_dir = backgrounds_dir
        self.bg_files = sorted(bg_files) if (profile.background_round_robin or profile.per_image_seed) else list(bg_files)
        self.rect_report: Optional[Dict[str, Any]] = None
        self.visibility_report: Optional[Dict[str, Any]] = None
        self.real: Optional[copy_paste.RealAnnotations] = None
        self.crop_sizes = crop_scale.CropSizes()
        self.matcher = background_match.Matcher(profile, backgrounds_dir) if profile.background_match else None
        if profile.copy_paste_real:
            self.real = copy_paste.RealAnnotations.load(copy_paste.annotation_path(profile, backgrounds_dir))

    def scale_filter(self, cls: str, quantile: float, bg_wh: Tuple[int, int]):
        """Function keeping the crops whose needed scale falls in `crop_scale_range` (None if not applicable)."""
        p = self.profile
        if p.crop_scale_range is None or not p.faithful_scale:
            return None
        target = crop_scale.target_px_for(self.stats, cls, quantile, bg_wh)
        if target is None:
            return None
        return lambda candidates: crop_scale.pick_candidates(candidates, self.crop_sizes, target, p.crop_scale_range)

    def seed_for(self, attempt: int) -> int:
        return derive_image_seed(self.base_seed, attempt)

    def make_rng(self, attempt: int):
        """Planning rng: local and reproducible if per_image_seed, otherwise the global one."""
        return random.Random(self.seed_for(attempt)) if self.profile.per_image_seed else random

    def pick_background(self, attempt: int, rng) -> str:
        """Background of the attempt: round-robin shuffled per epoch (uniform +-1) or random."""
        if not self.profile.background_round_robin:
            return rng.choice(self.bg_files)
        n = len(self.bg_files)
        order = list(range(n))
        random.Random(f"{self.base_seed}:{attempt // n}").shuffle(order)
        return self.bg_files[order[attempt % n]]

    def background_name(self, bg_file: str) -> str:
        import os
        return os.path.relpath(bg_file, self.backgrounds_dir).replace("\\", "/")

    def make_plan(self, attempt: int, bg_file: str, bg_wh: Tuple[int, int], rng) -> ImagePlan:
        p = self.profile
        plan = ImagePlan(profile=p, seed=self.seed_for(attempt) if p.per_image_seed else None,
                         background=self.background_name(bg_file), stats=self.stats)
        if self.real is not None:   # copy-paste: resolution and JPEG of the original
            plan.real_annotations = self.real.get(plan.background)
            plan.jpeg_quality = copy_paste.original_jpeg_quality(bg_file)
        elif p.output_resolution:
            plan.output_size = output_format.sample_output_size(rng, self.stats, bg_wh, p)
        if p.jpeg_recompress and plan.jpeg_quality is None:
            plan.jpeg_quality = output_format.sample_jpeg_quality(rng, self.stats, p)
        if p.haze_from_stats:
            plan.target_l_std = self.stats.l_std(rng.random())
        return plan

    def exhausted_report(self, counts: Dict[str, int], required: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
        """Classes that missed their target because the crops ran out (max_crop_reuse)."""
        if self.usage is None:
            return {}
        report = {}
        for cls, target in required.items():
            if counts.get(cls, 0) < target and not self.usage.has_free(cls):
                report[cls] = dict(self.usage.stats(cls), achieved=counts.get(cls, 0), target=target,
                                   max_crop_reuse=self.usage.max_reuse)
                logger.warning(f"Class {cls} exhausted by max_crop_reuse: {report[cls]}")
        return report

    def info(self) -> Dict[str, Any]:
        """`synth_generator` block of the COCO."""
        info = {"profile": self.profile.name, "config": self.profile.model_dump(),
                "batch_seed": self.base_seed, "real_stats_source": self.stats.source}
        if self.rect_report is not None:
            info["rect_filter"] = self.rect_report
        if self.visibility_report is not None:
            info["visibility_filter"] = self.visibility_report
        if self.matcher is not None:
            info["background_match"] = dict(self.matcher.stats)
        return info
