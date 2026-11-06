"""Planning of the objects of an image under an active profile.

Decides how many objects, of which class, which crop (respecting `max_crop_reuse`), with which augmentations and which
real-size quantile. Everything is sampled with the image's rng, so it is reproducible with `per_image_seed`.

Object count and classes (`faithful_count`): the count comes from the real training histogram conditioned on
<= `max_objects_cap`. Classes are chosen WITH replacement, weighted by what each class still needs for its target and
without exceeding it in the same image; so an image can hold several objects of the same class (as in real data). If
they do not fit without overlapping, the compositor discards the object after `try_count` attempts: the placed count
may be lower than the requested one, and the request is recorded (`objects_requested`). Objects are placed from
largest to smallest so small ones fill gaps.
"""

from typing import Dict, List, Optional, Tuple

from app import placement
from app.crop_augment import sample_augment
from app.models.schemas import ObjectPlacement
from app.synth_context import SynthContext


def _open_classes(ctx: SynthContext, still_needed: List[str]) -> List[str]:
    if ctx.usage is None:
        return list(still_needed)
    return [c for c in still_needed if ctx.usage.has_free(c)]


def choose_classes(
    ctx: SynthContext, rng, still_needed: List[str], remaining: Dict[str, int],
    max_objects: int,
) -> Tuple[List[str], int]:
    """(chosen classes, requested number of objects)."""
    open_classes = _open_classes(ctx, still_needed)
    if not open_classes:
        return [], 0
    p = ctx.profile
    if not p.faithful_count:
        k = rng.randint(1, min(max_objects, len(open_classes)))
        return rng.sample(open_classes, k=k), k
    requested = ctx.stats.sample_count(rng, p.max_objects_cap)
    left = {c: max(1, remaining.get(c, 1)) for c in open_classes}
    chosen: List[str] = []
    for _ in range(requested):
        pool = [c for c in open_classes if left[c] > 0]
        if not pool:
            break
        cls = rng.choices(pool, weights=[left[c] for c in pool], k=1)[0]
        left[cls] -= 1
        chosen.append(cls)
    return chosen, requested


def seabed_override(p, cls: str, rng) -> Optional[str]:
    """Behaviour forced by `seabed_fraction` (None = the class's own).

    float (legacy behaviour, no synthesis profile): with probability f, a non-'sink' object becomes 'sink'.
    dict (v10): p = probability of being on the seabed. With prob. p -> 'sink' (unchanged if the class is already
    'sink'); with 1-p -> the class's natural behaviour, except 'sink' classes, which go to `water_column_behavior`.
    Class absent from the dict: unchanged. The rng is only consumed when 0 < p < 1.
    """
    frac = p.seabed_fraction
    natural = placement.get_behavior(cls)
    if isinstance(frac, dict):
        if cls not in frac or natural is None:
            return None
        prob = frac[cls]
        on_seabed = prob >= 1.0 or (prob > 0.0 and rng.random() < prob)
        if on_seabed:
            return None if natural == "sink" else "sink"
        return p.water_column_behavior if natural == "sink" else None
    if frac > 0 and rng.random() < frac and natural not in (None, "sink"):
        return "sink"
    return None


def _build_object(ctx: SynthContext, rng, cls: str, path: str, quantile: Optional[float] = None) -> ObjectPlacement:
    p = ctx.profile
    obj = ObjectPlacement(image_path=path, class_name=cls)
    if p.faithful_scale and ctx.stats.has_class(cls):
        obj.size_quantile = rng.random() if quantile is None else quantile
    if p.paste_augment:
        obj.augment = sample_augment(rng, p)
        obj.rotation = obj.augment["rotation"]
    obj.behavior_override = seabed_override(p, cls, rng)
    return obj


def _size_hint(ctx: SynthContext, obj: ObjectPlacement) -> float:
    if obj.size_quantile is None:
        return 0.0
    return ctx.stats.quantile(obj.class_name, obj.size_quantile)


def _pick_crop(ctx: SynthContext, rng, cls: str, objects_by_class, bg_wh,
               bg_file: Optional[str] = None) -> Tuple[Optional[str], Optional[float], bool]:
    """(crop path, already sampled size quantile or None, reserved). With `crop_scale_range` and the background
    size, the quantile is sampled BEFORE so the crop is chosen by the scale it needs (docs/SYNTHESIS_PROFILE.md)."""
    quantile, narrow = None, None
    if bg_wh is not None and ctx.profile.crop_scale_range is not None and ctx.stats.has_class(cls):
        quantile = rng.random()
        narrow = ctx.scale_filter(cls, quantile, bg_wh)
    if ctx.matcher is not None and bg_file is not None:   # crop chosen by colour similarity to the background (v11, q)
        narrow = ctx.matcher.chain(narrow, bg_file, rng)
    if ctx.usage is not None:
        return ctx.usage.acquire(cls, rng, narrow), quantile, True
    paths = [o["path"] for o in objects_by_class[cls]]
    return rng.choice(narrow(paths) if narrow else paths), quantile, False


def plan_objects(
    ctx: SynthContext, rng, still_needed: List[str], remaining: Dict[str, int],
    objects_by_class: Dict[str, List[dict]], max_objects: int, bg_wh: Optional[Tuple[int, int]] = None,
    bg_file: Optional[str] = None,
) -> Tuple[List[ObjectPlacement], List[str], int]:
    """(objects, reserved crop paths, requested count). Synchronous: reserves without awaits.

    `bg_wh` (background width, height): used by `crop_scale_range`. `bg_file`: used by `background_match`.
    """
    classes, requested = choose_classes(ctx, rng, still_needed, remaining, max_objects)
    objects: List[ObjectPlacement] = []
    reserved: List[str] = []
    for cls in classes:
        path, quantile, held = _pick_crop(ctx, rng, cls, objects_by_class, bg_wh, bg_file)
        if path is None:
            continue
        if held:
            reserved.append(path)
        objects.append(_build_object(ctx, rng, cls, path, quantile))
    if ctx.profile.faithful_scale and len(objects) > 1:
        objects.sort(key=lambda o: _size_hint(ctx, o), reverse=True)
    return objects, reserved, requested
