"""Crop-to-background pairing by the colour of the crop's SOURCE image (profile v11, variants q; see docs/SYNTHESIS_PROFILE.md).

Most of the real-vs-synthetic gap per object appears when the crop's source image changes, and it sits mainly in the
ring around the box. Each crop carries the colour and light of its own source image (and a halo of its background).
Choosing a destination whose global colour resembles the source lowers the discriminator AUC. Two mechanisms, both off
by default:

  * `background_match = "lab_knn"`: the flow picks the background first, so the CROPS are chosen: among the free
    candidates (usage cap and scale already applied) the `k` whose source image has the closest global LAB mean and
    standard deviation to the background's (image at 64 px, L1 over the 6 values) are kept, and one is sampled with
    weights exp(-(d - d1) / (T * (dk - d1))) (T -> 0: the closest; large T: uniform among the k).
    The background's own image is never used; images of the same group are only excluded with
    `background_match_exclude_same_group`.
  * `local_patch_match`: among the non-colliding candidate positions of `position_from_stats`, the one whose ring
    around the box has the LAB statistics (mean and std) closest to the crop's ring in its source image.

The descriptors come from `scripts/build_background_match.py` (training split only) as a small NPZ. A crop or
background without a descriptor breaks nothing: a crop with unknown source is not paired (if none is left, the normal
choice is used and it is counted in `stats`).
"""

import math
import os
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_MATCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "background_match_train.npz")
DESC_SIDE = 64          # side of the downscaled image used for the global descriptor
HIST_BINS = 4           # bins per LAB channel of the coarse histogram (4**3 = 64)
RING_SIDE = 512         # longest side the image is downscaled to for ring measurements
RING_MIN_PX = 16        # minimum pixel count of a measurable ring
HIST_SCALE = 100.0      # weight of the histogram L1 (0..2) relative to the LAB mean/std L1


# ------------------------------------------------------------------ descriptors
def _lab_small(bgr: np.ndarray, side: int = DESC_SIDE) -> np.ndarray:
    """uint8 LAB of the image resized to side x side (OpenCV LAB convention on uint8)."""
    return cv2.cvtColor(cv2.resize(bgr, (side, side), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)


def lab_descriptor(bgr: np.ndarray) -> np.ndarray:
    """(6,) mean and std of L, a, b of the whole image (downscaled to 64 px)."""
    lab = _lab_small(bgr).astype(np.float32)
    return np.r_[lab.mean(axis=(0, 1)), lab.std(axis=(0, 1))].astype(np.float32)


def lab_histogram(bgr: np.ndarray) -> np.ndarray:
    """(64,) normalised coarse LAB histogram (sums to 1)."""
    lab = _lab_small(bgr).astype(np.int64) * HIST_BINS // 256
    idx = (lab[..., 0] * HIST_BINS + lab[..., 1]) * HIST_BINS + lab[..., 2]
    h = np.bincount(idx.ravel(), minlength=HIST_BINS ** 3).astype(np.float32)
    return h / max(float(h.sum()), 1.0)


class RingMeter:
    """LAB mean and std of the ring around a box, using integral images (O(1) per box)."""

    def __init__(self, bgr: np.ndarray, side: int = RING_SIDE):
        h, w = bgr.shape[:2]
        self.scale = min(1.0, side / max(h, w))
        small = bgr if self.scale == 1.0 else cv2.resize(
            bgr, (max(1, round(w * self.scale)), max(1, round(h * self.scale))), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float64)
        self.h, self.w = lab.shape[:2]
        self._s = cv2.integral(lab)
        self._q = cv2.integral(lab * lab)

    def _sum(self, table: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        return table[y2, x2] - table[y1, x2] - table[y2, x1] + table[y1, x1]

    def stats(self, box: Tuple[float, float, float, float], ring_frac: float) -> Optional[np.ndarray]:
        """(6,) LAB mean and std of the box ring (x1, y1, x2, y2 in original-image pixels).

        Ring = box grown by `ring_frac` x its longest side on each side, minus the box, clipped to the image.
        None if fewer than RING_MIN_PX pixels remain."""
        s = self.scale
        x1, y1, x2, y2 = (box[0] * s, box[1] * s, box[2] * s, box[3] * s)
        r = max(2.0, ring_frac * max(x2 - x1, y2 - y1))
        ox1, oy1 = max(0, int(math.floor(x1 - r))), max(0, int(math.floor(y1 - r)))
        ox2, oy2 = min(self.w, int(math.ceil(x2 + r))), min(self.h, int(math.ceil(y2 + r)))
        ix1, iy1 = min(max(int(round(x1)), ox1), ox2), min(max(int(round(y1)), oy1), oy2)
        ix2, iy2 = min(max(int(round(x2)), ix1), ox2), min(max(int(round(y2)), iy1), oy2)
        n = (ox2 - ox1) * (oy2 - oy1) - (ix2 - ix1) * (iy2 - iy1)
        if n < RING_MIN_PX:
            return None
        sums = self._sum(self._s, ox1, oy1, ox2, oy2) - self._sum(self._s, ix1, iy1, ix2, iy2)
        sqs = self._sum(self._q, ox1, oy1, ox2, oy2) - self._sum(self._q, ix1, iy1, ix2, iy2)
        mean = sums / n
        std = np.sqrt(np.maximum(sqs / n - mean * mean, 0.0))
        return np.r_[mean, std].astype(np.float32)


def crop_key(path: str) -> str:
    """Table key of a crop: `<Class>/<name>` (the `poly_` prefix tells the libraries apart)."""
    p = path.replace("\\", "/").rstrip("/")
    parts = p.split("/")
    return "/".join(parts[-2:])


# ------------------------------------------------------------------ table
class MatchTable:
    """Descriptors of the training images and the source (image, ring) of each crop."""

    def __init__(self, z):
        self.files: List[str] = [str(f) for f in z["files"]]
        self.index: Dict[str, int] = {f: i for i, f in enumerate(self.files)}
        self.group = np.asarray(z["group"], dtype=np.int64)
        self.lab = np.asarray(z["lab"], dtype=np.float32)
        self.hist = np.asarray(z["hist"], dtype=np.float32) / 255.0
        self.crop_keys: Dict[str, int] = {str(k): i for i, k in enumerate(z["crop_keys"])}
        self.crop_img = np.asarray(z["crop_img"], dtype=np.int64)
        self.crop_ring = np.asarray(z["crop_ring"], dtype=np.float32)
        self.ring_frac = float(z["ring_frac"])

    def source_image(self, path: str) -> int:
        """Index of the crop's source image, or -1 if unknown."""
        i = self.crop_keys.get(crop_key(path))
        return -1 if i is None else int(self.crop_img[i])

    def ring_of(self, path: str) -> Optional[np.ndarray]:
        i = self.crop_keys.get(crop_key(path))
        if i is None or not np.all(np.isfinite(self.crop_ring[i])):
            return None
        return self.crop_ring[i]


@lru_cache(maxsize=2)
def load_match_table(path: Optional[str] = None) -> MatchTable:
    p = path or DEFAULT_MATCH_PATH
    if not os.path.isfile(p):
        raise FileNotFoundError(f"background_match: descriptor table not found: {p} "
                                "(scripts/build_background_match.py)")
    with np.load(p, allow_pickle=False) as z:
        return MatchTable(z)


def descriptor_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """L1 between a (6,) descriptor and an (n, 6) matrix (or between two (6,))."""
    return np.abs(np.asarray(b) - np.asarray(a)).sum(axis=-1)


def ring_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(descriptor_distance(a, b))


# ------------------------------------------------------------------ crop selection
def knn_weights(dist: np.ndarray, temperature: float) -> np.ndarray:
    """Sampling weights among the k neighbours: exp(-(d - d1) / (T * (dk - d1))); T <= 0 keeps only the closest."""
    d = np.asarray(dist, dtype=np.float64)
    if temperature <= 0:
        w = np.zeros_like(d)
        w[int(np.argmin(d))] = 1.0
        return w
    spread = float(d.max() - d.min())
    if spread <= 1e-9:
        return np.ones_like(d)
    return np.exp(-(d - d.min()) / (temperature * spread))


class Matcher:
    """Chooses crops by similarity of their source image to the background (`background_match = "lab_knn"`)."""

    def __init__(self, profile, backgrounds_dir: str = ""):
        self.profile = profile
        self.table = load_match_table(profile.background_match_path)
        self.backgrounds_dir = backgrounds_dir
        self._bg: Dict[str, tuple] = {}
        self.stats = {"matched": 0, "unmatched_no_source": 0, "unmatched_no_eligible": 0}

    def background(self, bg_file: str) -> Optional[tuple]:
        """(table index or -1, (6,) descriptor, (64,) histogram) of the background; measured if not in the table."""
        if bg_file in self._bg:
            return self._bg[bg_file]
        name = os.path.relpath(bg_file, self.backgrounds_dir).replace("\\", "/") if self.backgrounds_dir else ""
        idx = self.table.index.get(name, -1)
        if idx >= 0:
            out = (idx, self.table.lab[idx], self.table.hist[idx])
        else:
            img = cv2.imread(bg_file)
            out = None if img is None else (-1, lab_descriptor(img), lab_histogram(img))
        self._bg[bg_file] = out
        return out

    def eligible(self, candidates: Sequence[str], bg: tuple) -> Tuple[List[str], np.ndarray]:
        """(candidates with known source that satisfy the exclusions, indices of their source image)."""
        p, t = self.profile, self.table
        bg_idx = bg[0]
        out, src = [], []
        for c in candidates:
            s = t.source_image(c)
            if s < 0:
                continue
            if bg_idx >= 0 and not p.background_match_allow_same_image and s == bg_idx:
                continue
            if bg_idx >= 0 and p.background_match_exclude_same_group and t.group[s] >= 0 and t.group[s] == t.group[bg_idx]:
                continue
            out.append(c)
            src.append(s)
        return out, np.asarray(src, dtype=np.int64)

    def distances(self, bg: tuple, src: np.ndarray) -> np.ndarray:
        d = descriptor_distance(bg[1], self.table.lab[src])
        w = self.profile.background_match_hist_weight
        if w > 0:
            d = d + w * HIST_SCALE * np.abs(self.table.hist[src] - bg[2]).sum(axis=-1)
        return d

    def pick(self, candidates: List[str], bg_file: str, rng) -> List[str]:
        """One crop (as a list) among the `k` most similar; the unpaired candidates if nothing is eligible."""
        bg = self.background(bg_file)
        if bg is None:
            return candidates
        known = [c for c in candidates if self.table.source_image(c) >= 0]
        if not known:
            self.stats["unmatched_no_source"] += 1
            return candidates
        pool, src = self.eligible(known, bg)
        if not pool:
            self.stats["unmatched_no_eligible"] += 1
            return self._without_same_image(candidates, bg)
        d = self.distances(bg, src)
        k = min(self.profile.background_match_k, len(pool))
        order = np.argsort(d, kind="stable")[:k]
        w = knn_weights(d[order], self.profile.background_match_temperature)
        self.stats["matched"] += 1
        return [pool[int(order[rng.choices(range(k), weights=w.tolist(), k=1)[0]])]]

    def _without_same_image(self, candidates: List[str], bg: tuple) -> List[str]:
        """Unpaired, but without the background's own image (if anything is left)."""
        if bg[0] < 0:
            return candidates
        rest = [c for c in candidates if self.table.source_image(c) != bg[0]]
        return rest or candidates

    def chain(self, narrow: Optional[Callable], bg_file: str, rng) -> Callable:
        """A previous `narrow` (e.g. scale) followed by the similarity choice."""
        return lambda cands: self.pick(narrow(cands) if narrow else cands, bg_file, rng)


# ------------------------------------------------------------------ local ring
def patch_scorer(profile, crop_path: str, composite: np.ndarray) -> Optional[Callable]:
    """f(box x1,y1,x2,y2) -> LAB distance between that box's ring in `composite` and the crop's ring in its source.

    None if inactive or the crop has no known source ring (the position is chosen as usual)."""
    if not getattr(profile, "local_patch_match", False):
        return None
    table = load_match_table(profile.background_match_path)
    target = table.ring_of(crop_path)
    if target is None:
        return None
    meter = RingMeter(composite)

    def score(box) -> float:
        r = meter.stats(box, table.ring_frac)
        return float("inf") if r is None else ring_distance(target, r)
    return score
