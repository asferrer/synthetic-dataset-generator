"""Uncompressed COCO RLE (`{"size": [h, w], "counts": [...]}`, column-major order) to store object masks."""
from typing import Any, Dict

import numpy as np


def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    """2D mask (any value > 0 is object) -> RLE; `counts` starts with the run of zeros."""
    binary = np.asarray(mask) > 0
    h, w = binary.shape
    flat = binary.reshape(-1, order="F").astype(np.int8)
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0] == 1:
        counts = [0] + counts
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """Inverse of `encode_rle`: uint8 {0, 1} mask of shape (h, w)."""
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in rle["counts"]:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape((h, w), order="F")
