"""Commit point and recovery for batch generation.

The COCO file is the commit point of a batch: the composer rewrites it
atomically every FLUSH_EVERY accepted images, so a crash loses at most the
images accepted since the last flush. On resume the job state (images, class
counts, next free index) is rebuilt from that file, not from progress.json.

Files on disk that never reached the COCO (accepted after the last flush,
still unvalidated when the job died, or annotation files left by rejected
attempts) are moved to images_unflushed/, never deleted, so the dataset only
contains images that went through validation.
"""

import json
import os
import re
import shutil
from collections import Counter
from typing import Any, Dict, Optional

from loguru import logger

COCO_NAME = "synthetic_dataset.json"
FLUSH_EVERY = 50
UNFLUSHED_DIR = "images_unflushed"
ANN_SUFFIX = "_annotations.json"

_INDEX_RE = re.compile(r"^synthetic_(\d+)(?:_annotations)?\.\w+$")


def write_json_atomic(path: str, data: Any, indent: Optional[int] = 2) -> None:
    """Write JSON through a temporary file so a crash never leaves it truncated."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def image_index(file_name: str) -> Optional[int]:
    """Index of a synthetic_XXXXX file (image or annotation), or None."""
    match = _INDEX_RE.match(os.path.basename(file_name))
    return int(match.group(1)) if match else None


def _base_name(file_name: str) -> str:
    if file_name.endswith(ANN_SUFFIX):
        return file_name[: -len(ANN_SUFFIX)]
    return os.path.splitext(file_name)[0]


def load_coco(output_dir: str) -> Optional[dict]:
    """Committed COCO of a batch, or None if missing or unreadable."""
    coco_path = os.path.join(output_dir, COCO_NAME)
    if not os.path.exists(coco_path):
        return None
    try:
        with open(coco_path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Unreadable COCO at {coco_path}: {e}")
        return None


def class_counts(coco: Optional[dict]) -> Dict[str, int]:
    """Instances per class name in a COCO dict."""
    if not coco:
        return {}
    names = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    return dict(Counter(names.get(ann["category_id"], "unknown") for ann in coco.get("annotations", [])))


def _move_unflushed(images_dir: str, committed: set) -> int:
    """Move files whose image is not in the COCO to images_unflushed/."""
    target_dir = os.path.join(os.path.dirname(images_dir), UNFLUSHED_DIR)
    moved = 0
    for name in sorted(os.listdir(images_dir)):
        if image_index(name) is None or _base_name(name) in committed:
            continue
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(os.path.join(images_dir, name), os.path.join(target_dir, name))
        moved += 1
    return moved


def recover_batch_state(output_dir: str) -> Dict[str, Any]:
    """Rebuild the state of an interrupted batch from its committed COCO.

    Returns generated (committed images), counts (instances per class),
    next_image_index (first index never used on disk) and moved (files set
    aside in images_unflushed/).
    """
    coco = load_coco(output_dir)
    images_dir = os.path.join(output_dir, "images")
    committed = {_base_name(img["file_name"]) for img in (coco or {}).get("images", [])}

    used = [image_index(n) for n in os.listdir(images_dir)] if os.path.isdir(images_dir) else []
    unflushed_dir = os.path.join(output_dir, UNFLUSHED_DIR)
    if os.path.isdir(unflushed_dir):
        used += [image_index(n) for n in os.listdir(unflushed_dir)]
    used += [image_index(img["file_name"]) for img in (coco or {}).get("images", [])]
    used = [i for i in used if i is not None]

    moved = _move_unflushed(images_dir, committed) if os.path.isdir(images_dir) else 0
    if moved:
        logger.warning(f"Moved {moved} uncommitted file(s) to {unflushed_dir}")

    return {
        "generated": len(committed),
        "counts": class_counts(coco),
        "next_image_index": max(used) + 1 if used else 0,
        "moved": moved,
    }
