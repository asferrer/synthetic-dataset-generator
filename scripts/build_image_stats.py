"""Image statistics of the REAL training data (JPEG quality and contrast).

Samples n real training images and measures, per image:
  - the JPEG quality estimated from the luminance quantisation table
    (libjpeg scale over the standard table; JPEG files only),
  - the standard deviation of L (LAB) of the whole image (global contrast),
and adds the key `image_stats` with their quantiles to the real_stats JSON.
With --compare-dir it measures the same on a synthetic batch and prints it for
comparison (it is not saved in the JSON).

It only reads images and refuses to work with a COCO that is not the training one. Meant to run
at low priority (IDLE is set if psutil is available).

Usage:
  python -X utf8 scripts/build_image_stats.py --train gen_inputs/real_train/annotations.json \
      --data-root gen_inputs/real_train \
      --stats services/augmentor/app/data/real_stats_train.json --n 400 \
      [--compare-dir output/synthetic_batch/images]
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional

import cv2
from PIL import Image

from build_real_stats import check_is_train, quantiles

# Standard luminance quantisation table (quality 50), natural order.
_STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
             14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
             18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
             49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def estimate_jpeg_quality(path: str) -> Optional[float]:
    """JPEG quality (1-100) from the quantisation table, or None if the file is not a JPEG."""
    try:
        with Image.open(path) as im:
            if im.format != "JPEG" or not getattr(im, "quantization", None):
                return None
            table = im.quantization[0]
    except OSError:
        return None
    scale = 100.0 * sum(table) / sum(_STD_LUMA)
    quality = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return float(min(100.0, max(1.0, quality)))


def l_std(path: str) -> Optional[float]:
    """Standard deviation of L (0-255) of the image resized to 224 px."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].std())


def measure(paths: List[str]) -> Dict[str, List[float]]:
    qual, std = [], []
    for p in paths:
        q, s = estimate_jpeg_quality(p), l_std(p)
        if q is not None:
            qual.append(q)
        if s is not None:
            std.append(s)
    return {"jpeg_quality": qual, "l_std": std}


def summarize(m: Dict[str, List[float]]) -> dict:
    return {
        "n": len(m["l_std"]),
        "n_jpeg": len(m["jpeg_quality"]),
        "l_std_quantiles": quantiles(m["l_std"]) if m["l_std"] else [],
        "jpeg_quality_quantiles": quantiles(m["jpeg_quality"]) if m["jpeg_quality"] else [],
    }


def real_paths(train: str, data_root: str, n: int, seed: int) -> List[str]:
    with open(train, "r", encoding="utf-8") as f:
        names = [im["file_name"] for im in json.load(f)["images"]]
    paths = [os.path.join(data_root, name) for name in names]
    paths = [p for p in paths if os.path.exists(p)]
    random.Random(seed).shuffle(paths)
    return paths[:n]


def lower_priority() -> None:
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--data-root", required=True, help="directory the COCO file_name entries are relative to")
    ap.add_argument("--stats", required=True, help="real_stats JSON to update")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare-dir", default=None)
    args = ap.parse_args()
    check_is_train(args.train)
    lower_priority()
    cv2.setNumThreads(1)
    real = summarize(measure(real_paths(args.train, args.data_root, args.n, args.seed)))
    with open(args.stats, "r", encoding="utf-8") as f:
        stats = json.load(f)
    stats["image_stats"] = dict(real, sample_seed=args.seed)
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print("REAL", json.dumps({k: (v[::10] if isinstance(v, list) else v) for k, v in real.items()}))
    if args.compare_dir:
        files = sorted(os.path.join(args.compare_dir, f) for f in os.listdir(args.compare_dir)
                       if f.lower().endswith(".jpg"))
        random.Random(args.seed).shuffle(files)
        synth = summarize(measure(files[:args.n]))
        print("SYNTH", json.dumps({k: (v[::10] if isinstance(v, list) else v) for k, v in synth.items()}))


if __name__ == "__main__":
    main()
