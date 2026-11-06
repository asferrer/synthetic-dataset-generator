"""Summarise the rows of measure_bbox_inflation.py: extra margin and IoU of the annotated box vs the mask box.

Usage: python -X utf8 tools/summarize_bbox_inflation.py bbox_check/legacy_objects.json [...]
COCO size class by the tight box area (the mask is not available: the box is an upper bound of its area):
small < 32^2, medium < 96^2, large otherwise.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_bbox_inflation import iou  # noqa: E402  (imports numpy and run_variants; launches nothing)


def size_class(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def stats(rows, key):
    """Extra margin (px and % of the side) of rows[key] vs the tight box, plus IoU."""
    extra_px, extra_pct, ious = [], [], []
    for r in rows:
        t, b = r["tight"], r[key]
        w, h = t[2] - t[0], t[3] - t[1]
        for e, side in ((t[0] - b[0], w), (t[1] - b[1], h), (b[2] - t[2], w), (b[3] - t[3], h)):
            extra_px.append(e)
            extra_pct.append(100.0 * e / side)
        ious.append(iou(t, b))
    ious = np.array(ious)
    q = lambda v, p: float(np.percentile(v, p))
    return {"n": len(rows), "px_mean": float(np.mean(extra_px)), "px_p50": q(extra_px, 50), "px_p90": q(extra_px, 90),
            "px_max": float(np.max(extra_px)), "pct_mean": float(np.mean(extra_pct)), "pct_p50": q(extra_pct, 50),
            "pct_p90": q(extra_pct, 90), "iou_mean": float(ious.mean()), "iou_lt09": float((ious < 0.9).mean()),
            "iou_lt08": float((ious < 0.8).mean())}


def fmt(s):
    return (f"n={s['n']:5d} extra/side px mean {s['px_mean']:5.1f} p50 {s['px_p50']:4.1f} p90 {s['px_p90']:4.1f} "
            f"max {s['px_max']:4.0f} | % of side mean {s['pct_mean']:5.1f} p50 {s['pct_p50']:5.1f} p90 {s['pct_p90']:5.1f} | "
            f"IoU mean {s['iou_mean']:.3f} <0.9 {100 * s['iou_lt09']:5.1f}% <0.8 {100 * s['iou_lt08']:5.1f}%")


def main(paths):
    out = {}
    for p in paths:
        rows = [r for r in json.load(open(p)) if r["tight"]]
        print(f"\n== {Path(p).stem}")
        res = {}
        for key in ("shadow", "final"):
            res[key] = stats(rows, key)
            print(f"[{key:6s}] {fmt(res[key])}")
        for group, fn in (("class", lambda r: r["class"]), ("size", lambda r: size_class(r["tight"]))):
            buckets = defaultdict(list)
            for r in rows:
                buckets[fn(r)].append(r)
            res[group] = {k: stats(v, "final") for k, v in sorted(buckets.items())}
            for k, s in res[group].items():
                print(f"  {group:6s} {k:15s} {fmt(s)}")
        out[Path(p).stem] = res
    json.dump(out, open(Path(paths[0]).parent / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
