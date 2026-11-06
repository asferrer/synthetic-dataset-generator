"""Helpers shared by the synthesis-profile tests (v10)."""

import itertools
import json
import random
from types import SimpleNamespace

import numpy as np

from app.synth_context import SynthContext
from app.synth_profile import SynthProfile


_COUNTER = itertools.count()


def make_stats_file(tmp_path, counts=None, jpeg=None, l_std=None):
    """JSON de estadisticas minimo: Bottle 0.1..0.9, Can 0.02..0.1."""
    lin = lambda a, b: [round(a + (b - a) * i / 100, 6) for i in range(101)]
    data = {
        "source": "test_train.json",
        "classes": {
            "Bottle": {"n": 100, "sqrt_area_rel_quantiles": lin(0.1, 0.9)},
            "Can": {"n": 100, "sqrt_area_rel_quantiles": lin(0.02, 0.1)},
        },
        "objects_per_image": counts or {"1": 70, "2": 20, "30": 10},
        "image_sizes": {"416x416": 90, "1920x1080": 10},
    }
    if jpeg or l_std:
        data["image_stats"] = {
            "jpeg_quality_quantiles": jpeg or [],
            "l_std_quantiles": l_std or [],
        }
    path = tmp_path / f"stats_{next(_COUNTER)}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def make_objects(classes=("Bottle", "Can"), crops=4):
    return {c: [{"path": f"/objs/{c}/crop_{i}.png"} for i in range(crops)] for c in classes}


def make_context(tmp_path, objects=None, base_seed=7, bgs=None, **profile_kwargs):
    if "stats_path" not in profile_kwargs:
        profile_kwargs["stats_path"] = make_stats_file(tmp_path)
    profile = SynthProfile(name="t", **profile_kwargs)
    objects = objects or make_objects()
    bgs = bgs or [f"/bgs/bg_{i}.jpg" for i in range(5)]
    return SynthContext(profile, objects, "/bgs", bgs, base_seed=base_seed)


def rgba_crop(w=60, h=40):
    """RGBA crop with a coloured opaque ellipse."""
    img = np.zeros((h, w, 4), dtype=np.uint8)
    import cv2
    cv2.ellipse(img, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, (30, 120, 200, 255), -1)
    return img


def fake_annotation(class_name, source_object=None):
    return SimpleNamespace(class_name=class_name, x=0, y=0, width=4, height=4, area=16,
                           source_object=source_object)


def seeded_rng(seed=1):
    return random.Random(seed)
