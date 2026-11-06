"""Tests of per-class seabed_fraction and of the additional crop libraries (profile v10)."""

import random

import pytest

from app import placement
from app.object_planner import plan_objects, seabed_override
from app.synth_context import merge_extra_libraries
from app.synth_profile import V10_SEABED, SynthProfile, get_profile
from tests.synth_helpers import make_context, make_objects, seeded_rng

ALL = ["Bottle", "Can", "Fishing_Net", "Glove", "Mask", "Metal_Debris", "Plastic_Debris", "Tire"]


def freq(profile, cls, n=4000):
    rng = random.Random(3)
    out = [seabed_override(profile, cls, rng) for _ in range(n)]
    return {k: out.count(k) / n for k in set(out)}


def test_v10_defaults_cover_the_eight_classes():
    p = get_profile("v10")
    assert p.seabed_fraction == V10_SEABED and set(p.seabed_fraction) == set(ALL)
    assert p.seabed_fraction == {
        "Bottle": 0.5, "Can": 1.0, "Metal_Debris": 1.0, "Tire": 1.0, "Fishing_Net": 0.5,
        "Plastic_Debris": 0.35, "Glove": 0.0, "Mask": 0.0}


def test_per_class_probabilities_are_respected():
    p = get_profile("v10")
    assert placement.get_behavior("Bottle") == "sink"
    b = freq(p, "Bottle")                       # 50 % seabed (natural sink, None) / 50 % water column
    assert abs(b.get("float", 0) - 0.5) < 0.04 and abs(b.get(None, 0) - 0.5) < 0.04
    pl = freq(p, "Plastic_Debris")              # 35 % forced sink, the rest keep their behaviour
    assert abs(pl.get("sink", 0) - 0.35) < 0.04 and set(pl) <= {"sink", None}
    assert abs(freq(p, "Fishing_Net").get("sink", 0) - 0.5) < 0.04


def test_certain_values_do_not_consume_rng():
    p = get_profile("v10")
    rng = seeded_rng(5)
    state = rng.getstate()
    assert [seabed_override(p, c, rng) for c in ("Can", "Tire", "Metal_Debris", "Glove", "Mask")] == [None] * 5
    assert rng.getstate() == state


def test_zero_means_class_behavior_never_forced():
    p = get_profile("v10")
    assert freq(p, "Glove") == {None: 1.0} and freq(p, "Mask") == {None: 1.0}   # natural float, untouched
    p0 = SynthProfile(name="t", seabed_fraction={"Tire": 0.0})
    assert freq(p0, "Tire") == {"float": 1.0}   # a 'sink' class with 0.0 is never on the seabed


def test_class_missing_from_dict_is_unchanged_and_water_behavior_configurable():
    p = SynthProfile(name="t", seabed_fraction={"Bottle": 0.0}, water_column_behavior="hover")
    assert freq(p, "Glove") == {None: 1.0}
    assert freq(p, "Bottle") == {"hover": 1.0}


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        SynthProfile(seabed_fraction={"Bottle": 1.5})
    with pytest.raises(ValueError):
        SynthProfile(seabed_fraction=-0.1)


def test_legacy_float_matches_original_algorithm():
    """With a float the planner consumes the same rng and decides like the legacy code."""
    for frac in (0.0, 0.3, 1.0):
        p = SynthProfile(name="t", seabed_fraction=frac)
        for cls in ("Glove", "Tire", "Fishing_Net", "Bottle"):
            a, b = random.Random(9), random.Random(9)
            for _ in range(200):
                old = None
                if frac > 0 and a.random() < frac and placement.get_behavior(cls) not in (None, "sink"):
                    old = "sink"
                assert seabed_override(p, cls, b) == old
            assert a.getstate() == b.getstate()


def test_plan_objects_applies_dict(tmp_path):
    objs = make_objects(["Tire", "Glove"])
    ctx = make_context(tmp_path, seabed_fraction={"Tire": 1.0, "Glove": 0.0}, objects=objs)
    planned, _, _ = plan_objects(ctx, seeded_rng(0), ["Tire", "Glove"], {"Tire": 9, "Glove": 9}, objs, 2)
    assert planned and all(o.behavior_override is None for o in planned)


def test_legacy_profile_has_no_extra_libraries_or_dict():
    p = SynthProfile()
    assert p.seabed_fraction == 0.0 and p.extra_objects_dirs == [] and p.active is False
    assert get_profile("v10").extra_objects_dirs == []


def make_lib(root, layout):
    for cls, names in layout.items():
        (root / cls).mkdir(parents=True)
        for n in names:
            (root / cls / n).write_bytes(b"x")


def test_extra_libraries_are_merged_without_copying(tmp_path):
    main, poly = tmp_path / "main_lib", tmp_path / "crops_poly"
    make_lib(poly, {"Can": ["poly_1.png", "poly_1.json"], "Tire": ["poly_2.png"]})
    objs = {"Can": [{"path": str(main / "Can" / "5.png")}]}
    merge_extra_libraries(objs, SynthProfile(name="t", extra_objects_dirs=[str(poly)]))
    assert [o["path"] for o in objs["Can"]] == [str(main / "Can" / "5.png"), str(poly / "Can" / "poly_1.png")]
    assert [o["path"] for o in objs["Tire"]] == [str(poly / "Tire" / "poly_2.png")]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["crops_poly"]   # nothing copied or created


def test_no_extra_dirs_is_a_noop_and_missing_root_fails(tmp_path):
    objs = make_objects(["Can"])
    before = {c: list(v) for c, v in objs.items()}
    merge_extra_libraries(objs, None)
    merge_extra_libraries(objs, SynthProfile(name="t"))
    assert objs == before
    with pytest.raises(FileNotFoundError):
        merge_extra_libraries(objs, SynthProfile(name="t", extra_objects_dirs=[str(tmp_path / "no")]))


def test_merged_crops_enter_the_usage_cap(tmp_path):
    poly = tmp_path / "crops_poly"
    make_lib(poly, {"Can": ["poly_1.png", "poly_2.png"]})
    objs = make_objects(["Can"], crops=1)
    merge_extra_libraries(objs, SynthProfile(name="t", extra_objects_dirs=[str(poly)]))
    ctx = make_context(tmp_path, objects=objs, max_crop_reuse=2)
    rng = seeded_rng(0)
    got = [ctx.usage.acquire("Can", rng) for _ in range(6)]
    assert None not in got and ctx.usage.acquire("Can", rng) is None   # 3 recortes x 2 usos
