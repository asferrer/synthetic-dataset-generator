"""Unit tests for semantic buoyancy-aware placement."""

import numpy as np
import pytest

from app import placement


@pytest.fixture
def region_map():
    """100x100 map: top third surface(3), middle open_water(1), bottom seafloor(2)."""
    rm = np.zeros((100, 100), dtype=np.uint8)
    rm[:33, :] = 3
    rm[33:66, :] = 1
    rm[66:, :] = 2
    return rm


class TestBehaviors:
    def test_known_class_behaviors(self):
        assert placement.get_behavior("Tire") == "sink"
        assert placement.get_behavior("Plastic_Bag") == "float"
        assert placement.get_behavior("Fishing_Net") == "hover"

    def test_case_insensitive(self):
        assert placement.get_behavior("tire") == placement.get_behavior("TIRE")

    def test_unknown_class_returns_none(self):
        assert placement.get_behavior("Unicorn") is None

    def test_behavior_regions_cover_all_behaviors(self):
        for behavior in set(placement._DEFAULT_BEHAVIORS.values()):
            assert behavior in placement.BEHAVIOR_REGIONS


class TestSamplePosition:
    def test_sink_lands_on_floor_regions(self, region_map):
        for _ in range(20):
            pos = placement.sample_position(region_map, "Tire", (10, 10))
            assert pos is not None
            x, y = pos
            # Bottom-center anchor must be in seafloor (rows >= 66)
            anchor_y = min(y + 10, 99)
            assert anchor_y >= 60, f"sink object anchored too high: y={y}"

    def test_float_lands_in_upper_regions(self, region_map):
        for _ in range(20):
            pos = placement.sample_position(region_map, "Plastic_Bag", (10, 10))
            assert pos is not None
            x, y = pos
            center_y = y + 5
            # float regions are surface(3) rows 0-32 and open water(1) rows 33-65
            assert center_y < 70, f"float object centered too low: y={y}"

    def test_position_inside_bounds(self, region_map):
        for cls in ("Tire", "Plastic_Bag", "Fishing_Net"):
            for _ in range(10):
                pos = placement.sample_position(region_map, cls, (30, 30))
                assert pos is not None
                x, y = pos
                assert 0 <= x <= 70
                assert 0 <= y <= 70

    def test_unknown_class_returns_none(self, region_map):
        assert placement.sample_position(region_map, "Unicorn", (10, 10)) is None

    def test_no_compatible_region_returns_none(self):
        # Map with only seafloor: float classes have nowhere to go
        rm = np.full((50, 50), 2, dtype=np.uint8)
        assert placement.sample_position(rm, "Plastic_Bag", (10, 10)) is None
        assert placement.sample_position(rm, "Tire", (10, 10)) is not None

    def test_object_larger_than_map_returns_none(self, region_map):
        assert placement.sample_position(region_map, "Tire", (200, 200)) is None

    def test_zero_size_returns_none(self, region_map):
        assert placement.sample_position(region_map, "Tire", (0, 10)) is None


class TestRegionMapAt:
    def test_none_passthrough(self):
        assert placement.region_map_at(None, (10, 10)) is None

    def test_resize_preserves_labels(self, region_map):
        resized = placement.region_map_at(region_map, (200, 200))
        assert resized.shape == (200, 200)
        # Nearest-neighbour: no new label values introduced
        assert set(np.unique(resized)) <= set(np.unique(region_map))

    def test_same_size_is_identity(self, region_map):
        out = placement.region_map_at(region_map, (100, 100))
        assert out is region_map
