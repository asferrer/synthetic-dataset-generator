"""Unit tests for the gateway effects_config flattening.

The gateway must forward ONLY keys explicitly present in the request so
the augmentor schema defaults (tuned via parameter studies) stay
authoritative.
"""

import importlib.util
import os
import sys

import pytest

# Import the router module directly without booting the FastAPI app
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GATEWAY_ROOT = os.path.join(_REPO_ROOT, "services", "gateway")
_MODULE_PATH = os.path.join(_GATEWAY_ROOT, "app", "routers", "augment.py")


def _load_flatten():
    # Make the gateway's `app` package importable for augment.py's imports
    sys.path.insert(0, _GATEWAY_ROOT)
    try:
        spec = importlib.util.spec_from_file_location("gateway_augment", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as e:
            pytest.skip(
                f"gateway deps not installed locally: {e}", allow_module_level=True
            )
        return module
    finally:
        sys.path.remove(_GATEWAY_ROOT)
        # Drop any gateway 'app' modules so they don't shadow other services
        for name in [n for n in sys.modules if n == "app" or n.startswith("app.")]:
            del sys.modules[name]


module = _load_flatten()
_flatten = module._flatten_effects_config


class TestFlattenEffectsConfig:
    def test_none_returns_empty(self):
        assert _flatten(None) == {}

    def test_non_dict_returns_empty(self):
        assert _flatten("garbage") == {}

    def test_empty_dict_stays_empty_no_default_injection(self):
        # The old code injected stale defaults (0.12/0.5/0.15...) here
        assert _flatten({}) == {}

    def test_nested_copies_only_present_keys(self):
        ec = {"color_correction": {"enabled": True, "color_intensity": 0.25}}
        flat = _flatten(ec)
        assert flat == {"color_intensity": 0.25}

    def test_nested_full_frontend_shape(self):
        ec = {
            "color_correction": {"enabled": True, "color_intensity": 0.25},
            "blur_matching": {"enabled": True, "blur_strength": 0.65},
            "underwater": {
                "enabled": True,
                "underwater_intensity": 0.33,
                "water_color": [180, 120, 30],
                "water_clarity": "clear",
            },
            "caustics": {"enabled": True, "caustics_intensity": 0.12},
            "shadows": {"enabled": True, "shadow_opacity": 0.12, "shadow_blur": 28},
            "edge_smoothing": {"enabled": True, "edge_feather": 12},
        }
        flat = _flatten(ec)
        assert flat["color_intensity"] == 0.25
        assert flat["blur_strength"] == 0.65
        assert flat["underwater_intensity"] == 0.33
        assert flat["water_color"] == [180, 120, 30]
        assert flat["shadow_blur"] == 28
        assert flat["edge_feather"] == 12
        assert "lighting_type" not in flat  # not sent -> not forwarded

    def test_flat_passthrough(self):
        ec = {"color_intensity": 0.3, "edge_feather": 10}
        assert _flatten(ec) == ec

    def test_mixed_nested_overrides_flat(self):
        ec = {
            "color_intensity": 0.1,
            "color_correction": {"color_intensity": 0.25},
        }
        assert _flatten(ec)["color_intensity"] == 0.25

    def test_unknown_keys_dropped(self):
        ec = {"bogus_param": 1, "color_intensity": 0.2}
        flat = _flatten(ec)
        assert "bogus_param" not in flat
        assert flat["color_intensity"] == 0.2

    def test_physical_realism_flags_forwarded(self):
        ec = {
            "underwater": {"underwater_physical": True},
            "caustics": {"caustics_depth_aware": True},
            "depth_occlusion": {"depth_occlusion": True},
        }
        flat = _flatten(ec)
        assert flat["underwater_physical"] is True
        assert flat["caustics_depth_aware"] is True
        assert flat["depth_occlusion"] is True
