"""Scene understanding: depth map, lighting estimation and heuristic scene analysis."""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import httpx
from loguru import logger

from app.models.schemas import LightingInfo, LightSourceInfo, LightType, DepthZone, WaterClarity


class SceneMixin:
    """Scene understanding: depth map, lighting estimation and heuristic scene analysis."""

    async def estimate_lighting(
        self,
        image_path: str,
        max_light_sources: int = 3,
        intensity_threshold: float = 0.6,
        estimate_hdr: bool = False,
        apply_water_attenuation: bool = False,
        depth_category: DepthZone = DepthZone.MID,
        water_clarity: WaterClarity = WaterClarity.CLEAR,
    ) -> LightingInfo:
        """Estimate lighting conditions from an image."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        return await self._estimate_lighting(
            image,
            max_light_sources,
            intensity_threshold,
            apply_water_attenuation,
            depth_category,
            water_clarity,
        )

    async def _get_depth_map(
        self, image: np.ndarray, cache_key: Optional[str] = None
    ) -> Tuple[np.ndarray, str]:
        """Get depth map and its source: "service" or "fallback_gradient".

        With cache_key (background path), service results are cached so a
        batch reusing the same background skips the HTTP round-trip.
        """
        if cache_key:
            cached = self._depth_cache.get(cache_key)
            if cached is not None:
                return cached
        try:
            # Try depth service first
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Encode image to JPEG
                _, buffer = cv2.imencode('.jpg', image)

                response = await client.post(
                    f"{self.depth_service_url}/depth/estimate",
                    files={"file": ("image.jpg", buffer.tobytes(), "image/jpeg")},
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'depth_map' in data:
                        # Decode base64 depth map
                        import base64
                        depth_bytes = base64.b64decode(data['depth_map'])
                        depth_array = np.frombuffer(depth_bytes, dtype=np.float32)
                        h, w = data.get('height', image.shape[0]), data.get('width', image.shape[1])
                        depth_map = depth_array.reshape((h, w))
                        logger.info(f"Depth map received from service: {w}x{h}, range [{depth_map.min():.3f}, {depth_map.max():.3f}]")
                        if cache_key:
                            self._depth_cache.put(cache_key, (depth_map, "service"))
                        return depth_map, "service"
                else:
                    logger.warning(f"Depth service returned status {response.status_code}: {response.text[:200]}")

        except Exception as e:
            logger.warning(f"Depth service call failed: {e}, using fallback gradient")

        # Fallback: simple gradient depth (NOT suitable for production)
        logger.warning(
            "Depth service unavailable — using vertical gradient fallback. "
            "Depth-aware placement/effects will NOT reflect real scene depth."
        )
        h, w = image.shape[:2]
        depth_map = np.linspace(0, 1, h)[:, np.newaxis]
        depth_map = np.repeat(depth_map, w, axis=1).astype(np.float32)
        return depth_map, "fallback_gradient"

    async def _estimate_lighting(
        self,
        image: np.ndarray,
        max_light_sources: int = 3,
        intensity_threshold: float = 0.6,
        apply_water_attenuation: bool = False,
        depth_category: DepthZone = DepthZone.MID,
        water_clarity: WaterClarity = WaterClarity.CLEAR,
    ) -> LightingInfo:
        """Estimate lighting from image."""
        if self.lighting_estimator:
            try:
                result = self.lighting_estimator.estimate_lighting(image)
                light_sources = []
                for ls in result.light_sources[:max_light_sources]:
                    if ls.intensity >= intensity_threshold:
                        # Compute shadow_softness from the light type
                        # LightSource has no shadow_softness, so it is derived here
                        if ls.light_type == 'directional':
                            shadow_softness = 0.3  # Hard shadows (sun)
                        elif ls.light_type == 'point':
                            shadow_softness = 0.5  # Medium shadows
                        elif ls.light_type == 'area':
                            # Softer shadows for large lights
                            shadow_softness = min(0.8, 0.5 + ls.angular_size * 0.3)
                        else:
                            shadow_softness = 0.5  # Default

                        # Convert LightSource (2D position, float colors) to LightSourceInfo (3D position, int colors)
                        # LightSource.position is (azimuth, elevation); LightSourceInfo expects (azimuth, elevation, distance)
                        position_3d = (ls.position[0], ls.position[1], 2.0)  # Assume a distance of 2 metres

                        # LightSource.color is List[float]; LightSourceInfo expects Tuple[int, int, int]
                        color_int = (int(ls.color[0]), int(ls.color[1]), int(ls.color[2]))

                        light_sources.append(LightSourceInfo(
                            light_type=LightType(ls.light_type),
                            position=position_3d,
                            intensity=ls.intensity,
                            color=color_int,
                            shadow_softness=shadow_softness,
                        ))
                return LightingInfo(
                    light_sources=light_sources,
                    dominant_direction=result.dominant_direction,
                    color_temperature=result.color_temperature,
                    ambient_intensity=result.ambient_intensity,
                )
            except Exception as e:
                logger.warning(f"Lighting estimation failed: {e}")

        # Fallback: simple analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Find brightest region
        kernel_size = max(h, w) // 10
        blurred = cv2.GaussianBlur(gray, (kernel_size | 1, kernel_size | 1), 0)
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)

        # Calculate dominant direction
        cx, cy = w // 2, h // 2
        dx, dy = max_loc[0] - cx, max_loc[1] - cy
        azimuth = np.degrees(np.arctan2(dy, dx))
        elevation = 45.0  # Default

        # Estimate color temperature from image
        b, g, r = cv2.split(image)
        r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()

        if r_mean > b_mean:
            color_temp = 3000 + (r_mean - b_mean) * 10
        else:
            color_temp = 6500 + (b_mean - r_mean) * 20
        color_temp = np.clip(color_temp, 2000, 12000)

        return LightingInfo(
            light_sources=[
                LightSourceInfo(
                    light_type=LightType.DIRECTIONAL,
                    position=(azimuth, elevation, 1.0),
                    intensity=max_val / 255.0,
                    color=(int(r_mean), int(g_mean), int(b_mean)),
                    shadow_softness=0.5,
                )
            ],
            dominant_direction=(azimuth, elevation),
            color_temperature=float(color_temp),
            ambient_intensity=float(gray.mean() / 255.0),
        )

    def _create_heuristic_scene_analysis(self, image: np.ndarray):
        """
        Create a basic scene analysis using heuristics when segmentation service fails.

        This is a fallback that analyzes basic image properties to estimate scene regions.
        It's not as accurate as SAM3 but provides reasonable defaults.

        Args:
            image: BGR image array

        Returns:
            SceneAnalysis-like object with basic scene information
        """
        from dataclasses import dataclass
        from typing import Dict, Optional
        import numpy as np

        @dataclass
        class HeuristicSceneAnalysis:
            dominant_region: str
            region_scores: Dict[str, float]
            depth_zones: Dict[str, tuple]
            scene_brightness: float
            water_clarity: str
            color_temperature: str
            region_map: Optional[np.ndarray] = None

        try:
            h, w = image.shape[:2]

            # Analyze color distribution for underwater scene classification
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(hsv)

            # Calculate brightness
            brightness = val.mean() / 255.0

            # Analyze color temperature from BGR
            b, g, r = cv2.split(image)
            blue_dominance = b.mean() / max(r.mean(), 1)

            # Determine water clarity based on saturation and contrast
            contrast = val.std() / 127.0  # Normalize
            avg_saturation = sat.mean() / 255.0

            if contrast > 0.3 and avg_saturation > 0.3:
                water_clarity = "clear"
            elif contrast > 0.15:
                water_clarity = "moderate"
            else:
                water_clarity = "murky"

            # Color temperature estimation
            if blue_dominance > 1.3:
                color_temperature = "cool"
            elif blue_dominance < 0.8:
                color_temperature = "warm"
            else:
                color_temperature = "neutral"

            # Analyze vertical gradient to detect seafloor vs open water
            # Darker bottom typically indicates seafloor
            top_third = val[:h//3, :].mean()
            bottom_third = val[2*h//3:, :].mean()

            # Estimate region scores based on image characteristics
            region_scores = {
                "open_water": 0.0,
                "seafloor": 0.0,
                "surface": 0.0,
                "vegetation": 0.0,
                "rocky": 0.0,
                "sandy": 0.0,
                "murky": 0.0,
            }

            # Score based on vertical gradient (darker bottom = seafloor likely)
            if bottom_third < top_third * 0.7:
                region_scores["seafloor"] += 0.4
                region_scores["open_water"] += 0.3
            elif top_third > bottom_third * 1.2:
                region_scores["surface"] += 0.3
                region_scores["open_water"] += 0.4
            else:
                region_scores["open_water"] += 0.5

            # Blue dominance suggests open water
            if blue_dominance > 1.2:
                region_scores["open_water"] += 0.3

            # Green tint suggests vegetation
            green_ratio = g.mean() / max((r.mean() + b.mean()) / 2, 1)
            if green_ratio > 1.1:
                region_scores["vegetation"] += 0.2

            # Low saturation/contrast suggests murky water
            if water_clarity == "murky":
                region_scores["murky"] += 0.3
                region_scores["open_water"] -= 0.1

            # Brown tones suggest sandy bottom
            if r.mean() > b.mean() and g.mean() > b.mean():
                brown_score = (r.mean() + g.mean()) / (2 * max(b.mean(), 1))
                if brown_score > 1.2:
                    region_scores["sandy"] += 0.25

            # Normalize scores
            total = sum(region_scores.values())
            if total > 0:
                region_scores = {k: max(0, v/total) for k, v in region_scores.items()}
            else:
                region_scores["open_water"] = 1.0

            # Determine dominant region
            dominant_region = max(region_scores.items(), key=lambda x: x[1])[0]

            # Create simple depth zones based on image thirds
            depth_zones = {
                "near": (0.0, 0.33),
                "mid": (0.33, 0.66),
                "far": (0.66, 1.0),
            }

            logger.info(f"Heuristic scene analysis: dominant={dominant_region}, brightness={brightness:.2f}, clarity={water_clarity}")

            return HeuristicSceneAnalysis(
                dominant_region=dominant_region,
                region_scores=region_scores,
                depth_zones=depth_zones,
                scene_brightness=brightness,
                water_clarity=water_clarity,
                color_temperature=color_temperature,
                region_map=None,  # Heuristic doesn't provide pixel-level map
            )

        except Exception as e:
            logger.error(f"Heuristic scene analysis failed: {e}")
            # Ultimate fallback - return neutral defaults
            return HeuristicSceneAnalysis(
                dominant_region="open_water",
                region_scores={"open_water": 0.6, "seafloor": 0.2, "surface": 0.2},
                depth_zones={"mid": (0.3, 0.7)},
                scene_brightness=0.5,
                water_clarity="moderate",
                color_temperature="neutral",
                region_map=None,
            )

    def _create_multi_object_summary(
        self,
        background: np.ndarray,
        composite: np.ndarray,
        object_debug_results: List[dict],
        scene_analysis,
        depth_map: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Create a consolidated visualization showing all objects and their placement decisions.

        Layout (improved to prevent overlap):
        ┌─────────────────────────────────────────────┐
        │  TITLE BAR                                  │
        ├─────────────────────┬───────────────────────┤
        │                     │  OBJECT LIST          │
        │  COMPOSITE IMAGE    │  (scrollable info)    │
        │  (with bboxes)      │                       │
        │                     │                       │
        ├─────────────────────┴───────────────────────┤
        │  THUMBNAILS (horizontal strip)              │
        ├─────────────────────┬───────────────────────┤
        │  BACKGROUND         │  DEPTH MAP            │
        └─────────────────────┴───────────────────────┘
        """
        try:
            h, w = background.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_small = cv2.FONT_HERSHEY_SIMPLEX

            num_objects = len(object_debug_results)

            # Fixed dimensions for cleaner layout
            info_panel_w = 280
            thumb_h = 100
            footer_h = 150
            title_h = 35

            # Color palette for objects
            colors = [
                (0, 255, 0),    # Green
                (255, 165, 0),  # Orange
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 255, 0),  # Yellow
                (255, 0, 0),    # Blue
                (0, 0, 255),    # Red
                (128, 0, 128),  # Purple
            ]

            # =====================================================
            # 1. Create title bar
            # =====================================================
            total_w = w + info_panel_w
            title_bar = np.zeros((title_h, total_w, 3), dtype=np.uint8)
            title_bar[:] = (50, 50, 50)
            cv2.putText(title_bar, "MULTI-OBJECT PIPELINE DEBUG SUMMARY",
                       (10, 25), font, 0.7, (255, 255, 255), 2)
            cv2.putText(title_bar, f"Objects: {num_objects}",
                       (total_w - 150, 25), font, 0.5, (200, 200, 200), 1)

            # =====================================================
            # 2. Create composite panel with bounding boxes
            # =====================================================
            composite_panel = composite.copy()

            # Draw bounding boxes with smart label placement
            label_positions = []  # Track label positions to avoid overlap

            for i, obj_info in enumerate(object_debug_results):
                color = colors[i % len(colors)]
                bbox = obj_info['bbox']
                class_name = obj_info['class_name'][:15]  # Truncate long names
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # Draw bounding box
                cv2.rectangle(composite_panel, (x1, y1), (x2, y2), color, 2)

                # Smart label placement (avoid overlap)
                label = f"{i}:{class_name}"
                (label_w, label_h), _ = cv2.getTextSize(label, font_small, 0.4, 1)

                # Try positions: above, below, inside-top, inside-bottom
                label_x = x1
                label_y = y1 - 8
                if label_y < label_h + 5:
                    label_y = y1 + label_h + 5  # Below top edge

                # Check for overlap with existing labels
                for (lx, ly, lw, lh) in label_positions:
                    if abs(label_x - lx) < lw and abs(label_y - ly) < lh:
                        label_y = y2 + label_h + 5  # Move to bottom
                        break

                label_positions.append((label_x, label_y, label_w + 10, label_h + 5))

                # Draw label background
                cv2.rectangle(composite_panel,
                             (label_x, label_y - label_h - 3),
                             (label_x + label_w + 6, label_y + 3),
                             color, -1)
                cv2.putText(composite_panel, label, (label_x + 3, label_y),
                           font_small, 0.4, (0, 0, 0), 1)

            # =====================================================
            # 3. Create info panel (right side)
            # =====================================================
            info_panel = np.zeros((h, info_panel_w, 3), dtype=np.uint8)
            info_panel[:] = (35, 35, 35)

            # Title
            cv2.putText(info_panel, "PLACED OBJECTS", (10, 25),
                       font, 0.5, (255, 255, 255), 1)
            cv2.line(info_panel, (10, 35), (info_panel_w - 10, 35), (80, 80, 80), 1)

            # Object list with proper spacing
            y_pos = 55
            max_objects_display = (h - 150) // 50  # Calculate how many fit

            for i, obj_info in enumerate(object_debug_results[:max_objects_display]):
                color = colors[i % len(colors)]
                class_name = obj_info['class_name'][:18]
                bbox = obj_info['bbox']
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # Object name with color indicator
                cv2.circle(info_panel, (15, y_pos - 4), 5, color, -1)
                cv2.putText(info_panel, f"{i}. {class_name}",
                           (25, y_pos), font_small, 0.4, (220, 220, 220), 1)
                y_pos += 18

                # Size info
                cv2.putText(info_panel, f"   Size: {x2-x1}x{y2-y1}px",
                           (25, y_pos), font_small, 0.32, (150, 150, 150), 1)
                y_pos += 15

                # Placement decision
                placement = obj_info.get('placement_decision')
                if placement:
                    decision_color = (100, 200, 100) if placement.decision == 'accepted' else (200, 200, 100)
                    score = getattr(placement, 'compatibility_score', 0)
                    cv2.putText(info_panel, f"   {placement.decision} ({score:.0%})",
                               (25, y_pos), font_small, 0.32, decision_color, 1)
                    y_pos += 15

                y_pos += 8  # Spacing between objects

            # Show "more" indicator if truncated
            if num_objects > max_objects_display:
                cv2.putText(info_panel, f"   ... +{num_objects - max_objects_display} more",
                           (25, y_pos), font_small, 0.35, (120, 120, 120), 1)

            # Scene analysis at bottom
            if scene_analysis is not None:
                y_pos = h - 80
                cv2.line(info_panel, (10, y_pos - 10), (info_panel_w - 10, y_pos - 10), (80, 80, 80), 1)
                cv2.putText(info_panel, "SCENE INFO", (10, y_pos),
                           font_small, 0.4, (180, 180, 180), 1)
                y_pos += 20
                dominant = scene_analysis.dominant_region
                dominant_str = dominant.value if hasattr(dominant, 'value') else str(dominant)
                cv2.putText(info_panel, f"Region: {dominant_str}",
                           (15, y_pos), font_small, 0.35, (140, 140, 140), 1)
                y_pos += 16
                cv2.putText(info_panel, f"Clarity: {scene_analysis.water_clarity}",
                           (15, y_pos), font_small, 0.35, (140, 140, 140), 1)

            # =====================================================
            # 4. Combine composite + info panel
            # =====================================================
            main_panel = np.hstack([composite_panel, info_panel])

            # =====================================================
            # 5. Create thumbnail strip
            # =====================================================
            thumb_strip = np.zeros((thumb_h + 30, total_w, 3), dtype=np.uint8)
            thumb_strip[:] = (45, 45, 45)
            cv2.putText(thumb_strip, "Object Thumbnails:", (10, 18),
                       font_small, 0.4, (180, 180, 180), 1)

            x_offset = 10
            thumb_max_w = 80
            for i, obj_info in enumerate(object_debug_results):
                if x_offset + thumb_max_w + 10 > total_w - 50:
                    # Show count of remaining
                    remaining = num_objects - i
                    cv2.putText(thumb_strip, f"+{remaining}",
                               (x_offset + 5, 75), font_small, 0.4, (150, 150, 150), 1)
                    break

                color = colors[i % len(colors)]
                debug_dir = obj_info.get('debug_dir')

                thumb = None
                if debug_dir:
                    orig_path = os.path.join(debug_dir, "03_object_original.png")
                    if os.path.exists(orig_path):
                        thumb_orig = cv2.imread(orig_path, cv2.IMREAD_UNCHANGED)
                        if thumb_orig is not None:
                            if len(thumb_orig.shape) == 3 and thumb_orig.shape[2] == 4:
                                alpha = thumb_orig[:, :, 3:4].astype(np.float32) / 255.0
                                thumb = (thumb_orig[:, :, :3].astype(np.float32) * alpha +
                                        60 * (1 - alpha)).astype(np.uint8)
                            else:
                                thumb = thumb_orig[:, :, :3] if len(thumb_orig.shape) == 3 else cv2.cvtColor(thumb_orig, cv2.COLOR_GRAY2BGR)

                            # Resize maintaining aspect ratio
                            th, tw = thumb.shape[:2]
                            scale = min(thumb_max_w / tw, (thumb_h - 5) / th)
                            thumb = cv2.resize(thumb, (int(tw * scale), int(th * scale)))

                if thumb is None:
                    thumb = np.full((thumb_h - 10, thumb_max_w - 10, 3), 60, dtype=np.uint8)

                th, tw = thumb.shape[:2]
                y_thumb = 28
                thumb_strip[y_thumb:y_thumb + th, x_offset:x_offset + tw] = thumb
                cv2.rectangle(thumb_strip, (x_offset - 1, y_thumb - 1),
                             (x_offset + tw + 1, y_thumb + th + 1), color, 1)

                x_offset += tw + 8

            # =====================================================
            # 6. Create footer with background + depth
            # =====================================================
            half_w = total_w // 2
            bg_small = cv2.resize(background, (half_w, footer_h))

            if depth_map is not None:
                depth_norm = (depth_map * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                depth_small = cv2.resize(depth_color, (total_w - half_w, footer_h))
            else:
                depth_small = np.zeros((footer_h, total_w - half_w, 3), dtype=np.uint8)
                cv2.putText(depth_small, "No Depth", (10, footer_h // 2),
                           font_small, 0.5, (100, 100, 100), 1)

            footer = np.hstack([bg_small, depth_small])
            cv2.putText(footer, "Background", (5, 15), font_small, 0.4, (255, 255, 255), 1)
            cv2.putText(footer, "Depth Map", (half_w + 5, 15), font_small, 0.4, (255, 255, 255), 1)

            # =====================================================
            # 7. Stack all panels vertically
            # =====================================================
            final = np.vstack([title_bar, main_panel, thumb_strip, footer])

            return final

        except Exception as e:
            logger.warning(f"Multi-object summary visualization failed: {e}")
            return composite
