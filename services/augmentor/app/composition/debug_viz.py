"""Debug visualizations of lighting, scene analysis and placement decisions."""

import cv2
import numpy as np
from loguru import logger

from app.models.schemas import LightingInfo


class DebugVizMixin:
    """Debug visualizations of lighting, scene analysis and placement decisions."""

    def _visualize_lighting(
        self,
        image: np.ndarray,
        lighting_info: LightingInfo,
    ) -> np.ndarray:
        """
        Create a visualization of detected lighting for documentation.

        Draws:
        - Light source positions as colored circles
        - Light direction arrows
        - Ambient intensity indicator
        - Color temperature bar
        """
        try:
            vis = image.copy()
            h, w = vis.shape[:2]
            cx, cy = w // 2, h // 2

            # Draw info panel background
            panel_h = 120
            overlay = vis.copy()
            cv2.rectangle(overlay, (10, h - panel_h - 10), (350, h - 10), (0, 0, 0), -1)
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            # Draw each light source
            for i, ls in enumerate(lighting_info.light_sources):
                azimuth, elevation, distance = ls.position
                intensity = ls.intensity

                # Calculate arrow endpoint from center
                arrow_len = min(w, h) // 4 * intensity
                rad_azimuth = np.radians(azimuth)
                end_x = int(cx + arrow_len * np.cos(rad_azimuth))
                end_y = int(cy + arrow_len * np.sin(rad_azimuth))

                # Light source color (warm = yellow/orange, cool = blue)
                if lighting_info.color_temperature < 4000:
                    arrow_color = (0, 165, 255)  # Orange
                elif lighting_info.color_temperature > 6500:
                    arrow_color = (255, 200, 100)  # Light blue
                else:
                    arrow_color = (255, 255, 255)  # White

                # Draw light direction arrow
                cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), arrow_color, 3, tipLength=0.3)

                # Draw light source indicator at origin point
                cv2.circle(vis, (end_x, end_y), 15, arrow_color, -1)
                cv2.circle(vis, (end_x, end_y), 15, (255, 255, 255), 2)

                # Label
                label = f"L{i+1}: {ls.light_type.value}"
                cv2.putText(vis, label, (end_x + 20, end_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw text info in panel
            y_text = h - panel_h + 10
            line_height = 22

            # Dominant direction
            dom_az, dom_el = lighting_info.dominant_direction
            cv2.putText(vis, f"Dominant Direction: {dom_az:.1f}deg az, {dom_el:.1f}deg el",
                       (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_text += line_height

            # Color temperature
            ct = lighting_info.color_temperature
            ct_label = "Warm" if ct < 4000 else ("Neutral" if ct < 6500 else "Cool")
            cv2.putText(vis, f"Color Temp: {ct:.0f}K ({ct_label})",
                       (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_text += line_height

            # Ambient intensity
            amb = lighting_info.ambient_intensity
            cv2.putText(vis, f"Ambient Intensity: {amb:.2f}",
                       (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_text += line_height

            # Light sources count
            cv2.putText(vis, f"Light Sources: {len(lighting_info.light_sources)}",
                       (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw color temperature bar
            bar_x, bar_y = 200, h - panel_h + 35
            bar_w, bar_h = 130, 15

            # Create gradient bar (warm to cool)
            for i in range(bar_w):
                ratio = i / bar_w
                if ratio < 0.5:
                    color = (int(255 * ratio * 2), int(165 * ratio * 2), 0)  # Black to orange
                else:
                    ratio2 = (ratio - 0.5) * 2
                    color = (int(255 - 55 * ratio2), int(165 + 90 * ratio2), int(255 * ratio2))  # Orange to light blue
                cv2.line(vis, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_h), color, 1)

            cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

            # Marker for current temperature
            ct_ratio = np.clip((ct - 2000) / 10000, 0, 1)
            marker_x = int(bar_x + ct_ratio * bar_w)
            cv2.drawMarker(vis, (marker_x, bar_y + bar_h + 8), (255, 255, 255),
                          cv2.MARKER_TRIANGLE_UP, 10, 2)

            return vis

        except Exception as e:
            logger.warning(f"Lighting visualization failed: {e}")
            return image

    def _visualize_scene_analysis(
        self,
        image: np.ndarray,
        scene_analysis,
        debug_info,
    ) -> np.ndarray:
        """
        Create visualization of scene analysis showing detected regions.

        Creates a 3-panel visualization:
        - Original image
        - Region overlay with colors
        - Region scores info panel
        """
        try:
            h, w = image.shape[:2]

            # Color map for scene regions (BGR format)
            region_colors = {
                'open_water': (255, 150, 50),     # Blue
                'seafloor': (50, 150, 200),       # Sandy brown
                'surface': (255, 255, 200),       # Light cyan
                'vegetation': (50, 200, 50),      # Green
                'rocky': (100, 100, 100),         # Gray
                'sandy': (100, 180, 220),         # Sand
                'murky': (80, 80, 60),            # Dark murky
                'unknown': (128, 128, 128),       # Neutral gray
            }

            # Create region visualization based on scores (since region_map may not be available via HTTP)
            region_viz = np.zeros((h, w, 3), dtype=np.uint8)

            # Check if region_map is available
            if hasattr(scene_analysis, 'region_map') and scene_analysis.region_map is not None:
                # Region value to name mapping
                region_value_map = {
                    1: 'open_water', 2: 'seafloor', 3: 'surface',
                    4: 'vegetation', 5: 'rocky', 6: 'sandy', 7: 'murky',
                }
                for value, region_name in region_value_map.items():
                    mask = (scene_analysis.region_map == value)
                    color = region_colors.get(region_name, (128, 128, 128))
                    region_viz[mask] = color
            else:
                # Create approximate visualization based on region scores
                # Use horizontal bands for top regions (FALLBACK - not real segmentation)
                logger.warning("Using fallback band visualization - region_map not available from SAM3")
                sorted_regions = sorted(scene_analysis.region_scores.items(), key=lambda x: -x[1])
                if sorted_regions:
                    y_pos = 0
                    for region_name, score in sorted_regions[:4]:
                        if score > 0.05:
                            band_height = int(h * score)
                            color = region_colors.get(region_name, (128, 128, 128))
                            region_viz[y_pos:y_pos + band_height, :] = color
                            y_pos += band_height
                    # Add warning text on the visualization
                    cv2.putText(region_viz, "FALLBACK VIZ (no SAM3 map)", (10, h - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Create overlay (original + colored regions at 40% opacity)
            overlay = cv2.addWeighted(image, 0.6, region_viz, 0.4, 0)

            # Create info panel
            info_panel = np.zeros((h, w, 3), dtype=np.uint8)
            info_panel[:] = (40, 40, 40)  # Dark gray background

            # Stack: Original | Overlay | Info Panel
            combined = np.hstack([image, overlay, info_panel])

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Scene Regions", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Analysis Info", (2*w + 10, 30), font, 0.7, (255, 255, 255), 2)

            # Add dominant region
            dominant = scene_analysis.dominant_region
            dominant_name = dominant.value if hasattr(dominant, 'value') else str(dominant)
            cv2.putText(combined, f"Dominant: {dominant_name}", (2*w + 10, 60), font, 0.5, (255, 255, 255), 1)

            # Add region scores legend
            y_offset = 85
            for region, score in sorted(scene_analysis.region_scores.items(), key=lambda x: -x[1])[:5]:
                if score > 0.01:
                    color = region_colors.get(region, (255, 255, 255))
                    cv2.putText(combined, f"{region}: {score:.1%}", (2*w + 10, y_offset), font, 0.4, color, 1)
                    y_offset += 20

            # Add analysis method
            method = debug_info.analysis_method if debug_info else "unknown"
            cv2.putText(combined, f"Method: {method}", (2*w + 10, y_offset + 10), font, 0.4, (200, 200, 200), 1)

            # Add water clarity and brightness info
            y_offset += 35
            cv2.putText(combined, f"Brightness: {scene_analysis.scene_brightness:.2f}", (2*w + 10, y_offset), font, 0.4, (200, 200, 200), 1)
            y_offset += 18
            cv2.putText(combined, f"Clarity: {scene_analysis.water_clarity}", (2*w + 10, y_offset), font, 0.4, (200, 200, 200), 1)
            y_offset += 18
            cv2.putText(combined, f"Temp: {scene_analysis.color_temperature}", (2*w + 10, y_offset), font, 0.4, (200, 200, 200), 1)

            # Add processing time if available
            if debug_info and hasattr(debug_info, 'processing_time_ms'):
                y_offset += 25
                cv2.putText(combined, f"Time: {debug_info.processing_time_ms:.1f}ms", (2*w + 10, y_offset), font, 0.4, (200, 200, 200), 1)

            return combined

        except Exception as e:
            logger.warning(f"Scene analysis visualization failed: {e}")
            return image

    def _visualize_placement_decisions(
        self,
        image: np.ndarray,
        scene_analysis,
        placement_decisions: list,
        bboxes: list,
    ) -> np.ndarray:
        """
        Visualize placement decisions on the composed image.

        Shows:
        - Bounding boxes with decision color coding
        - Decision labels (accepted/rejected/relocated)
        - Compatibility scores
        - Alternative positions
        """
        try:
            vis = image.copy()
            h, w = vis.shape[:2]

            # Color coding for decisions
            decision_colors = {
                'accepted': (0, 255, 0),     # Green
                'relocated': (0, 165, 255),  # Orange
                'rejected': (0, 0, 255),     # Red
            }

            font = cv2.FONT_HERSHEY_SIMPLEX

            # Draw bounding boxes and labels
            for i, (decision, bbox) in enumerate(zip(placement_decisions, bboxes)):
                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = decision_colors.get(decision.decision, (255, 255, 255))

                # Draw bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                label = f"{decision.object_class}"
                score_label = f"{decision.compatibility_score:.0%}"
                decision_label = f"[{decision.decision}]"
                region_label = f"@{decision.region_at_position}"

                # Calculate label size
                (label_w, label_h), baseline = cv2.getTextSize(label, font, 0.5, 1)
                label_bg_y = max(y1 - 45, 0)

                # Draw label background
                cv2.rectangle(vis, (x1, label_bg_y), (x1 + max(label_w + 10, 120), label_bg_y + 45), (0, 0, 0), -1)
                cv2.rectangle(vis, (x1, label_bg_y), (x1 + max(label_w + 10, 120), label_bg_y + 45), color, 1)

                # Draw labels
                cv2.putText(vis, label, (x1 + 5, label_bg_y + 12), font, 0.4, (255, 255, 255), 1)
                cv2.putText(vis, f"{decision_label} {score_label}", (x1 + 5, label_bg_y + 26), font, 0.35, color, 1)
                cv2.putText(vis, region_label, (x1 + 5, label_bg_y + 40), font, 0.35, (180, 180, 180), 1)

                # Draw alternative positions (small circles) if relocated
                if decision.decision == 'relocated' and decision.alternative_positions:
                    for alt_x, alt_y, alt_score in decision.alternative_positions[:3]:
                        alt_x, alt_y = int(alt_x), int(alt_y)
                        if 0 <= alt_x < w and 0 <= alt_y < h:
                            cv2.circle(vis, (alt_x, alt_y), 8, (0, 165, 255), 2)
                            cv2.circle(vis, (alt_x, alt_y), 3, (0, 165, 255), -1)

            # Add legend panel
            panel_h = 100
            panel_y = h - panel_h - 10
            overlay = vis.copy()
            cv2.rectangle(overlay, (10, panel_y), (280, h - 10), (0, 0, 0), -1)
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            # Draw legend
            cv2.putText(vis, "PLACEMENT DECISIONS", (20, panel_y + 20), font, 0.5, (255, 255, 255), 1)
            y_leg = panel_y + 40
            for decision_type, color in decision_colors.items():
                cv2.circle(vis, (30, y_leg), 6, color, -1)
                cv2.putText(vis, decision_type.capitalize(), (45, y_leg + 5), font, 0.4, color, 1)
                y_leg += 20

            # Statistics
            accepted = sum(1 for d in placement_decisions if d.decision == 'accepted')
            relocated = sum(1 for d in placement_decisions if d.decision == 'relocated')
            total = len(placement_decisions)
            cv2.putText(vis, f"Total: {total} | OK: {accepted} | Moved: {relocated}",
                       (150, panel_y + 40), font, 0.35, (200, 200, 200), 1)

            return vis

        except Exception as e:
            logger.warning(f"Placement decisions visualization failed: {e}")
            return image
