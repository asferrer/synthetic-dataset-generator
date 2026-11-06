"""
Codificación de mapas de regiones para las respuestas de la API.
"""

import base64

import cv2
import numpy as np


def encode_region_map_base64(region_map: np.ndarray) -> str:
    """Encode region map as base64 PNG string.

    The region_map is a uint8 array with values:
    0=unknown, 1=open_water, 2=seafloor, 3=surface, 4=vegetation, 5=rocky, 6=sandy, 7=murky
    """
    # Encode as PNG (lossless, efficient for indexed/mask images)
    success, encoded = cv2.imencode('.png', region_map)
    if success:
        return base64.b64encode(encoded.tobytes()).decode('utf-8')
    return None
