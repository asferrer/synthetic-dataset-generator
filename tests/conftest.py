"""
Shared fixtures for all tests.

This module provides common fixtures used across unit, integration, and e2e tests.
"""
import pytest
import base64
import subprocess
import time
import sys
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sample_image_array():
    """Generate a random sample image as numpy array."""
    np.random.seed(42)
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def sample_image_pil(sample_image_array):
    """Generate a sample PIL Image."""
    return Image.fromarray(sample_image_array)


@pytest.fixture(scope="session")
def sample_image_base64(sample_image_pil):
    """Generate a sample image in base64 format."""
    buffer = BytesIO()
    sample_image_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture(scope="session")
def sample_rgba_image():
    """Generate a sample RGBA image (with alpha channel)."""
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGBA")


@pytest.fixture(scope="session")
def sample_rgba_base64(sample_rgba_image):
    """Generate a sample RGBA image in base64 format."""
    buffer = BytesIO()
    sample_rgba_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def sample_images(tmp_path):
    """Create temporary sample images for testing."""
    np.random.seed(42)

    # Background image (larger)
    bg_array = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    bg_img = Image.fromarray(bg_array)
    bg_path = tmp_path / "background.jpg"
    bg_img.save(bg_path, format="JPEG")

    # Object image with alpha channel
    obj_array = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
    obj_img = Image.fromarray(obj_array, mode="RGBA")
    obj_path = tmp_path / "object.png"
    obj_img.save(obj_path, format="PNG")

    return bg_path, obj_path


# =============================================================================
# Docker Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def docker_compose_file():
    """Get path to docker-compose file."""
    return PROJECT_ROOT / "docker-compose.microservices.yml"


@pytest.fixture(scope="module")
def docker_services(docker_compose_file):
    """
    Start and stop Docker services for integration tests.

    This fixture starts all services before tests and stops them after.
    Use with integration and e2e tests only.
    """
    compose_file = str(docker_compose_file)

    # Build images first
    build_result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "build"],
        capture_output=True,
        text=True,
        timeout=1800  # 30 minutes for build
    )

    if build_result.returncode != 0:
        pytest.fail(f"Docker build failed: {build_result.stderr}")

    # Start services
    up_result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "up", "-d"],
        capture_output=True,
        text=True,
        timeout=120
    )

    if up_result.returncode != 0:
        pytest.fail(f"Docker compose up failed: {up_result.stderr}")

    # Wait for services to be healthy
    _wait_for_services()

    yield

    # Cleanup: stop services
    subprocess.run(
        ["docker", "compose", "-f", compose_file, "down"],
        capture_output=True,
        text=True,
        timeout=120
    )


def _wait_for_services(timeout: int = 300, interval: int = 5):
    """Wait for all services to be healthy."""
    import requests

    services = {
        "gateway": "http://localhost:8000/health",
        "depth": "http://localhost:8001/health",
        "segmentation": "http://localhost:8002/health",
        "augmentor": "http://localhost:8004/health",
    }

    start_time = time.time()

    for name, url in services.items():
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(interval)
        else:
            pytest.fail(f"Service {name} did not become healthy within {timeout}s")


# =============================================================================
# Service URL Fixtures
# =============================================================================

@pytest.fixture
def gateway_url():
    """Gateway service URL."""
    return "http://localhost:8000"


@pytest.fixture
def depth_url():
    """Depth service URL."""
    return "http://localhost:8001"


@pytest.fixture
def segmentation_url():
    """Segmentation service URL."""
    return "http://localhost:8002"


@pytest.fixture
def augmentor_url():
    """Augmentor service URL."""
    return "http://localhost:8004"
