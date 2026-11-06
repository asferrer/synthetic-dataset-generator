"""
Pruebas de humo de la API de segmentación.

Comparan la tabla de rutas y las operaciones de OpenAPI con una instantánea tomada
antes de dividir ``app/main.py`` en routers, y comprueban las rutas de salud sin GPU
ni modelo cargado (el cliente de pruebas no ejecuta el ciclo de vida de la aplicación).
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SNAPSHOT = json.loads((Path(__file__).parent / "routes_snapshot.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _current_routes():
    return [
        {
            "path": r.path,
            "methods": sorted(getattr(r, "methods", None) or []),
            "name": r.name,
            "tags": getattr(r, "tags", None) or [],
        }
        for r in app.routes
    ]


def test_route_table_matches_snapshot():
    """Mismas rutas, métodos, nombres y etiquetas, en el mismo orden."""
    assert _current_routes() == SNAPSHOT["routes"]


def test_openapi_operations_match_snapshot():
    paths = app.openapi()["paths"]
    operations = [
        {"path": p, "method": m.upper(), "operation_id": d["operationId"], "tags": d.get("tags", [])}
        for p, methods in paths.items()
        for m, d in methods.items()
    ]
    assert operations == SNAPSHOT["openapi_operations"]


def test_openapi_endpoint_served(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Segmentation Service"


def test_health_without_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"]
    assert body["sam3_available"] is False


def test_model_status_without_model(client):
    response = client.get("/model-status")
    assert response.status_code == 200


def test_unknown_job_returns_404(client):
    for path in ("/extract/jobs/inexistente", "/sam3/jobs/inexistente", "/labeling/jobs/inexistente"):
        assert client.get(path).status_code == 404
