"""
src/tests/test_api.py
Integration tests for FastAPI inference service.
Uses httpx TestClient with a mocked model.
"""

import io
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def make_jpeg_bytes(width=224, height=224):
    """Create a minimal valid JPEG image as bytes."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# Patch model loading before importing app
@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """Inject a fake model so tests don't need an actual checkpoint."""
    import src.api.app as app_module
    fake_model = MagicMock()
    app_module.MODEL = fake_model
    app_module.DEVICE = "cpu"

    # patch predict to return a canned result
    monkeypatch.setattr(
        "src.api.app.predict",
        lambda model, tensor, device: {
            "label": "cat",
            "confidence": 0.92,
            "probabilities": {"cat": 0.92, "dog": 0.08},
        },
    )
    monkeypatch.setattr("src.api.app.preprocess_pil_image", lambda img: MagicMock())
    yield fake_model


@pytest.fixture
def client(mock_model):
    from src.api.app import app
    with TestClient(app) as c:
        yield c


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_key(self, client):
        resp = client.get("/health")
        assert "status" in resp.json()

    def test_health_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True


# ── Predict endpoint ──────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("cat.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_response_keys(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("cat.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        data = resp.json()
        for key in ("label", "confidence", "probabilities", "latency_ms"):
            assert key in data

    def test_predict_label_is_string(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("cat.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        assert isinstance(resp.json()["label"], str)

    def test_predict_invalid_file_type(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("data.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 422

    def test_predict_latency_non_negative(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("cat.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.json()["latency_ms"] >= 0


# ── Metrics endpoint ──────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]
