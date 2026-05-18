"""
tests/test_api.py
============================================================
Smoke tests for the FastAPI inference service
============================================================

WHY THESE TESTS:
    Not exhaustive testing - just enough to confirm the four
    things that MUST work before you commit:
      1. The app starts without crashing.
      2. /health responds.
      3. /predict returns the expected shape.
      4. /predict_batch returns the expected shape.

    These run with pytest + fastapi's TestClient, which spins up
    the app in-process - no separate server needed.

RUN:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.api import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """
    TestClient triggers the lifespan hook on enter, so the
    model is loaded once per test module - same as real serving.
    """
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_after_model_load(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_predict_returns_expected_shape(client):
    r = client.post("/predict", json={
        "text": "Bank of Canada holds overnight rate at 5%."
    })
    assert r.status_code == 200, r.text
    body = r.json()

    # Top-level shape
    for key in ["text", "cleaned_text", "sentiment", "intent"]:
        assert key in body, f"missing key: {key}"

    # Each head's shape
    for head in ["sentiment", "intent"]:
        h = body[head]
        assert "label" in h
        assert isinstance(h["label"], str)
        assert 0.0 <= h["confidence"] <= 1.0
        assert isinstance(h["all_scores"], dict)
        # Probabilities should sum to ~1.
        assert abs(sum(h["all_scores"].values()) - 1.0) < 0.01


def test_predict_strips_urls(client):
    """The preprocessor's main job - confirm URLs are stripped."""
    r = client.post("/predict", json={
        "text": "TSX closes flat https://t.co/abc123def"
    })
    assert r.status_code == 200
    body = r.json()
    assert "http" not in body["cleaned_text"]
    assert "t.co" not in body["cleaned_text"]


def test_predict_batch(client):
    r = client.post("/predict_batch", json={
        "texts": [
            "Shopify Q3 revenue rose 25%.",
            "Manulife flags U.S. real estate exposure.",
            "TSX closed flat.",
        ]
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 3
    for item in body:
        assert "sentiment" in item
        assert "intent" in item


def test_predict_rejects_empty_input(client):
    r = client.post("/predict", json={"text": ""})
    # pydantic returns 422 for validation errors
    assert r.status_code == 422


def test_batch_size_cap(client):
    """Should reject batches larger than 64."""
    r = client.post("/predict_batch", json={
        "texts": ["test"] * 100,
    })
    assert r.status_code == 422
