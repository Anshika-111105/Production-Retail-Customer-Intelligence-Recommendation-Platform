"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app, load_artifacts


@pytest.fixture(scope="module", autouse=True)
def init_api():
    """Ensure artifacts are loaded."""
    load_artifacts()
    client = TestClient(app)
    return client


def test_health_endpoint(init_api):
    """Test /health endpoint."""
    response = init_api.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["version"] == "1.0.0"


def test_customer_profile_endpoint(init_api):
    """Test /customers/{customer_id} endpoint."""
    # Test valid customer
    response = init_api.get("/customers/20000")
    if response.status_code == 200:
        data = response.json()
        assert data["customer_id"] == 20000
        assert "segment_label" in data
        assert "frequency" in data
        assert "monetary_value" in data

    # Test unknown customer
    response_404 = init_api.get("/customers/999999")
    assert response_404.status_code == 404


def test_recommendations_endpoint(init_api):
    """Test /recommendations/{customer_id} endpoint."""
    response = init_api.get("/recommendations/20000?n=5")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == 20000
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 5
    if data["recommendations"]:
        first_rec = data["recommendations"][0]
        assert "product_id" in first_rec
        assert "score" in first_rec
        assert "explanation" in first_rec

    # Test invalid n
    response_invalid = init_api.get("/recommendations/20000?n=0")
    assert response_invalid.status_code == 422


def test_similar_products_endpoint(init_api):
    """Test /products/{product_id}/similar endpoint."""
    response = init_api.get("/products/1000/similar?n=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_segments_and_model_info_endpoints(init_api):
    """Test /segments and /model-info endpoints."""
    resp_seg = init_api.get("/segments")
    assert resp_seg.status_code == 200
    assert isinstance(resp_seg.json(), list)

    resp_info = init_api.get("/model-info")
    assert resp_info.status_code == 200
    data = resp_info.json()
    assert "algorithms" in data
    assert "evaluation_metrics" in data
