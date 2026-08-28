import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from backend.app.main import app
from backend.app.services.inference import inference_service

@pytest.fixture(scope="session", autouse=True)
def load_test_models():
    """Ensure models are loaded for testing."""
    if not inference_service.is_ready:
        inference_service.load_models()

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def create_test_image_bytes(format="JPEG", size=(200, 200), color=(128, 128, 128)) -> bytes:
    """Helper to generate in-memory synthetic image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.read()

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert "environment" in data["details"]

def test_analyze_valid_image(client):
    img_bytes = create_test_image_bytes(format="JPEG", color=(100, 150, 200))
    files = {"image": ("test_sample.jpg", img_bytes, "image/jpeg")}
    
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 201
    data = response.json()

    assert "id" in data
    assert data["filename"] == "test_sample.jpg"
    assert 0.0 <= data["quality_score"] <= 100.0
    assert data["quality_label"] in ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]
    assert isinstance(data["issues"], list)
    assert "statistics" in data
    assert "laplacian_variance" in data["statistics"]
    assert "image_url" in data and data["image_url"].startswith("/uploads/images/")
    assert "heatmap_url" in data and data["heatmap_url"].startswith("/uploads/heatmaps/")

def test_analyze_unsupported_file_extension(client):
    files = {"image": ("document.txt", b"Hello world text", "text/plain")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_analyze_corrupted_image_header(client):
    corrupt_bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00corrupted_garbage_bytes_here"
    files = {"image": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "Corrupted or invalid image" in response.json()["detail"]

def test_analyze_empty_file(client):
    files = {"image": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]

def test_get_results_pagination(client):
    img_bytes = create_test_image_bytes(format="PNG", color=(50, 50, 50))
    client.post("/api/analyze", files={"image": ("pagination_test.png", img_bytes, "image/png")})

    response = client.get("/api/results?page=1&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "page" in data
    assert "items" in data
    assert len(data["items"]) >= 1
    assert data["page"] == 1

def test_get_result_by_id_and_delete(client):
    img_bytes = create_test_image_bytes(format="JPEG", color=(200, 200, 200))
    post_resp = client.post("/api/analyze", files={"image": ("detail_test.jpg", img_bytes, "image/jpeg")})
    assert post_resp.status_code == 201
    record_id = post_resp.json()["id"]

    get_resp = client.get(f"/api/results/{record_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == record_id
    assert get_resp.json()["filename"] == "detail_test.jpg"

    del_resp = client.delete(f"/api/results/{record_id}")
    assert del_resp.status_code == 204

    get_after = client.get(f"/api/results/{record_id}")
    assert get_after.status_code == 404

def test_get_nonexistent_result_404(client):
    response = client.get("/api/results/99999999")
    assert response.status_code == 404
