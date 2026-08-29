from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data

def test_health_head_endpoint():
    response = client.head("/api/health")
    assert response.status_code == 200

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs" in data

def test_root_head_endpoint():
    response = client.head("/")
    assert response.status_code == 200
