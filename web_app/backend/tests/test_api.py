from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_read_root():
    # Health check is at root /health or /api/health? 
    # In main.py: @app.get("/health") is at root. 
    # But router is at /api. Let's check main.py again.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_get_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_mock():
    # Test valid prediction with mock model
    payload = {
        "model_name": "TestModel",
        "signal_data": [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "diagnosis" in data
    assert "confidence" in data
    assert "model_used" in data

def test_predict_invalid():
    # Test missing field
    payload = {
        "model_name": "TestModel"
        # missing signal_data
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
