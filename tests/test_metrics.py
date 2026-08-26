import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_metrics_endpoint_exposition(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    text = response.text

    # Verify expected metric names exist in exposition
    assert "http_requests_total" in text
    assert "http_request_duration_seconds" in text
    assert "model_inference_duration_seconds" in text
    assert "model_predictions_total" in text
    assert "model_prediction_confidence_bucket" in text
    assert "model_oos_predictions_total" in text
    assert "model_in_scope_predictions_total" in text
    assert "model_info" in text

def test_predict_metrics_increment(client):
    # Fetch initial metrics
    m0 = client.get("/metrics").text

    # Make single predict request
    resp = client.post("/predict", json={"text": "what is my credit card balance"})
    assert resp.status_code == 200

    # Fetch updated metrics
    m1 = client.get("/metrics").text

    assert "endpoint=\"/predict\"" in m1
    assert "intent=\"balance\"" in m1

def test_oos_metrics_increment(client):
    m0 = client.get("/metrics").text
    
    # Make OOS request
    resp = client.post("/predict", json={"text": "who won the 1998 World Cup in France"})
    assert resp.status_code == 200
    assert resp.json()["is_oos"] is True

    m1 = client.get("/metrics").text
    assert "model_oos_predictions_total" in m1
    assert "intent=\"oos\"" in m1

def test_batch_metrics_recording(client):
    # Batch request with 3 items
    payload = {
        "inputs": [
            "what is my credit card balance",
            "book a table for 4 at an Italian restaurant",
            "who won the 1998 World Cup in France"
        ]
    }
    
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200

    metrics_text = client.get("/metrics").text

    assert "endpoint=\"/predict/batch\"" in metrics_text
    assert "intent=\"balance\"" in metrics_text
    assert "intent=\"restaurant_reservation\"" in metrics_text
    assert "intent=\"oos\"" in metrics_text
