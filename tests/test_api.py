import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"

def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "distilbert_clinc150"
    assert data["model_version"] == "1.0.0"
    assert data["num_classes"] == 151
    assert data["max_sequence_length"] == 48
    assert "ONNX Runtime" in data["inference_backend"]

def test_metrics_placeholder_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200

def test_valid_single_predict(client):
    payload = {"text": "what is my credit card balance"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "confidence" in data
    assert "is_oos" in data
    assert data["model_version"] == "1.0.0"
    assert data["intent"] == "balance"
    assert data["confidence"] > 0.4
    assert data["is_oos"] is False

def test_predict_empty_text_validation(client):
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code in (400, 422)

def test_predict_missing_text_payload(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_valid_batch_predict(client):
    payload = {
        "inputs": [
            "what is my credit card balance",
            "book a table for 4 at an Italian restaurant",
            "what is the weather like in New York today"
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 3
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["intent"] == "balance"
    assert data["predictions"][1]["intent"] == "restaurant_reservation"
    assert data["predictions"][2]["intent"] == "weather"

def test_predict_empty_batch_validation(client):
    payload = {"inputs": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code in (400, 422)

def test_oos_prediction_query(client):
    payload = {"text": "who won the 1998 World Cup in France"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "oos"
    assert data["is_oos"] is True

def test_batch_vs_individual_inference_equivalence(client):
    queries = [
        "what is my credit card balance",
        "book a table for 4 at an Italian restaurant",
        "what is the weather like in New York today",
        "who won the 1998 World Cup in France",
        "how do I change my oil in my car"
    ]

    # 1. Individual predictions
    individual_preds = []
    for q in queries:
        resp = client.post("/predict", json={"text": q})
        assert resp.status_code == 200
        individual_preds.append(resp.json())

    # 2. Batch predictions
    batch_resp = client.post("/predict/batch", json={"inputs": queries})
    assert batch_resp.status_code == 200
    batch_preds = batch_resp.json()["predictions"]

    # 3. Equivalence assertion
    assert len(individual_preds) == len(batch_preds)
    for ind, bat in zip(individual_preds, batch_preds):
        assert ind["intent"] == bat["intent"], f"Intent mismatch: {ind['intent']} != {bat['intent']}"
        assert abs(ind["confidence"] - bat["confidence"]) < 1e-4, f"Confidence mismatch: {ind['confidence']} != {bat['confidence']}"
        assert ind["is_oos"] == bat["is_oos"], f"OOS flag mismatch: {ind['is_oos']} != {bat['is_oos']}"
        assert ind["model_version"] == bat["model_version"]
