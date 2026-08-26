import time
from fastapi.testclient import TestClient
from app.main import app

def test_api_server_sanity():
    print("==========================================")
    print("      FastAPI Server & Metrics Check      ")
    print("==========================================")

    with TestClient(app) as client:
        # 1. GET /health
        t0 = time.perf_counter()
        resp = client.get("/health")
        t1 = time.perf_counter()
        assert resp.status_code == 200
        print(f"GET /health        : {resp.status_code} ({(t1 - t0)*1000:.2f} ms) -> {resp.json()}")

        # 2. POST /predict (In-scope query)
        payload_single = {"text": "what is my credit card balance"}
        t0 = time.perf_counter()
        resp = client.post("/predict", json=payload_single)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        print(f"POST /predict (IS) : {resp.status_code} ({(t1 - t0)*1000:.2f} ms) -> {resp.json()}")

        # 3. POST /predict (OOS query)
        payload_oos = {"text": "who won the 1998 World Cup in France"}
        t0 = time.perf_counter()
        resp = client.post("/predict", json=payload_oos)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        print(f"POST /predict (OOS): {resp.status_code} ({(t1 - t0)*1000:.2f} ms) -> {resp.json()}")

        # 4. POST /predict/batch (3 queries)
        payload_batch = {
            "inputs": [
                "what is my credit card balance",
                "book a table for 4 at an Italian restaurant",
                "who won the 1998 World Cup in France"
            ]
        }
        t0 = time.perf_counter()
        resp = client.post("/predict/batch", json=payload_batch)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        print(f"POST /predict/batch: {resp.status_code} ({(t1 - t0)*1000:.2f} ms for 3 items)")

        # 5. GET /metrics (Prometheus exposition)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        metrics_text = resp.text

        print("\n==========================================")
        print("    Prometheus Exposition Text Snippet    ")
        print("==========================================")
        relevant_lines = [
            line for line in metrics_text.splitlines()
            if any(k in line for k in [
                "http_requests_total",
                "model_predictions_total",
                "model_oos_predictions_total",
                "model_in_scope_predictions_total",
                "model_info",
                "model_inference_duration_seconds_count"
            ]) and not line.startswith("#")
        ]
        for line in relevant_lines[:15]:
            print(f"  {line}")
        print("==========================================")

if __name__ == "__main__":
    test_api_server_sanity()
