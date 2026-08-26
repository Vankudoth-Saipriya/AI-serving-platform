# AI Model Serving, Evaluation & Observability Platform

Production-grade Transformer model inference, ONNX optimization, vectorized batch serving, Prometheus model observability, Locust load testing, and CI/CD quality gate platform for high-throughput intent classification and out-of-scope (OOS) detection.

---

## 🚀 Key AI Engineering & MLOps Highlights

- **ONNX Model Optimization**: 1.92x inference latency reduction via ONNX Runtime CPU execution (`35.37 ms` PyTorch $\rightarrow$ `18.42 ms` ONNX) with $5.48 \times 10^{-6}$ max absolute logit equivalence.
- **FastAPI Vectorized Batch Serving**: True 2D matrix tensor batch inference (`POST /predict/batch`) avoiding loop overhead.
- **AI-Aware Prometheus Observability**: Custom metrics tracking model confidence distribution, OOS prediction ratio, in-scope vs out-of-scope rates, and pure ONNX runtime execution latency vs HTTP overhead.
- **Auto-Provisioned Grafana Dashboard**: 10-panel reproducible dashboard answering *"Is the AI model behaving correctly under load?"*.
- **Empirical Load Testing Envelope**: Locust load testing characterizing throughput saturation plateau (**~56 predictions/sec**, 0% error rate across 10, 50, 100 concurrent users).
- **CI-Based ML Evaluation Quality Gate**: Machine-readable evaluation engine enforcing mandatory regression protection (Macro F1: 88.89%, Accuracy: 86.18%, OOS F1: 70.42%) in GitHub Actions.

---

## 🏗️ System Architecture

```
[ Client / Locust Load Tester ]
               │
               ▼
┌──────────────────────────────────────────┐
│      FastAPI Intent Serving Service      │  (Port 8000)
│  ┌────────────────────────────────────┐  │
│  │ ModelContainer Singleton           │  │
│  │ (ONNX Runtime + HF Tokenizer)      │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ Prometheus Client Registry         │  │
│  └────────────────────────────────────┘  │
└──────────────────┬───────────────────────┘
                   │ Scrape (5s)
                   ▼
┌──────────────────────────────────────────┐
│            Prometheus Server             │  (Port 9090)
└──────────────────┬───────────────────────┘
                   │ PromQL Queries
                   ▼
┌──────────────────────────────────────────┐
│    Grafana AI Observability Dashboard    │  (Port 3000)
└──────────────────────────────────────────┘
```

---

## ⚡ Quick Start & Deployment Guide

### 1. Prerequisites & Model Artifact Setup
Large binary model files (`model.onnx` ~256 MB and `model.safetensors` ~268 MB) are explicitly excluded from Git history via `.gitignore`. 

Before starting the service, ensure the ONNX model binary and tokenizer files are present in `model/`:

```
model/
├── distilbert_clinc150/
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
├── intent_names.json
├── model_metadata.json
└── model.onnx                  <-- Required ONNX Model Binary
```

*Note: If `model.onnx` is missing at startup, the container emits an explicit critical error log and fails cleanly.*

---

### 2. Docker Compose Deployment (Recommended)

To launch the full stack (FastAPI + Prometheus + Grafana):

```bash
# 1. Build and start all services in background
docker compose up -d --build

# 2. View running container logs
docker compose logs -f api

# 3. Teardown stack
docker compose down
```

#### Exposed Service Ports:
- **FastAPI Model Inference Service**: `http://localhost:8000`
- **Interactive Swagger OpenAPI Docs**: `http://localhost:8000/docs`
- **Prometheus Telemetry Server**: `http://localhost:9090`
- **Grafana AI Observability Dashboard**: `http://localhost:3000` (User: `admin`, Password: `admin`)

---

### 3. Local Development Setup (Without Docker)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start FastAPI server locally
$env:PYTHONPATH="."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 📊 API Endpoint Specification

### `POST /predict`
Single text intent prediction.

**Request**:
```json
{
  "text": "what is my account balance?"
}
```

**Response**:
```json
{
  "intent": "balance",
  "confidence": 0.9984,
  "is_oos": false,
  "model_version": "1.0.0"
}
```

---

### `POST /predict/batch`
True 2D matrix vectorized batch intent prediction.

**Request**:
```json
{
  "inputs": [
    "transfer 100 dollars to savings",
    "what is the capital of Mars?",
    "schedule a doctor appointment"
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "intent": "transfer",
      "confidence": 0.9972,
      "is_oos": false,
      "model_version": "1.0.0"
    },
    {
      "intent": "oos",
      "confidence": 0.9845,
      "is_oos": true,
      "model_version": "1.0.0"
    },
    {
      "intent": "schedule_appointment",
      "confidence": 0.9910,
      "is_oos": false,
      "model_version": "1.0.0"
    }
  ],
  "batch_size": 3
}
```

---

### Additional Service Endpoints:
- `GET /health`: Service health and model container load readiness check.
- `GET /model/info`: Active model metadata, architecture details, parameter counts, and versioning.
- `GET /metrics`: Standard Prometheus metrics exposition endpoint.

---

## 📈 Performance Benchmarks & Observability

### ONNX Runtime CPU Speedup (PyTorch vs ONNX)
- **Batch Size 1**: `35.37 ms` $\rightarrow$ `18.42 ms` (**1.92x Speedup**)
- **Batch Size 8**: `158.52 ms` $\rightarrow$ `135.73 ms` (**1.17x Speedup**)
- **Batch Size 32**: `602.81 ms` $\rightarrow$ `153.61 ms` (**1.09x Speedup**)
- **Numerical Equivalence**: Max absolute logit difference = $5.48 \times 10^{-6}$ (100% prediction match rate).

### Locust Load Testing Profile
- **10 Concurrent Users**: 24.54 API RPS | 30.6 preds/sec | p50: 82 ms | p95: 340 ms | 0% error rate
- **50 Concurrent Users**: 51.75 API RPS | 56.4 preds/sec | p50: 460 ms | p95: 2,000 ms | 0% error rate
- **100 Concurrent Users**: 44.81 API RPS | 56.0 preds/sec | p50: 1,500 ms | p95: 3,800 ms | 0% error rate
- **System Observation**: Throughput plateaued around **~56 predictions/sec** while latency increased under higher concurrency, maintaining 0.00% error rate.

---

## 🛡️ Machine Learning Quality Gate & Testing

The platform includes a machine-readable quality gate (`evaluation/quality_gate.py`) protecting model quality in CI/CD:

```bash
# Run full unit and integration test suite
$env:PYTHONPATH="."
pytest -v

# Run live ML quality gate evaluation
python evaluation/quality_gate.py --evaluate-live --tolerance 0.0050

# Run regression failure simulation test
python scripts/run_failure_simulation.py

# Run Locust load test suite
python scripts/run_load_test.py
```

### Protected Quality Baseline Metrics:
- **Macro F1**: Baseline `88.89%` (Min threshold: `88.39%`)
- **Overall Accuracy**: Baseline `86.18%` (Min threshold: `85.68%`)
- **OOS F1 Score**: Baseline `70.42%` (Min threshold: `69.92%`)
