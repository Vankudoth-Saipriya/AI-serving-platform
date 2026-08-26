# AI Inference & Model Observability Architecture

This document details the observability architecture, metric collection pipeline, PromQL query catalog, and Grafana dashboard design for the ONNX-backed Intent Serving Platform.

---

## 1. Overview & Observability Rationale

Traditional infrastructure monitoring answers:  
> *"Is the server process running and are CPU/memory levels normal?"*

**AI Model Observability** expands this boundary to continuously answer:  
> *"Is the AI inference model behaving correctly under dynamic traffic, and are prediction semantics, confidence distributions, and out-of-scope (OOS) detection remaining stable?"*

### Key Operational Questions Addressed:
1. **Model Confidence Drift**: Is the model losing prediction confidence over time?
2. **Out-of-Scope (OOS) Traffic Spikes**: Are users submitting unhandled intent queries beyond the 150 target classes?
3. **Inference Execution Overhead**: What fraction of total end-to-end HTTP latency is spent executing pure ONNX Runtime tensor computations versus HTTP/Pydantic serialization overhead?
4. **Vectorized Batching Efficiency**: How does single inference latency compare against batched inference execution under load?

---

## 2. Telemetry & Metric Pipeline Architecture

```
[ HTTP Clients / Locust Load Generator ]
               │
               ▼
┌──────────────────────────────────────────┐
│      FastAPI Intent Serving Service      │
│  ┌────────────────────────────────────┐  │
│  │ prometheus_metrics_middleware      │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ ONNX Runtime Engine (DistilBERT)   │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ Prometheus Client Registry (/metrics) ││
│  └────────────────────────────────────┘  │
└──────────────────┬───────────────────────┘
                   │ Scrape (5s interval)
                   ▼
┌──────────────────────────────────────────┐
│            Prometheus Server             │
│        (Time-Series Data Store)          │
└──────────────────┬───────────────────────┘
                   │ PromQL Queries
                   ▼
┌──────────────────────────────────────────┐
│    Grafana AI Observability Dashboard    │
└──────────────────────────────────────────┘
```

---

## 3. Prometheus Metric Catalog

| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `http_requests_total` | Counter | `endpoint`, `method`, `status_code` | Total HTTP API request count. |
| `http_request_duration_seconds` | Histogram | `endpoint` | End-to-end API HTTP response latency. |
| `model_inference_duration_seconds` | Histogram | `inference_type` (`single`, `batch`) | Pure ONNX Runtime execution latency. |
| `model_predictions_total` | Counter | `intent` | Total predictions generated per intent class. |
| `model_prediction_confidence` | Histogram | None | Distribution of prediction confidence scores. |
| `model_oos_predictions_total` | Counter | None | Out-of-scope prediction count (`class_id == 42`). |
| `model_in_scope_predictions_total` | Counter | None | In-scope prediction count (`class_id != 42`). |
| `model_info` | Gauge | `model_name`, `model_version`, `task`, `backend` | Active checkpoint metadata. |

---

## 4. Grafana Dashboard Panels & PromQL Query Catalog

The Grafana dashboard ([`monitoring/grafana/dashboards/ai_model_observability.json`](file:///c:/Users/saipr/Desktop/ResumeProjects/ai-serving-platform/monitoring/grafana/dashboards/ai_model_observability.json)) organizes 10 dedicated panels:

### Panel 1: Active Model Info & Version
- **Type**: Stat / Banner
- **PromQL**: `model_info`
- **Purpose**: Displays active checkpoint metadata (`distilbert_clinc150 v1.0.0 (onnxruntime)`).

### Panel 2: HTTP Error Rate (%)
- **Type**: Stat (Gauge)
- **PromQL**: `(sum(rate(http_requests_total{status_code=~"4..|5.."}[1m])) / sum(rate(http_requests_total[1m]))) * 100 or vector(0)`
- **Purpose**: Monitors API reliability (Green <1%, Yellow 1-5%, Red >5%).

### Panel 3: Out-of-Scope (OOS) Rate (%)
- **Type**: Stat (Gauge)
- **PromQL**: `(sum(rate(model_oos_predictions_total[1m])) / sum(rate(model_predictions_total[1m]))) * 100 or vector(0)`
- **Purpose**: Tracks OOS prediction ratio relative to total prediction volume.

### Panel 4: Total Prediction Throughput (Predictions/sec)
- **Type**: Stat
- **PromQL**: `sum(rate(model_predictions_total[1m])) or vector(0)`
- **Purpose**: Displays total prediction throughput across single and batch requests.

### Panel 5: API Request Rate by Endpoint
- **Type**: Time Series
- **PromQL**: `sum(rate(http_requests_total[1m])) by (endpoint)`
- **Purpose**: Differentiates `/predict`, `/predict/batch`, and `/health` call rates.

### Panel 6: End-to-End API Request Latency Quantiles
- **Type**: Time Series
- **PromQL (p50)**: `histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[1m])) by (le)) * 1000`
- **PromQL (p95)**: `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[1m])) by (le)) * 1000`
- **PromQL (p99)**: `histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[1m])) by (le)) * 1000`
- **Purpose**: Tracks system response tail latencies in milliseconds.

### Panel 7: ONNX Model Pure Inference Latency (Single vs Batch p95)
- **Type**: Time Series
- **PromQL (Single)**: `histogram_quantile(0.95, sum(rate(model_inference_duration_seconds_bucket{inference_type="single"}[1m])) by (le)) * 1000`
- **PromQL (Batch)**: `histogram_quantile(0.95, sum(rate(model_inference_duration_seconds_bucket{inference_type="batch"}[1m])) by (le)) * 1000`
- **Purpose**: Isolates raw ONNX execution latency from HTTP/network overhead.

### Panel 8: Prediction Traffic Distribution (In-Scope vs Out-of-Scope)
- **Type**: Stacked Time Series
- **PromQL (In-Scope)**: `sum(rate(model_in_scope_predictions_total[1m]))`
- **PromQL (OOS)**: `sum(rate(model_oos_predictions_total[1m]))`
- **Purpose**: Visualizes prediction composition over time.

### Panel 9: Top 5 Predicted Intent Classes
- **Type**: Horizontal Bar Chart
- **PromQL**: `topk(5, sum(rate(model_predictions_total[1m])) by (intent))`
- **Purpose**: Surfaces dominant user intent distribution without high-cardinality label explosion.

### Panel 10: Prediction Confidence Score Distribution
- **Type**: Time Series
- **PromQL**: `sum(rate(model_prediction_confidence_bucket[1m])) by (le)`
- **Purpose**: Visualizes model prediction certainty buckets (0.1 to 1.0).

---

## 5. Load-Test Observation Summary (Locust Baseline Metrics)

Empirical load testing under controlled concurrency levels produced the following performance profile:

| Concurrency Level | Total Requests | Failure Rate | API Throughput (RPS) | Prediction Throughput (QPS) | Median Latency (p50) | p95 Latency | Pure ONNX Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10 Users** | **346** | **0.00%** | **24.54 reqs/sec** | **30.6 preds/sec** | **82.0 ms** | **340.0 ms** | **306.2 ms** |
| **50 Users** | **733** | **0.00%** | **51.75 reqs/sec** | **56.4 preds/sec** | **460.0 ms** | **2,000.0 ms** | **1,701.9 ms** |
| **100 Users** | **696** | **0.00%** | **44.81 reqs/sec** | **56.0 preds/sec** | **1,500.0 ms** | **3,800.0 ms** | **2,217.3 ms** |

### Key System Characteristics:
- **Throughput Saturation Pattern**: Throughput plateaued around **56 predictions/sec** while response latency increased under higher concurrency (p50 increased from 82 ms at 10 users to 1,500 ms at 100 users).
- **Service Reliability**: 0.00% error rate maintained across all concurrency levels.

---

## 6. Docker & Deployment Integration Setup

The provisioning directory layout is structured for seamless integration with Docker Compose:

```
monitoring/
├── prometheus.yml
└── grafana/
    ├── dashboards/
    │   └── ai_model_observability.json
    └── provisioning/
        ├── dashboards/
        │   └── dashboards.yaml
        └── datasources/
            └── prometheus.yaml
```

When deployed in Docker Compose, mount:
- `monitoring/prometheus.yml` $\rightarrow$ `/etc/prometheus/prometheus.yml`
- `monitoring/grafana/provisioning/` $\rightarrow$ `/etc/grafana/provisioning/`
- `monitoring/grafana/dashboards/` $\rightarrow$ `/var/lib/grafana/dashboards/`
