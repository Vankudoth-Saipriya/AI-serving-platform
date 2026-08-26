import json
import yaml
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMETHEUS_CONFIG_PATH = PROJECT_ROOT / "monitoring" / "prometheus.yml"
GRAFANA_DS_CONFIG_PATH = PROJECT_ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yaml"
GRAFANA_DASHBOARD_PROVIDER_PATH = PROJECT_ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboards.yaml"
GRAFANA_DASHBOARD_JSON_PATH = PROJECT_ROOT / "monitoring" / "grafana" / "dashboards" / "ai_model_observability.json"

def test_prometheus_yaml_config():
    assert PROMETHEUS_CONFIG_PATH.exists()
    with open(PROMETHEUS_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "scrape_configs" in cfg
    targets = cfg["scrape_configs"][0]["static_configs"][0]["targets"]
    assert "127.0.0.1:8000" in targets or "localhost:8000" in targets or "app:8000" in targets

def test_grafana_provisioning_yaml():
    assert GRAFANA_DS_CONFIG_PATH.exists()
    with open(GRAFANA_DS_CONFIG_PATH, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)
    assert ds_cfg.get("apiVersion") == 1
    assert len(ds_cfg.get("datasources", [])) >= 1

    assert GRAFANA_DASHBOARD_PROVIDER_PATH.exists()
    with open(GRAFANA_DASHBOARD_PROVIDER_PATH, "r", encoding="utf-8") as f:
        dp_cfg = yaml.safe_load(f)
    assert len(dp_cfg.get("providers", [])) >= 1

def test_grafana_dashboard_json_structure():
    assert GRAFANA_DASHBOARD_JSON_PATH.exists()
    with open(GRAFANA_DASHBOARD_JSON_PATH, "r", encoding="utf-8") as f:
        dashboard = json.load(f)

    assert dashboard.get("title") == "AI Inference & Model Observability Dashboard"
    panels = dashboard.get("panels", [])
    assert len(panels) == 10

    panel_titles = [p.get("title", "") for p in panels]
    assert any("Active Model Info" in t for t in panel_titles)
    assert any("HTTP Error Rate" in t for t in panel_titles)
    assert any("Out-of-Scope" in t for t in panel_titles)
    assert any("Prediction Throughput" in t for t in panel_titles)
    assert any("API Request Rate" in t for t in panel_titles)
    assert any("End-to-End API Request Latency" in t for t in panel_titles)
    assert any("ONNX Model Pure Inference Latency" in t for t in panel_titles)
    assert any("Prediction Traffic Distribution" in t for t in panel_titles)
    assert any("Top Predicted Intent Classes" in t for t in panel_titles)
    assert any("Prediction Confidence Score Distribution" in t for t in panel_titles)

def test_grafana_promql_query_validity():
    with open(GRAFANA_DASHBOARD_JSON_PATH, "r", encoding="utf-8") as f:
        dashboard = json.load(f)

    promql_exprs = []
    for panel in dashboard.get("panels", []):
        for target in panel.get("targets", []):
            promql_exprs.append(target.get("expr", ""))

    full_expr_str = " ".join(promql_exprs)

    # Check key PromQL metric references
    assert "http_requests_total" in full_expr_str
    assert "http_request_duration_seconds_bucket" in full_expr_str
    assert "model_inference_duration_seconds_bucket" in full_expr_str
    assert "model_predictions_total" in full_expr_str
    assert "model_prediction_confidence_bucket" in full_expr_str
    assert "model_oos_predictions_total" in full_expr_str
    assert "model_in_scope_predictions_total" in full_expr_str
    assert "model_info" in full_expr_str

    # Ensure histogram_quantile is used for latency calculations
    assert "histogram_quantile(0.50" in full_expr_str
    assert "histogram_quantile(0.95" in full_expr_str
    assert "histogram_quantile(0.99" in full_expr_str
