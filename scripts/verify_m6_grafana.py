import os
import sys
import time
import json
import yaml
import subprocess
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMETHEUS_CONFIG_PATH = PROJECT_ROOT / "monitoring" / "prometheus.yml"
GRAFANA_DS_CONFIG_PATH = PROJECT_ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yaml"
GRAFANA_DASHBOARD_PROVIDER_PATH = PROJECT_ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboards.yaml"
GRAFANA_DASHBOARD_JSON_PATH = PROJECT_ROOT / "monitoring" / "grafana" / "dashboards" / "ai_model_observability.json"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

EXPECTED_PANEL_COUNT = 10
REQUIRED_METRIC_PROMPT_TARGETS = [
    "model_info",
    "http_requests_total",
    "http_request_duration_seconds_bucket",
    "model_inference_duration_seconds_bucket",
    "model_predictions_total",
    "model_prediction_confidence_bucket",
    "model_oos_predictions_total",
    "model_in_scope_predictions_total"
]

def verify_provisioning_files():
    print("\n1. Verifying Prometheus & Grafana Provisioning Files...")

    # Check Prometheus config
    assert PROMETHEUS_CONFIG_PATH.exists(), f"Missing {PROMETHEUS_CONFIG_PATH}"
    with open(PROMETHEUS_CONFIG_PATH, "r", encoding="utf-8") as f:
        prom_cfg = yaml.safe_load(f)
    assert "scrape_configs" in prom_cfg, "Prometheus config missing scrape_configs"
    print("  [OK] monitoring/prometheus.yml parsed successfully.")

    # Check Grafana Datasource config
    assert GRAFANA_DS_CONFIG_PATH.exists(), f"Missing {GRAFANA_DS_CONFIG_PATH}"
    with open(GRAFANA_DS_CONFIG_PATH, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)
    assert ds_cfg.get("apiVersion") == 1, "Invalid Grafana datasource apiVersion"
    print("  [OK] monitoring/grafana/provisioning/datasources/prometheus.yaml parsed successfully.")

    # Check Grafana Dashboard Provider config
    assert GRAFANA_DASHBOARD_PROVIDER_PATH.exists(), f"Missing {GRAFANA_DASHBOARD_PROVIDER_PATH}"
    with open(GRAFANA_DASHBOARD_PROVIDER_PATH, "r", encoding="utf-8") as f:
        dp_cfg = yaml.safe_load(f)
    assert "providers" in dp_cfg, "Missing providers in dashboards.yaml"
    print("  [OK] monitoring/grafana/provisioning/dashboards/dashboards.yaml parsed successfully.")

    # Check Dashboard JSON
    assert GRAFANA_DASHBOARD_JSON_PATH.exists(), f"Missing {GRAFANA_DASHBOARD_JSON_PATH}"
    with open(GRAFANA_DASHBOARD_JSON_PATH, "r", encoding="utf-8") as f:
        dash_json = json.load(f)

    panels = dash_json.get("panels", [])
    assert len(panels) == EXPECTED_PANEL_COUNT, f"Expected {EXPECTED_PANEL_COUNT} panels, found {len(panels)}"

    promql_expressions = []
    for panel in panels:
        targets = panel.get("targets", [])
        for target in targets:
            expr = target.get("expr", "")
            promql_expressions.append(expr)

    all_promql_str = " ".join(promql_expressions)
    for req_metric in REQUIRED_METRIC_PROMPT_TARGETS:
        assert req_metric in all_promql_str, f"Required metric '{req_metric}' missing from Grafana dashboard PromQL queries!"

    print(f"  [OK] monitoring/grafana/dashboards/ai_model_observability.json parsed ({len(panels)} panels verified).")

def verify_live_prometheus_metrics():
    print("\n2. Testing FastAPI Metrics Endpoint Live Integration...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", SERVER_HOST, "--port", str(SERVER_PORT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        # Wait for server
        for _ in range(30):
            try:
                r = requests.get(f"{SERVER_URL}/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)

        # Generate sample inference traffic
        print("  Generating sample inference traffic...")
        requests.post(f"{SERVER_URL}/predict", json={"text": "what is my account balance?"})
        requests.post(f"{SERVER_URL}/predict", json={"text": "asdfghjkl zxcvbnm qwertyuiop"}) # OOS query
        requests.post(f"{SERVER_URL}/predict/batch", json={"inputs": ["transfer 100 dollars", "cancel my card", "what is the weather today in Mars"]})

        # Scrape metrics
        resp = requests.get(f"{SERVER_URL}/metrics", timeout=5)
        assert resp.status_code == 200, f"Expected HTTP 200 from /metrics, got {resp.status_code}"
        metrics_text = resp.text

        for req_metric in REQUIRED_METRIC_PROMPT_TARGETS:
            assert req_metric in metrics_text, f"Metric '{req_metric}' not found in live /metrics endpoint response!"

        print("  [OK] All 8 required PromQL metric targets actively populated on /metrics endpoint!")

    finally:
        server_process.terminate()
        server_process.wait()
        print("  Uvicorn test server stopped cleanly.")

def main():
    print("==================================================")
    print("  Milestone M6: Grafana AI Observability Verification ")
    print("==================================================")
    verify_provisioning_files()
    verify_live_prometheus_metrics()
    print("\n==================================================")
    print("  M6 Verification SUCCESSFUL! All checks passed.  ")
    print("==================================================")

if __name__ == "__main__":
    main()
