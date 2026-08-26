import os
import sys
import time
import json
import subprocess
import requests
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCUST_FILE = PROJECT_ROOT / "load_test" / "locustfile.py"
OUTPUT_JSON_PATH = PROJECT_ROOT / "evaluation" / "load_test_results.json"

CONCURRENCY_LEVELS = [10, 50, 100]
RUN_TIME_SECONDS = 15
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

def wait_for_server(timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{SERVER_URL}/health", timeout=2)
            if resp.status_code == 200 and resp.json().get("model_loaded"):
                return True
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server at {SERVER_URL} failed to start.")

def scrape_prometheus_metrics():
    try:
        resp = requests.get(f"{SERVER_URL}/metrics", timeout=5)
        if resp.status_code == 200:
            lines = resp.text.splitlines()
            metrics = {}
            for line in lines:
                if line.startswith("model_inference_duration_seconds_sum"):
                    metrics["onnx_sum_sec"] = float(line.split()[-1])
                elif line.startswith("model_inference_duration_seconds_count"):
                    metrics["onnx_count"] = float(line.split()[-1])
                elif line.startswith("http_request_duration_seconds_sum"):
                    metrics["http_sum_sec"] = float(line.split()[-1])
                elif line.startswith("http_request_duration_seconds_count"):
                    metrics["http_count"] = float(line.split()[-1])
            
            if metrics.get("onnx_count", 0) > 0:
                metrics["avg_onnx_ms"] = round((metrics["onnx_sum_sec"] / metrics["onnx_count"]) * 1000, 2)
            if metrics.get("http_count", 0) > 0:
                metrics["avg_http_ms"] = round((metrics["http_sum_sec"] / metrics["http_count"]) * 1000, 2)
            return metrics
    except Exception:
        pass
    return {}

def run_single_level_benchmark(users):
    spawn_rate = min(users, 20)
    csv_prefix = PROJECT_ROOT / "load_test" / f"locust_run_u{users}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    # Start fresh server per level
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", SERVER_HOST, "--port", str(SERVER_PORT), "--log-level", "warning"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        wait_for_server()

        cmd = [
            sys.executable, "-m", "locust",
            "-f", str(LOCUST_FILE),
            "--headless",
            "-u", str(users),
            "-r", str(spawn_rate),
            "--run-time", f"{RUN_TIME_SECONDS}s",
            "--host", SERVER_URL,
            "--csv", str(csv_prefix),
            "--stop-timeout", "5"
        ]

        t0 = time.time()
        locust_res = subprocess.run(cmd, capture_output=True, text=True)
        t1 = time.time()

        time.sleep(1.0)
        prom_metrics = scrape_prometheus_metrics()

        stats_csv = Path(f"{csv_prefix}_stats.csv")
        total_reqs = 0
        failed_reqs = 0
        p50_ms = 0.0
        p95_ms = 0.0
        p99_ms = 0.0
        max_ms = 0.0
        rps = 0.0
        single_reqs = 0
        batch_reqs = 0

        if stats_csv.exists():
            df = pd.read_csv(stats_csv)
            for _, row in df.iterrows():
                name = str(row["Name"])
                if name == "Aggregated":
                    total_reqs = int(row.get("Request Count", 0))
                    failed_reqs = int(row.get("Failure Count", 0))
                    p50_ms = round(float(row.get("50%", 0)), 2)
                    p95_ms = round(float(row.get("95%", 0)), 2)
                    p99_ms = round(float(row.get("99%", 0)), 2)
                    max_ms = round(float(row.get("Max Response Time", 0)), 2)
                    rps = round(float(row.get("Requests/s", 0)), 2)
                elif "/predict (single)" in name:
                    single_reqs = int(row.get("Request Count", 0))
                elif "/predict/batch" in name:
                    batch_reqs = int(row.get("Request Count", 0))

        # Cleanup CSVs
        for ext in ["_stats.csv", "_failures.csv", "_stats_history.csv"]:
            p = Path(f"{csv_prefix}{ext}")
            if p.exists():
                try: p.unlink()
                except Exception: pass

        error_rate = round((failed_reqs / total_reqs * 100.0), 2) if total_reqs > 0 else 0.0
        total_predictions = (single_reqs * 1) + (batch_reqs * 5)
        prediction_qps = round(total_predictions / (t1 - t0), 1)

        level_results = {
            "concurrent_users": users,
            "duration_seconds": RUN_TIME_SECONDS,
            "total_requests": total_reqs,
            "successful_requests": total_reqs - failed_reqs,
            "failed_requests": failed_reqs,
            "error_rate_percent": error_rate,
            "api_throughput_rps": rps,
            "prediction_throughput_qps": prediction_qps,
            "latency_metrics_ms": {
                "median_p50": p50_ms,
                "p95": p95_ms,
                "p99": p99_ms,
                "max": max_ms
            },
            "latency_breakdown_ms": {
                "avg_http_end_to_end": prom_metrics.get("avg_http_ms", 0.0),
                "avg_onnx_runtime_pure": prom_metrics.get("avg_onnx_ms", 0.0),
                "overhead_ms": round(max(0, prom_metrics.get("avg_http_ms", 0.0) - prom_metrics.get("avg_onnx_ms", 0.0)), 2)
            }
        }
        return level_results

    finally:
        server_process.terminate()
        server_process.wait()

def run_all_load_tests():
    print("==========================================")
    print("    Locust AI Model Load Testing Suite    ")
    print("==========================================")

    all_data = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "server_host": SERVER_HOST,
            "server_port": SERVER_PORT,
            "inference_backend": "ONNX Runtime (CPUExecutionProvider)",
            "onnx_threads": {"intra_op": 2, "inter_op": 2}
        },
        "concurrency_levels": {}
    }

    for users in CONCURRENCY_LEVELS:
        print(f"\n--- Testing Concurrency Level: {users} Concurrent Users ---")
        res = run_single_level_benchmark(users)
        all_data["concurrency_levels"][f"users_{users}"] = res
        
        print(f"  Requests  : {res['total_requests']} total ({res['failed_requests']} failed, Error Rate: {res['error_rate_percent']}%)")
        print(f"  API RPS   : {res['api_throughput_rps']} reqs/sec")
        print(f"  Pred QPS  : {res['prediction_throughput_qps']} predictions/sec")
        print(f"  p50 / p95 : {res['latency_metrics_ms']['median_p50']} ms / {res['latency_metrics_ms']['p95']} ms")
        print(f"  ONNX Pure : {res['latency_breakdown_ms']['avg_onnx_runtime_pure']} ms (avg)")

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nSaved load test results to: {OUTPUT_JSON_PATH}")
    print("==========================================")
    print("   Load Testing Execution Completed!      ")
    print("==========================================")

if __name__ == "__main__":
    run_all_load_tests()
