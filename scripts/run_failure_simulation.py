import sys
import json
import tempfile
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HEALTHY_METRICS = {
    "overall_accuracy": 0.8618,
    "macro_f1": 0.8889,
    "in_scope_accuracy": 0.9298,
    "oos_precision": 0.9603,
    "oos_recall": 0.5560,
    "oos_f1": 0.7042
}

DEGRADED_METRICS = {
    "overall_accuracy": 0.8200,  # Below 0.8618 - 0.0050 (0.8568)
    "macro_f1": 0.8150,         # Below 0.8889 - 0.0050 (0.8839)
    "in_scope_accuracy": 0.8800,
    "oos_precision": 0.9000,
    "oos_recall": 0.4000,
    "oos_f1": 0.5500           # Below 0.7042 - 0.0050 (0.6992)
}

def main():
    print("======================================================================")
    print("        ML QUALITY GATE REGRESSION FAILURE SIMULATION TEST            ")
    print("======================================================================")

    quality_gate_script = PROJECT_ROOT / "evaluation" / "quality_gate.py"

    with tempfile.TemporaryDirectory() as tmpdir:
        healthy_file = Path(tmpdir) / "healthy_metrics.json"
        degraded_file = Path(tmpdir) / "degraded_metrics.json"

        with open(healthy_file, "w", encoding="utf-8") as f:
            json.dump(HEALTHY_METRICS, f, indent=2)

        with open(degraded_file, "w", encoding="utf-8") as f:
            json.dump(DEGRADED_METRICS, f, indent=2)

        print("\n1. Testing Quality Gate with HEALTHY Metrics (Expect Exit Code 0 - PASS)...")
        res_pass = subprocess.run(
            [sys.executable, str(quality_gate_script), "--current-metrics-file", str(healthy_file)],
            capture_output=True,
            text=True
        )
        print(res_pass.stdout)
        assert res_pass.returncode == 0, f"Expected exit code 0 for healthy metrics, got {res_pass.returncode}"
        print("[OK] Healthy metrics test passed successfully (Exit code 0).")

        print("\n2. Testing Quality Gate with DEGRADED Metrics (Expect Exit Code 1 - FAIL)...")
        res_fail = subprocess.run(
            [sys.executable, str(quality_gate_script), "--current-metrics-file", str(degraded_file)],
            capture_output=True,
            text=True
        )
        print(res_fail.stdout)
        assert res_fail.returncode == 1, f"Expected exit code 1 for degraded metrics, got {res_fail.returncode}"
        print("[OK] Degraded metrics regression detection passed successfully (Exit code 1 - FAIL).")

    print("======================================================================")
    print("   ML Quality Gate Failure Simulation Verification SUCCESSFUL!        ")
    print("======================================================================")

if __name__ == "__main__":
    main()
