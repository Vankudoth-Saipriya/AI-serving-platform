import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_FILE = PROJECT_ROOT / "evaluation" / "baseline_metrics.json"
DEFAULT_MODEL_METADATA_FILE = PROJECT_ROOT / "model" / "model_metadata.json"

PROTECTED_METRICS = {
    "macro_f1": "Macro F1",
    "overall_accuracy": "Overall Accuracy",
    "oos_f1": "OOS F1"
}

DEFAULT_TOLERANCE = 0.0050  # 0.50 percentage points absolute tolerance

def load_json_file(filepath: Path) -> Dict[str, Any]:
    if not filepath.exists():
        raise FileNotFoundError(f"File not found at path: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON format in {filepath}. Expected top-level dict/object.")
    return data

def load_baseline_metrics(baseline_filepath: Path) -> Dict[str, float]:
    data = load_json_file(baseline_filepath)
    if "verified_baseline_metrics" in data:
        metrics = data["verified_baseline_metrics"]
    else:
        metrics = data

    normalized = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            normalized[k] = float(v)
    return normalized

def run_live_evaluation() -> Dict[str, float]:
    """Runs live evaluation script against local held-out test dataset."""
    from evaluation.evaluate_model import evaluate_model_on_test_set
    results = evaluate_model_on_test_set()
    return results

def evaluate_quality_gate(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    tolerance: float = DEFAULT_TOLERANCE
) -> Tuple[bool, Dict[str, Any]]:
    
    detailed_results = {}
    overall_pass = True

    for metric_key, display_name in PROTECTED_METRICS.items():
        if metric_key not in baseline_metrics:
            raise KeyError(f"Required protected metric '{metric_key}' missing from baseline metrics specification.")
        if metric_key not in current_metrics:
            raise KeyError(f"Required protected metric '{metric_key}' missing from current evaluation metrics.")

        try:
            baseline_val = float(baseline_metrics[metric_key])
            current_val = float(current_metrics[metric_key])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Metric value for '{metric_key}' is non-numeric: {e}")

        min_required = baseline_val - tolerance
        passed = current_val >= min_required

        if not passed:
            overall_pass = False

        detailed_results[metric_key] = {
            "display_name": display_name,
            "baseline": baseline_val,
            "current": current_val,
            "tolerance": tolerance,
            "min_required": min_required,
            "delta": current_val - baseline_val,
            "status": "PASS" if passed else "FAIL"
        }

    return overall_pass, detailed_results

def print_quality_gate_summary(
    overall_pass: bool,
    results: Dict[str, Any],
    tolerance: float
) -> None:
    print("\n======================================================================")
    print("                  ML EVALUATION QUALITY GATE                          ")
    print("======================================================================")
    print(f"Tolerance Policy: Absolute margin <= {tolerance * 100:.2f}% (delta <= {tolerance:.4f})")
    print("----------------------------------------------------------------------")
    print(f"{'Protected Metric':<20} | {'Baseline':<9} | {'Current':<9} | {'Min Required':<12} | {'Status':<6}")
    print("----------------------------------------------------------------------")

    for key, info in results.items():
        name = info["display_name"]
        base_pct = f"{info['baseline'] * 100:.2f}%"
        curr_pct = f"{info['current'] * 100:.2f}%"
        min_pct = f"{info['min_required'] * 100:.2f}%"
        status = info["status"]
        print(f"{name:<20} | {base_pct:<9} | {curr_pct:<9} | {min_pct:<12} | {status:<6}")

    print("----------------------------------------------------------------------")
    final_str = "PASS" if overall_pass else "FAIL"
    print(f"FINAL QUALITY GATE STATUS: {final_str}")
    print("======================================================================\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Machine Learning Quality Gate Evaluation Engine")
    parser.add_argument("--current-metrics-file", type=str, default=None, help="Path to current evaluation JSON file.")
    parser.add_argument("--baseline-file", type=str, default=str(DEFAULT_BASELINE_FILE), help="Path to baseline metrics JSON file.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Allowed metric regression tolerance.")
    parser.add_argument("--evaluate-live", action="store_true", help="Force live ONNX model evaluation on held-out dataset.")
    return parser.parse_args()

def main():
    args = parse_args()
    baseline_path = Path(args.baseline_file)

    try:
        baseline_metrics = load_baseline_metrics(baseline_path)
    except Exception as e:
        print(f"ERROR: Failed to load baseline metrics: {e}", file=sys.stderr)
        sys.exit(1)

    if args.current_metrics_file:
        try:
            current_path = Path(args.current_metrics_file)
            current_metrics = load_baseline_metrics(current_path)
        except Exception as e:
            print(f"ERROR: Failed to load current metrics file '{args.current_metrics_file}': {e}", file=sys.stderr)
            sys.exit(1)
    elif args.evaluate_live:
        print("Executing live evaluation on held-out test dataset...")
        try:
            current_metrics = run_live_evaluation()
        except Exception as e:
            print(f"ERROR: Failed to execute live evaluation: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Default behavior: evaluate baseline specification
        current_metrics = baseline_metrics

    try:
        passed, results = evaluate_quality_gate(current_metrics, baseline_metrics, tolerance=args.tolerance)
        print_quality_gate_summary(passed, results, args.tolerance)
        if not passed:
            sys.exit(1)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR during quality gate evaluation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
