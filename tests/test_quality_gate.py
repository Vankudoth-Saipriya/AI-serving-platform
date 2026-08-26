import json
import pytest
from pathlib import Path
from evaluation.quality_gate import (
    load_baseline_metrics,
    evaluate_quality_gate,
    DEFAULT_TOLERANCE,
    PROTECTED_METRICS
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = PROJECT_ROOT / "evaluation" / "baseline_metrics.json"
MODEL_METADATA_FILE = PROJECT_ROOT / "model" / "model_metadata.json"

@pytest.fixture
def baseline_metrics_data():
    return {
        "overall_accuracy": 0.8618,
        "macro_f1": 0.8889,
        "in_scope_accuracy": 0.9298,
        "oos_precision": 0.9603,
        "oos_recall": 0.5560,
        "oos_f1": 0.7042
    }

# 1. Baseline metrics schema test
def test_baseline_metrics_schema():
    assert BASELINE_FILE.exists()
    assert MODEL_METADATA_FILE.exists()
    metrics = load_baseline_metrics(BASELINE_FILE)

    assert "macro_f1" in metrics
    assert "overall_accuracy" in metrics
    assert "oos_f1" in metrics
    assert isinstance(metrics["macro_f1"], float)
    assert isinstance(metrics["overall_accuracy"], float)
    assert isinstance(metrics["oos_f1"], float)
    assert metrics["macro_f1"] == pytest.approx(0.8889, abs=1e-4)
    assert metrics["overall_accuracy"] == pytest.approx(0.8618, abs=1e-4)

# 2. Current metrics loading test
def test_current_metrics_loading(tmp_path):
    sample_file = tmp_path / "sample_metrics.json"
    sample_data = {"macro_f1": 0.8889, "overall_accuracy": 0.8618, "oos_f1": 0.7042}
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f)

    loaded = load_baseline_metrics(sample_file)
    assert loaded["macro_f1"] == 0.8889
    assert loaded["overall_accuracy"] == 0.8618
    assert loaded["oos_f1"] == 0.7042

# 3. Metric comparison test
def test_metric_comparison_details(baseline_metrics_data):
    current_metrics = baseline_metrics_data.copy()
    passed, details = evaluate_quality_gate(current_metrics, baseline_metrics_data, tolerance=0.0050)
    assert passed is True
    assert len(details) == 3
    assert details["macro_f1"]["status"] == "PASS"
    assert details["overall_accuracy"]["status"] == "PASS"
    assert details["oos_f1"]["status"] == "PASS"

# 4. Passing quality gate test (Exact match & Slight gain)
def test_passing_quality_gate(baseline_metrics_data):
    # Slight gain
    better_metrics = baseline_metrics_data.copy()
    better_metrics["macro_f1"] += 0.0010
    passed, details = evaluate_quality_gate(better_metrics, baseline_metrics_data, tolerance=0.0050)
    assert passed is True

# 5. Failing quality gate test (Degraded metric)
def test_failing_quality_gate(baseline_metrics_data):
    degraded_metrics = baseline_metrics_data.copy()
    degraded_metrics["macro_f1"] = 0.8500  # Well below 0.8889 - 0.0050
    passed, details = evaluate_quality_gate(degraded_metrics, baseline_metrics_data, tolerance=0.0050)
    assert passed is False
    assert details["macro_f1"]["status"] == "FAIL"

# 6. Tolerance boundary handling test
def test_tolerance_boundary_handling(baseline_metrics_data):
    boundary_metrics = baseline_metrics_data.copy()
    # Right on boundary: 0.8889 - 0.0050 = 0.8839
    boundary_metrics["macro_f1"] = 0.8839
    passed, details = evaluate_quality_gate(boundary_metrics, baseline_metrics_data, tolerance=0.0050)
    assert passed is True
    assert details["macro_f1"]["status"] == "PASS"

    # Just below boundary: 0.8838
    below_boundary = baseline_metrics_data.copy()
    below_boundary["macro_f1"] = 0.8838
    passed_below, details_below = evaluate_quality_gate(below_boundary, baseline_metrics_data, tolerance=0.0050)
    assert passed_below is False
    assert details_below["macro_f1"]["status"] == "FAIL"

# 7. Malformed metric input test
def test_malformed_metric_input(baseline_metrics_data):
    malformed_metrics = baseline_metrics_data.copy()
    malformed_metrics["macro_f1"] = "not-a-number"
    with pytest.raises(ValueError, match="non-numeric"):
        evaluate_quality_gate(malformed_metrics, baseline_metrics_data, tolerance=0.0050)

# 8. Missing required metric test
def test_missing_required_metric(baseline_metrics_data):
    incomplete_metrics = {"overall_accuracy": 0.8618}  # Missing macro_f1 and oos_f1
    with pytest.raises(KeyError, match="missing from current evaluation metrics"):
        evaluate_quality_gate(incomplete_metrics, baseline_metrics_data, tolerance=0.0050)
