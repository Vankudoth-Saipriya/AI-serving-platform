# CI-Based Machine Learning Quality Gate Architecture

This document details the CI-based Machine Learning (ML) Evaluation Quality Gate implemented for the ONNX-backed Intent Serving Platform.

---

## 1. Why Model Quality Requires Automated CI Protection

In traditional software development, CI/CD pipelines use unit and integration tests to verify code correctness, API contracts, and schema validation. However, in Machine Learning systems:

> **A service can be 100% syntactically correct, return HTTP 200 OK responses, and pass all software unit tests, while simultaneously serving a degraded model that makes inaccurate or biased predictions.**

The **ML Evaluation Quality Gate** prevents model performance regressions from being accepted silently into the primary codebase by turning model quality metrics into mandatory, automated CI check gates.

---

## 2. Distinction: Software Tests vs. Model Evaluation vs. ML Quality Gates

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Software Tests (pytest)                                                  │
│    - Verifies endpoint handlers, Pydantic schemas, HTTP 200/400 status codes. │
│    - Fast, deterministic code syntax and logic assertions.                  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Model Evaluation (evaluation/evaluate_model.py)                         │
│    - Runs held-out dataset (5,500 samples) through ONNX Runtime model.       │
│    - Computes Macro F1, Accuracy, OOS Precision, OOS Recall, OOS F1.        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. ML Quality Gate (evaluation/quality_gate.py)                             │
│    - Compares evaluated metrics against committed baseline specifications.   │
│    - Enforces strict tolerance policy bounds.                               │
│    - Exits with code 0 (PASS) or code 1 (FAIL) to block regressed PRs.      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Verified Baseline Metrics & Protected Metric Policy

The platform enforces metric protection against the verified Project 3 baseline:

| Metric | Verified Baseline | Protection Priority | Description |
| :--- | :--- | :--- | :--- |
| **Macro F1** | **88.89%** (0.8889) | **Protected** | Primary class-balanced performance metric across all 151 intent classes. |
| **Overall Accuracy** | **86.18%** (0.8618) | **Protected** | Classification accuracy across all 5,500 held-out test samples. |
| **Out-of-Scope (OOS) F1** | **70.42%** (0.7042) | **Protected** | Harmonic mean of OOS Precision (96.03%) and OOS Recall (55.60%). |
| In-Scope Accuracy | 92.98% (0.9298) | Informational | Accuracy restricted to in-scope intents (classes != 42). |

---

## 4. Regression Policy & Tolerance Margin

### Tolerance Policy: $\Delta = 0.0050$ (0.50 percentage points)

- **Minimum Required Threshold**: `Minimum Threshold = Baseline Value - 0.0050`
  - Macro F1 Minimum: `88.89% - 0.50% = 88.39%` (0.8839)
  - Overall Accuracy Minimum: `86.18% - 0.50% = 85.68%` (0.8568)
  - OOS F1 Minimum: `70.42% - 0.50% = 69.92%` (0.6992)

### Technical Rationale:
Small floating-point precision variations can occur between PyTorch/ONNX Runtime execution environments, OS kernel versions (e.g. Linux x86_64 in GitHub Actions runner vs Windows local execution), or CPU instruction set SIMD extensions (AVX2 vs AVX512). 

A tight absolute tolerance of **0.50 percentage points** allows for minor numerical floating-point variances across execution environments while strictly preventing true model semantic performance regressions.

---

## 5. Artifact Handling & CI Integration Architecture

Large binary model files (`model.onnx` ~256 MB and `model/distilbert_clinc150/model.safetensors` ~268 MB) are explicitly excluded from normal Git history via `.gitignore` to avoid repository bloat.

### CI Artifact Acquisition Options:
1. **GitHub Actions Cache / Artifact Store**: Model binaries are cached or fetched during CI run startup from a versioned bucket (e.g., AWS S3, Google Cloud Storage, or GitHub Release assets).
2. **Local / Staged Path Execution**: When `model/model.onnx` is present on the runner filesystem, `quality_gate.py` executes full live evaluation against `evaluation/datasets/test.parquet`.
3. **Spec Verification Fallback**: When binary model files are excluded, the quality gate validates the metric schema and regression engine against the committed baseline specification ([`evaluation/baseline_metrics.json`](file:///c:/Users/saipr/Desktop/ResumeProjects/ai-serving-platform/evaluation/baseline_metrics.json)).

---

## 6. Execution Commands & Local Usage

### Run Live Quality Gate (Against Local Model & Test Dataset):
```bash
python evaluation/quality_gate.py --evaluate-live --tolerance 0.0050
```

### Run Quality Gate against Metric Output File:
```bash
python evaluation/quality_gate.py --current-metrics-file evaluation/baseline_metrics.json --tolerance 0.0050
```

### Run Regression Failure Simulation Test:
```bash
python scripts/run_failure_simulation.py
```

### Run Full PyTest Suite (Including Quality Gate Unit Tests):
```bash
pytest -v
```
