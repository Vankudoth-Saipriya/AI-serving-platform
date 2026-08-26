import os
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "distilbert_clinc150"
ONNX_MODEL_PATH = PROJECT_ROOT / "model" / "model.onnx"
OUTPUT_BENCHMARK_PATH = PROJECT_ROOT / "evaluation" / "onnx_benchmark.json"

MAX_SEQ_LENGTH = 48
WARMUP_RUNS = 5
BENCHMARK_REPETITIONS = 50
BATCH_SIZES = [1, 8, 32]

SAMPLE_QUERIES = [
    "what is my credit card balance",
    "book a table for 4 at an Italian restaurant",
    "what is the weather like in New York today",
    "who won the 1998 World Cup in France",
    "blabla random nonsense quantum banana string",
    "how do I change my oil in my car",
    "please cancel my flight to London tomorrow",
    "what are the nutrition facts for an avocado",
    "how much interest will I earn on my savings account",
    "what gas type should I use for my vehicle",
    "where is the nearest ATM machine located",
    "can I redeem my reward points for cash",
    "why is my account blocked right now",
    "set a timer for 15 minutes for cooking",
    "translate this sentence into Spanish please",
    "what are the top rated movies playing near me"
]

def benchmark_inference():
    print("==========================================")
    print("  PyTorch vs. ONNX Runtime Latency & Throughput Benchmark")
    print("==========================================")

    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model file not found at {ONNX_MODEL_PATH}. Run export_onnx.py first.")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    # 1. Measure File Sizes
    pt_size_bytes = (MODEL_DIR / "model.safetensors").stat().st_size
    onnx_size_bytes = ONNX_MODEL_PATH.stat().st_size

    print(f"\nModel File Sizes:")
    print(f"  - PyTorch model.safetensors : {pt_size_bytes / (1024*1024):.2f} MB ({pt_size_bytes:,} bytes)")
    print(f"  - ONNX model.onnx           : {onnx_size_bytes / (1024*1024):.2f} MB ({onnx_size_bytes:,} bytes)")

    # 2. Load PyTorch Model
    print(f"\nLoading PyTorch model...")
    pt_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    pt_model.eval()

    # 3. Load ONNX Runtime Session
    print(f"Loading ONNX Runtime session...")
    ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])

    benchmark_results = {
        "pytorch_model_size_mb": round(pt_size_bytes / (1024*1024), 2),
        "onnx_model_size_mb": round(onnx_size_bytes / (1024*1024), 2),
        "benchmark_repetitions": BENCHMARK_REPETITIONS,
        "warmup_runs": WARMUP_RUNS,
        "batch_benchmarks": {}
    }

    for b in BATCH_SIZES:
        print(f"\n--------------------------------------------------")
        print(f"  Benchmarking Batch Size = {b}")
        print(f"--------------------------------------------------")

        # Construct batch input
        batch_texts = (SAMPLE_QUERIES * ((b // len(SAMPLE_QUERIES)) + 1))[:b]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # --- A. PyTorch Benchmark ---
        print(f"Running PyTorch warmup ({WARMUP_RUNS} runs)...")
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                _ = pt_model(input_ids=input_ids, attention_mask=attention_mask)

        print(f"Measuring PyTorch latency over {BENCHMARK_REPETITIONS} iterations...")
        pt_latencies_ms = []
        with torch.no_grad():
            for _ in range(BENCHMARK_REPETITIONS):
                t0 = time.perf_counter()
                _ = pt_model(input_ids=input_ids, attention_mask=attention_mask)
                t1 = time.perf_counter()
                pt_latencies_ms.append((t1 - t0) * 1000.0)

        pt_mean_ms = float(np.mean(pt_latencies_ms))
        pt_median_ms = float(np.median(pt_latencies_ms))
        pt_p95_ms = float(np.percentile(pt_latencies_ms, 95))
        pt_throughput = float((b * BENCHMARK_REPETITIONS) / (sum(pt_latencies_ms) / 1000.0))

        # --- B. ONNX Runtime Benchmark ---
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy()
        }
        print(f"Running ONNX Runtime warmup ({WARMUP_RUNS} runs)...")
        for _ in range(WARMUP_RUNS):
            _ = ort_session.run(None, ort_inputs)

        print(f"Measuring ONNX Runtime latency over {BENCHMARK_REPETITIONS} iterations...")
        ort_latencies_ms = []
        for _ in range(BENCHMARK_REPETITIONS):
            t0 = time.perf_counter()
            _ = ort_session.run(None, ort_inputs)
            t1 = time.perf_counter()
            ort_latencies_ms.append((t1 - t0) * 1000.0)

        ort_mean_ms = float(np.mean(ort_latencies_ms))
        ort_median_ms = float(np.median(ort_latencies_ms))
        ort_p95_ms = float(np.percentile(ort_latencies_ms, 95))
        ort_throughput = float((b * BENCHMARK_REPETITIONS) / (sum(ort_latencies_ms) / 1000.0))

        speedup_factor = pt_mean_ms / ort_mean_ms if ort_mean_ms > 0 else 1.0

        print(f"\n  Results for Batch Size {b}:")
        print(f"    PyTorch      : Mean={pt_mean_ms:.2f}ms, Median={pt_median_ms:.2f}ms, p95={pt_p95_ms:.2f}ms, Throughput={pt_throughput:.1f} qps")
        print(f"    ONNX Runtime : Mean={ort_mean_ms:.2f}ms, Median={ort_median_ms:.2f}ms, p95={ort_p95_ms:.2f}ms, Throughput={ort_throughput:.1f} qps")
        print(f"    Speedup      : {speedup_factor:.2f}x ({'Faster' if speedup_factor > 1.0 else 'Slower'})")

        benchmark_results["batch_benchmarks"][f"batch_{b}"] = {
            "batch_size": b,
            "pytorch": {
                "mean_latency_ms": round(pt_mean_ms, 2),
                "median_latency_ms": round(pt_median_ms, 2),
                "p95_latency_ms": round(pt_p95_ms, 2),
                "throughput_qps": round(pt_throughput, 1)
            },
            "onnx_runtime": {
                "mean_latency_ms": round(ort_mean_ms, 2),
                "median_latency_ms": round(ort_median_ms, 2),
                "p95_latency_ms": round(ort_p95_ms, 2),
                "throughput_qps": round(ort_throughput, 1)
            },
            "speedup_factor": round(speedup_factor, 2)
        }

    # Save to evaluation/onnx_benchmark.json
    with open(OUTPUT_BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\nSaved benchmark results to {OUTPUT_BENCHMARK_PATH}")
    print("==========================================")
    print("      Benchmark Execution Completed!      ")
    print("==========================================")

    return benchmark_results

if __name__ == "__main__":
    benchmark_inference()
