import os
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "distilbert_clinc150"
OUTPUT_ONNX_PATH = PROJECT_ROOT / "model" / "model.onnx"

MAX_SEQ_LENGTH = 48
OPSET_VERSION = 14

def export_model_to_onnx():
    print("==========================================")
    print("        ONNX Model Export Pipeline        ")
    print("==========================================")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"PyTorch model directory not found at: {MODEL_DIR}")

    print(f"\n1. Loading PyTorch model & tokenizer from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    # Create dummy inputs for ONNX tracing
    dummy_text = "what is my credit card balance"
    encodings = tokenizer(
        dummy_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    print(f"\n2. Exporting PyTorch model to ONNX at: {OUTPUT_ONNX_PATH}")
    print(f"   - Inputs        : input_ids {input_ids.shape}, attention_mask {attention_mask.shape}")
    print(f"   - Dynamic Axes  : batch_size, sequence_length")
    print(f"   - Opset Version : {OPSET_VERSION}")

    start_time = time.time()
    
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(OUTPUT_ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True
    )
    
    elapsed = time.time() - start_time
    onnx_size = OUTPUT_ONNX_PATH.stat().st_size
    print(f"   - Export completed in {elapsed:.2f}s!")
    print(f"   - ONNX Model File Size: {onnx_size:,} bytes ({onnx_size / (1024*1024):.2f} MB)")

    # 3. Validate ONNX Graph Integrity
    print("\n3. Verifying ONNX model graph structure with onnx.checker...")
    onnx_model = onnx.load(str(OUTPUT_ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("   - ONNX model graph structure is 100% VALID!")

    print("\n==========================================")
    print("        ONNX Export Completed!            ")
    print("==========================================")
    
    return str(OUTPUT_ONNX_PATH)

if __name__ == "__main__":
    export_model_to_onnx()
