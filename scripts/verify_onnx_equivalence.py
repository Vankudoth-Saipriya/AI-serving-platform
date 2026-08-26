import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "distilbert_clinc150"
ONNX_MODEL_PATH = PROJECT_ROOT / "model" / "model.onnx"
INTENT_NAMES_PATH = PROJECT_ROOT / "model" / "intent_names.json"

OOS_LABEL_ID = 42
MAX_SEQ_LENGTH = 48
TOLERANCE = 1e-4

def verify_equivalence():
    print("==========================================")
    print("  PyTorch vs. ONNX Equivalence Verification")
    print("==========================================")

    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model file not found at {ONNX_MODEL_PATH}. Run export_onnx.py first.")

    # 1. Load Intent Names
    with open(INTENT_NAMES_PATH, "r", encoding="utf-8") as f:
        intent_names = json.load(f)

    # 2. Load PyTorch Model & Tokenizer
    print(f"\n1. Loading PyTorch model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    pt_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    pt_model.eval()

    # 3. Load ONNX Runtime Session
    print(f"\n2. Loading ONNX Runtime session from {ONNX_MODEL_PATH}...")
    ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])

    # 4. Test Queries across different domains & lengths
    test_queries = [
        "what is my credit card balance",
        "book a table for 4 at an Italian restaurant",
        "what is the weather like in New York today",
        "who won the 1998 World Cup in France",
        "blabla random nonsense quantum banana string",
        "how do I change my oil in my car",
        "please cancel my flight to London tomorrow",
        "what are the nutrition facts for an avocado"
    ]

    print(f"\n3. Running equivalence verification across {len(test_queries)} test queries...")
    print("----------------------------------------------------------------------------------")

    encodings = tokenizer(
        test_queries,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )

    pt_input_ids = encodings["input_ids"]
    pt_attention_mask = encodings["attention_mask"]

    # PyTorch Inference
    with torch.no_grad():
        pt_outputs = pt_model(input_ids=pt_input_ids, attention_mask=pt_attention_mask)
        pt_logits = pt_outputs.logits.numpy()
        pt_probs = torch.softmax(pt_outputs.logits, dim=-1).numpy()
        pt_preds = np.argmax(pt_probs, axis=-1)

    # ONNX Runtime Inference
    ort_inputs = {
        "input_ids": pt_input_ids.numpy(),
        "attention_mask": pt_attention_mask.numpy()
    }
    ort_outs = ort_session.run(None, ort_inputs)
    ort_logits = ort_outs[0]
    
    # Softmax on ORT Logits
    exp_logits = np.exp(ort_logits - np.max(ort_logits, axis=-1, keepdims=True))
    ort_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    ort_preds = np.argmax(ort_probs, axis=-1)

    # Numerical Comparisons
    max_abs_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    mean_abs_diff = float(np.mean(np.abs(pt_logits - ort_logits)))
    max_prob_diff = float(np.max(np.abs(pt_probs - ort_probs)))

    class_id_matches = (pt_preds == ort_preds).all()
    
    print(f"   - Max Absolute Logits Difference  : {max_abs_diff:.6e}")
    print(f"   - Mean Absolute Logits Difference : {mean_abs_diff:.6e}")
    print(f"   - Max Probability Difference     : {max_prob_diff:.6e}")
    print(f"   - Class ID Matches (100% target)  : {class_id_matches}")

    print("\nDetailed Query Comparisons:")
    for i, q in enumerate(test_queries):
        pt_class = intent_names[pt_preds[i]]
        ort_class = intent_names[ort_preds[i]]
        pt_conf = pt_probs[i, pt_preds[i]]
        ort_conf = ort_probs[i, ort_preds[i]]
        match_str = "MATCH" if (pt_preds[i] == ort_preds[i]) else "MISMATCH"
        print(f"  [{match_str}] \"{q[:40]}...\"")
        print(f"     PyTorch : {pt_class:<25} (conf: {pt_conf:.4f})")
        print(f"     ONNX    : {ort_class:<25} (conf: {ort_conf:.4f})")

    assert max_abs_diff < TOLERANCE, f"Max absolute difference {max_abs_diff} exceeds tolerance {TOLERANCE}"
    assert class_id_matches, "Class ID predictions between PyTorch and ONNX Runtime do not match!"

    print("\n==========================================")
    print("  ONNX Equivalence Verification PASSED!   ")
    print("==========================================")

    return {
        "max_absolute_logits_difference": max_abs_diff,
        "mean_absolute_logits_difference": mean_abs_diff,
        "max_probability_difference": max_prob_diff,
        "class_id_matches": bool(class_id_matches),
        "tolerance": TOLERANCE,
        "queries_tested": len(test_queries)
    }

if __name__ == "__main__":
    verify_equivalence()
