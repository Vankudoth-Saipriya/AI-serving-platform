import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "distilbert_clinc150"
INTENT_NAMES_PATH = PROJECT_ROOT / "model" / "intent_names.json"
TEST_DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "test.parquet"
OUTPUT_METRICS_PATH = PROJECT_ROOT / "evaluation" / "baseline_metrics.json"

OOS_LABEL_ID = 42
MAX_SEQ_LEN = 48
BATCH_SIZE = 64

def evaluate_model_on_test_set():
    print(f"Loading PyTorch model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Running evaluation on device: {device}")

    # Load Intent Names
    with open(INTENT_NAMES_PATH, "r", encoding="utf-8") as f:
        intent_names = json.load(f)
    print(f"Loaded {len(intent_names)} intent classes (Class {OOS_LABEL_ID} = '{intent_names[OOS_LABEL_ID]}').")

    # Load Test Dataset
    if not TEST_DATASET_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATASET_PATH}")
    
    df_test = pd.read_parquet(TEST_DATASET_PATH)
    print(f"Loaded test set with {len(df_test)} samples.")

    texts = df_test["text"].tolist()
    labels = df_test["intent"].tolist()

    all_preds = []
    all_probs = []

    start_time = time.time()
    
    # Process in batches
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    elapsed_time = time.time() - start_time
    print(f"Inference completed in {elapsed_time:.2f}s ({len(texts)/elapsed_time:.1f} samples/sec).")

    y_true = np.array(labels)
    y_pred = np.array(all_preds)

    # 1. Overall Metrics (all 151 classes)
    overall_acc = accuracy_score(y_true, y_pred)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    # 2. In-Scope Metrics (excluding OOS label 42)
    in_scope_mask = (y_true != OOS_LABEL_ID)
    in_scope_acc = accuracy_score(y_true[in_scope_mask], y_pred[in_scope_mask])
    
    # In-Scope Macro F1
    in_scope_labels = [idx for idx in range(151) if idx != OOS_LABEL_ID]
    prec_is, rec_is, f1_is, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=in_scope_labels, average="macro", zero_division=0
    )
    in_scope_macro_f1 = f1_is

    # 3. OOS Class 42 Metrics
    oos_precision, oos_recall, oos_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[OOS_LABEL_ID], average="macro", zero_division=0
    )

    metrics = {
        "overall_accuracy": round(float(overall_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "in_scope_accuracy": round(float(in_scope_acc), 4),
        "in_scope_macro_f1": round(float(in_scope_macro_f1), 4),
        "oos_precision": round(float(oos_precision), 4),
        "oos_recall": round(float(oos_recall), 4),
        "oos_f1": round(float(oos_f1), 4),
        "num_test_samples": len(y_true),
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    # Save to baseline_metrics.json
    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n==========================================")
    print("      PyTorch Model Evaluation Results     ")
    print("==========================================")
    print(f"Overall Accuracy   : {metrics['overall_accuracy']*100:.2f}% (Documented: 86.18%)")
    print(f"Macro F1           : {metrics['macro_f1']*100:.2f}% (Documented: 88.89%)")
    print(f"In-Scope Accuracy  : {metrics['in_scope_accuracy']*100:.2f}% (Documented: 92.98%)")
    print(f"In-Scope Macro F1  : {metrics['in_scope_macro_f1']*100:.2f}% (Documented: 92.52%)")
    print(f"OOS Precision      : {metrics['oos_precision']*100:.2f}% (Documented: 96.03%)")
    print(f"OOS Recall         : {metrics['oos_recall']*100:.2f}% (Documented: 55.60%)")
    print(f"OOS F1             : {metrics['oos_f1']*100:.2f}% (Documented: 70.42%)")
    print("==========================================\n")
    
    return metrics

if __name__ == "__main__":
    evaluate_model_on_test_set()
