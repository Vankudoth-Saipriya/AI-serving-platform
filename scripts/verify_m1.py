import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "distilbert_clinc150"
INTENT_NAMES_PATH = PROJECT_ROOT / "model" / "intent_names.json"
METADATA_PATH = PROJECT_ROOT / "model" / "model_metadata.json"

OOS_LABEL_ID = 42
MAX_SEQ_LENGTH = 48

def verify_m1():
    print("==========================================")
    print("         M1 Verification Script           ")
    print("==========================================")

    # 1. Load Metadata
    print(f"\n1. Checking model_metadata.json at {METADATA_PATH}...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"   - Model Name: {metadata['model_name']} (v{metadata['model_version']})")
    print(f"   - Base Model: {metadata['base_model']}")
    print(f"   - Num Classes: {metadata['num_classes']}")
    print(f"   - OOS Label ID: {metadata['oos_label_id']} ('{metadata['oos_label_name']}')")
    print(f"   - Max Seq Length: {metadata['max_seq_length']}")

    # 2. Load Label Names
    print(f"\n2. Loading intent_names.json at {INTENT_NAMES_PATH}...")
    with open(INTENT_NAMES_PATH, "r", encoding="utf-8") as f:
        intent_names = json.load(f)
    assert len(intent_names) == 151, f"Expected 151 intent names, found {len(intent_names)}"
    assert intent_names[OOS_LABEL_ID] == "oos", f"Expected class 42 to be 'oos', got {intent_names[OOS_LABEL_ID]}"
    print(f"   - Verified {len(intent_names)} classes.")
    print(f"   - Class 42: '{intent_names[OOS_LABEL_ID]}'")
    print(f"   - Sample classes [0..4]: {intent_names[:5]}")

    # 3. Load Tokenizer & Model
    print(f"\n3. Loading DistilBERT Tokenizer & PyTorch Model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"   - Model loaded successfully onto device: {device}")

    # 4. Representative Prediction Verification
    test_queries = [
        "what is my credit card balance",
        "book a table for 4 at an Italian restaurant",
        "what is the weather like in New York today",
        "who won the 1998 World Cup in France",
        "blabla random nonsense quantum banana string"
    ]

    print("\n4. Running predictions on representative queries:")
    print("--------------------------------------------------")
    for query in test_queries:
        encodings = tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        predicted_intent = intent_names[pred_idx]
        is_oos = (pred_idx == OOS_LABEL_ID)

        print(f"Query: \"{query}\"")
        print(f"  -> Predicted Intent : '{predicted_intent}' (ID: {pred_idx})")
        print(f"  -> Confidence       : {confidence:.4f}")
        print(f"  -> Is OOS           : {is_oos}")
        print()

    print("==========================================")
    print("   M1 Verification Completed Successfully!")
    print("==========================================")

if __name__ == "__main__":
    verify_m1()
