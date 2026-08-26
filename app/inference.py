import time
from typing import List
import numpy as np
from app.config import settings
from app.model_loader import model_container
from app.schemas import PredictResponse, BatchPredictResponse
from app.metrics import (
    MODEL_INFERENCE_DURATION_SECONDS,
    MODEL_PREDICTIONS_TOTAL,
    MODEL_PREDICTION_CONFIDENCE,
    MODEL_OOS_PREDICTIONS_TOTAL,
    MODEL_IN_SCOPE_PREDICTIONS_TOTAL
)

def predict_single(text: str) -> PredictResponse:
    if not model_container.is_loaded:
        model_container.load()

    cleaned_text = text.strip()
    
    encodings = model_container.tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=settings.MAX_SEQ_LENGTH,
        return_tensors="np"
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    t0 = time.perf_counter()
    ort_outs = model_container.ort_session.run(None, ort_inputs)
    t1 = time.perf_counter()
    
    # Observe ONNX single inference latency
    MODEL_INFERENCE_DURATION_SECONDS.labels(inference_type="single").observe(t1 - t0)

    logits = ort_outs[0][0]  # shape: (151,)

    # Softmax probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    predicted_id = int(np.argmax(probs))
    confidence = round(float(probs[predicted_id]), 4)
    intent_name = model_container.intent_names[predicted_id]
    is_oos = bool(predicted_id == settings.OOS_LABEL_ID)

    # Instrument model behavior metrics
    MODEL_PREDICTIONS_TOTAL.labels(intent=intent_name).inc()
    MODEL_PREDICTION_CONFIDENCE.observe(confidence)
    if is_oos:
        MODEL_OOS_PREDICTIONS_TOTAL.inc()
    else:
        MODEL_IN_SCOPE_PREDICTIONS_TOTAL.inc()

    return PredictResponse(
        intent=intent_name,
        confidence=confidence,
        is_oos=is_oos,
        model_version=settings.MODEL_VERSION
    )

def predict_batch(texts: List[str]) -> BatchPredictResponse:
    if not model_container.is_loaded:
        model_container.load()

    if len(texts) > settings.MAX_BATCH_SIZE:
        raise ValueError(f"Batch size {len(texts)} exceeds maximum allowed batch size ({settings.MAX_BATCH_SIZE}).")

    cleaned_texts = [t.strip() for t in texts]

    # True Vectorized Tokenization
    encodings = model_container.tokenizer(
        cleaned_texts,
        truncation=True,
        padding="max_length",
        max_length=settings.MAX_SEQ_LENGTH,
        return_tensors="np"
    )

    input_ids = encodings["input_ids"]        # shape: (N, 48)
    attention_mask = encodings["attention_mask"]# shape: (N, 48)

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    # True Vectorized ONNX Session Execution
    t0 = time.perf_counter()
    ort_outs = model_container.ort_session.run(None, ort_inputs)
    t1 = time.perf_counter()

    # Observe ONNX batch inference latency
    MODEL_INFERENCE_DURATION_SECONDS.labels(inference_type="batch").observe(t1 - t0)

    logits = ort_outs[0]  # shape: (N, 151)

    # Vectorized Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    predicted_ids = np.argmax(probs, axis=-1)

    predictions = []
    for idx, pred_id in enumerate(predicted_ids):
        pred_id_int = int(pred_id)
        confidence = round(float(probs[idx, pred_id_int]), 4)
        intent_name = model_container.intent_names[pred_id_int]
        is_oos = bool(pred_id_int == settings.OOS_LABEL_ID)

        # Instrument prediction metrics per batch item
        MODEL_PREDICTIONS_TOTAL.labels(intent=intent_name).inc()
        MODEL_PREDICTION_CONFIDENCE.observe(confidence)
        if is_oos:
            MODEL_OOS_PREDICTIONS_TOTAL.inc()
        else:
            MODEL_IN_SCOPE_PREDICTIONS_TOTAL.inc()

        predictions.append(
            PredictResponse(
                intent=intent_name,
                confidence=confidence,
                is_oos=is_oos,
                model_version=settings.MODEL_VERSION
            )
        )

    return BatchPredictResponse(
        predictions=predictions,
        batch_size=len(predictions),
        model_version=settings.MODEL_VERSION
    )
