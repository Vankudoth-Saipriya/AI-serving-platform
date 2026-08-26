from typing import List
from pydantic import BaseModel, Field, field_validator, ConfigDict

class BaseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

class PredictRequest(BaseSchema):
    text: str = Field(..., json_schema_extra={"example": "what is my credit card balance"})

    @field_validator("text")
    def validate_text_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Input query text cannot be empty or whitespace only.")
        return v.strip()

class PredictResponse(BaseSchema):
    intent: str = Field(..., json_schema_extra={"example": "balance"})
    confidence: float = Field(..., json_schema_extra={"example": 0.8542})
    is_oos: bool = Field(..., json_schema_extra={"example": False})
    model_version: str = Field(..., json_schema_extra={"example": "1.0.0"})

class BatchPredictRequest(BaseSchema):
    inputs: List[str] = Field(..., json_schema_extra={"example": ["what is my balance", "book a table"]})

    @field_validator("inputs")
    def validate_inputs_not_empty(cls, v: List[str]) -> List[str]:
        if not v or len(v) == 0:
            raise ValueError("Batch inputs list cannot be empty.")
        cleaned = [item.strip() for item in v if item and item.strip()]
        if not cleaned:
            raise ValueError("Batch inputs list must contain at least one valid non-empty text string.")
        return cleaned

class BatchPredictResponse(BaseSchema):
    predictions: List[PredictResponse]
    batch_size: int = Field(..., json_schema_extra={"example": 2})
    model_version: str = Field(..., json_schema_extra={"example": "1.0.0"})

class HealthResponse(BaseSchema):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})

class ModelInfoResponse(BaseSchema):
    model_name: str = Field(..., json_schema_extra={"example": "distilbert_clinc150"})
    model_version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    num_classes: int = Field(..., json_schema_extra={"example": 151})
    max_sequence_length: int = Field(..., json_schema_extra={"example": 48})
    inference_backend: str = Field(..., json_schema_extra={"example": "ONNX Runtime (CPUExecutionProvider)"})
