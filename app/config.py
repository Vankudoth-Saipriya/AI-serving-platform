import json
from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "AI Model Serving Platform"
    MODEL_VERSION: str = "1.0.0"
    MODEL_DIR: Path = PROJECT_ROOT / "model" / "distilbert_clinc150"
    ONNX_MODEL_PATH: Path = PROJECT_ROOT / "model" / "model.onnx"
    INTENT_NAMES_PATH: Path = PROJECT_ROOT / "model" / "intent_names.json"
    METADATA_PATH: Path = PROJECT_ROOT / "model" / "model_metadata.json"
    
    MAX_SEQ_LENGTH: int = 48
    OOS_LABEL_ID: int = 42
    MAX_BATCH_SIZE: int = 128

settings = Settings()
