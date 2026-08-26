import json
import logging
from pathlib import Path
from typing import List, Optional
import onnxruntime as ort
from transformers import AutoTokenizer
from app.config import settings

logger = logging.getLogger(__name__)

class ModelContainer:
    _instance: Optional["ModelContainer"] = None

    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.ort_session: Optional[ort.InferenceSession] = None
        self.intent_names: List[str] = []
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ModelContainer":
        if cls._instance is None:
            cls._instance = ModelContainer()
        return cls._instance

    def load(self) -> None:
        if self.is_loaded:
            return

        model_dir = Path(settings.MODEL_DIR)
        onnx_path = Path(settings.ONNX_MODEL_PATH)
        intent_path = Path(settings.INTENT_NAMES_PATH)

        if not onnx_path.exists():
            error_msg = (
                f"CRITICAL STARTUP ERROR: ONNX model binary not found at '{onnx_path}'. "
                f"Please ensure 'model.onnx' is placed in the model/ directory or MODEL_DIR environment variable is set."
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        if not model_dir.exists():
            error_msg = (
                f"CRITICAL STARTUP ERROR: Tokenizer model directory not found at '{model_dir}'. "
                f"Please ensure tokenizer artifacts (vocab.txt, tokenizer_config.json) exist in '{model_dir}'."
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        if not intent_path.exists():
            error_msg = f"CRITICAL STARTUP ERROR: Intent class names JSON not found at '{intent_path}'."
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Loading HuggingFace Tokenizer from: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        logger.info(f"Loading ONNX Runtime session from: {onnx_path}")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 2
        
        self.ort_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

        logger.info(f"Loading Intent Class Names from: {intent_path}")
        with open(intent_path, "r", encoding="utf-8") as f:
            self.intent_names = json.load(f)

        assert len(self.intent_names) == 151, f"Expected 151 intent classes, found {len(self.intent_names)}"
        self.is_loaded = True
        logger.info("ModelContainer initialization complete.")

model_container = ModelContainer.get_instance()
