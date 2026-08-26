import time
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException, status, Request, Response
from fastapi.responses import JSONResponse
from app.config import settings
from app.model_loader import model_container
from app.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse
)
from app.inference import predict_single, predict_batch
from app.metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    get_metrics_content
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing FastAPI service & loading model artifacts...")
    model_container.load()
    yield
    logger.info("Shutting down FastAPI service...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.MODEL_VERSION,
    description="High-performance AI model serving platform for DistilBERT Intent & OOS Classification using ONNX Runtime.",
    lifespan=lifespan
)

@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    
    endpoint = request.url.path
    if endpoint != "/metrics":
        HTTP_REQUESTS_TOTAL.labels(
            endpoint=endpoint,
            method=request.method,
            status_code=response.status_code
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(duration)
        
    return response

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error encountered during inference processing."}
    )

@app.get("/health", response_model=HealthResponse, tags=["Service"])
def health_check():
    return HealthResponse(
        status="healthy" if model_container.is_loaded else "degraded",
        model_loaded=model_container.is_loaded,
        version=settings.MODEL_VERSION
    )

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Service"])
def model_info():
    return ModelInfoResponse(
        model_name="distilbert_clinc150",
        model_version=settings.MODEL_VERSION,
        num_classes=151,
        max_sequence_length=settings.MAX_SEQ_LENGTH,
        inference_backend="ONNX Runtime (CPUExecutionProvider)"
    )

@app.get("/metrics", tags=["Observability"])
def get_metrics():
    content, media_type = get_metrics_content()
    return Response(content=content, media_type=media_type)

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_endpoint(payload: PredictRequest):
    return predict_single(payload.text)

@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch_endpoint(payload: BatchPredictRequest):
    return predict_batch(payload.inputs)
