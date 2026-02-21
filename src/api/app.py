"""
src/api/app.py
FastAPI inference service for Cats vs Dogs classification.

Endpoints:
  GET  /health   – liveness / readiness probe
  POST /predict  – classify an uploaded image
  GET  /metrics  – Prometheus metrics
"""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from src.data.preprocess import preprocess_pil_image
from src.models.model import load_model, predict

# ── Logging ───────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["label"],
)
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
ERROR_COUNT = Counter("prediction_errors_total", "Total prediction errors")

# ── App State ─────────────────────────────────────────────────────────────────
MODEL = None
DEVICE = "cpu"

MODEL_PATH = os.getenv("MODEL_PATH", "models/artifacts/best_model.pt")
MODEL_TYPE = os.getenv("MODEL_TYPE", "cnn")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading model", path=MODEL_PATH, device=DEVICE)
    try:
        MODEL = load_model(MODEL_PATH, model_type=MODEL_TYPE, device=DEVICE)
        log.info("Model loaded successfully")
    except Exception as exc:
        log.error("Failed to load model", error=str(exc))
        MODEL = None
    yield
    MODEL = None


app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Binary image classification REST API – MLOps Assignment 2",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health():
    """Liveness and readiness probe."""
    status = "healthy" if MODEL is not None else "unhealthy"
    code = 200 if MODEL is not None else 503
    return JSONResponse(
        content={
            "status": status,
            "model_loaded": MODEL is not None,
            "device": DEVICE,
        },
        status_code=code,
    )


@app.post("/predict", tags=["inference"])
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Classify an uploaded image as cat or dog.

    - **file**: JPEG / PNG image file

    Returns:
    - **label**: predicted class ("cat" or "dog")
    - **confidence**: probability of predicted class
    - **probabilities**: per-class probabilities
    - **latency_ms**: inference latency in milliseconds
    """
    if MODEL is None:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        ERROR_COUNT.inc()
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    try:
        raw = await file.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=422, detail=f"Cannot read image: {exc}")

    t0 = time.perf_counter()
    try:
        tensor = preprocess_pil_image(pil_image)
        result = predict(MODEL, tensor, device=DEVICE)
    except Exception as exc:
        ERROR_COUNT.inc()
        log.error("Inference error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    latency = time.perf_counter() - t0
    REQUEST_LATENCY.observe(latency)
    REQUEST_COUNT.labels(label=result["label"]).inc()

    log.info(
        "Prediction complete",
        label=result["label"],
        confidence=round(result["confidence"], 4),
        latency_ms=round(latency * 1000, 2),
        filename=file.filename,
    )

    return {
        **result,
        "latency_ms": round(latency * 1000, 2),
    }


@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Cats vs Dogs Classifier API – see /docs for usage"}
