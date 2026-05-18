"""
app/api.py
============================================================
FastAPI inference service for the Canadian finance classifier
============================================================

WHY FASTAPI:
    - Automatic Swagger UI at /docs (instant interactive demo)
    - Built-in pydantic validation (no manual request parsing)
    - Async-ready (so the server stays responsive under load)
    - One of the most common Python web frameworks in ML
      production - good to have on your resume

ENDPOINTS:
    GET  /            redirects to /docs
    GET  /health      liveness check (always returns ok)
    GET  /ready       readiness check (only ok once model is loaded)
    POST /predict     single-text prediction
    POST /predict_batch  list of texts -> list of predictions

WHY SEPARATE /health AND /ready:
    Deployment platforms (Render, Railway, k8s) hit /health to
    decide if the process is alive. /ready tells them whether
    it's ALSO finished loading the model. If you only have
    /health, the platform routes traffic to you before the
    model loads and the first user gets a 5-second wait.

RUN LOCALLY:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
    Then visit http://localhost:8000/docs
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# Make `src.*` imports work whether the app runs from project
# root or from app/ directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference.predictor import get_predictor  # noqa: E402

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("finbert-api")


class _State:
    model_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI's modern startup/shutdown hook (replaces the
    deprecated @app.on_event decorators).

    We load the model HERE, before the server starts accepting
    traffic - so the first request never pays the 5-10 second
    cold-start cost.
    """
    log.info("Loading FinBERT predictor (this takes ~5-10s on CPU)...")
    try:
        get_predictor()
        _State.model_ready = True
        log.info("Predictor loaded. Ready to serve.")
    except Exception as exc:
        log.exception(f"Failed to load predictor: {exc}")
        _State.model_ready = False

    yield

    log.info("Shutting down.")


# Request / response schemas
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The finance text to classify",
        examples=["Bank of Canada holds overnight rate at 5%."],
    )


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="A list of texts to classify (max 64 per call)",
    )


class HeadPrediction(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]


class PredictResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: HeadPrediction
    intent: HeadPrediction


# App setup
app = FastAPI(
    title="Canadian Finance Sentiment & Intent Classifier",
    description=(
        "Fine-tuned FinBERT with two heads (sentiment + intent) "
        "for Canadian financial text. Test F1: 0.86 sentiment, "
        "0.91 intent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS so the Streamlit demo (different domain) can call this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Endpoints
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """Liveness probe - process is up."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness probe - model is loaded and serving."""
    if not _State.model_ready:
        raise HTTPException(503, detail="Model not loaded yet")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Classify a single piece of finance text."""
    if not _State.model_ready:
        raise HTTPException(503, detail="Model not loaded yet")
    try:
        result = get_predictor().predict(req.text)[0]
        return result
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(500, detail=str(exc))


@app.post("/predict_batch", response_model=list[PredictResponse])
async def predict_batch(req: BatchPredictRequest):
    """Classify up to 64 pieces of finance text in one call."""
    if not _State.model_ready:
        raise HTTPException(503, detail="Model not loaded yet")
    try:
        return get_predictor().predict(req.texts)
    except Exception as exc:
        log.exception("Batch prediction failed")
        raise HTTPException(500, detail=str(exc))
