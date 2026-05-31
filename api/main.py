"""
FastAPI REST API for churn prediction.

Endpoints:
  GET  /health             → liveness check
  POST /predict            → single prediction
  POST /predict/batch      → batch prediction (list of customers)

Run locally:
    uvicorn api.main:app --reload --port 8000

Docker:
    docker compose up api
"""
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import BatchPredictionResponse, CustomerFeatures, PredictionResponse
from src.config import MODEL_PATH

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="XGBoost + SHAP pipeline — trained on IBM Telco Churn dataset.",
    version="1.0.0",
)

# ── Load model once at startup ─────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python -m src.train` first."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


# ── Helper ─────────────────────────────────────────────────────────────────

def _risk_label(prob: float) -> str:
    if prob >= 0.65:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    return "Low"


def _predict_one(customer: CustomerFeatures) -> PredictionResponse:
    df   = pd.DataFrame([customer.model_dump()])
    prob = float(get_pipeline().predict_proba(df)[0, 1])
    return PredictionResponse(
        churn_probability = round(prob, 4),
        churn_prediction  = prob >= 0.5,
        risk_label        = _risk_label(prob),
    )


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: CustomerFeatures):
    """Single customer churn prediction."""
    try:
        return _predict_one(customer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(customers: list[CustomerFeatures]):
    """Batch prediction for a list of customers (max 500)."""
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Max batch size is 500.")
    try:
        preds = [_predict_one(c) for c in customers]
        churners = sum(1 for p in preds if p.churn_prediction)
        return BatchPredictionResponse(
            predictions = preds,
            n_churners  = churners,
            churn_rate  = round(churners / len(preds), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
