from fastapi import FastAPI, HTTPException

from app.schemas import (
    CustomerFeatures,
    SegmentResponse,
    ChurnResponse,
    ForecastRequest,
    ForecastResponse,
)
from app.utils import predict_segment, predict_churn, predict_forecast

app = FastAPI(
    title="Customer Intelligence Platform API",
    description="Serves K-Means segmentation, churn prediction, and revenue forecasting models.",
    version="1.0.0",)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Customer Intelligence Platform API"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.post("/segment", response_model=SegmentResponse, tags=["Segmentation"])
def get_segment(features: CustomerFeatures):
    """Predict which of the 4 K-Means customer segments a customer belongs to."""
    result = predict_segment(features.model_dump())
    return result


@app.post("/churn-score", response_model=ChurnResponse, tags=["Churn"])
def get_churn_score(features: CustomerFeatures):
    """Predict a customer's churn probability using the Random Forest model."""
    result = predict_churn(features.model_dump())
    return result


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def get_forecast(request: ForecastRequest):
    """Forecast revenue for a target month using the Seasonal Naive model."""
    try:
        result = predict_forecast(request.year, request.month)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result