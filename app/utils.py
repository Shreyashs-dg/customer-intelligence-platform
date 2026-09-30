import sys
from pathlib import Path

import joblib
import pandas as pd

# Make src/ importable (src sits one level up from app/, as a sibling folder)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import (
    apply_log_transforms,
    build_kmeans_feature_row,
    build_churn_feature_row,
    SEGMENT_NAMES,)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Loaded once at import time (i.e. once when the API starts), not per-request
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
kmeans_model = joblib.load(MODELS_DIR / "kmeans_model.pkl")
churn_model = joblib.load(MODELS_DIR / "churn_model.pkl")
monthly_revenue: pd.Series = joblib.load(MODELS_DIR / "forecast_model.pkl")

def predict_segment(raw: dict) -> dict:
    raw_with_logs = apply_log_transforms(raw)
    feature_row = build_kmeans_feature_row(raw_with_logs)
    scaled_row = scaler.transform(feature_row)
    cluster = int(kmeans_model.predict(scaled_row)[0])
    return {"cluster": cluster, "segment_name": SEGMENT_NAMES[cluster]}


def predict_churn(raw: dict) -> dict:
    segment_result = predict_segment(raw)
    raw_with_logs = apply_log_transforms(raw)
    churn_row = build_churn_feature_row(raw_with_logs, cluster=segment_result["cluster"])
    churn_probability = float(churn_model.predict_proba(churn_row)[0][1])
    return {
        "cluster": segment_result["cluster"],
        "segment_name": segment_result["segment_name"],
        "churn_probability": round(churn_probability, 4),
        "churned": churn_probability >= 0.5,}

def predict_forecast(year: int, month: int) -> dict:
    target_month = pd.Timestamp(year=year, month=month, day=1)
    reference_month = target_month - pd.DateOffset(years=1)

    if reference_month not in monthly_revenue.index:
        available_start = monthly_revenue.index.min().strftime("%Y-%m")
        available_end = monthly_revenue.index.max().strftime("%Y-%m")
        raise ValueError(
            f"Cannot forecast {target_month:%Y-%m}: no historical data for "
            f"{reference_month:%Y-%m} (one year earlier). Historical data covers "
            f"{available_start} to {available_end}, so forecasts are only available "
            f"for months from {(monthly_revenue.index.min() + pd.DateOffset(years=1)):%Y-%m} onward.")

    predicted = float(monthly_revenue.loc[reference_month])
    return {
        "target_month": target_month.strftime("%Y-%m"),
        "predicted_revenue": round(predicted, 2),
        "method": "seasonal_naive (same month, previous year)",}