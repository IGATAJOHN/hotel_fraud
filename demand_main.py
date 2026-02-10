import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel

from datetime import date
app = FastAPI(title="Hotel AI API")
# ---- Load Demand Forecast Artifacts ----
try:
    demand_model = joblib.load("demand_forecast_lgbm_v1.pkl")
    demand_features = joblib.load("demand_forecast_features_v1.pkl")
    demand_metadata = joblib.load("demand_forecast_metadata_v1.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load demand model artifacts: {e}")


class DemandHistoryRow(BaseModel):
    date: date
    occupancy: float
    adr: float
    revpar: float
    avg_lead_time: float
    short_lead_ratio: float
    cancellation_rate: float
    refund_rate: float
    local_event: int
    tourism_trend: float
    competitor_price: float

class DemandForecastRequest(BaseModel):
    history: List[DemandHistoryRow]
    forecast_days: int = 7

def prepare_demand_features(history: List[DemandHistoryRow]) -> pd.DataFrame:
    df = pd.DataFrame([h.dict() for h in history])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Time features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Lag features
    df["occupancy_lag_1"] = df["occupancy"].shift(1)
    df["occupancy_lag_7"] = df["occupancy"].shift(7)
    df["adr_lag_7"] = df["adr"].shift(7)

    # Rolling features
    df["occ_roll_3"] = df["occupancy"].rolling(3).mean()
    df["occ_roll_7"] = df["occupancy"].rolling(7).mean()
    df["adr_roll_7"] = df["adr"].rolling(7).mean()
    df["occ_std_7"] = df["occupancy"].rolling(7).std()

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("Not enough historical data to compute features")

    return df

@app.post("/forecast/demand")
def forecast_demand(request: DemandForecastRequest):
    try:
        df = prepare_demand_features(request.history)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Use last available row as forecast base
    X = df[demand_features].tail(1)

    forecasts = []
    current_date = df["date"].iloc[-1]

    for step in range(request.forecast_days):
        pred = float(demand_model.predict(X)[0])
        pred = max(0.0, min(1.0, pred))  # clamp occupancy

        forecasts.append({
            "date": (current_date + pd.Timedelta(days=step + 1)).date(),
            "predicted_occupancy": round(pred, 3)
        })

    return {
        "model_version": demand_metadata["version"],
        "mae": demand_metadata["mae"],
        "forecast_days": request.forecast_days,
        "forecast": forecasts
    }

