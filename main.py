from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

import pandas as pd
import numpy as np
import joblib
import shap

# Load fraud detection artifacts
try:
    model = joblib.load("model.pkl")
    explainer = joblib.load("explainer.pkl")
    FEATURE_COLUMNS = joblib.load("feature_columns.pkl")
except Exception as e:
    print(f"Error loading fraud detection artifacts: {e}")
    raise RuntimeError("Failed to load required fraud detection model artifacts.")

# Load demand forecasting artifacts
try:
    demand_model = joblib.load("demand_forecast_lgbm_v1.pkl")
    demand_features = joblib.load("demand_forecast_features_v1.pkl")
    demand_metadata = joblib.load("demand_forecast_metadata_v1.pkl")
except Exception as e:
    print(f"Error loading demand forecasting artifacts: {e}")
    raise RuntimeError("Failed to load required demand forecasting model artifacts.")

BEST_THRESHOLD = 0.42  # <-- optimized threshold


app = FastAPI(title="Hotel Fraud Detection API")

# FAQ bot (Gemini); routes in app.faq.routes
from app.faq.routes import router as faq_router
from app.guest_risk.routes import router as guest_risk_router

app.include_router(faq_router)
app.include_router(guest_risk_router)


@app.get("/")
def root():
    """API info and links. Use /docs for interactive API docs."""
    return {
        "name": "Hotel Fraud Detection API",
        "docs": "/docs",
        "redoc": "/redoc",
        "faq": {
            "list_hotels": "GET /faq/hotels",
            "ask_question": "POST /faq/ask (body: {\"question\": \"...\", \"hotel_id\": \"...\"})",
        },
        "guest_risk": {
            "compute_profile": "POST /api/v1/guest/risk-profile (body: {\"guest_id\": \"...\"})",
            "get_profile": "GET /api/v1/guest/risk-profile/{guest_id}",
            "record_booking": "POST /api/v1/guest/booking (body: {\"booking\": {...}})",
        },
    }


# Allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BookingRequest(BaseModel):
    data: dict

class Room(BaseModel):
    room_number: str
    floor: int
    is_clean: bool
    is_occupied: bool
    distance_to_elevator: float  # in meters/arbitrary units

class AssignmentRequest(BaseModel):
    rooms: List[Room]

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

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
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

    return df

def prepare_demand_features(history: List[DemandHistoryRow]) -> pd.DataFrame:
    df = pd.DataFrame([h.dict() for h in history])
    df["date"] = pd.to_datetime(df["date"])
    
    df = compute_features(df)
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("Not enough historical data to compute features")

    return df

@app.post("/assign-room")
def assign_room(request: AssignmentRequest):
    """
    Assigns the optimal room based on:
    1. Clean status (must be clean)
    2. Vacancy (must be unoccupied)
    3. Lowest floor (preference for lower floors)
    4. Proximity to elevator (nearest first)
    """
    # Filter for Clean and Vacant rooms
    available_rooms = [
        r for r in request.rooms 
        if r.is_clean and not r.is_occupied
    ]

    if not available_rooms:
        return {"status": "error", "message": "No clean and vacant rooms available"}

    # Sort based on rules: Floor ASC, then Distance to Elevator ASC
    # Python's sort is stable, but we can just use a multi-key lambda
    available_rooms.sort(key=lambda x: (x.floor, x.distance_to_elevator))

    assigned_room = available_rooms[0]

    return {
        "status": "success",
        "assigned_room": assigned_room,
        "criteria": {
            "floor": assigned_room.floor,
            "dist_to_elevator": assigned_room.distance_to_elevator,
            "total_available": len(available_rooms)
        }
    }

@app.post("/score")
def score_booking(request: BookingRequest):
    # Convert input to DataFrame
    X = pd.DataFrame([request.data])

    # Map common aliases (optional but helpful)
    ALIASES = {
        "lead_time": "lead_time_days",
        "booking_hour": "checkin_time_hour"
    }
    X = X.rename(columns=ALIASES)

    # Ensure all required columns exist, fill missing with 0, and order correctly
    # Note: Using 0 is a placeholder. For higher accuracy, you should use the 
    # median/mode from your training data, but 0 is safe for stability.
    X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0)




    # Predict probability
    fraud_proba = model.predict_proba(X)[0, 1]

    # Decision
    flagged = fraud_proba >= BEST_THRESHOLD

    # SHAP explanation
    try:
        shap_values = explainer.shap_values(X)
        # Handle different SHAP output formats (sometimes it's a list for multiclass)
        if isinstance(shap_values, list):
            # For binary classification, typically index [1] or just [0] depending on explainer
            # Using [0] if it's a single array in a list
            vals = shap_values[0] if len(shap_values) == 1 else shap_values[1]
        else:
            vals = shap_values

        contrib = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "shap_value": vals[0]
        })

        contrib["impact"] = contrib["shap_value"].abs()
        top_reasons = contrib.sort_values(
            "impact", ascending=False
        ).head(5)
        top_features = top_reasons["feature"].tolist()
    except Exception as e:
        print(f"SHAP explanation error: {e}")
        top_features = ["Explanation unavailable"]

    return {
        "fraud_probability": round(float(fraud_proba), 4),
        "flagged_for_review": bool(flagged),
        "top_reasons": top_features
    }

@app.post("/forecast/demand")
def forecast_demand(request: DemandForecastRequest):
    try:
        # Initial dataframe setup
        history_df = pd.DataFrame([h.dict() for h in request.history])
        history_df["date"] = pd.to_datetime(history_df["date"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    forecasts = []
    
    # Recursive forecasting loop
    for step in range(request.forecast_days):
        # Re-compute features on the current history
        feature_df = compute_features(history_df.copy())
        
        # Check if we have enough data to generate features for the last row
        last_row = feature_df.iloc[[-1]] 
        
        # Ensure required features are present and not NaN
        if last_row[demand_features].isnull().any().any():
             raise HTTPException(status_code=400, detail=f"Not enough history to forecast day {step+1}. Need more data for lag features.")

        # Predict
        X = last_row[demand_features]
        pred = float(demand_model.predict(X)[0])
        pred = max(0.0, min(1.0, pred))  # clamp occupancy
        
        # Next Forecast Date
        last_date = history_df["date"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        forecasts.append({
            "date": next_date.date(),
            "predicted_occupancy": round(pred, 3)
        })
        
        # Append prediction to history_df for next iteration
        # We carry forward exogenous variables from the last known day
        new_row = history_df.iloc[-1].copy()
        new_row["date"] = next_date
        new_row["occupancy"] = pred # Update with predicted occupancy
        
        # Append using loc to avoid deprecated append
        history_df.loc[len(history_df)] = new_row

    return {
        "model_version": demand_metadata["version"],
        "mae": demand_metadata["mae"],
        "forecast_days": request.forecast_days,
        "forecast": forecasts
    }
