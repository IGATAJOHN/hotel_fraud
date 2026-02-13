from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

import pandas as pd
import numpy as np
import joblib
import shap

import random
import uuid

# --- A/B Testing & Rollback Infrastructure ---
class ModelRegistry:
    def __init__(self):
        self.models = {
            "fraud": {
                "active_version": "v1",
                "versions": {
                    "v1": {"path": "model.pkl", "explainer": "explainer.pkl", "features": "feature_columns.pkl"},
                    "v2": {"path": "model.pkl", "explainer": "explainer.pkl", "features": "feature_columns.pkl"}
                },
                "traffic_split": {"v1": 1.0} # Version -> Percentage (0.0 to 1.0)
            },
            "demand": {
                "active_version": "v1",
                "versions": {
                    "v1": {"path": "demand_forecast_lgbm_v1.pkl", "features": "demand_forecast_features_v1.pkl", "metadata": "demand_forecast_metadata_v1.pkl"}
                },
                "traffic_split": {"v1": 1.0}
            },
            "in-stay": {
                "active_version": "v1",
                "versions": {
                    "v1": {
                        "path": "in_stay_anomaly_v1.pkl", 
                        "scaler": "in_stay_scaler_v1.pkl"
                    }
                },
                "traffic_split": {"v1": 1.0}
            }
        }
        self.loaded_artifacts = {}

    def load_version(self, service: str, version: str):
        config = self.models[service]["versions"][version]
        artifacts = {}
        try:
            for key, filename in config.items():
                artifacts[key] = joblib.load(filename)
            self.loaded_artifacts[f"{service}:{version}"] = artifacts
            print(f"Loaded {service} model {version} successfully.")
        except Exception as e:
            print(f"Error loading {service} model {version}: {e}")
            raise RuntimeError(f"Critical error loading {service}:{version}")

    def get_model(self, service: str):
        # Weighted random selection for A/B testing
        split = self.models[service]["traffic_split"]
        versions = list(split.keys())
        weights = list(split.values())
        selected_version = random.choices(versions, weights=weights)[0]
        
        key = f"{service}:{selected_version}"
        if key not in self.loaded_artifacts:
            self.load_version(service, selected_version)
            
        return selected_version, self.loaded_artifacts[key]

# Initialize Registry
registry = ModelRegistry()

# Pre-load production versions
try:
    registry.load_version("fraud", "v1")
    registry.load_version("demand", "v1")
    registry.load_version("in-stay", "v1")
except Exception as e:
    print(f"Initial model loading failed: {e}")

# --- App Initialization ---
app = FastAPI(title="Hotel Fraud Detection API")

# Allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Hotel AI API",
        "models": ["fraud_detection", "demand_forecasting", "in-stay_monitoring"]
    }

@app.get("/dashboard")
def dashboard():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

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

class SplitRequest(BaseModel):
    service: str
    split: dict

class RollbackRequest(BaseModel):
    service: str
    target_version: str = "v1"

class PricingRequest(BaseModel):
    current_adr: float
    forecast_occupancy: float
    lower_ci: float
    upper_ci: float
    competitor_price: float
    season: str
    elasticity: Optional[float] = None
    historical_data: Optional[List[dict]] = None

class InStayRequest(BaseModel):
    room_access_count: int
    dnd_hours: float
    visitor_count: int
    noise_complaints: int
    service_usage_intensity: float # Will map to 'service_usage_count'
    stay_length: int
    cleaning_duration_minutes: Optional[float] = 0.0
    maintenance_issues: Optional[int] = 0
    maintenance_resolution_hours: Optional[float] = 0.0
    staff_overtime_hours: Optional[float] = 0.0
    guest_rating_staff: Optional[float] = 5.0
    access_rate: Optional[float] = 0.0
    service_intensity: Optional[float] = 0.0

class RecommendationRequest(BaseModel):
    stay_length: int
    service_usage_count: int
    guest_rating_staff: float
    refund_rate: float
    room_price: float
    booking_history_count: int
    cancellation_history: int
    refund_history: int
    channel_source: str # e.g., "direct", "ota", "corporate"


class SARRecord(BaseModel):
    booking_id: str
    fraud_score: float
    timestamp: date
    narrative: str
    top_indicators: List[str]
    status: str = "Filed"

# In-memory storage for auto-filed reports (mock database)
sar_reports: List[SARRecord] = []

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
    # Get model from registry
    version, artifacts = registry.get_model("fraud")
    model = artifacts["path"]
    explainer = artifacts["explainer"]
    features = artifacts["features"]

    # Convert input to DataFrame
    X = pd.DataFrame([request.data])

    # Map common aliases (optional but helpful)
    ALIASES = {
        "lead_time": "lead_time_days",
        "booking_hour": "checkin_time_hour"
    }
    X = X.rename(columns=ALIASES)

    # Ensure all required columns exist, fill missing with 0, and order correctly
    X = X.reindex(columns=features, fill_value=0)

    # Predict probability
    fraud_proba = model.predict_proba(X)[0, 1]

    # Decision
    flagged = fraud_proba >= BEST_THRESHOLD

    # SHAP explanation
    try:
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            vals = shap_values[0] if len(shap_values) == 1 else shap_values[1]
        else:
            vals = shap_values

        contrib = pd.DataFrame({
            "feature": features,
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

    # SAR Auto-Filing (Critical Threshold = 0.85)
    filed_sar = False
    if fraud_proba >= 0.85:
        narrative = generate_sar_narrative(fraud_proba, top_features, request.data)
        import uuid
        sar_record = SARRecord(
            booking_id=str(uuid.uuid4())[:8],
            fraud_score=round(float(fraud_proba), 4),
            timestamp=date.today(),
            narrative=narrative,
            top_indicators=top_features
        )
        sar_reports.append(sar_record)
        filed_sar = True

    return {
        "fraud_probability": round(float(fraud_proba), 4),
        "flagged_for_review": bool(flagged),
        "top_reasons": top_features,
        "model_info": {
            "service": "fraud",
            "version": version
        },
        "compliance": {
            "sar_filed": filed_sar,
            "filing_status": "Success" if filed_sar else "N/A"
        }
    }

@app.get("/compliance/reports")
def get_compliance_reports():
    return {
        "total_filed": len(sar_reports),
        "reports": sar_reports
    }

def generate_sar_narrative(score: float, top_features: List[str], input_data: dict) -> str:
    """
    Generates a formal SAR narrative based on fraud indicators.
    """
    date_str = date.today().isoformat()
    indicators_str = ", ".join(top_features)
    
    narrative = (
        f"Suspicious Activity Report generated on {date_str}. "
        f"The system detected a high-risk booking with a probability score of {round(score, 4)}. "
        f"Primary indicators of concern include: {indicators_str}. "
    )
    
    # Add context from input data if available
    if "lead_time" in input_data or "lead_time_days" in input_data:
        lt = input_data.get("lead_time") or input_data.get("lead_time_days")
        narrative += f"Of particular note is the lead time of {lt} days. "
        
    narrative += "This booking has been intercepted and flagged for mandatory compliance review."
    return narrative

@app.post("/forecast/demand")
def forecast_demand(request: DemandForecastRequest):
    # Get model from registry
    version, artifacts = registry.get_model("demand")
    demand_model = artifacts["path"]
    demand_features = artifacts["features"]
    demand_metadata = artifacts["metadata"]

    try:
        # Initial dataframe setup
        history_df = pd.DataFrame([h.dict() for h in request.history])
        history_df["date"] = pd.to_datetime(history_df["date"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    forecasts = []
    residual_std = demand_metadata.get("residual_std", 0.1013)
    z_score = 1.96
    
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

        # Confidence Interval
        lower = pred - z_score * residual_std
        upper = pred + z_score * residual_std
        
        # Next Forecast Date
        last_date = history_df["date"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        forecasts.append({
            "date": next_date.date(),
            "predicted_occupancy": round(pred, 3),
            "confidence_interval": {
                "lower": round(max(0.0, lower), 3),
                "upper": round(min(1.0, upper), 3),
                "level": "95%"
            }
        })
        
        # Append prediction to history_df for next iteration
        new_row = history_df.iloc[-1].copy()
        new_row["date"] = next_date
        new_row["occupancy"] = pred # Update with predicted occupancy
        
        # Append using loc to avoid deprecated append
        history_df.loc[len(history_df)] = new_row

    return {
        "model_version": version,
        "model_metadata": {
            "version": demand_metadata["version"],
            "mae": demand_metadata["mae"]
        },
        "forecast_days": request.forecast_days,
        "forecast": forecasts
    }

def explain_anomaly(input_df: pd.DataFrame, artifacts: dict):
    explanations = []
    
    # Define features inside function to avoid global dependencies
    features = [
        'room_access_count', 'dnd_hours', 'visitor_count', 'noise_complaints', 
        'service_usage_count', 'cleaning_duration_minutes', 'maintenance_issues', 
        'maintenance_resolution_hours', 'staff_overtime_hours', 'guest_rating_staff', 
        'access_rate', 'service_intensity', 'stay_length'
    ]
    
    # Try to get means and stds from artifacts, fallback to simple placeholder if missing
    means = artifacts.get("means")
    stds = artifacts.get("stds")
    
    if not means or not stds:
        return [{"feature": "Stats Unavailable", "value": 0, "z_score": 0, "direction": "N/A"}]

    for col in features:
        if col not in input_df.columns:
            continue
        value = input_df.iloc[0][col]
        mean = means.get(col, 0)
        std = stds.get(col, 1)

        z_score = (value - mean) / std if std != 0 else 0

        explanations.append({
            "feature": col,
            "value": round(float(value), 3),
            "z_score": round(float(z_score), 3),
            "direction": "High" if z_score > 0 else "Low"
        })

    # sort by absolute deviation
    explanations = sorted(
        explanations,
        key=lambda x: abs(x["z_score"]),
        reverse=True
    )

    return explanations[:3]

@app.post("/monitor/in-stay")
def detect_in_stay_anomaly(request: InStayRequest):
    # Get model from registry
    version, artifacts = registry.get_model("in-stay")
    anomaly_model = artifacts["path"]
    anomaly_scaler = artifacts["scaler"]
    
    # Map API fields to model features (handling naming variations in artifacts)
    data = request.dict()
    # Map 'service_usage_intensity' to 'service_usage_count' as seen in the model
    data['service_usage_count'] = data.pop('service_usage_intensity')
    
    input_df = pd.DataFrame([data])
    
    input_df = input_df.reindex(columns=[
        'room_access_count', 'dnd_hours', 'visitor_count', 'noise_complaints', 
        'service_usage_count', 'cleaning_duration_minutes', 'maintenance_issues', 
        'maintenance_resolution_hours', 'staff_overtime_hours', 'guest_rating_staff', 
        'access_rate', 'service_intensity', 'stay_length'
    ], fill_value=0)

    X_scaled = anomaly_scaler.transform(input_df)

    score = anomaly_model.decision_function(X_scaled)[0]
    prediction = anomaly_model.predict(X_scaled)[0]

    explanation = explain_anomaly(input_df, artifacts)

    return {
        "anomaly_score": round(float(score), 4),
        "status": "Anomaly" if prediction == -1 else "Normal",
        "risk_level": (
            "High" if score < -0.05 else
            "Moderate" if score < 0 else
            "Low"
        ),
        "top_contributing_factors": explanation,
        "model_info": {
            "service": "in-stay",
            "version": version
        }
    }

# --- Management Endpoints ---
@app.get("/admin/models")
def list_models():
    """Returns the current model registry state."""
    return registry.models

@app.post("/admin/models/split")
def update_traffic_split(request: SplitRequest):
    """Updates the traffic split for a specific service."""
    service = request.service
    split = request.split
    
    if service not in registry.models:
        raise HTTPException(status_code=404, detail="Service not found")
    
    # Validation
    if not np.isclose(sum(split.values()), 1.0):
        raise HTTPException(status_code=400, detail="Traffic split must sum to 1.0")
    
    for v in split.keys():
        if v not in registry.models[service]["versions"]:
            raise HTTPException(status_code=400, detail=f"Version {v} not found for {service}")
            
    registry.models[service]["traffic_split"] = split
    return {"status": "success", "new_split": split}

@app.post("/admin/models/rollback")
def rollback_model(request: RollbackRequest):
    """Instantly reverts traffic to a stable version."""
    service = request.service
    target_version = request.target_version
    
    if service not in registry.models:
        raise HTTPException(status_code=404, detail="Service not found")
    
    if target_version not in registry.models[service]["versions"]:
        raise HTTPException(status_code=400, detail=f"Target version {target_version} not found")
        
    registry.models[service]["traffic_split"] = {target_version: 1.0}
    registry.models[service]["active_version"] = target_version
    
    return {
        "status": "success", 
        "message": f"Service {service} rolled back to {target_version}",
        "new_split": registry.models[service]["traffic_split"]
    }

def estimate_elasticity(history: List[dict]) -> float:
    """
    Estimates price elasticity of demand using simple log-linear regression:
    log(Occupancy) = alpha + beta * log(ADR)
    The coefficient 'beta' is the price elasticity.
    """
    if len(history) < 5:
        return -1.5 # Default inelastic assumption for hotels
    
    df = pd.DataFrame(history)
    
    # Filter for valid data
    df = df[(df["occupancy"] > 0) & (df["adr"] > 0)]
    
    if len(df) < 3:
        return -1.5

    # Log transformation
    y = np.log(df["occupancy"])
    X = np.log(df["adr"])
    
    # Simple linear regression (slope)
    # beta = sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2)
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean)**2)
    
    if denominator == 0:
        return -1.5
        
    beta = float(numerator / denominator)
    
    # Sanity check: Elasticity for hotels is typically between -0.5 and -3.0
    return max(-5.0, min(-0.1, beta))

def dynamic_pricing_logic(data: PricingRequest):
    # Determine elasticity
    elasticity = data.elasticity
    if elasticity is None and data.historical_data:
        elasticity = float(estimate_elasticity(data.historical_data))
    
    # Fallback to inelastic default if still None/0
    if not elasticity:
        elasticity = -1.5

    adjustment = 0.0
    demand_level = "Moderate"

    if data.upper_ci > 0.85:
        adjustment = 0.12
        demand_level = "Very High"
    elif data.upper_ci > 0.70:
        adjustment = 0.07
        demand_level = "High"
    elif data.upper_ci > 0.55:
        adjustment = 0.0
        demand_level = "Moderate"
    elif data.upper_ci > 0.40:
        adjustment = -0.06
        demand_level = "Soft"
    else:
        adjustment = -0.10
        demand_level = "Weak"

    # Elasticity Modifier: 
    # If guests are sensitive (e.g., -3.0), we dampen the increase.
    # If guests are inelastic (e.g., -0.5), we can push the increase higher.
    # Baseline elasticity = -1.5. 
    elasticity_modifier = float(-1.5 / elasticity)
    elasticity_modifier = max(0.5, min(2.0, elasticity_modifier))
    
    # Apply modifier only to price increases (to avoid dropping price too fast on sensitive demand)
    if adjustment > 0:
        adjustment *= elasticity_modifier

    # Seasonal boost
    if data.season.lower() in ["peak", "peak season", "holiday"]:
        adjustment += 0.05

    recommended_price = float(data.current_adr * (1 + adjustment))

    # Competitor guardrail
    max_price = float(data.competitor_price * 1.15)
    recommended_price = min(recommended_price, max_price)

    return {
        "demand_level": demand_level,
        "estimated_elasticity": round(float(elasticity), 3),
        "adjustment_percent": round(float(adjustment) * 100, 2),
        "recommended_price": round(float(recommended_price), 2),
        "guardrail_applied": bool(recommended_price == max_price)
    }

@app.post("/pricing/recommend")
def recommend_price(request: PricingRequest):
    result = dynamic_pricing_logic(request)

    return {
        "current_adr": request.current_adr,
        "forecast_occupancy": request.forecast_occupancy,
        "confidence_interval": {
            "lower": request.lower_ci,
            "upper": request.upper_ci
        },
        "pricing_decision": result
    }

@app.post("/recommendations")
def generate_recommendations(request: RecommendationRequest):
    upsell_score = compute_upsell_score(request)
    upsell_offer = generate_upsell_offer(upsell_score)
    
    loyalty_score = compute_loyalty_score(request)
    loyalty_tier = generate_loyalty_action(loyalty_score)
    
    return {
        "upsell": {
            "score": round(upsell_score, 2),
            "recommendation": upsell_offer
        },
        "loyalty": {
            "score": round(loyalty_score, 2),
            "tier": loyalty_tier
        }
    }

def compute_upsell_score(data: RecommendationRequest):
    score = 0
    if data.stay_length >= 3:
        score += 0.2
    if data.service_usage_count > 2:
        score += 0.2
    if data.guest_rating_staff >= 4:
        score += 0.2
    if data.refund_rate < 0.1:
        score += 0.2
    if data.room_price > 120:
        score += 0.2
    return min(score, 1.0)

def generate_upsell_offer(score: float):
    if score > 0.8:
        return {"offer": "Premium Suite Upgrade", "discount": 0, "confidence": "High"}
    elif score > 0.6:
        return {"offer": "Late Checkout + Breakfast Bundle", "discount": 5, "confidence": "Medium"}
    elif score > 0.4:
        return {"offer": "Spa Discount", "discount": 10, "confidence": "Moderate"}
    else:
        return {"offer": "No Upsell", "confidence": "Low"}

def compute_loyalty_score(data: RecommendationRequest):
    score = 0
    score += min(data.booking_history_count * 0.1, 0.4)
    if data.cancellation_history == 0:
        score += 0.2
    if data.refund_history == 0:
        score += 0.2
    if data.channel_source == "direct":
        score += 0.2
    return min(score, 1.0)

def generate_loyalty_action(score: float):
    if score > 0.8:
        return "Gold Tier – Exclusive Member Perks"
    elif score > 0.6:
        return "Silver Tier – 10% Rebooking Discount"
    elif score > 0.4:
        return "Bronze Tier – 5% Next Stay Coupon"
    else:
        return "Standard Guest"
