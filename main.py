from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap

# Load artifacts
try:
    model = joblib.load("model.pkl")
    explainer = joblib.load("explainer.pkl")
    FEATURE_COLUMNS = joblib.load("feature_columns.pkl")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    raise RuntimeError("Failed to load required model artifacts.")

BEST_THRESHOLD = 0.42  # <-- optimized threshold


app = FastAPI(title="Hotel Fraud Detection API")
class BookingRequest(BaseModel):
    data: dict
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

