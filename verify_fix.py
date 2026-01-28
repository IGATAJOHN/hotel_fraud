import joblib
import pandas as pd
import numpy as np
import shap

def test_user_problem():
    print("--- Verifying Fix for User's KeyError ---")
    try:
        # Load artifacts
        model = joblib.load("model.pkl")
        FEATURE_COLUMNS = joblib.load("feature_columns.pkl")
        
        # User's problematic input
        user_input = {
            "lead_time": 1,
            "device_risk_score": 0.91,
            "price_deviation": 0.32,
            "booking_hour": 2,
            "is_new_customer": 1
        }
        
        # Map common aliases (from main.py)
        ALIASES = {
            "lead_time": "lead_time_days",
            "booking_hour": "checkin_time_hour"
        }
        
        # 1. Convert to DataFrame
        X = pd.DataFrame([user_input])
        print(f"Initial columns: {X.columns.tolist()}")
        
        # 2. Map aliases
        X = X.rename(columns=ALIASES)
        print(f"After alias mapping: {X.columns.tolist()}")
        
        # 3. Use reindex (The Improved Fix)
        # Using NaN allows CatBoost to use its internal missing value logic.
        X = X.reindex(columns=FEATURE_COLUMNS, fill_value=np.nan)
        print(f"After reindex with NaN, shape: {X.shape}")

        
        # 4. Predict
        proba = model.predict_proba(X)[0, 1]
        print(f"Success! Prediction Probability: {proba}")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_user_problem()
