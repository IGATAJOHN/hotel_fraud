import joblib
import pandas as pd
import numpy as np
import shap
import sys

def test_logic():
    print("--- Testing main.py Logic ---")
    try:
        # Load artifacts using joblib as in main.py
        model = joblib.load("model.pkl")
        print("Model loaded successfully.")
        
        explainer = joblib.load("explainer.pkl")
        print("Explainer loaded successfully.")
        
        FEATURE_COLUMNS = joblib.load("feature_columns.pkl")
        print(f"Feature columns loaded. Count: {len(FEATURE_COLUMNS)}")
        
        # Create a dummy record based on feature columns
        # Using 0 for all features as a baseline test
        dummy_data = {col: 0 for col in FEATURE_COLUMNS}
        X = pd.DataFrame([dummy_data])
        
        # Ensure column order
        X = X[FEATURE_COLUMNS]
        print("DataFrame created and columns ordered.")
        
        # Predict probability
        proba = model.predict_proba(X)[0, 1]
        print(f"Predicted Probability: {proba}")
        
        # SHAP explanation
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            vals = shap_values[0] if len(shap_values) == 1 else shap_values[1]
        else:
            vals = shap_values

        contrib = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "shap_value": vals[0]
        })
        print("Contribution DataFrame created successfully.")
        print(contrib.head())
        
        contrib["impact"] = contrib["shap_value"].abs()
        top_reasons = contrib.sort_values("impact", ascending=False).head(5)
        print(f"Top Reasons: {top_reasons['feature'].tolist()}")

        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logic()

