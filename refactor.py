import re

with open("main.py", "r") as f:
    content = f.read()

# Paths and Imports
content = content.replace('"path": "model.pkl"', '"path": "models/booking_model.pkl"')
content = content.replace('"path": "in_stay_anomaly_v1.pkl"', '"path": "models/instay_model.pkl"')
content = content.replace('"explainer": "explainer.pkl"', '"explainer": "artifacts/booking_explainer.pkl"')
content = content.replace('"features": "feature_columns.pkl"', '"features": "artifacts/feature_columns.pkl"')
content = content.replace('"path": "demand_forecast_lgbm_v1.pkl"', '"path": "models/demand_forecast_lgbm_v1.pkl"')
content = content.replace('"features": "demand_forecast_features_v1.pkl"', '"features": "artifacts/demand_forecast_features_v1.pkl"')
content = content.replace('"metadata": "demand_forecast_metadata_v1.pkl"', '"metadata": "artifacts/demand_forecast_metadata_v1.pkl"')
content = content.replace('"scaler": "in_stay_scaler_v1.pkl"', '"scaler": "artifacts/in_stay_scaler_v1.pkl"')

imports = """import random
import uuid

from risk_engine.feature_builder import compute_features, prepare_demand_features
from risk_engine.policy import BEST_THRESHOLD, SARRecord, sar_reports, generate_sar_narrative
from risk_engine.composite import (
    explain_anomaly, estimate_elasticity, dynamic_pricing_logic,
    compute_upsell_score, generate_upsell_offer, compute_loyalty_score, generate_loyalty_action
)"""
content = re.sub(r'import random\s+import uuid', imports, content)

# Constants & SAR
content = re.sub(r'# Global Constants\s+BEST_THRESHOLD = \d+\.\d+', '', content)
content = re.sub(r'class SARRecord.*?sar_reports: List\[SARRecord\] = \[\]\n*', '', content, flags=re.DOTALL)

# Compute Features
content = re.sub(r'def compute_features\(.*?return df\n*', '', content, flags=re.DOTALL)
content = re.sub(r'def prepare_demand_features\(.*?return df\n*', '', content, flags=re.DOTALL)

# Generate SAR Narrative
content = re.sub(r'def generate_sar_narrative\(.*?return narrative\n*', '', content, flags=re.DOTALL)

# Explain Anomaly
content = re.sub(r'def explain_anomaly\(.*?return explanations\[:3\]\n*', '', content, flags=re.DOTALL)

# Pricing Logic
content = re.sub(r'def estimate_elasticity\(.*?return max\(-5\.0, min\(-0\.1, beta\)\)\n*', '', content, flags=re.DOTALL)
content = re.sub(r'def dynamic_pricing_logic\(.*?\"guardrail_applied\": bool\(recommended_price == max_price\)\n    \}\n*', '', content, flags=re.DOTALL)

# Recommendations
content = re.sub(r'def compute_upsell_score\(.*?(?=\Z)', '', content, flags=re.DOTALL)

# Remove extra newlines
content = re.sub(r'\n{4,}', '\n\n', content)

with open("main.py", "w") as f:
    f.write(content)
