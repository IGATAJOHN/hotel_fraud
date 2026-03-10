import pandas as pd
import numpy as np
from typing import List, Any

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
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean)**2)
    
    if denominator == 0:
        return -1.5
        
    beta = float(numerator / denominator)
    
    return max(-5.0, min(-0.1, beta))

def dynamic_pricing_logic(data: Any):
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

    elasticity_modifier = float(-1.5 / elasticity)
    elasticity_modifier = max(0.5, min(2.0, elasticity_modifier))
    
    if adjustment > 0:
        adjustment *= elasticity_modifier

    if data.season.lower() in ["peak", "peak season", "holiday"]:
        adjustment += 0.05

    recommended_price = float(data.current_adr * (1 + adjustment))
    max_price = float(data.competitor_price * 1.15)
    recommended_price = min(recommended_price, max_price)

    return {
        "demand_level": demand_level,
        "estimated_elasticity": round(float(elasticity), 3),
        "adjustment_percent": round(float(adjustment) * 100, 2),
        "recommended_price": round(float(recommended_price), 2),
        "guardrail_applied": bool(recommended_price == max_price)
    }

def compute_upsell_score(data: Any):
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

def compute_loyalty_score(data: Any):
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

WEIGHTS = {
    "booking": 0.45,
    "payment": 0.25,
    "behavior": 0.20,
    "network": 0.10
}

def compute_final_risk(booking, payment, behavior, network):

    final_score = (
        booking * WEIGHTS["booking"] +
        payment * WEIGHTS["payment"] +
        behavior * WEIGHTS["behavior"] +
        network * WEIGHTS["network"]
    )

    return round(final_score, 4)

