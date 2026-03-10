from pydantic import BaseModel
from typing import List
from datetime import date

# Global Constants
BEST_THRESHOLD = 0.42

class SARRecord(BaseModel):
    booking_id: str
    fraud_score: float
    timestamp: date
    narrative: str
    top_indicators: List[str]
    status: str = "Filed"

# In-memory storage for auto-filed reports (mock database)
sar_reports: List[SARRecord] = []

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

def determine_action(score: float):
    if score < 0.40:
        return {
            "tier": "Low",
            "action": "Auto Approve"
        }
    elif score < 0.70:
        return {
            "tier": "Medium",
            "action": "Manual Review"
        }
    elif score < 0.85:
        return {
            "tier": "High",
            "action": "Hold Payment"
        }
    else:
        return {
            "tier": "Critical",
            "action": "Auto SAR Filing"
        }
