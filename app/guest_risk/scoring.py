"""
Guest risk scoring: formula, tiers, trend, watchlist, feature aggregates.
MVP = weighted scoring formula (no trained model).
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

# Risk tier bounds (guide: 0-30 Low, 31-60 Medium, 61-85 High, 86-100 Critical)
TIER_BOUNDS = [(0, 30, "Low"), (31, 60, "Medium"), (61, 85, "High"), (86, 100, "Critical")]

# Align with main fraud model threshold
FRAUD_FLAG_THRESHOLD = 0.42


def risk_tier(score: float) -> str:
    """Map risk_score 0-100 to tier label."""
    for lo, hi, label in TIER_BOUNDS:
        if lo <= score <= hi:
            return label
    return "Critical" if score > 100 else "Low"


def weighted_guest_risk(
    avg_fraud_score: float,
    max_fraud_score: float,
    fraud_booking_ratio: float,
    cancellation_ratio: float,
    refund_ratio: float,
    no_show_ratio: float,
    dispute_factor: float,
    chargeback_factor: float,
    short_lead_ratio: float,
    anomaly_factor: float = 0.0,
) -> float:
    """
    Combined guest risk score (0-100). Weights are tunable.
    All ratios/factors in 0-1 except we scale fraud (0-1) into 0-100 space.
    """
    # Normalize fraud contribution (0-1 -> 0-50)
    fraud_component = (0.6 * avg_fraud_score + 0.4 * max_fraud_score) * 50
    cancellation_component = min(cancellation_ratio * 25, 25)
    refund_component = min(refund_ratio * 20, 20)
    no_show_component = min(no_show_ratio * 15, 15)
    dispute_component = min(dispute_factor * 10, 10)
    chargeback_component = min(chargeback_factor * 15, 15)
    short_lead_component = min(short_lead_ratio * 10, 10)
    anomaly_component = min(anomaly_factor * 10, 10)
    # Cap total and scale to 0-100
    raw = (
        fraud_component
        + cancellation_component
        + refund_component
        + no_show_component
        + dispute_component
        + chargeback_component
        + short_lead_component
        + anomaly_component
    )
    return round(min(100.0, max(0.0, raw)), 1)


def trend_slope_from_scores(recent_fraud_scores: List[float]) -> float:
    """Return slope of last 6 fraud scores (for trend detection and storage)."""
    if len(recent_fraud_scores) < 2:
        return 0.0
    n = min(6, len(recent_fraud_scores))
    recent = recent_fraud_scores[-n:]
    x = list(range(n))
    y = recent
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def trend_from_scores(recent_fraud_scores: List[float]) -> str:
    """Trend = slope of last 6 fraud scores. Simple linear slope."""
    slope = trend_slope_from_scores(recent_fraud_scores)
    if slope > 0.02:
        return "Increasing"
    if slope < -0.02:
        return "Decreasing"
    return "Stable"


def watchlist_rule(
    risk_score: float,
    high_risk_booking_count_90d: int,
    confirmed_fraud_count: int,
) -> bool:
    """Auto-add to watchlist: score >= 80 OR 3+ high-risk in 90d OR 1+ confirmed fraud."""
    if risk_score >= 80:
        return True
    if high_risk_booking_count_90d >= 3:
        return True
    if confirmed_fraud_count >= 1:
        return True
    return False


def build_risk_factors(
    cancellation_ratio: float,
    refund_ratio: float,
    short_lead_ratio: float,
    fraud_booking_ratio: float,
    high_risk_count: int,
    total_bookings: int,
    cancellations: int,
    refunds: int,
) -> List[str]:
    """Human-readable risk factors (SHAP-style aggregates)."""
    factors: List[str] = []
    if total_bookings >= 1 and high_risk_count >= 1:
        factors.append(f"{high_risk_count} high-risk booking(s) in last 12 months")
    if cancellations > 0:
        factors.append(f"{cancellations} cancellation(s) in 60 days")
    if refunds > 0:
        factors.append(f"{refunds} refund(s)")
    if short_lead_ratio > 0.5 and total_bookings >= 2:
        factors.append("Short lead-time pattern")
    if refund_ratio > 0.3 and total_bookings >= 2:
        factors.append("High refund rate")
    if cancellation_ratio > 0.3 and total_bookings >= 2:
        factors.append("High cancellation ratio")
    if fraud_booking_ratio > 0.2 and total_bookings >= 2:
        factors.append("Repeated high fraud-score bookings")
    if not factors:
        factors.append("Insufficient history or low-risk indicators")
    return factors
