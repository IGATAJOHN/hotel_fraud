"""
Guest risk profile aggregation: feature engineering and risk scoring.
Consumes booking_records; produces guest-level profile for storage and API.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

from app.guest_risk.scoring import (
    FRAUD_FLAG_THRESHOLD,
    build_risk_factors,
    risk_tier,
    trend_from_scores,
    trend_slope_from_scores,
    watchlist_rule,
    weighted_guest_risk,
)


def _parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD to datetime at midnight."""
    return datetime.strptime(s[:10], "%Y-%m-%d")


def _days_ago(n: int) -> str:
    """Date string N days ago (YYYY-MM-DD)."""
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")


def compute_profile(guest_id: str, bookings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate booking records into a guest risk profile.
    Uses last 12 months for main aggregates; 60d for cancellations; 90d for watchlist.
    Returns dict with risk_score, risk_tier, watchlist, risk_factors, trend, etc.
    """
    now = datetime.utcnow()
    cutoff_12m = _days_ago(365)
    cutoff_60d = _days_ago(60)
    cutoff_90d = _days_ago(90)

    # Filter by 12 months for core aggregates
    recent = [b for b in bookings if b.get("booking_date", "") >= cutoff_12m]
    total = len(recent)

    if total == 0:
        return _empty_profile(guest_id)

    fraud_scores = [b["fraud_probability"] for b in recent]
    avg_fraud = sum(fraud_scores) / total
    max_fraud = max(fraud_scores)

    high_risk = [b for b in recent if b["fraud_probability"] >= FRAUD_FLAG_THRESHOLD]
    fraud_booking_ratio = len(high_risk) / total
    fraud_flags = len(high_risk)

    cancellations = sum(1 for b in recent if b.get("cancelled"))
    refunds = sum(1 for b in recent if b.get("refunded"))
    no_shows = sum(1 for b in recent if b.get("no_show"))
    disputes = sum(1 for b in recent if b.get("dispute"))
    chargebacks = sum(1 for b in recent if b.get("chargeback"))

    cancellation_ratio = cancellations / total
    refund_ratio = refunds / total
    no_show_ratio = no_shows / total
    dispute_factor = disputes / total
    chargeback_factor = chargebacks / total if total > 0 else 0.0

    lead_times = [
        b["lead_time_days"]
        for b in recent
        if b.get("lead_time_days") is not None and b["lead_time_days"] is not None
    ]
    if lead_times:
        short_lead_count = sum(1 for lt in lead_times if lt < 7)
        short_lead_ratio = short_lead_count / len(lead_times)
    else:
        short_lead_ratio = 0.0

    # 90d high-risk count for watchlist
    bookings_90d = [b for b in recent if b.get("booking_date", "") >= cutoff_90d]
    high_risk_90d = sum(
        1 for b in bookings_90d if b["fraud_probability"] >= FRAUD_FLAG_THRESHOLD
    )

    # 60d cancellations for risk factors wording
    cancellations_60d = sum(
        1 for b in recent if b.get("cancelled") and b.get("booking_date", "") >= cutoff_60d
    )

    risk_score = weighted_guest_risk(
        avg_fraud_score=avg_fraud,
        max_fraud_score=max_fraud,
        fraud_booking_ratio=fraud_booking_ratio,
        cancellation_ratio=cancellation_ratio,
        refund_ratio=refund_ratio,
        no_show_ratio=no_show_ratio,
        dispute_factor=dispute_factor,
        chargeback_factor=chargeback_factor,
        short_lead_ratio=short_lead_ratio,
        anomaly_factor=0.0,
    )

    tier = risk_tier(risk_score)
    trend_str = trend_from_scores(fraud_scores)
    trend_slope = trend_slope_from_scores(fraud_scores)
    watchlist = watchlist_rule(
        risk_score=risk_score,
        high_risk_booking_count_90d=high_risk_90d,
        confirmed_fraud_count=chargebacks,
    )

    risk_factors = build_risk_factors(
        cancellation_ratio=cancellation_ratio,
        refund_ratio=refund_ratio,
        short_lead_ratio=short_lead_ratio,
        fraud_booking_ratio=fraud_booking_ratio,
        high_risk_count=fraud_flags,
        total_bookings=total,
        cancellations=cancellations_60d,
        refunds=refunds,
    )

    return {
        "guest_id": guest_id,
        "risk_score": risk_score,
        "risk_tier": tier,
        "watchlist": watchlist,
        "risk_factors": risk_factors,
        "trend": trend_str,
        "risk_trend_slope": trend_slope,
        "total_bookings": total,
        "fraud_flags": fraud_flags,
        "cancellations": cancellations_60d,
        "refunds": refunds,
    }


def _empty_profile(guest_id: str) -> Dict[str, Any]:
    """
    Return a minimal profile when guest has no booking history.
    """
    return {
        "guest_id": guest_id,
        "risk_score": 0.0,
        "risk_tier": "Low",
        "watchlist": False,
        "risk_factors": ["Insufficient history or low-risk indicators"],
        "trend": "Stable",
        "risk_trend_slope": 0.0,
        "total_bookings": 0,
        "fraud_flags": 0,
        "cancellations": 0,
        "refunds": 0,
    }
