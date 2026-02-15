"""
Guest Risk Profiling API. Prefix: /api/v1.
Endpoints: POST /guest/risk-profile, GET /guest/risk-profile/{guest_id}, POST /guest/booking.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.guest_risk.aggregation import compute_profile
from app.guest_risk.db import (
    get_bookings_for_guest,
    get_profile,
    init_db,
    record_booking,
    upsert_profile,
)
from app.guest_risk.schema import (
    BookingOutcomeRecord,
    RecordBookingRequest,
    RiskProfileRequest,
    RiskProfileResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Guest Risk"])


@router.post("/guest/risk-profile", response_model=RiskProfileResponse)
def post_guest_risk_profile(request: RiskProfileRequest):
    """
    Compute or refresh risk profile for a guest and return it.
    Uses booking_records for last 12 months; updates guest_risk_profiles.
    """
    init_db()
    guest_id = request.guest_id.strip()
    if not guest_id:
        raise HTTPException(status_code=400, detail="guest_id is required")

    bookings = get_bookings_for_guest(guest_id)
    profile = compute_profile(guest_id, bookings)

    # Persist
    upsert_profile(
        guest_id=guest_id,
        risk_score=profile["risk_score"],
        risk_tier=profile["risk_tier"],
        total_bookings=profile["total_bookings"],
        fraud_flags=profile["fraud_flags"],
        cancellations=profile["cancellations"],
        refunds=profile["refunds"],
        risk_trend=profile.get("risk_trend_slope"),
        watchlist_status=profile["watchlist"],
        risk_factors=profile["risk_factors"],
    )
    # Reload to get last_updated
    stored = get_profile(guest_id) or profile

    last_updated = None
    if stored.get("last_updated"):
        try:
            last_updated = datetime.fromisoformat(
                stored["last_updated"].replace("Z", "+00:00")
            )
        except Exception:
            pass

    return RiskProfileResponse(
        guest_id=guest_id,
        risk_score=profile["risk_score"],
        risk_tier=profile["risk_tier"],
        watchlist=profile["watchlist"],
        risk_factors=profile["risk_factors"],
        trend=profile["trend"],
        total_bookings=profile["total_bookings"],
        fraud_flags=profile["fraud_flags"],
        cancellations=profile["cancellations"],
        refunds=profile["refunds"],
        last_updated=last_updated,
    )


@router.get("/guest/risk-profile/{guest_id}", response_model=RiskProfileResponse)
def get_guest_risk_profile(guest_id: str):
    """
    Return stored risk profile for guest. Call POST /guest/risk-profile to compute/refresh.
    """
    init_db()
    profile = get_profile(guest_id)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"No risk profile for guest_id={guest_id}. Use POST /api/v1/guest/risk-profile to compute.",
        )
    last_updated = None
    if profile.get("last_updated"):
        try:
            last_updated = datetime.fromisoformat(
                profile["last_updated"].replace("Z", "+00:00")
            )
        except Exception:
            pass
    return RiskProfileResponse(
        guest_id=profile["guest_id"],
        risk_score=profile["risk_score"],
        risk_tier=profile["risk_tier"],
        watchlist=profile["watchlist_status"],
        risk_factors=profile["risk_factors"],
        trend=profile.get("trend") or "Stable",
        total_bookings=profile["total_bookings"],
        fraud_flags=profile["fraud_flags"],
        cancellations=profile["cancellations"],
        refunds=profile["refunds"],
        last_updated=last_updated,
    )


@router.post("/guest/booking")
def post_guest_booking(request: RecordBookingRequest):
    """
    Record a booking outcome for a guest (fraud score + ops flags).
    Use this after scoring a booking so guest risk can aggregate over time.
    """
    init_db()
    b: BookingOutcomeRecord = request.booking
    record_booking(
        guest_id=b.guest_id,
        booking_date=b.booking_date,
        fraud_probability=b.fraud_probability,
        property_id=b.property_id,
        amount=b.amount,
        lead_time_days=b.lead_time_days,
        cancelled=b.cancelled,
        refunded=b.refunded,
        no_show=b.no_show,
        dispute=b.dispute,
        chargeback=b.chargeback,
    )
    return {"status": "ok", "message": "Booking recorded for guest risk aggregation."}
