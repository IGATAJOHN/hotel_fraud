"""
Pydantic models for Guest Risk Profiling API.
Matches the developer guide: risk_score 0-100, risk_tier, risk_factors, trend, watchlist.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---- Request ----

class RiskProfileRequest(BaseModel):
    """Input for GET risk profile by guest."""
    guest_id: str = Field(..., description="Guest identifier")


class BookingOutcomeRecord(BaseModel):
    """Single booking outcome to feed into guest risk (from fraud model + ops)."""
    booking_id: Optional[str] = None
    guest_id: str
    booking_date: str  # ISO date "YYYY-MM-DD"
    property_id: Optional[str] = None
    fraud_probability: float = Field(..., ge=0, le=1)
    amount: Optional[float] = None
    lead_time_days: Optional[int] = None
    cancelled: bool = False
    refunded: bool = False
    no_show: bool = False
    dispute: bool = False
    chargeback: bool = False


class RecordBookingRequest(BaseModel):
    """Record a booking outcome to update guest risk profile."""
    booking: BookingOutcomeRecord


# ---- Response (guest_risk_profiles concept) ----

class RiskProfileResponse(BaseModel):
    """Output for POST /api/v1/guest/risk-profile."""
    guest_id: str
    risk_score: float = Field(..., ge=0, le=100)
    risk_tier: str  # Low | Medium | High | Critical
    watchlist: bool
    risk_factors: List[str] = Field(default_factory=list)
    trend: str  # "Stable" | "Increasing" | "Decreasing"
    total_bookings: int = 0
    fraud_flags: int = 0
    cancellations: int = 0
    refunds: int = 0
    last_updated: Optional[datetime] = None
