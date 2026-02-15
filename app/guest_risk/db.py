"""
Guest Risk persistence: booking_records + guest_risk_profiles.
SQLite backend, production-ready with proper indexing and connection handling.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional

# DB path: project root / guest_risk.db
_DB_PATH = Path(__file__).resolve().parent.parent.parent / "guest_risk.db"
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    """Thread-local style connection. For SQLite single writer, one conn is fine."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def init_db() -> None:
    """
    Create tables if not exist. Idempotent.
    Runs on startup and before each guest risk operation.
    """
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS booking_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_id TEXT NOT NULL,
            booking_date TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            property_id TEXT,
            amount REAL,
            lead_time_days INTEGER,
            cancelled INTEGER NOT NULL DEFAULT 0,
            refunded INTEGER NOT NULL DEFAULT 0,
            no_show INTEGER NOT NULL DEFAULT 0,
            dispute INTEGER NOT NULL DEFAULT 0,
            chargeback INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_booking_guest_date ON booking_records(guest_id, booking_date)"
    )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS guest_risk_profiles (
            guest_id TEXT PRIMARY KEY,
            risk_score REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            total_bookings INTEGER NOT NULL DEFAULT 0,
            fraud_flags INTEGER NOT NULL DEFAULT 0,
            cancellations INTEGER NOT NULL DEFAULT 0,
            refunds INTEGER NOT NULL DEFAULT 0,
            last_updated TEXT NOT NULL,
            risk_trend REAL,
            watchlist_status INTEGER NOT NULL DEFAULT 0,
            risk_factors TEXT,
            trend TEXT
        )
    """)

    # Migration: add trend column if missing (legacy schema)
    try:
        cur.execute("ALTER TABLE guest_risk_profiles ADD COLUMN trend TEXT")
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise

    conn.commit()


def record_booking(
    guest_id: str,
    booking_date: str,
    fraud_probability: float,
    property_id: Optional[str] = None,
    amount: Optional[float] = None,
    lead_time_days: Optional[int] = None,
    cancelled: bool = False,
    refunded: bool = False,
    no_show: bool = False,
    dispute: bool = False,
    chargeback: bool = False,
) -> None:
    """
    Insert a booking outcome. Call after fraud scoring or from ops.
    Enables guest risk aggregation.
    """
    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO booking_records (
            guest_id, booking_date, fraud_probability, property_id, amount,
            lead_time_days, cancelled, refunded, no_show, dispute, chargeback, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            guest_id,
            booking_date,
            fraud_probability,
            property_id,
            amount,
            lead_time_days,
            1 if cancelled else 0,
            1 if refunded else 0,
            1 if no_show else 0,
            1 if dispute else 0,
            1 if chargeback else 0,
            now,
        ),
    )
    conn.commit()


def get_bookings_for_guest(
    guest_id: str,
    months_back: int = 12,
) -> List[dict[str, Any]]:
    """
    Fetch booking records for a guest in the last N months.
    Used by aggregation to compute profile.
    """
    conn = _get_conn()
    cur = conn.cursor()
    cutoff = (datetime.utcnow() - timedelta(days=months_back * 31)).strftime("%Y-%m-%d")
    cur.execute(
        """
        SELECT guest_id, booking_date, fraud_probability, property_id, amount,
               lead_time_days, cancelled, refunded, no_show, dispute, chargeback
        FROM booking_records
        WHERE guest_id = ? AND booking_date >= ?
        ORDER BY booking_date ASC
        """,
        (guest_id, cutoff),
    )
    rows = cur.fetchall()
    return [
        {
            "guest_id": r["guest_id"],
            "booking_date": r["booking_date"],
            "fraud_probability": r["fraud_probability"],
            "property_id": r["property_id"],
            "amount": r["amount"],
            "lead_time_days": r["lead_time_days"],
            "cancelled": bool(r["cancelled"]),
            "refunded": bool(r["refunded"]),
            "no_show": bool(r["no_show"]),
            "dispute": bool(r["dispute"]),
            "chargeback": bool(r["chargeback"]),
        }
        for r in rows
    ]


def get_profile(guest_id: str) -> Optional[dict[str, Any]]:
    """
    Return stored guest risk profile, or None if not found.
    """
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT guest_id, risk_score, risk_tier, total_bookings, fraud_flags,
               cancellations, refunds, last_updated, risk_trend, watchlist_status,
               risk_factors, trend
        FROM guest_risk_profiles
        WHERE guest_id = ?
        """,
        (guest_id,),
    )
    row = cur.fetchone()
    if not row:
        return None

    factors_raw = row["risk_factors"]
    risk_factors: List[str] = []
    if factors_raw:
        try:
            risk_factors = json.loads(factors_raw)
        except json.JSONDecodeError:
            risk_factors = [factors_raw]

    return {
        "guest_id": row["guest_id"],
        "risk_score": row["risk_score"],
        "risk_tier": row["risk_tier"],
        "total_bookings": row["total_bookings"],
        "fraud_flags": row["fraud_flags"],
        "cancellations": row["cancellations"],
        "refunds": row["refunds"],
        "last_updated": row["last_updated"],
        "risk_trend": row["risk_trend"],
        "watchlist_status": bool(row["watchlist_status"]),
        "risk_factors": risk_factors,
        "trend": row["trend"] or "Stable",
    }


def upsert_profile(
    guest_id: str,
    risk_score: float,
    risk_tier: str,
    total_bookings: int,
    fraud_flags: int,
    cancellations: int,
    refunds: int,
    risk_trend: Optional[float] = None,
    watchlist_status: bool = False,
    risk_factors: Optional[List[str]] = None,
    trend: Optional[str] = None,
) -> None:
    """
    Insert or update guest_risk_profiles row.
    """
    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat() + "Z"
    factors_json = json.dumps(risk_factors or [])

    cur.execute(
        """
        INSERT INTO guest_risk_profiles (
            guest_id, risk_score, risk_tier, total_bookings, fraud_flags,
            cancellations, refunds, last_updated, risk_trend, watchlist_status,
            risk_factors, trend
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(guest_id) DO UPDATE SET
            risk_score = excluded.risk_score,
            risk_tier = excluded.risk_tier,
            total_bookings = excluded.total_bookings,
            fraud_flags = excluded.fraud_flags,
            cancellations = excluded.cancellations,
            refunds = excluded.refunds,
            last_updated = excluded.last_updated,
            risk_trend = excluded.risk_trend,
            watchlist_status = excluded.watchlist_status,
            risk_factors = excluded.risk_factors,
            trend = excluded.trend
        """,
        (
            guest_id,
            risk_score,
            risk_tier,
            total_bookings,
            fraud_flags,
            cancellations,
            refunds,
            now,
            risk_trend,
            1 if watchlist_status else 0,
            factors_json,
            trend or "Stable",
        ),
    )
    conn.commit()
