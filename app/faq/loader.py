"""
Load FAQ data from app/core/data/faqs.json.
Uses paths relative to this package so it works regardless of CWD.
"""

import json
from pathlib import Path
from typing import Any

# Resolve path: app/faq/loader.py -> app/core/data/faqs.json
_FAQ_DATA_PATH = Path(__file__).resolve().parent.parent / "core" / "data" / "faqs.json"


def load_faq_data() -> dict[str, Any]:
    """Load and return the full FAQ JSON (hotels list with FAQs)."""
    with open(_FAQ_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_faqs_for_hotel(hotel_id: str | None) -> list[dict[str, Any]]:
    """
    Return FAQ items for one hotel, or all FAQs from all hotels if hotel_id is None.
    Each item has keys: id, intents, answer.
    """
    data = load_faq_data()
    hotels = data.get("hotels") or []

    if hotel_id:
        for h in hotels:
            if h.get("hotel_id") == hotel_id:
                return list(h.get("faqs") or [])
        return []

    # No hotel_id: flatten all FAQs and include hotel name for context
    out: list[dict[str, Any]] = []
    for h in hotels:
        name = h.get("hotel_name") or h.get("hotel_id") or "Hotel"
        for faq in h.get("faqs") or []:
            out.append({
                **faq,
                "hotel_name": name,
            })
    return out


def get_hotel_ids() -> list[str]:
    """Return list of known hotel_id values."""
    data = load_faq_data()
    hotels = data.get("hotels") or []
    return [h.get("hotel_id") for h in hotels if h.get("hotel_id")]
