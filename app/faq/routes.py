"""
FAQ bot API routes. Mounted under /faq in the main app.
Uses app.faq.gemini_bot and app.faq.loader; data lives in app/core/data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.faq.gemini_bot import ask as faq_ask, GeminiQuotaError
from app.faq.loader import get_hotel_ids

router = APIRouter(prefix="/faq", tags=["FAQ Bot"])


@router.get("")
def faq_root():
    """FAQ bot info. Use POST /faq/ask to ask a question, GET /faq/hotels to list hotels."""
    return {
        "message": "FAQ bot (Gemini). Ask a question with POST /faq/ask.",
        "endpoints": {
            "ask": "POST /faq/ask — body: {\"question\": \"...\", \"hotel_id\": \"...\"}",
            "hotels": "GET /faq/hotels — list hotel IDs with FAQs",
        },
    }


@router.get("/ask")
def ask_faq_get():
    """FAQ ask is POST-only. Use POST /faq/ask with JSON body {\"question\": \"...\", \"hotel_id\": \"...\"}."""
    return {
        "message": "Use POST with JSON body: {\"question\": \"your question\", \"hotel_id\": \"hotel_001\"} (hotel_id optional).",
        "docs": "/docs#/FAQ%20Bot/ask_faq_ask_faq_post",
    }


class FAQAskRequest(BaseModel):
    """Request body for the FAQ bot."""

    question: str = Field(..., min_length=1, description="Guest question")
    hotel_id: Optional[str] = Field(
        default=None,
        description="Optional hotel ID to scope FAQs; if omitted, all hotels are used",
    )


class FAQAskResponse(BaseModel):
    """Response from the FAQ bot."""

    answer: str
    hotel_id: Optional[str] = None


@router.post("/ask", response_model=FAQAskResponse)
def ask_faq(request: FAQAskRequest):
    """
    Ask the Gemini-powered FAQ bot a question.
    Replies are based on hotel FAQs in app/core/data and are meant to be smart and human-like.
    """
    try:
        answer = faq_ask(question=request.question, hotel_id=request.hotel_id)
        return FAQAskResponse(answer=answer, hotel_id=request.hotel_id)
    except ValueError as e:
        if "GROQ_API_KEY" in str(e) or "GEMINI_API_KEY" in str(e):
            raise HTTPException(
                status_code=503,
                detail="FAQ bot is not configured. Set GROQ_API_KEY or GEMINI_API_KEY.",
            ) from e
        raise HTTPException(status_code=400, detail=str(e)) from e
    except GeminiQuotaError as e:
        # 429 / quota exceeded – return 503 with Retry-After so clients can back off
        retry_after = int(e.retry_after_seconds) if e.retry_after_seconds else 60
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": str(retry_after)},
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAQ bot error: {e}") from e


@router.get("/hotels")
def list_hotels():
    """Return list of hotel IDs that have FAQs (for dropdowns or validation)."""
    return {"hotel_ids": get_hotel_ids()}
