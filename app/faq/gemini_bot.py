"""
FAQ bot: Gemini or Groq. Uses FAQ data from app/core/data and generates
smart, human-like replies. Backend chosen by env: GROQ_API_KEY → Groq, else GEMINI_API_KEY → Gemini.
"""

import os
import json
import re
from pathlib import Path
from typing import Optional

from app.faq.loader import get_faqs_for_hotel

# Backend: "groq" if GROQ_API_KEY set, else "gemini" if GEMINI_API_KEY set
def _backend() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    raise ValueError("Set GROQ_API_KEY or GEMINI_API_KEY for the FAQ bot.")


class GeminiQuotaError(Exception):
    """Raised when the LLM API returns 429 / quota exceeded (Gemini or Groq)."""

    def __init__(self, message: str, retry_after_seconds: Optional[float] = None):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


def _load_system_prompt() -> str:
    """Load system prompt from app/prompts (used by both backends)."""
    path = Path(__file__).resolve().parent.parent / "prompts" / "gemini_faq_system.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return (data.get("content") or "").strip()


def _faq_context_for_hotel(hotel_id: Optional[str]) -> str:
    """Build a single string of FAQ Q&A for the given hotel (or all hotels)."""
    faqs = get_faqs_for_hotel(hotel_id)
    if not faqs:
        return "No FAQs available."
    lines = []
    for item in faqs:
        hotel_name = item.get("hotel_name", "")
        prefix = f"[{hotel_name}] " if hotel_name else ""
        intents = item.get("intents", [])
        answer = item.get("answer", "")
        q_preview = intents[0] if intents else item.get("id", "?")
        lines.append(f"{prefix}Q: {q_preview}\nA: {answer}")
    return "\n\n".join(lines)


def _parse_retry_seconds(err_text: str) -> Optional[float]:
    """Try to extract retry-after seconds from error message."""
    m = re.search(r"[Rr]etry in (\d+(?:\.\d+)?)\s*s", err_text)
    return float(m.group(1)) if m else None


def _is_rate_limit_error(err_msg: str) -> bool:
    return (
        "429" in err_msg
        or "quota" in err_msg.lower()
        or "rate" in err_msg.lower()
        or "rate limit" in err_msg.lower()
    )


def _ask_groq(question: str, hotel_id: Optional[str], model_name: str) -> str:
    """Use Groq (OpenAI-compatible API)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    system_prompt = _load_system_prompt()
    faq_context = _faq_context_for_hotel(hotel_id)
    user_content = (
        "Here are the hotel FAQs you must use to answer:\n\n"
        f"{faq_context}\n\n"
        "Guest question:\n"
        f'"{question}"\n\n'
        "Reply in a friendly, human way based only on the FAQs above."
    )
    try:
        r = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
    except Exception as e:
        err_msg = str(e)
        if _is_rate_limit_error(err_msg):
            raise GeminiQuotaError(
                "We're getting a lot of questions right now. Please try again in a minute, or contact the front desk.",
                retry_after_seconds=_parse_retry_seconds(err_msg) or 60,
            ) from e
        raise
    if not r.choices or not r.choices[0].message.content:
        return "I couldn't generate a reply right now. Please contact the front desk."
    return r.choices[0].message.content.strip()


def _ask_gemini(question: str, hotel_id: Optional[str], model_name: str) -> str:
    """Use Google Gemini."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    system_prompt = _load_system_prompt()
    faq_context = _faq_context_for_hotel(hotel_id)
    user_content = (
        "Here are the hotel FAQs you must use to answer:\n\n"
        f"{faq_context}\n\n"
        "Guest question:\n"
        f'"{question}"\n\n'
        "Reply in a friendly, human way based only on the FAQs above."
    )
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )
    try:
        response = model.generate_content(user_content)
    except Exception as e:
        err_msg = str(e)
        if _is_rate_limit_error(err_msg):
            raise GeminiQuotaError(
                "We're getting a lot of questions right now. Please try again in a minute, or contact the front desk.",
                retry_after_seconds=_parse_retry_seconds(err_msg),
            ) from e
        raise
    if not response or not response.text:
        return "I couldn't generate a reply right now. Please contact the front desk."
    return response.text.strip()


def ask(
    question: str,
    hotel_id: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Answer a guest question using FAQ data. Uses Groq if GROQ_API_KEY is set,
    otherwise Gemini. Returns a single string reply.
    Raises GeminiQuotaError on 429 / quota exceeded.
    """
    backend = _backend()
    if backend == "groq":
        model_name = model_name or os.environ.get("GROQ_FAQ_MODEL", "llama-3.3-70b-versatile")
        return _ask_groq(question, hotel_id, model_name)
    else:
        model_name = model_name or os.environ.get("GEMINI_FAQ_MODEL", "gemini-2.0-flash")
        return _ask_gemini(question, hotel_id, model_name)
