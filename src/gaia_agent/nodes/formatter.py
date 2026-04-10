from __future__ import annotations

import re
from gaia_agent.models import get_strong_model
from gaia_agent.prompts import FORMATTER_SYSTEM

_PREFIX_RE = re.compile(
    r"^(?:final answer\s*[:\-]\s*|answer\s*[:\-]\s*|the answer is\s*)",
    re.IGNORECASE,
)


def _normalize_regex(answer: str) -> str:
    normalized = _PREFIX_RE.sub("", answer.strip()).strip()
    normalized = re.sub(r"[.!?]+$", "", normalized).strip()
    return normalized


def formatter(state) -> dict:
    raw = state["final_answer"] or state["draft_answer"] or ""
    if not raw:
        return {"final_answer": ""}
    
    try:
        model = get_strong_model()
        response = model.invoke([
            {"role": "system", "content": FORMATTER_SYSTEM},
            {"role": "user", "content": f"Format this answer precisely: {raw}"}
        ])
        final_answer = response.content.strip()
        # Ensure we didn't get a hallucinated long story, if so, fallback
        if len(final_answer) > len(raw) + 10:
             final_answer = _normalize_regex(raw)
    except Exception:
        final_answer = _normalize_regex(raw)

    return {"final_answer": final_answer}
