from __future__ import annotations

import re

_PREFIX_RE = re.compile(
    r"^(?:final answer\s*[:\-]\s*|answer\s*[:\-]\s*|the answer is\s*)",
    re.IGNORECASE,
)


def _normalize(answer: str) -> str:
    normalized = _PREFIX_RE.sub("", answer.strip()).strip()
    normalized = re.sub(r"[.!?]+$", "", normalized).strip()
    return normalized


def formatter(state) -> dict:
    raw = state["final_answer"] or state["draft_answer"] or ""
    return {"final_answer": _normalize(raw)}
