from __future__ import annotations

import re
from gaia_agent.prompts import FORMATTER_SYSTEM, apply_caveman

_PREFIX_RE = re.compile(
    r"^(?:final answer\s*[:\-]\s*|answer\s*[:\-]\s*|the answer is\s*)",
    re.IGNORECASE,
)


def _normalize_regex(answer: str) -> str:
    normalized = _PREFIX_RE.sub("", answer.strip()).strip()
    normalized = re.sub(r"[.!?]+$", "", normalized).strip()
    return normalized


def make_formatter_node(model, caveman: bool = False, caveman_mode: str = "full"):
    def formatter(state) -> dict:
        raw = state["final_answer"] or state["draft_answer"] or ""
        if not raw:
            return {"final_answer": ""}
        
        try:
            from gaia_agent.llm_utils import extract_text
            formatter_prompt = apply_caveman(FORMATTER_SYSTEM, caveman, caveman_mode)
            
            response = model.invoke([
                {"role": "system", "content": formatter_prompt},
                {"role": "user", "content": f"Format this answer precisely: {raw}"}
            ])
            # Handle both string content and message objects
            final_answer = extract_text(getattr(response, "content", response)).strip()
            # Remove markdown backticks if they are wrapping the answer
            final_answer = final_answer.replace("`", "").strip()
            
            # Ensure we didn't get a hallucinated long story, if so, fallback
            if len(final_answer) > len(raw) + 10:
                 final_answer = _normalize_regex(raw)
        except Exception:
            final_answer = _normalize_regex(raw)

        return {"final_answer": final_answer}
    
    return formatter
