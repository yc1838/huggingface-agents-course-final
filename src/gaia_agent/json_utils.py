"""Robust JSON extraction from LLM responses.

LLMs (especially local/small models) often return malformed JSON:
single-quoted keys, trailing commas, Python literals (True/False/None),
or markdown-fenced blocks.  This module normalises those patterns
before parsing.
"""

from __future__ import annotations

import json
import re


# Strip markdown fences: ```json ... ``` or ``` ... ```
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Find the first top-level { ... } block
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _repair(text: str) -> str:
    """Best-effort repair of common LLM JSON mistakes."""
    # Python-style booleans/None → JSON
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    # Single-quoted strings → double-quoted strings
    # Only outside of already-double-quoted regions.
    # Simple heuristic: replace ' with " when it looks like a JSON key/value boundary.
    text = re.sub(r"(?<=[\[{,:\s])'|'(?=[\]},:.\s])", '"', text)

    # Trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text


def extract_json(text: str) -> dict:
    """Extract and parse JSON from an LLM response string."""
    # 1. Try LangChain's parser first (handles ```json ... ``` well)
    try:
        from langchain_core.utils.json import parse_json_markdown
        res = parse_json_markdown(text)
        if res:
            return res
    except Exception:
        pass

    # 2. Aggressive search for the first { and last }
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = text[start_idx : end_idx + 1]
        try:
            return json.loads(_repair(json_candidate))
        except Exception:
            # Try once more without repair if repair failed
            try:
                return json.loads(json_candidate)
            except Exception:
                pass

    sample = text[:500] + "..." if len(text) > 500 else text
    raise ValueError(f"No valid JSON found in LLM response. Raw snippet:\n{sample}")
