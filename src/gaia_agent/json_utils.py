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
    """Extract and parse JSON from an LLM response string.

    Tries in order:
    1. ``parse_json_markdown`` from LangChain (handles fenced blocks)
    2. Repair common mistakes (single quotes, trailing commas, Python
       literals) then ``json.loads``
    3. Regex-extract the first ``{ ... }`` block and retry
    """
    # 1. Try LangChain's parser first (handles ```json ... ``` well)
    try:
        from langchain_core.utils.json import parse_json_markdown

        return parse_json_markdown(text)
    except Exception:
        pass

    # 2. Try repairing the raw text
    try:
        return json.loads(_repair(text))
    except json.JSONDecodeError:
        pass

    # 3. Strip markdown fences, then try repair
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            return json.loads(_repair(fence_match.group(1)))
        except json.JSONDecodeError:
            pass

    # 4. Extract first { ... } block and try repair
    brace_match = _BRACE_RE.search(text)
    if brace_match:
        return json.loads(_repair(brace_match.group(0)))

    raise ValueError(f"No valid JSON found in LLM response: {text[:300]}")
