"""Utilities for working with LLM response content."""

from __future__ import annotations


def extract_text(content) -> str:
    """Extract plain text from LLM response content.

    Handles both plain strings and Gemini-style list-of-blocks
    like [{'type': 'text', 'text': '...', 'extras': {...}}].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "".join(parts)
    return str(content)
