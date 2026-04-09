from __future__ import annotations

import httpx
import trafilatura


def fetch_url(url: str, max_chars: int = 8000, timeout: float = 20.0) -> str:
    resp = httpx.get(
        url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "gaia-agent/0.1"},
    )
    resp.raise_for_status()
    extracted = trafilatura.extract(
        resp.text, include_comments=False, include_tables=True
    )
    text = extracted or resp.text
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text
