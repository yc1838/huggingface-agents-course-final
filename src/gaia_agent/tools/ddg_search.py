"""DuckDuckGo web search – zero API key required.

Uses the duckduckgo-search library (pip install duckduckgo-search).
Falls back gracefully if the library is not installed.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo. Returns title/url/snippet results."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "ERROR: duckduckgo-search not installed. Run: pip install duckduckgo-search"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        log.warning("[web_search] DuckDuckGo error: %s", e)
        return f"ERROR: DuckDuckGo search failed: {e}"

    if not results:
        return "No results."

    lines: list[str] = []
    for item in results:
        title = item.get("title", "")
        url = item.get("href", "")
        body = item.get("body", "")
        lines.append(f"- {title} ({url})\n  {body}")
    return "\n".join(lines)
