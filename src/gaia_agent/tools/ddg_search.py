"""DuckDuckGo web search – zero API key required.

Uses the ddgs library (pip install ddgs).
Falls back gracefully if the library is not installed.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo. Returns title/url/snippet results."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "ERROR: ddgs not installed. Run: pip install ddgs"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        log.warning("[web_search] DuckDuckGo error: %s", e)
        return f"ERROR: DuckDuckGo search failed: {e}"

    if not results:
        return "No results."

    lines: list[str] = []
    for r in results:
        title = r.get("title", "No Title")
        url = r.get("href", "No URL")
        body = r.get("body", "No snippet available")
        lines.append(f"- {title} ({url})\n  {body}")

    import json
    metadata = {
        "value": len(results),
        "data_source": "duckduckgo",
        "record_type": "web-snippet",
        "type_strictness": "broad",
        "note": "Web snippets are unstructured and may contain irrelevant or duplicate information."
    }
    res_text = "\n".join(lines)
    return f"{res_text}\n\nMETADATA:\n{json.dumps(metadata, indent=2)}"
