import io
import httpx
import trafilatura
from pypdf import PdfReader


def fetch_url(url: str, max_chars: int = 8000, timeout: float = 20.0) -> str:
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
        )
        resp.raise_for_status()

        # Handle PDF
        content_type = resp.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(resp.content))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            extracted = trafilatura.extract(
                resp.text, include_comments=False, include_tables=True
            )
            text = extracted or resp.text

        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text
    except Exception as e:
        return f"Fetch Failed: {e}"
