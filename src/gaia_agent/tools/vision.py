import base64
import os
from pathlib import Path

import httpx
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from gaia_agent.config import Config


def inspect_visual_content(file_path_or_url: str, prompt: str, cfg: Config) -> str:
    """Use Gemini Multimodal Vision to analyze an image (PNG, JPG) or video (MP4)."""
    # Use a strong model for vision tasks
    model = ChatGoogleGenerativeAI(
        model=cfg.strong_model, 
        google_api_key=cfg.google_api_key
    )

    content = []
    content.append({"type": "text", "text": prompt})

    if file_path_or_url.startswith(("http://", "https://")):
        # If it's a URL, we might want to download it first or pass URL if supported.
        # Most reliable is to download it.
        try:
            resp = httpx.get(file_path_or_url, follow_redirects=True, timeout=30.0)
            resp.raise_for_status()
            data = resp.content
            mime_type = resp.headers.get("Content-Type", "image/png")
        except Exception as e:
            return f"Error downloading visual content from {file_path_or_url}: {e}"
    else:
        path = Path(file_path_or_url)
        if not path.exists():
            return f"File not found: {file_path_or_url}"
        data = path.read_bytes()
        ext = path.suffix.lower()
        if ext in (".png", ".jpg", ".jpeg"):
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
        elif ext == ".mp4":
            mime_type = "video/mp4"
        else:
            mime_type = "image/png" # fallback

    b64_data = base64.b64encode(data).decode("utf-8")
    content.append({
        "type": "media",
        "mime_type": mime_type,
        "data": b64_data
    })

    try:
        msg = HumanMessage(content=content)
        response = model.invoke([msg])
        return str(response.content)
    except Exception as e:
        return f"Vision Analysis Failed: {e}"
