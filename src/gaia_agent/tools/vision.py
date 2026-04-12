import base64
import os
from pathlib import Path

import httpx
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from gaia_agent.config import Config


def inspect_visual_content(file_path_or_url: str, prompt: str, cfg: Config) -> str:
    """Analyze image/video with remapping and prioritized fallback logic."""
    
    # 1. Model Remapping Table
    MODEL_MAP = {
        "claude-sonnet-4-6": "claude-3-5-sonnet-20240620",
        "llava": "fal-ai/moondream-next", # Correct fal.ai hosted ID
        "gemini-flash": "gemini-1.5-flash",
    }
    
    requested_model = cfg.vision_model
    target_model = MODEL_MAP.get(requested_model, requested_model)
    requested_provider = cfg.vision_provider

    def _call_vision(model_name: str, provider: str) -> str:
        if provider == "fal":
            try:
                import fal_client
                os.environ["FAL_KEY"] = cfg.fal_vision_api_key
                image_data_url = f"data:{mime_type};base64,{b64_data}"
                print(f"[vision] calling fal model={model_name}")
                result = fal_client.subscribe(
                    model_name,
                    arguments={"prompt": prompt, "image_url": image_data_url}
                )
                return result.get("text") or result.get("description") or str(result)
            except Exception as e:
                return f"ERROR:{e}"
        else:
            # Default Google
            try:
                print(f"[vision] calling google model={model_name}")
                model = ChatGoogleGenerativeAI(model=model_name, google_api_key=cfg.google_api_key)
                response = model.invoke([HumanMessage(content=content)])
                return str(response.content)
            except Exception as e:
                return f"ERROR:{e}"

    # Preparation
    content = [{"type": "text", "text": prompt}]
    try:
        if file_path_or_url.startswith(("http://", "https://")):
            resp = httpx.get(file_path_or_url, follow_redirects=True, timeout=30.0)
            resp.raise_for_status()
            data = resp.content
            mime_type = resp.headers.get("Content-Type", "image/png")
        else:
            path = Path(file_path_or_url)
            data = path.read_bytes()
            mime_type = "video/mp4" if path.suffix.lower() == ".mp4" else "image/png"
        b64_data = base64.b64encode(data).decode("utf-8")
        content.append({"type": "media", "mime_type": mime_type, "data": b64_data})
    except Exception as e:
        return f"File Preparation Failed: {e}"

    # 2. Attempt 1: Targeted Model on Configured Provider
    res = _call_vision(target_model, requested_provider)
    if not res.startswith("ERROR:"):
        return res

    # 3. Attempt 2: Stable Provider-Specific Fallback
    if requested_provider == "fal":
        print(f"[vision] FAL model {target_model} failed. Trying stable FAL fallback: fal-ai/moondream-next")
        fallback_res = _call_vision("fal-ai/moondream-next", "fal")
    else:
        print(f"[vision] Google model {target_model} failed. Trying stable Google fallback: gemini-1.5-flash")
        fallback_res = _call_vision("gemini-1.5-flash", "google")
    
    if not fallback_res.startswith("ERROR:"):
        return fallback_res

    # 4. Attempt 3: Ultimate Last Resort (Gemini Flash)
    if requested_provider == "fal":
        print(f"[vision] All FAL attempts failed. Last resort attempt: gemini-1.5-flash on Google")
        ultimate_res = _call_vision("gemini-1.5-flash", "google")
        if not ultimate_res.startswith("ERROR:"):
            return ultimate_res
        return f"All Vision Attempts Failed. Final Error: {ultimate_res}"
    
    return f"All Vision Attempts Failed. Final Error: {fallback_res}"
