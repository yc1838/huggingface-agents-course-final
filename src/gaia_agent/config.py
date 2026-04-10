from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    cheap_provider: str
    cheap_model: str
    strong_provider: str
    strong_model: str
    api_url: str
    checkpoint_dir: str
    whisper_model: str
    anthropic_api_key: str
    google_api_key: str
    huggingface_api_key: str
    tavily_api_key: str
    lmstudio_base_url: str
    max_tokens: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            cheap_provider=os.getenv("GAIA_CHEAP_PROVIDER", "ollama"),
            cheap_model=os.getenv("GAIA_CHEAP_MODEL", "gemma3:4b"),
            strong_provider=os.getenv("GAIA_STRONG_PROVIDER", "anthropic"),
            strong_model=os.getenv("GAIA_STRONG_MODEL", "claude-sonnet-4-6"),
            api_url=os.getenv(
                "GAIA_API_URL", "https://agents-course-unit4-scoring.hf.space"
            ),
            checkpoint_dir=os.getenv("GAIA_CHECKPOINT_DIR", ".checkpoints"),
            whisper_model=os.getenv("GAIA_WHISPER_MODEL", "base"),
            anthropic_api_key=os.getenv("GAIA_ANTHROPIC_API_KEY", ""),
            google_api_key=os.getenv("GAIA_GOOGLE_API_KEY", ""),
            huggingface_api_key=os.getenv("GAIA_HUGGINGFACE_API_KEY", ""),
            tavily_api_key=os.getenv("GAIA_TAVILY_API_KEY", ""),
            lmstudio_base_url=os.getenv("GAIA_LMSTUDIO_BASE_URL", ""),
            max_tokens=int(os.getenv("GAIA_MAX_TOKENS", "1024")),
        )
