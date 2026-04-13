from gaia_agent.config import Config


def test_config_from_env_uses_defaults(monkeypatch):
    keys = [
        "GAIA_CHEAP_PROVIDER",
        "GAIA_CHEAP_MODEL",
        "GAIA_STRONG_PROVIDER",
        "GAIA_STRONG_MODEL",
        "GAIA_API_URL",
        "GAIA_CHECKPOINT_DIR",
        "GAIA_ANTHROPIC_API_KEY",
        "GAIA_GOOGLE_API_KEY",
        "GAIA_HUGGINGFACE_API_KEY",
        "GAIA_TAVILY_API_KEY",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    config = Config.from_env()

    assert config.cheap_provider == "ollama"
    assert config.cheap_model == "gemma3:4b"
    assert config.strong_provider == "anthropic"
    assert config.strong_model == "claude-sonnet-4-6"
    assert config.extra_strong_provider == "anthropic"
    assert config.extra_strong_model == "claude-sonnet-4-6"
    assert config.vision_provider == "fal"
    assert config.vision_model == "gemini-3-flash-preview"
    assert config.fal_vision_api_key == ""
    assert config.api_url == "https://agents-course-unit4-scoring.hf.space"
    assert config.checkpoint_dir == ".checkpoints"
    assert config.whisper_model == "base"
    assert config.anthropic_api_key == ""
    assert config.google_api_key == ""
    assert config.huggingface_api_key == ""
    assert config.tavily_api_key == ""
    assert config.lmstudio_base_url == ""
    assert config.max_tokens == 1024


def test_config_from_env_honors_gaia_prefixed_overrides(monkeypatch):
    monkeypatch.setenv("GAIA_CHEAP_PROVIDER", "anthropic")
    monkeypatch.setenv("GAIA_CHEAP_MODEL", "haiku")
    monkeypatch.setenv("GAIA_STRONG_PROVIDER", "google")
    monkeypatch.setenv("GAIA_STRONG_MODEL", "gemini-pro")
    monkeypatch.setenv("GAIA_EXTRA_STRONG_PROVIDER", "google")
    monkeypatch.setenv("GAIA_EXTRA_STRONG_MODEL", "gemini-pro-1.5")
    monkeypatch.setenv("GAIA_VISION_PROVIDER", "google")
    monkeypatch.setenv("GAIA_VISION_MODEL", "gemini-pro-vision")
    monkeypatch.setenv("GAIA_FAL_VISION_API_KEY", "fal-123")
    monkeypatch.setenv("GAIA_API_URL", "https://example.test")
    monkeypatch.setenv("GAIA_CHECKPOINT_DIR", "/tmp/checkpoints")
    monkeypatch.setenv("GAIA_ANTHROPIC_API_KEY", "anth-key")
    monkeypatch.setenv("GAIA_GOOGLE_API_KEY", "goog-key")
    monkeypatch.setenv("GAIA_HUGGINGFACE_API_KEY", "hf-key")
    monkeypatch.setenv("GAIA_TAVILY_API_KEY", "tv-key")
    monkeypatch.setenv("GAIA_WHISPER_MODEL", "small")
    monkeypatch.setenv("GAIA_LMSTUDIO_BASE_URL", "http://host:1234/v1")
    monkeypatch.setenv("GAIA_MAX_TOKENS", "2048")

    config = Config.from_env()

    assert config == Config(
        cheap_provider="anthropic",
        cheap_model="haiku",
        strong_provider="google",
        strong_model="gemini-pro",
        extra_strong_provider="google",
        extra_strong_model="gemini-pro-1.5",
        vision_provider="google",
        vision_model="gemini-pro-vision",
        fal_vision_api_key="fal-123",
        api_url="https://example.test",
        checkpoint_dir="/tmp/checkpoints",
        whisper_model="small",
        anthropic_api_key="anth-key",
        google_api_key="goog-key",
        huggingface_api_key="hf-key",
        tavily_api_key="tv-key",
        lmstudio_base_url="http://host:1234/v1",
        max_tokens=2048,
    )
