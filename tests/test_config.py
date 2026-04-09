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
    assert config.api_url == "https://agents-course-unit4-scoring.hf.space"
    assert config.checkpoint_dir == ".checkpoints"
    assert config.anthropic_api_key == ""
    assert config.google_api_key == ""
    assert config.huggingface_api_key == ""
    assert config.tavily_api_key == ""


def test_config_from_env_honors_gaia_prefixed_overrides(monkeypatch):
    monkeypatch.setenv("GAIA_CHEAP_PROVIDER", "anthropic")
    monkeypatch.setenv("GAIA_CHEAP_MODEL", "haiku")
    monkeypatch.setenv("GAIA_STRONG_PROVIDER", "google")
    monkeypatch.setenv("GAIA_STRONG_MODEL", "gemini-pro")
    monkeypatch.setenv("GAIA_API_URL", "https://example.test")
    monkeypatch.setenv("GAIA_CHECKPOINT_DIR", "/tmp/checkpoints")
    monkeypatch.setenv("GAIA_ANTHROPIC_API_KEY", "anth-key")
    monkeypatch.setenv("GAIA_GOOGLE_API_KEY", "goog-key")
    monkeypatch.setenv("GAIA_HUGGINGFACE_API_KEY", "hf-key")
    monkeypatch.setenv("GAIA_TAVILY_API_KEY", "tv-key")

    config = Config.from_env()

    assert config == Config(
        cheap_provider="anthropic",
        cheap_model="haiku",
        strong_provider="google",
        strong_model="gemini-pro",
        api_url="https://example.test",
        checkpoint_dir="/tmp/checkpoints",
        anthropic_api_key="anth-key",
        google_api_key="goog-key",
        huggingface_api_key="hf-key",
        tavily_api_key="tv-key",
    )
