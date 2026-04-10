from gaia_agent.config import Config
from gaia_agent.tools import build_tools


def _cfg() -> Config:
    return Config(
        cheap_provider="ollama",
        cheap_model="gemma3:4b",
        strong_provider="anthropic",
        strong_model="claude-sonnet-4-6",
        api_url="https://example.test",
        checkpoint_dir=".checkpoints",
        whisper_model="base",
        anthropic_api_key="",
        google_api_key="",
        huggingface_api_key="",
        tavily_api_key="",
        lmstudio_base_url="",
        max_tokens=1024,
    )


def test_build_tools_returns_expected_names():
    tools = build_tools(_cfg())
    names = {t.name for t in tools}
    assert names == {
        "tavily_search",
        "fetch_url",
        "run_python",
        "read_file",
        "transcribe_audio",
        "youtube_transcript",
    }


def test_build_tools_all_have_descriptions():
    tools = build_tools(_cfg())
    for t in tools:
        assert t.description, f"{t.name} missing description"
