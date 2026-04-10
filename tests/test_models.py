from unittest.mock import patch, MagicMock

import pytest

from gaia_agent.config import Config
from gaia_agent.models import get_cheap_model, get_strong_model


def _cfg(**overrides):
    base = dict(
        cheap_provider="ollama",
        cheap_model="gemma3:4b",
        strong_provider="anthropic",
        strong_model="claude-sonnet-4-6",
        api_url="",
        checkpoint_dir="",
        whisper_model="base",
        anthropic_api_key="sk-ant-xxx",
        google_api_key="",
        huggingface_api_key="",
        tavily_api_key="",
        lmstudio_base_url="",
        max_tokens=1024,
    )
    base.update(overrides)
    return Config(**base)


def test_get_cheap_model_ollama():
    with patch("gaia_agent.models.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock()
        model = get_cheap_model(_cfg())
        mock_cls.assert_called_once_with(model="gemma3:4b", num_predict=1024)
        assert model is mock_cls.return_value


def test_get_strong_model_anthropic():
    with patch("gaia_agent.models.ChatAnthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        get_strong_model(_cfg())
        mock_cls.assert_called_once_with(
            model="claude-sonnet-4-6", api_key="sk-ant-xxx", max_tokens=1024
        )


def test_get_strong_model_google():
    cfg = _cfg(
        strong_provider="google",
        strong_model="gemini-2.5-pro",
        google_api_key="gkey",
    )
    with patch("gaia_agent.models.ChatGoogleGenerativeAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        get_strong_model(cfg)
        mock_cls.assert_called_once_with(model="gemini-2.5-pro", google_api_key="gkey", max_output_tokens=1024)


def test_unknown_provider_raises():
    cfg = _cfg(cheap_provider="bogus")
    with pytest.raises(ValueError, match="Unknown provider"):
        get_cheap_model(cfg)


def test_get_cheap_model_lmstudio_default_base_url():
    cfg = _cfg(cheap_provider="lmstudio", cheap_model="qwen2.5-7b-instruct")
    with patch("gaia_agent.models.ChatOpenAI") as mock_openai, \
         patch("gaia_agent.models._NoThinkWrapper") as mock_wrapper:
        mock_openai.return_value = MagicMock()
        mock_wrapper.return_value = MagicMock()
        get_cheap_model(cfg)
        mock_openai.assert_called_once_with(
            model="qwen2.5-7b-instruct",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            temperature=0,
            max_tokens=1024,
        )


def test_get_cheap_model_lmstudio_custom_base_url():
    cfg = _cfg(
        cheap_provider="lmstudio",
        cheap_model="llama-3.3-70b",
        lmstudio_base_url="http://192.168.1.10:1234/v1",
    )
    with patch("gaia_agent.models.ChatOpenAI") as mock_openai, \
         patch("gaia_agent.models._NoThinkWrapper") as mock_wrapper:
        mock_openai.return_value = MagicMock()
        mock_wrapper.return_value = MagicMock()
        get_cheap_model(cfg)
        mock_openai.assert_called_once_with(
            model="llama-3.3-70b",
            base_url="http://192.168.1.10:1234/v1",
            api_key="lm-studio",
            temperature=0,
            max_tokens=1024,
        )
