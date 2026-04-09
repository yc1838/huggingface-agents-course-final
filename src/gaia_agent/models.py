from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from gaia_agent.config import Config

LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1"


def _build(provider: str, model: str, cfg: Config) -> BaseChatModel:
    if provider == "ollama":
        return ChatOllama(model=model)
    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=cfg.anthropic_api_key)
    if provider == "google":
        return ChatGoogleGenerativeAI(model=model, google_api_key=cfg.google_api_key)
    if provider == "huggingface":
        endpoint = HuggingFaceEndpoint(
            repo_id=model, huggingfacehub_api_token=cfg.huggingface_api_key
        )
        return ChatHuggingFace(llm=endpoint)
    if provider == "lmstudio":
        # LM Studio exposes an OpenAI-compatible API. Use ChatOpenAI pointed at
        # the local server; api_key is required by the client but unused.
        return ChatOpenAI(
            model=model,
            base_url=cfg.lmstudio_base_url or LMSTUDIO_DEFAULT_BASE_URL,
            api_key="lm-studio",
        )
    raise ValueError(f"Unknown provider: {provider}")


def get_cheap_model(cfg: Config) -> BaseChatModel:
    return _build(cfg.cheap_provider, cfg.cheap_model, cfg)


def get_strong_model(cfg: Config) -> BaseChatModel:
    return _build(cfg.strong_provider, cfg.strong_model, cfg)
