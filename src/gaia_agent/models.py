from __future__ import annotations

from typing import Any, Sequence

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from gaia_agent.config import Config

LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1"
_NO_THINK = "/no_think"


class _NoThinkWrapper(BaseChatModel):
    """Wraps a ChatOpenAI model to append /no_think to every HumanMessage.

    Qwen3 Instruct in LM Studio enters thinking (chain-of-thought) mode by
    default, producing a huge reasoning_content blob and empty content.
    Appending /no_think disables it per Qwen3's chat template spec.
    """

    inner: ChatOpenAI

    @property
    def _llm_type(self) -> str:
        return "lmstudio-no-think"

    def _inject(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        out = []
        for msg in messages:
            if isinstance(msg, HumanMessage) and not str(msg.content).endswith(_NO_THINK):
                content = str(msg.content) + f" {_NO_THINK}"
                out.append(HumanMessage(content=content))
            else:
                out.append(msg)
        return out

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self.inner._generate(self._inject(messages), stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await self.inner._agenerate(self._inject(messages), stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any):
        # Delegate tool binding to inner model; return a wrapper that injects /no_think
        bound = self.inner.bind_tools(tools, **kwargs)
        wrapper = _BoundNoThinkWrapper(bound=bound, inject=self._inject)
        return wrapper


class _BoundNoThinkWrapper:
    """Thin callable wrapper around a tool-bound model that injects /no_think."""

    def __init__(self, bound, inject):
        self._bound = bound
        self._inject = inject

    def invoke(self, messages, **kwargs):
        return self._bound.invoke(self._inject(messages), **kwargs)

    def __getattr__(self, name):
        return getattr(self._bound, name)


def _build(provider: str, model: str, cfg: Config) -> BaseChatModel:
    max_tokens = cfg.max_tokens
    if provider == "ollama":
        return ChatOllama(model=model, num_predict=max_tokens)
    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=cfg.anthropic_api_key, max_tokens=max_tokens)
    if provider == "google":
        return ChatGoogleGenerativeAI(model=model, google_api_key=cfg.google_api_key, max_output_tokens=max_tokens)
    if provider == "huggingface":
        endpoint = HuggingFaceEndpoint(
            repo_id=model, huggingfacehub_api_token=cfg.huggingface_api_key, max_new_tokens=max_tokens
        )
        return ChatHuggingFace(llm=endpoint)
    if provider == "lmstudio":
        # LM Studio exposes an OpenAI-compatible API.
        inner = ChatOpenAI(
            model=model,
            base_url=cfg.lmstudio_base_url or LMSTUDIO_DEFAULT_BASE_URL,
            api_key="lm-studio",
            temperature=0,
            max_tokens=max_tokens,
        )
        # Wrap with /no_think injection to disable Qwen3 chain-of-thought mode
        return _NoThinkWrapper(inner=inner)
    raise ValueError(f"Unknown provider: {provider}")


def get_cheap_model(cfg: Config) -> BaseChatModel:
    return _build(cfg.cheap_provider, cfg.cheap_model, cfg)


def get_strong_model(cfg: Config) -> BaseChatModel:
    return _build(cfg.strong_provider, cfg.strong_model, cfg)
