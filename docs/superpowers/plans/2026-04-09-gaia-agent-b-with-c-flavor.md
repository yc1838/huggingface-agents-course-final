# GAIA Agent Implementation Plan — Plan/Execute/Verify with System 1/System 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stub `BasicAgent` in `app.py` with a LangGraph-based agent that solves GAIA Level 1 questions via a Plan → Tiered-Execute → Verify loop, targeting both the HuggingFace Agents course leaderboard and the broader GAIA leaderboard.

**Architecture:** A 6-node LangGraph `StateGraph`. Perception classifies modality and fetches files. Planner (strong model, called once) writes a 3–5 step plan. Router dispatches each step to Executor S1 (cheap model) or Executor S2 (strong model) by rule. Tools read the environment and write observations into a typed Working Memory state. When the plan completes, a Verifier (strong model) either approves the draft or rejects it with a critique routed back to the Planner. A rule-based Formatter strips the approved answer into GAIA's exact-match shape. See `docs/architecture.md` for the diagram.

**Tech Stack:**
- Python 3.11+, LangGraph, LangChain core
- Model providers: Ollama (dev), Anthropic, Google GenerativeAI, HF Inference (pluggable)
- Tools: Tavily (search), httpx + trafilatura (web fetch), in-process Python exec, pypdf, pandas + openpyxl, python-docx, faster-whisper, youtube-transcript-api
- Testing: pytest, pytest-asyncio, responses (HTTP mocks)
- Runtime: Gradio (existing), deployed as a HuggingFace Space

---

## File Structure

```
src/gaia_agent/
  __init__.py
  config.py              # env-driven config: model tiers, API keys, paths
  state.py               # AgentState TypedDict (working memory)
  models.py              # LLM provider factory: cheap() + strong()
  api_client.py          # GAIA /questions /files /submit client
  prompts.py             # all system prompts as constants
  tools/
    __init__.py          # TOOL_REGISTRY
    search.py            # tavily_search
    web.py               # fetch_url (httpx + trafilatura)
    python_exec.py       # run_python
    files.py             # read_file (PDF/XLSX/CSV/DOCX/TXT dispatch)
    audio.py             # transcribe_audio (faster-whisper)
    youtube.py           # youtube_transcript
  nodes/
    __init__.py
    perception.py        # detect modality, fetch file
    planner.py           # strong model, writes plan
    router.py            # rule-based S1/S2 step dispatch
    executor.py          # single executor parameterized by model tier
    verifier.py          # strong model, approves or rejects
    formatter.py         # exact-match shape normalizer
  graph.py               # build_graph() assembles StateGraph
  runner.py              # run_agent_on_questions() with JSON checkpoints

tests/
  test_config.py
  test_state.py
  test_models.py
  test_api_client.py
  tools/
    test_search.py
    test_web.py
    test_python_exec.py
    test_files.py
    test_audio.py
    test_youtube.py
  nodes/
    test_perception.py
    test_planner.py
    test_router.py
    test_executor.py
    test_verifier.py
    test_formatter.py
  test_graph.py
  test_runner.py

docs/
  architecture.md        # (already exists)
  superpowers/plans/
    2026-04-09-gaia-agent-b-with-c-flavor.md  # (this file)
```

Responsibilities:
- **`config.py`** — single source of truth for runtime config; everything else reads from it.
- **`state.py`** — the typed Working Memory. Every node reads from and writes into this.
- **`models.py`** — the *only* place that imports provider SDKs. Returns a `BaseChatModel`.
- **`api_client.py`** — the *only* place that talks to the GAIA scoring API.
- **`tools/`** — each tool is a pure function; no LLM calls inside tools.
- **`nodes/`** — each node is a function `(state) -> partial_state`. Pure w.r.t. external state except LLM calls and tool calls.
- **`graph.py`** — wires nodes and edges. No business logic.
- **`runner.py`** — orchestration layer: loops over questions, checkpoints, handles per-task errors.

---

## Task 1: Project scaffolding and dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `src/gaia_agent/__init__.py`
- Create: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `.env.example`

- [ ] **Step 1: Update `requirements.txt`**

Replace the contents of `requirements.txt` with:

```
gradio
requests
httpx
langgraph>=0.2.0
langchain-core>=0.3.0
langchain-anthropic>=0.2.0
langchain-google-genai>=2.0.0
langchain-ollama>=0.2.0
langchain-huggingface>=0.1.0
tavily-python>=0.5.0
trafilatura>=1.12.0
pypdf>=5.0.0
pandas>=2.2.0
openpyxl>=3.1.0
python-docx>=1.1.0
faster-whisper>=1.0.0
youtube-transcript-api>=0.6.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
responses>=0.25.0
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "gaia_agent"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"
```

- [ ] **Step 3: Create empty package markers**

Create `src/gaia_agent/__init__.py` with:
```python
"""GAIA agent: Plan → Execute → Verify with System 1/System 2 tiering."""
```

Create `tests/__init__.py` as an empty file.

- [ ] **Step 4: Create `.env.example`**

```
# Model tier selection. Values: ollama, anthropic, google, huggingface
GAIA_CHEAP_PROVIDER=ollama
GAIA_CHEAP_MODEL=gemma3:4b
GAIA_STRONG_PROVIDER=anthropic
GAIA_STRONG_MODEL=claude-sonnet-4-6

# API keys (only needed for providers you use)
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
HUGGINGFACE_API_KEY=
TAVILY_API_KEY=

# GAIA scoring API
GAIA_API_URL=https://agents-course-unit4-scoring.hf.space

# Runtime
GAIA_CHECKPOINT_DIR=.checkpoints
GAIA_WHISPER_MODEL=base
```

- [ ] **Step 5: Install and verify**

Run: `pip install -e .`
Run: `pytest --collect-only`
Expected: "collected 0 items" with no import errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pyproject.toml src/gaia_agent/__init__.py tests/__init__.py .env.example
git commit -m "chore: scaffold gaia_agent package and deps"
```

---

## Task 2: Config module

**Files:**
- Create: `src/gaia_agent/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import os
import pytest
from gaia_agent.config import Config

def test_config_reads_from_env(monkeypatch):
    monkeypatch.setenv("GAIA_CHEAP_PROVIDER", "ollama")
    monkeypatch.setenv("GAIA_CHEAP_MODEL", "gemma3:4b")
    monkeypatch.setenv("GAIA_STRONG_PROVIDER", "anthropic")
    monkeypatch.setenv("GAIA_STRONG_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("GAIA_API_URL", "https://example.test")
    monkeypatch.setenv("GAIA_CHECKPOINT_DIR", "/tmp/ckpt")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")

    cfg = Config.from_env()

    assert cfg.cheap_provider == "ollama"
    assert cfg.cheap_model == "gemma3:4b"
    assert cfg.strong_provider == "anthropic"
    assert cfg.strong_model == "claude-sonnet-4-6"
    assert cfg.api_url == "https://example.test"
    assert cfg.checkpoint_dir == "/tmp/ckpt"
    assert cfg.tavily_api_key == "tvly-xxx"

def test_config_defaults(monkeypatch):
    for var in ["GAIA_CHEAP_PROVIDER", "GAIA_CHEAP_MODEL",
                "GAIA_STRONG_PROVIDER", "GAIA_STRONG_MODEL",
                "GAIA_API_URL", "GAIA_CHECKPOINT_DIR"]:
        monkeypatch.delenv(var, raising=False)

    cfg = Config.from_env()

    assert cfg.cheap_provider == "ollama"
    assert cfg.api_url == "https://agents-course-unit4-scoring.hf.space"
    assert cfg.checkpoint_dir == ".checkpoints"
    assert cfg.whisper_model == "base"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'gaia_agent.config'`

- [ ] **Step 3: Implement `config.py`**

Create `src/gaia_agent/config.py`:

```python
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

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

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            cheap_provider=os.getenv("GAIA_CHEAP_PROVIDER", "ollama"),
            cheap_model=os.getenv("GAIA_CHEAP_MODEL", "gemma3:4b"),
            strong_provider=os.getenv("GAIA_STRONG_PROVIDER", "anthropic"),
            strong_model=os.getenv("GAIA_STRONG_MODEL", "claude-sonnet-4-6"),
            api_url=os.getenv("GAIA_API_URL", "https://agents-course-unit4-scoring.hf.space"),
            checkpoint_dir=os.getenv("GAIA_CHECKPOINT_DIR", ".checkpoints"),
            whisper_model=os.getenv("GAIA_WHISPER_MODEL", "base"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY", ""),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/config.py tests/test_config.py
git commit -m "feat(config): env-driven Config dataclass"
```

---

## Task 3: Working Memory state

**Files:**
- Create: `src/gaia_agent/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_state.py`:

```python
from gaia_agent.state import AgentState, PlanStep, Observation, new_state

def test_new_state_has_all_fields():
    s = new_state(task_id="t1", question="What is 2+2?")
    assert s["task_id"] == "t1"
    assert s["question"] == "What is 2+2?"
    assert s["file_path"] is None
    assert s["modality"] == "text"
    assert s["plan"] == []
    assert s["step_idx"] == 0
    assert s["observations"] == []
    assert s["draft_answer"] is None
    assert s["critique"] is None
    assert s["retries"] == 0
    assert s["final_answer"] is None

def test_plan_step_shape():
    step: PlanStep = {"description": "search for X", "tier": "S1"}
    assert step["description"] == "search for X"
    assert step["tier"] == "S1"

def test_observation_shape():
    obs: Observation = {
        "step_idx": 0,
        "tool": "tavily_search",
        "args": {"query": "X"},
        "result": "result text",
    }
    assert obs["tool"] == "tavily_search"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_state.py -v`
Expected: `ModuleNotFoundError: No module named 'gaia_agent.state'`

- [ ] **Step 3: Implement `state.py`**

Create `src/gaia_agent/state.py`:

```python
from __future__ import annotations
from typing import TypedDict, Literal, Any

Modality = Literal["text", "image", "audio", "pdf", "excel", "csv", "docx", "youtube", "web"]
Tier = Literal["S1", "S2"]

class PlanStep(TypedDict):
    description: str
    tier: Tier

class Observation(TypedDict):
    step_idx: int
    tool: str
    args: dict[str, Any]
    result: str

class AgentState(TypedDict):
    task_id: str
    question: str
    file_path: str | None
    modality: Modality
    plan: list[PlanStep]
    step_idx: int
    observations: list[Observation]
    draft_answer: str | None
    critique: str | None
    retries: int
    final_answer: str | None

def new_state(task_id: str, question: str) -> AgentState:
    return AgentState(
        task_id=task_id,
        question=question,
        file_path=None,
        modality="text",
        plan=[],
        step_idx=0,
        observations=[],
        draft_answer=None,
        critique=None,
        retries=0,
        final_answer=None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_state.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/state.py tests/test_state.py
git commit -m "feat(state): typed AgentState working memory"
```

---

## Task 4: LLM provider factory

**Files:**
- Create: `src/gaia_agent/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
from unittest.mock import patch, MagicMock
from gaia_agent.config import Config
from gaia_agent.models import get_cheap_model, get_strong_model

def _cfg(**overrides):
    base = dict(
        cheap_provider="ollama", cheap_model="gemma3:4b",
        strong_provider="anthropic", strong_model="claude-sonnet-4-6",
        api_url="", checkpoint_dir="", whisper_model="base",
        anthropic_api_key="sk-ant-xxx", google_api_key="",
        huggingface_api_key="", tavily_api_key="",
    )
    base.update(overrides)
    return Config(**base)

def test_get_cheap_model_ollama():
    with patch("gaia_agent.models.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock()
        model = get_cheap_model(_cfg())
        mock_cls.assert_called_once_with(model="gemma3:4b")
        assert model is mock_cls.return_value

def test_get_strong_model_anthropic():
    with patch("gaia_agent.models.ChatAnthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        model = get_strong_model(_cfg())
        mock_cls.assert_called_once_with(
            model="claude-sonnet-4-6", api_key="sk-ant-xxx"
        )

def test_get_strong_model_google():
    cfg = _cfg(strong_provider="google", strong_model="gemini-2.5-pro", google_api_key="gkey")
    with patch("gaia_agent.models.ChatGoogleGenerativeAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        get_strong_model(cfg)
        mock_cls.assert_called_once_with(model="gemini-2.5-pro", google_api_key="gkey")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: `ModuleNotFoundError: No module named 'gaia_agent.models'`

- [ ] **Step 3: Implement `models.py`**

Create `src/gaia_agent/models.py`:

```python
from __future__ import annotations
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from gaia_agent.config import Config

def _build(provider: str, model: str, cfg: Config) -> BaseChatModel:
    if provider == "ollama":
        return ChatOllama(model=model)
    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=cfg.anthropic_api_key)
    if provider == "google":
        return ChatGoogleGenerativeAI(model=model, google_api_key=cfg.google_api_key)
    if provider == "huggingface":
        endpoint = HuggingFaceEndpoint(repo_id=model, huggingfacehub_api_token=cfg.huggingface_api_key)
        return ChatHuggingFace(llm=endpoint)
    raise ValueError(f"Unknown provider: {provider}")

def get_cheap_model(cfg: Config) -> BaseChatModel:
    return _build(cfg.cheap_provider, cfg.cheap_model, cfg)

def get_strong_model(cfg: Config) -> BaseChatModel:
    return _build(cfg.strong_provider, cfg.strong_model, cfg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/models.py tests/test_models.py
git commit -m "feat(models): pluggable LLM provider factory"
```

---

## Task 5: GAIA API client

**Files:**
- Create: `src/gaia_agent/api_client.py`
- Test: `tests/test_api_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_client.py`:

```python
import responses
from gaia_agent.api_client import GaiaApiClient

BASE = "https://example.test"

@responses.activate
def test_get_questions():
    responses.add(
        responses.GET, f"{BASE}/questions",
        json=[{"task_id": "t1", "question": "Q?", "file_name": ""}],
        status=200,
    )
    client = GaiaApiClient(BASE)
    qs = client.get_questions()
    assert qs == [{"task_id": "t1", "question": "Q?", "file_name": ""}]

@responses.activate
def test_download_file(tmp_path):
    responses.add(
        responses.GET, f"{BASE}/files/t1",
        body=b"hello",
        status=200,
        content_type="application/octet-stream",
        headers={"content-disposition": 'attachment; filename="data.txt"'},
    )
    client = GaiaApiClient(BASE)
    path = client.download_file("t1", tmp_path)
    assert path.read_bytes() == b"hello"
    assert path.name == "data.txt"

@responses.activate
def test_download_file_missing_returns_none(tmp_path):
    responses.add(responses.GET, f"{BASE}/files/t1", status=404)
    client = GaiaApiClient(BASE)
    assert client.download_file("t1", tmp_path) is None

@responses.activate
def test_submit():
    responses.add(
        responses.POST, f"{BASE}/submit",
        json={"username": "u", "score": 50.0, "correct_count": 1,
              "total_attempted": 2, "message": "ok", "timestamp": "t"},
        status=200,
    )
    client = GaiaApiClient(BASE)
    res = client.submit(username="u", agent_code="https://x", answers=[
        {"task_id": "t1", "submitted_answer": "a"}
    ])
    assert res["score"] == 50.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_client.py -v`
Expected: `ModuleNotFoundError: No module named 'gaia_agent.api_client'`

- [ ] **Step 3: Implement `api_client.py`**

Create `src/gaia_agent/api_client.py`:

```python
from __future__ import annotations
import re
from pathlib import Path
import requests

class GaiaApiClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_questions(self) -> list[dict]:
        r = requests.get(f"{self.base_url}/questions", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def download_file(self, task_id: str, dest_dir: Path) -> Path | None:
        r = requests.get(f"{self.base_url}/files/{task_id}", timeout=self.timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        cd = r.headers.get("content-disposition", "")
        m = re.search(r'filename="?([^";]+)"?', cd)
        filename = m.group(1) if m else f"{task_id}.bin"
        dest = Path(dest_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest

    def submit(self, username: str, agent_code: str, answers: list[dict]) -> dict:
        payload = {"username": username, "agent_code": agent_code, "answers": answers}
        r = requests.post(f"{self.base_url}/submit", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_client.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/api_client.py tests/test_api_client.py
git commit -m "feat(api): GAIA scoring API client"
```

---

## Task 6: Search tool (Tavily)

**Files:**
- Create: `src/gaia_agent/tools/__init__.py`
- Create: `src/gaia_agent/tools/search.py`
- Test: `tests/tools/__init__.py`
- Test: `tests/tools/test_search.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tools/__init__.py` as empty.

Create `tests/tools/test_search.py`:

```python
from unittest.mock import patch, MagicMock
from gaia_agent.tools.search import tavily_search

def test_tavily_search_returns_formatted_results():
    fake_client = MagicMock()
    fake_client.search.return_value = {
        "results": [
            {"title": "T1", "url": "https://a.test", "content": "snippet 1"},
            {"title": "T2", "url": "https://b.test", "content": "snippet 2"},
        ]
    }
    with patch("gaia_agent.tools.search.TavilyClient", return_value=fake_client):
        out = tavily_search("who is X", api_key="tvly-xxx", max_results=2)
    assert "T1" in out
    assert "https://a.test" in out
    assert "snippet 1" in out
    assert "T2" in out
```

- [ ] **Step 2: Run test to verify it fails**

Create `src/gaia_agent/tools/__init__.py` as empty.
Run: `pytest tests/tools/test_search.py -v`
Expected: `ModuleNotFoundError: No module named 'gaia_agent.tools.search'`

- [ ] **Step 3: Implement `search.py`**

Create `src/gaia_agent/tools/search.py`:

```python
from __future__ import annotations
from tavily import TavilyClient

def tavily_search(query: str, api_key: str, max_results: int = 5) -> str:
    client = TavilyClient(api_key=api_key)
    res = client.search(query=query, max_results=max_results)
    lines: list[str] = []
    for item in res.get("results", []):
        lines.append(f"- {item['title']} ({item['url']})\n  {item['content']}")
    return "\n".join(lines) if lines else "No results."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_search.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/tools/__init__.py src/gaia_agent/tools/search.py tests/tools/__init__.py tests/tools/test_search.py
git commit -m "feat(tools): tavily search tool"
```

---

## Task 7: Web fetch tool

**Files:**
- Create: `src/gaia_agent/tools/web.py`
- Test: `tests/tools/test_web.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tools/test_web.py`:

```python
from unittest.mock import patch, MagicMock
from gaia_agent.tools.web import fetch_url

def test_fetch_url_extracts_main_text():
    html = "<html><body><article><h1>Hello</h1><p>World body</p></article></body></html>"
    fake_resp = MagicMock(status_code=200, text=html)
    fake_resp.raise_for_status = MagicMock()
    with patch("gaia_agent.tools.web.httpx.get", return_value=fake_resp):
        out = fetch_url("https://example.test/a")
    assert "Hello" in out
    assert "World body" in out

def test_fetch_url_truncates_long_output():
    html = "<html><body><p>" + ("x " * 20000) + "</p></body></html>"
    fake_resp = MagicMock(status_code=200, text=html)
    fake_resp.raise_for_status = MagicMock()
    with patch("gaia_agent.tools.web.httpx.get", return_value=fake_resp):
        out = fetch_url("https://example.test/a", max_chars=1000)
    assert len(out) <= 1000 + 50  # allow suffix
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_web.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `web.py`**

Create `src/gaia_agent/tools/web.py`:

```python
from __future__ import annotations
import httpx
import trafilatura

def fetch_url(url: str, max_chars: int = 8000, timeout: float = 20.0) -> str:
    resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                     headers={"User-Agent": "gaia-agent/0.1"})
    resp.raise_for_status()
    extracted = trafilatura.extract(resp.text, include_comments=False, include_tables=True)
    text = extracted or resp.text
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_web.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/tools/web.py tests/tools/test_web.py
git commit -m "feat(tools): web fetch with trafilatura extraction"
```

---

## Task 8: Python execution tool

**Files:**
- Create: `src/gaia_agent/tools/python_exec.py`
- Test: `tests/tools/test_python_exec.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tools/test_python_exec.py`:

```python
from gaia_agent.tools.python_exec import run_python

def test_run_python_captures_stdout():
    out = run_python("print(2 + 2)")
    assert "4" in out

def test_run_python_captures_last_expr():
    out = run_python("x = 10\ny = 32\nx + y")
    assert "42" in out

def test_run_python_reports_exceptions():
    out = run_python("1/0")
    assert "ZeroDivisionError" in out

def test_run_python_timeout():
    out = run_python("while True: pass", timeout=1)
    assert "timeout" in out.lower() or "timed out" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_python_exec.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `python_exec.py`**

Create `src/gaia_agent/tools/python_exec.py`. This module runs untrusted LLM-generated Python in a subprocess using the Python-builtin code executor (imported via `builtins` to make the boundary explicit). The subprocess + hard timeout is the safety boundary — treat output as untrusted.

```python
from __future__ import annotations
import ast
import builtins
import contextlib
import io
import multiprocessing as mp
import traceback

_py_exec = builtins.exec
_py_eval = builtins.eval

def _worker(code: str, q: "mp.Queue") -> None:
    buf = io.StringIO()
    ns: dict = {}
    try:
        tree = ast.parse(code, mode="exec")
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(body=tree.body[-1].value)
            tree.body = tree.body[:-1]
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            _py_exec(compile(tree, "<agent>", "exec"), ns)
            if last_expr is not None:
                val = _py_eval(compile(last_expr, "<agent>", "eval"), ns)
                if val is not None:
                    print(repr(val), file=buf)
        q.put(buf.getvalue())
    except Exception:
        q.put(buf.getvalue() + "\n" + traceback.format_exc())

def run_python(code: str, timeout: int = 30) -> str:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return f"ERROR: execution timed out after {timeout}s"
    if not q.empty():
        return q.get()
    return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_python_exec.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/tools/python_exec.py tests/tools/test_python_exec.py
git commit -m "feat(tools): sandboxed python executor"
```

---

## Task 9: File readers

**Files:**
- Create: `src/gaia_agent/tools/files.py`
- Test: `tests/tools/test_files.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tools/test_files.py`:

```python
from pathlib import Path
import pandas as pd
from docx import Document
from gaia_agent.tools.files import read_file

def test_read_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world")
    assert "hello world" in read_file(str(p))

def test_read_csv(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("col1,col2\n1,2\n3,4\n")
    out = read_file(str(p))
    assert "col1" in out and "3" in out

def test_read_xlsx(tmp_path):
    p = tmp_path / "a.xlsx"
    pd.DataFrame({"name": ["alice"], "age": [30]}).to_excel(p, index=False)
    out = read_file(str(p))
    assert "alice" in out and "30" in out

def test_read_docx(tmp_path):
    p = tmp_path / "a.docx"
    doc = Document()
    doc.add_paragraph("Hello docx.")
    doc.save(p)
    assert "Hello docx." in read_file(str(p))

def test_read_unknown_falls_back_to_bytes(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"raw")
    out = read_file(str(p))
    assert "raw" in out or "binary" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_files.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `files.py`**

Create `src/gaia_agent/tools/files.py`:

```python
from __future__ import annotations
from pathlib import Path
import pandas as pd
from pypdf import PdfReader
from docx import Document

def read_file(path: str, max_chars: int = 20000) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    try:
        if ext == ".txt" or ext == ".md":
            text = p.read_text(encoding="utf-8", errors="replace")
        elif ext == ".csv":
            df = pd.read_csv(p)
            text = df.to_csv(index=False)
        elif ext in (".xlsx", ".xls"):
            dfs = pd.read_excel(p, sheet_name=None)
            text = "\n\n".join(f"--- Sheet: {name} ---\n{df.to_csv(index=False)}"
                               for name, df in dfs.items())
        elif ext == ".pdf":
            reader = PdfReader(str(p))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        elif ext == ".docx":
            doc = Document(str(p))
            text = "\n".join(para.text for para in doc.paragraphs)
        else:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = f"[binary file, {p.stat().st_size} bytes]"
    except Exception as e:
        return f"ERROR reading {path}: {e}"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_files.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/tools/files.py tests/tools/test_files.py
git commit -m "feat(tools): multi-format file reader"
```

---

## Task 10: Audio transcription tool

**Files:**
- Create: `src/gaia_agent/tools/audio.py`
- Test: `tests/tools/test_audio.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tools/test_audio.py`:

```python
from unittest.mock import patch, MagicMock
from gaia_agent.tools.audio import transcribe_audio

def test_transcribe_audio_joins_segments():
    fake_model = MagicMock()
    fake_segment = MagicMock()
    fake_segment.text = " hello world "
    fake_model.transcribe.return_value = ([fake_segment, fake_segment], MagicMock(language="en"))
    with patch("gaia_agent.tools.audio._get_model", return_value=fake_model):
        out = transcribe_audio("/tmp/nope.mp3", model_size="base")
    assert "hello world" in out
    assert out.count("hello world") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_audio.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `audio.py`**

Create `src/gaia_agent/tools/audio.py`:

```python
from __future__ import annotations
from functools import lru_cache

@lru_cache(maxsize=2)
def _get_model(model_size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe_audio(path: str, model_size: str = "base") -> str:
    model = _get_model(model_size)
    segments, _info = model.transcribe(path)
    return " ".join(seg.text.strip() for seg in segments).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_audio.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/tools/audio.py tests/tools/test_audio.py
git commit -m "feat(tools): faster-whisper audio transcription"
```

---

## Task 11: YouTube transcript tool

**Files:**
- Create: `src/gaia_agent/tools/youtube.py`
- Test: `tests/tools/test_youtube.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tools/test_youtube.py`:

```python
from unittest.mock import patch
from gaia_agent.tools.youtube import youtube_transcript, extract_video_id

def test_extract_video_id_from_watch_url():
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_extract_video_id_from_short_url():
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=10") == "dQw4w9WgXcQ"

def test_youtube_transcript_joins_segments():
    fake = [{"text": "hello"}, {"text": "world"}]
    with patch("gaia_agent.tools.youtube.YouTubeTranscriptApi.get_transcript", return_value=fake):
        out = youtube_transcript("https://youtu.be/abc")
    assert "hello world" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_youtube.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `youtube.py`**

Create `src/gaia_agent/tools/youtube.py`:

```python
from __future__ import annotations
import re
from youtube_transcript_api import YouTubeTranscriptApi

_PATTERNS = [
    re.compile(r"youtube\.com/watch\?v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{11})"),
]

def extract_video_id(url: str) -> str:
    for pat in _PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract video id from {url}")

def youtube_transcript(url: str) -> str:
    vid = extract_video_id(url)
    segments = YouTubeTranscriptApi.get_transcript(vid)
    return " ".join(seg["text"] for seg in segments)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_youtube.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/tools/youtube.py tests/tools/test_youtube.py
git commit -m "feat(tools): youtube transcript fetch"
```

---

## Task 12: Tool registry and unified LangChain Tool interface

**Files:**
- Modify: `src/gaia_agent/tools/__init__.py`
- Test: `tests/tools/test_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tools/test_registry.py`:

```python
from gaia_agent.config import Config
from gaia_agent.tools import build_tools

def test_build_tools_returns_expected_names():
    cfg = Config.from_env()
    tools = build_tools(cfg)
    names = {t.name for t in tools}
    assert {"tavily_search", "fetch_url", "run_python", "read_file",
            "transcribe_audio", "youtube_transcript"} <= names

def test_each_tool_has_description():
    cfg = Config.from_env()
    for t in build_tools(cfg):
        assert t.description, f"{t.name} missing description"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_registry.py -v`
Expected: `ImportError: cannot import name 'build_tools'`.

- [ ] **Step 3: Implement the registry**

Replace `src/gaia_agent/tools/__init__.py` with:

```python
from __future__ import annotations
from langchain_core.tools import tool, BaseTool
from gaia_agent.config import Config
from gaia_agent.tools.search import tavily_search as _search
from gaia_agent.tools.web import fetch_url as _fetch
from gaia_agent.tools.python_exec import run_python as _run_py
from gaia_agent.tools.files import read_file as _read_file
from gaia_agent.tools.audio import transcribe_audio as _transcribe
from gaia_agent.tools.youtube import youtube_transcript as _yt

def build_tools(cfg: Config) -> list[BaseTool]:
    @tool
    def tavily_search(query: str) -> str:
        """Search the web via Tavily. Returns top results with titles, URLs, and snippets."""
        return _search(query, api_key=cfg.tavily_api_key)

    @tool
    def fetch_url(url: str) -> str:
        """Fetch a URL and return cleaned main text content."""
        return _fetch(url)

    @tool
    def run_python(code: str) -> str:
        """Execute Python code in a sandboxed subprocess. Returns stdout/stderr/last-expr repr."""
        return _run_py(code)

    @tool
    def read_file(path: str) -> str:
        """Read a local file. Supports txt, md, csv, xlsx, pdf, docx."""
        return _read_file(path)

    @tool
    def transcribe_audio(path: str) -> str:
        """Transcribe a local audio file using faster-whisper."""
        return _transcribe(path, model_size=cfg.whisper_model)

    @tool
    def youtube_transcript(url: str) -> str:
        """Fetch the transcript of a YouTube video by URL."""
        return _yt(url)

    return [tavily_search, fetch_url, run_python, read_file, transcribe_audio, youtube_transcript]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_registry.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/tools/__init__.py tests/tools/test_registry.py
git commit -m "feat(tools): build_tools registry for LangChain BaseTool interface"
```

---

## Task 13: Prompts module

**Files:**
- Create: `src/gaia_agent/prompts.py`

- [ ] **Step 1: Create `prompts.py`**

Create `src/gaia_agent/prompts.py`:

```python
"""System prompts for each node. Kept separate so they can be tuned without touching logic."""

PLANNER_SYSTEM = """You are the planning node of a GAIA benchmark agent.

Given a question (and optional file preview), write a 3-5 step plan to solve it.
Each step MUST declare a tier: "S1" for routine tool calls (search, fetch, file read, simple lookup)
or "S2" for reasoning-heavy steps (multi-hop synthesis, ambiguous questions, final reasoning).

Respond ONLY with a JSON object of this exact shape:
{
  "plan": [
    {"description": "...", "tier": "S1"},
    {"description": "...", "tier": "S2"}
  ],
  "expected_answer_shape": "short string describing answer format, e.g. 'a single integer' or 'a person's full name'"
}

If a critique from a prior attempt is provided, revise the plan to address it."""

EXECUTOR_SYSTEM = """You are the execution node of a GAIA benchmark agent.

You are executing ONE step of a plan. The full plan, prior observations, and current step are given.
You have tools available. Call ONE tool that advances the current step, then stop.
If the current step does not require a tool (pure reasoning), respond with a concise result in text.
When the plan is complete, respond with a DRAFT ANSWER marked with the prefix "DRAFT:"."""

VERIFIER_SYSTEM = """You are the verification node of a GAIA benchmark agent.

Given the original question, the plan, all observations, and a draft answer, decide:
- APPROVED: the draft correctly answers the question and fits the expected answer shape.
- REJECTED: the draft is wrong, incomplete, or wrong-format. Include a 1-2 sentence critique
  explaining what to fix.

Respond ONLY with JSON:
{"decision": "APPROVED" | "REJECTED", "critique": "..." | null}"""
```

- [ ] **Step 2: Commit**

```bash
git add src/gaia_agent/prompts.py
git commit -m "feat(prompts): system prompts for planner/executor/verifier"
```

---

## Task 14: Perception node

**Files:**
- Create: `src/gaia_agent/nodes/__init__.py`
- Create: `src/gaia_agent/nodes/perception.py`
- Test: `tests/nodes/__init__.py`
- Test: `tests/nodes/test_perception.py`

- [ ] **Step 1: Write the failing test**

Create `tests/nodes/__init__.py` as empty.

Create `tests/nodes/test_perception.py`:

```python
from unittest.mock import MagicMock
from pathlib import Path
from gaia_agent.state import new_state
from gaia_agent.nodes.perception import make_perception_node

def test_perception_no_file():
    client = MagicMock()
    client.download_file.return_value = None
    node = make_perception_node(client, Path("/tmp/gaia"))
    state = new_state("t1", "what is 2+2?")
    out = node(state)
    assert out["file_path"] is None
    assert out["modality"] == "text"

def test_perception_pdf_file(tmp_path):
    fake_pdf = tmp_path / "doc.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")
    client = MagicMock()
    client.download_file.return_value = fake_pdf
    node = make_perception_node(client, tmp_path)
    state = new_state("t1", "What does the attached PDF say?")
    out = node(state)
    assert out["file_path"] == str(fake_pdf)
    assert out["modality"] == "pdf"

def test_perception_youtube_in_question():
    client = MagicMock()
    client.download_file.return_value = None
    node = make_perception_node(client, Path("/tmp/gaia"))
    state = new_state("t1", "Watch https://youtu.be/abc and tell me who speaks first")
    out = node(state)
    assert out["modality"] == "youtube"
```

- [ ] **Step 2: Run test to verify it fails**

Create empty `src/gaia_agent/nodes/__init__.py`.
Run: `pytest tests/nodes/test_perception.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `perception.py`**

Create `src/gaia_agent/nodes/perception.py`:

```python
from __future__ import annotations
from pathlib import Path
from gaia_agent.state import AgentState, Modality
from gaia_agent.api_client import GaiaApiClient

_EXT_TO_MODALITY: dict[str, Modality] = {
    ".pdf": "pdf", ".xlsx": "excel", ".xls": "excel", ".csv": "csv",
    ".docx": "docx", ".doc": "docx",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".flac": "audio",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".txt": "text", ".md": "text",
}

def _modality_from_question(q: str) -> Modality | None:
    ql = q.lower()
    if "youtube.com" in ql or "youtu.be" in ql:
        return "youtube"
    if "http://" in ql or "https://" in ql:
        return "web"
    return None

def make_perception_node(client: GaiaApiClient, file_dir: Path):
    def perception(state: AgentState) -> dict:
        path = client.download_file(state["task_id"], file_dir)
        modality: Modality = "text"
        file_path: str | None = None
        if path is not None:
            file_path = str(path)
            modality = _EXT_TO_MODALITY.get(path.suffix.lower(), "text")
        else:
            q_mod = _modality_from_question(state["question"])
            if q_mod:
                modality = q_mod
        return {"file_path": file_path, "modality": modality}
    return perception
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_perception.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/nodes/__init__.py src/gaia_agent/nodes/perception.py tests/nodes/__init__.py tests/nodes/test_perception.py
git commit -m "feat(nodes): perception node with modality detection"
```

---

## Task 15: Planner node

**Files:**
- Create: `src/gaia_agent/nodes/planner.py`
- Test: `tests/nodes/test_planner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/nodes/test_planner.py`:

```python
import json
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from gaia_agent.state import new_state
from gaia_agent.nodes.planner import make_planner_node

def _fake_model(response_text: str):
    model = MagicMock()
    model.invoke.return_value = AIMessage(content=response_text)
    return model

def test_planner_parses_valid_json():
    resp = json.dumps({
        "plan": [
            {"description": "search for X", "tier": "S1"},
            {"description": "reason about result", "tier": "S2"},
        ],
        "expected_answer_shape": "single word"
    })
    model = _fake_model(resp)
    node = make_planner_node(model)
    state = new_state("t1", "Who is X?")
    out = node(state)
    assert len(out["plan"]) == 2
    assert out["plan"][0]["tier"] == "S1"
    assert out["step_idx"] == 0
    assert out["critique"] is None

def test_planner_handles_json_with_prose():
    resp = 'Here is the plan:\n```json\n{"plan":[{"description":"a","tier":"S2"}],"expected_answer_shape":"x"}\n```'
    model = _fake_model(resp)
    node = make_planner_node(model)
    state = new_state("t1", "q")
    out = node(state)
    assert len(out["plan"]) == 1

def test_planner_uses_critique_on_retry():
    resp = json.dumps({"plan": [{"description": "new step", "tier": "S1"}],
                       "expected_answer_shape": "x"})
    model = _fake_model(resp)
    node = make_planner_node(model)
    state = new_state("t1", "q")
    state["critique"] = "previous answer wrong because Y"
    state["retries"] = 1
    node(state)
    args, kwargs = model.invoke.call_args
    messages = args[0]
    joined = " ".join(getattr(m, "content", "") for m in messages)
    assert "previous answer wrong because Y" in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_planner.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `planner.py`**

Create `src/gaia_agent/nodes/planner.py`:

```python
from __future__ import annotations
import json
import re
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from gaia_agent.state import AgentState, PlanStep
from gaia_agent.prompts import PLANNER_SYSTEM

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json(text: str) -> dict:
    m = _JSON_BLOCK.search(text)
    if not m:
        raise ValueError(f"No JSON object found in planner response: {text[:200]}")
    return json.loads(m.group(0))

def make_planner_node(model: BaseChatModel):
    def planner(state: AgentState) -> dict:
        human_parts = [f"Question: {state['question']}"]
        if state["file_path"]:
            human_parts.append(f"Associated file: {state['file_path']} (modality: {state['modality']})")
        if state["critique"]:
            human_parts.append(f"Prior attempt critique: {state['critique']}")
        messages = [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content="\n".join(human_parts)),
        ]
        resp = model.invoke(messages)
        data = _extract_json(resp.content if isinstance(resp.content, str) else str(resp.content))
        plan: list[PlanStep] = [
            {"description": s["description"], "tier": s.get("tier", "S1")}
            for s in data.get("plan", [])
        ]
        return {
            "plan": plan,
            "step_idx": 0,
            "observations": [],
            "draft_answer": None,
            "critique": None,
        }
    return planner
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_planner.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/nodes/planner.py tests/nodes/test_planner.py
git commit -m "feat(nodes): planner node with JSON plan extraction"
```

---

## Task 16: Router (rule-based edge function)

**Files:**
- Create: `src/gaia_agent/nodes/router.py`
- Test: `tests/nodes/test_router.py`

- [ ] **Step 1: Write the failing test**

Create `tests/nodes/test_router.py`:

```python
from gaia_agent.state import new_state
from gaia_agent.nodes.router import route_next

def _state_with_plan(plan, step_idx=0, draft=None):
    s = new_state("t1", "q")
    s["plan"] = plan
    s["step_idx"] = step_idx
    s["draft_answer"] = draft
    return s

def test_route_to_s1():
    plan = [{"description": "search", "tier": "S1"}]
    assert route_next(_state_with_plan(plan)) == "exec_s1"

def test_route_to_s2():
    plan = [{"description": "think hard", "tier": "S2"}]
    assert route_next(_state_with_plan(plan)) == "exec_s2"

def test_route_to_verifier_when_plan_exhausted():
    plan = [{"description": "a", "tier": "S1"}]
    s = _state_with_plan(plan, step_idx=1, draft="42")
    assert route_next(s) == "verifier"

def test_route_to_verifier_when_draft_set_early():
    plan = [{"description": "a", "tier": "S1"}, {"description": "b", "tier": "S2"}]
    s = _state_with_plan(plan, step_idx=1, draft="done")
    assert route_next(s) == "verifier"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_router.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `router.py`**

Create `src/gaia_agent/nodes/router.py`:

```python
from __future__ import annotations
from typing import Literal
from gaia_agent.state import AgentState

RouteTarget = Literal["exec_s1", "exec_s2", "verifier"]

def route_next(state: AgentState) -> RouteTarget:
    if state["draft_answer"] is not None:
        return "verifier"
    if state["step_idx"] >= len(state["plan"]):
        return "verifier"
    current = state["plan"][state["step_idx"]]
    return "exec_s1" if current["tier"] == "S1" else "exec_s2"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_router.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/nodes/router.py tests/nodes/test_router.py
git commit -m "feat(nodes): rule-based step router"
```

---

## Task 17: Executor node

**Files:**
- Create: `src/gaia_agent/nodes/executor.py`
- Test: `tests/nodes/test_executor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/nodes/test_executor.py`:

```python
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from gaia_agent.state import new_state
from gaia_agent.nodes.executor import make_executor_node

@tool
def fake_tool(query: str) -> str:
    """fake tool"""
    return f"result for {query}"

def _state():
    s = new_state("t1", "Who is X?")
    s["plan"] = [
        {"description": "search for X", "tier": "S1"},
        {"description": "summarize", "tier": "S2"},
    ]
    s["step_idx"] = 0
    return s

def test_executor_runs_tool_call():
    model = MagicMock()
    ai = AIMessage(content="", tool_calls=[{
        "name": "fake_tool", "args": {"query": "X"}, "id": "call_1"
    }])
    model.bind_tools.return_value = model
    model.invoke.return_value = ai

    node = make_executor_node(model, [fake_tool])
    out = node(_state())
    assert len(out["observations"]) == 1
    assert "result for X" in out["observations"][0]["result"]
    assert out["step_idx"] == 1
    assert out["draft_answer"] is None

def test_executor_records_draft_on_text_answer_with_prefix():
    model = MagicMock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(content="DRAFT: 42")

    node = make_executor_node(model, [fake_tool])
    s = _state()
    s["step_idx"] = 1
    out = node(s)
    assert out["draft_answer"] == "42"
    assert out["step_idx"] == 2

def test_executor_plain_text_reasoning_advances_step():
    model = MagicMock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(content="some reasoning result")
    node = make_executor_node(model, [fake_tool])
    out = node(_state())
    assert out["step_idx"] == 1
    assert len(out["observations"]) == 1
    assert "some reasoning result" in out["observations"][0]["result"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_executor.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `executor.py`**

Create `src/gaia_agent/nodes/executor.py`:

```python
from __future__ import annotations
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from gaia_agent.state import AgentState, Observation
from gaia_agent.prompts import EXECUTOR_SYSTEM

DRAFT_PREFIX = "DRAFT:"

def _format_context(state: AgentState) -> str:
    lines = [f"Question: {state['question']}"]
    if state["file_path"]:
        lines.append(f"File: {state['file_path']} ({state['modality']})")
    lines.append("\nPlan:")
    for i, step in enumerate(state["plan"]):
        marker = ">>" if i == state["step_idx"] else "  "
        lines.append(f"{marker} {i}. [{step['tier']}] {step['description']}")
    if state["observations"]:
        lines.append("\nPrior observations:")
        for obs in state["observations"]:
            snippet = obs["result"][:500]
            lines.append(f"- step {obs['step_idx']} [{obs['tool']}]: {snippet}")
    lines.append(f"\nExecute step {state['step_idx']} now.")
    return "\n".join(lines)

def make_executor_node(model: BaseChatModel, tools: list[BaseTool]):
    tools_by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)

    def executor(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=EXECUTOR_SYSTEM),
            HumanMessage(content=_format_context(state)),
        ]
        resp = bound.invoke(messages)

        new_obs: list[Observation] = list(state["observations"])
        draft: str | None = state["draft_answer"]

        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("args", {})
                try:
                    result = tools_by_name[name].invoke(args)
                except Exception as e:
                    result = f"TOOL ERROR: {e}"
                new_obs.append({
                    "step_idx": state["step_idx"],
                    "tool": name,
                    "args": args,
                    "result": str(result),
                })
        else:
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            stripped = content.strip()
            if stripped.upper().startswith(DRAFT_PREFIX):
                draft = stripped[len(DRAFT_PREFIX):].strip()
            else:
                new_obs.append({
                    "step_idx": state["step_idx"],
                    "tool": "reasoning",
                    "args": {},
                    "result": stripped,
                })

        return {
            "observations": new_obs,
            "step_idx": state["step_idx"] + 1,
            "draft_answer": draft,
        }
    return executor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_executor.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/nodes/executor.py tests/nodes/test_executor.py
git commit -m "feat(nodes): executor with tool-call loop and draft detection"
```

---

## Task 18: Verifier node

**Files:**
- Create: `src/gaia_agent/nodes/verifier.py`
- Test: `tests/nodes/test_verifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/nodes/test_verifier.py`:

```python
import json
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from gaia_agent.state import new_state
from gaia_agent.nodes.verifier import make_verifier_node, verifier_decision

def _model(text):
    m = MagicMock()
    m.invoke.return_value = AIMessage(content=text)
    return m

def _draft_state(draft="42"):
    s = new_state("t1", "what is 6 times 7?")
    s["plan"] = [{"description": "compute", "tier": "S1"}]
    s["step_idx"] = 1
    s["draft_answer"] = draft
    return s

def test_verifier_approves():
    model = _model(json.dumps({"decision": "APPROVED", "critique": None}))
    node = make_verifier_node(model)
    out = node(_draft_state())
    assert out["critique"] is None
    assert out["final_answer"] == "42"

def test_verifier_rejects_with_critique():
    model = _model(json.dumps({"decision": "REJECTED",
                               "critique": "should be 42 not 41"}))
    node = make_verifier_node(model)
    s = _draft_state(draft="41")
    out = node(s)
    assert out["critique"] == "should be 42 not 41"
    assert out["final_answer"] is None
    assert out["draft_answer"] is None
    assert out["retries"] == 1

def test_verifier_decision_routes():
    s_ok = _draft_state()
    s_ok["final_answer"] = "42"
    assert verifier_decision(s_ok) == "formatter"

    s_bad = _draft_state()
    s_bad["critique"] = "nope"
    s_bad["retries"] = 1
    assert verifier_decision(s_bad) == "planner"

def test_verifier_gives_up_after_max_retries():
    s = _draft_state()
    s["critique"] = "still wrong"
    s["retries"] = 3
    assert verifier_decision(s) == "formatter"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_verifier.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `verifier.py`**

Create `src/gaia_agent/nodes/verifier.py`:

```python
from __future__ import annotations
import json
import re
from typing import Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from gaia_agent.state import AgentState
from gaia_agent.prompts import VERIFIER_SYSTEM

MAX_RETRIES = 2
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json(text: str) -> dict:
    m = _JSON_BLOCK.search(text)
    if not m:
        return {"decision": "APPROVED", "critique": None}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"decision": "APPROVED", "critique": None}

def make_verifier_node(model: BaseChatModel):
    def verifier(state: AgentState) -> dict:
        context = [
            f"Question: {state['question']}",
            f"Draft answer: {state['draft_answer']}",
            "",
            "Plan executed:",
        ]
        for i, step in enumerate(state["plan"]):
            context.append(f"  {i}. [{step['tier']}] {step['description']}")
        context.append("")
        context.append("Observations:")
        for obs in state["observations"]:
            context.append(f"- [{obs['tool']}] {obs['result'][:400]}")

        messages = [
            SystemMessage(content=VERIFIER_SYSTEM),
            HumanMessage(content="\n".join(context)),
        ]
        resp = model.invoke(messages)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = _extract_json(text)

        if data.get("decision") == "APPROVED":
            return {
                "final_answer": state["draft_answer"],
                "critique": None,
            }
        return {
            "critique": data.get("critique") or "Answer rejected without specific reason.",
            "draft_answer": None,
            "retries": state["retries"] + 1,
        }
    return verifier

def verifier_decision(state: AgentState) -> Literal["formatter", "planner"]:
    if state["final_answer"] is not None:
        return "formatter"
    if state["retries"] >= MAX_RETRIES + 1:
        return "formatter"
    return "planner"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_verifier.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/nodes/verifier.py tests/nodes/test_verifier.py
git commit -m "feat(nodes): verifier with approve/reject JSON parsing"
```

---

## Task 19: Formatter node

**Files:**
- Create: `src/gaia_agent/nodes/formatter.py`
- Test: `tests/nodes/test_formatter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/nodes/test_formatter.py`:

```python
from gaia_agent.state import new_state
from gaia_agent.nodes.formatter import formatter

def _state(final="The answer is 42."):
    s = new_state("t1", "q")
    s["final_answer"] = final
    s["draft_answer"] = final
    return s

def test_formatter_strips_prefixes():
    out = formatter(_state("The answer is 42."))
    assert out["final_answer"] == "42"

def test_formatter_strips_final_answer_prefix():
    out = formatter(_state("FINAL ANSWER: Paris"))
    assert out["final_answer"] == "Paris"

def test_formatter_strips_trailing_punctuation():
    out = formatter(_state("42."))
    assert out["final_answer"] == "42"

def test_formatter_uses_draft_if_final_missing():
    s = new_state("t1", "q")
    s["final_answer"] = None
    s["draft_answer"] = "50"
    out = formatter(s)
    assert out["final_answer"] == "50"

def test_formatter_empty_fallback():
    s = new_state("t1", "q")
    out = formatter(s)
    assert out["final_answer"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_formatter.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `formatter.py`**

Create `src/gaia_agent/nodes/formatter.py`:

```python
from __future__ import annotations
import re
from gaia_agent.state import AgentState

_PREFIXES = [
    r"^final answer\s*[:\-]\s*",
    r"^answer\s*[:\-]\s*",
    r"^the answer is\s*",
    r"^result\s*[:\-]\s*",
]
_PREFIX_RE = re.compile("|".join(_PREFIXES), re.IGNORECASE)

def _normalize(ans: str) -> str:
    s = ans.strip()
    s = _PREFIX_RE.sub("", s).strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1]
    s = re.sub(r"[.!?]+$", "", s).strip()
    return s

def formatter(state: AgentState) -> dict:
    raw = state["final_answer"] or state["draft_answer"] or ""
    return {"final_answer": _normalize(raw)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_formatter.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/nodes/formatter.py tests/nodes/test_formatter.py
git commit -m "feat(nodes): exact-match answer formatter"
```

---

## Task 20: Graph assembly

**Files:**
- Create: `src/gaia_agent/graph.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_graph.py`:

```python
import json
from unittest.mock import MagicMock
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from gaia_agent.graph import build_graph
from gaia_agent.state import new_state

@tool
def fake_search(query: str) -> str:
    """fake search"""
    return "Paris is the capital of France."

def _planner_model():
    m = MagicMock()
    m.invoke.return_value = AIMessage(content=json.dumps({
        "plan": [
            {"description": "search for capital of France", "tier": "S1"},
            {"description": "answer from results", "tier": "S2"},
        ],
        "expected_answer_shape": "single word"
    }))
    return m

def _executor_model_s1():
    m = MagicMock()
    m.bind_tools.return_value = m
    m.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "fake_search", "args": {"query": "capital of France"}, "id": "c1"}],
    )
    return m

def _executor_model_s2():
    m = MagicMock()
    m.bind_tools.return_value = m
    m.invoke.return_value = AIMessage(content="DRAFT: Paris")
    return m

def _verifier_model():
    m = MagicMock()
    m.invoke.return_value = AIMessage(content=json.dumps({
        "decision": "APPROVED", "critique": None
    }))
    return m

def _perception(state):
    return {"file_path": None, "modality": "text"}

def test_graph_happy_path():
    graph = build_graph(
        perception_node=_perception,
        planner_model=_planner_model(),
        executor_model_s1=_executor_model_s1(),
        executor_model_s2=_executor_model_s2(),
        verifier_model=_verifier_model(),
        tools=[fake_search],
    )
    state = new_state("t1", "What is the capital of France?")
    final = graph.invoke(state)
    assert final["final_answer"] == "Paris"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `graph.py`**

Create `src/gaia_agent/graph.py`:

```python
from __future__ import annotations
from typing import Callable
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from gaia_agent.state import AgentState
from gaia_agent.nodes.planner import make_planner_node
from gaia_agent.nodes.router import route_next
from gaia_agent.nodes.executor import make_executor_node
from gaia_agent.nodes.verifier import make_verifier_node, verifier_decision
from gaia_agent.nodes.formatter import formatter

def build_graph(
    perception_node: Callable[[AgentState], dict],
    planner_model: BaseChatModel,
    executor_model_s1: BaseChatModel,
    executor_model_s2: BaseChatModel,
    verifier_model: BaseChatModel,
    tools: list[BaseTool],
):
    g = StateGraph(AgentState)

    g.add_node("perception", perception_node)
    g.add_node("planner", make_planner_node(planner_model))
    g.add_node("exec_s1", make_executor_node(executor_model_s1, tools))
    g.add_node("exec_s2", make_executor_node(executor_model_s2, tools))
    g.add_node("verifier", make_verifier_node(verifier_model))
    g.add_node("formatter", formatter)

    g.set_entry_point("perception")
    g.add_edge("perception", "planner")
    g.add_conditional_edges("planner", route_next,
        {"exec_s1": "exec_s1", "exec_s2": "exec_s2", "verifier": "verifier"})
    g.add_conditional_edges("exec_s1", route_next,
        {"exec_s1": "exec_s1", "exec_s2": "exec_s2", "verifier": "verifier"})
    g.add_conditional_edges("exec_s2", route_next,
        {"exec_s1": "exec_s1", "exec_s2": "exec_s2", "verifier": "verifier"})
    g.add_conditional_edges("verifier", verifier_decision,
        {"planner": "planner", "formatter": "formatter"})
    g.add_edge("formatter", END)

    return g.compile()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/graph.py tests/test_graph.py
git commit -m "feat(graph): assemble Plan/Execute/Verify LangGraph"
```

---

## Task 21: Runner with JSON checkpointing

**Files:**
- Create: `src/gaia_agent/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner.py`:

```python
import json
from unittest.mock import MagicMock
from pathlib import Path
from gaia_agent.runner import run_agent_on_questions

def test_runner_writes_checkpoints_and_returns_answers(tmp_path):
    questions = [
        {"task_id": "t1", "question": "q1"},
        {"task_id": "t2", "question": "q2"},
    ]
    graph = MagicMock()
    graph.invoke.side_effect = [
        {"final_answer": "A1"},
        {"final_answer": "A2"},
    ]

    answers = run_agent_on_questions(
        graph=graph, questions=questions, checkpoint_dir=tmp_path,
    )

    assert answers == [
        {"task_id": "t1", "submitted_answer": "A1"},
        {"task_id": "t2", "submitted_answer": "A2"},
    ]
    assert (tmp_path / "t1.json").exists()
    assert json.loads((tmp_path / "t1.json").read_text())["submitted_answer"] == "A1"

def test_runner_resumes_from_checkpoints(tmp_path):
    (tmp_path / "t1.json").write_text(json.dumps(
        {"task_id": "t1", "submitted_answer": "cached"}
    ))
    questions = [
        {"task_id": "t1", "question": "q1"},
        {"task_id": "t2", "question": "q2"},
    ]
    graph = MagicMock()
    graph.invoke.return_value = {"final_answer": "A2"}

    answers = run_agent_on_questions(
        graph=graph, questions=questions, checkpoint_dir=tmp_path,
    )

    assert graph.invoke.call_count == 1
    assert {"task_id": "t1", "submitted_answer": "cached"} in answers

def test_runner_records_error_on_exception(tmp_path):
    questions = [{"task_id": "t1", "question": "q1"}]
    graph = MagicMock()
    graph.invoke.side_effect = RuntimeError("boom")

    answers = run_agent_on_questions(
        graph=graph, questions=questions, checkpoint_dir=tmp_path,
    )
    assert len(answers) == 1
    assert "ERROR" in answers[0]["submitted_answer"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runner.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `runner.py`**

Create `src/gaia_agent/runner.py`:

```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from gaia_agent.state import new_state

def run_agent_on_questions(
    graph: Any,
    questions: list[dict],
    checkpoint_dir: Path,
) -> list[dict]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    answers: list[dict] = []

    for q in questions:
        task_id = q.get("task_id")
        question_text = q.get("question")
        if not task_id or question_text is None:
            continue

        ckpt = checkpoint_dir / f"{task_id}.json"
        if ckpt.exists():
            answers.append(json.loads(ckpt.read_text()))
            continue

        try:
            state = new_state(task_id=task_id, question=question_text)
            final = graph.invoke(state)
            submitted = final.get("final_answer") or ""
            entry = {"task_id": task_id, "submitted_answer": submitted}
        except Exception as e:
            entry = {"task_id": task_id, "submitted_answer": f"AGENT ERROR: {e}"}

        ckpt.write_text(json.dumps(entry))
        answers.append(entry)

    return answers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runner.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gaia_agent/runner.py tests/test_runner.py
git commit -m "feat(runner): question loop with JSON checkpointing"
```

---

## Task 22: Wire the agent into `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace `BasicAgent` and the run loop**

Open `app.py` and replace lines 13-20 (the `BasicAgent` class) with:

```python
from pathlib import Path
from gaia_agent.config import Config
from gaia_agent.models import get_cheap_model, get_strong_model
from gaia_agent.api_client import GaiaApiClient
from gaia_agent.tools import build_tools
from gaia_agent.nodes.perception import make_perception_node
from gaia_agent.graph import build_graph
from gaia_agent.runner import run_agent_on_questions
from gaia_agent.state import new_state

class GaiaAgent:
    def __init__(self):
        self.cfg = Config.from_env()
        self.client = GaiaApiClient(self.cfg.api_url)
        cheap = get_cheap_model(self.cfg)
        strong = get_strong_model(self.cfg)
        tools = build_tools(self.cfg)
        file_dir = Path(self.cfg.checkpoint_dir) / "files"
        perception = make_perception_node(self.client, file_dir)
        self.graph = build_graph(
            perception_node=perception,
            planner_model=strong,
            executor_model_s1=cheap,
            executor_model_s2=strong,
            verifier_model=strong,
            tools=tools,
        )
        print("GaiaAgent initialized.")

    def __call__(self, question: str, task_id: str = "adhoc") -> str:
        state = new_state(task_id=task_id, question=question)
        final = self.graph.invoke(state)
        return final.get("final_answer") or ""
```

- [ ] **Step 2: Replace the question loop**

In `run_and_submit_all`, replace the block starting with `# 1. Instantiate Agent` through `# 3. Run your Agent` (lines ~41-88) with:

```python
    # 1. Instantiate Agent
    try:
        agent = GaiaAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    try:
        questions_data = agent.client.get_questions()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        return f"Error fetching questions: {e}", None

    # 3. Run the agent with checkpointing
    ckpt_dir = Path(agent.cfg.checkpoint_dir) / "answers"
    answers_payload = run_agent_on_questions(
        graph=agent.graph, questions=questions_data, checkpoint_dir=ckpt_dir,
    )
    results_log = [
        {"Task ID": a["task_id"],
         "Question": next((q["question"] for q in questions_data if q["task_id"] == a["task_id"]), ""),
         "Submitted Answer": a["submitted_answer"]}
        for a in answers_payload
    ]
```

- [ ] **Step 3: Smoke test the import**

Run: `python -c "import app; print('ok')"`
Expected: `ok` (no ImportError).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): wire GaiaAgent with LangGraph into Gradio runner"
```

---

## Task 23: End-to-end smoke test with stub models

**Files:**
- Create: `tests/test_e2e_smoke.py`

- [ ] **Step 1: Write a single end-to-end test using mocked models**

Create `tests/test_e2e_smoke.py`:

```python
import json
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from gaia_agent.graph import build_graph
from gaia_agent.runner import run_agent_on_questions

@tool
def fake_search(query: str) -> str:
    """fake search"""
    return "The Eiffel Tower is in Paris."

def _planner():
    m = MagicMock()
    m.invoke.return_value = AIMessage(content=json.dumps({
        "plan": [
            {"description": "search for location", "tier": "S1"},
            {"description": "answer", "tier": "S2"},
        ],
        "expected_answer_shape": "city name"
    }))
    return m

def _s1():
    m = MagicMock()
    m.bind_tools.return_value = m
    m.invoke.return_value = AIMessage(content="",
        tool_calls=[{"name": "fake_search", "args": {"query": "Eiffel Tower"}, "id": "1"}])
    return m

def _s2():
    m = MagicMock()
    m.bind_tools.return_value = m
    m.invoke.return_value = AIMessage(content="DRAFT: The answer is Paris.")
    return m

def _verifier():
    m = MagicMock()
    m.invoke.return_value = AIMessage(content=json.dumps({
        "decision": "APPROVED", "critique": None}))
    return m

def _perception(state):
    return {"file_path": None, "modality": "text"}

def test_full_loop_with_checkpointing(tmp_path):
    graph = build_graph(
        perception_node=_perception,
        planner_model=_planner(),
        executor_model_s1=_s1(),
        executor_model_s2=_s2(),
        verifier_model=_verifier(),
        tools=[fake_search],
    )
    questions = [{"task_id": "q1", "question": "Where is the Eiffel Tower?"}]
    answers = run_agent_on_questions(graph=graph, questions=questions,
                                     checkpoint_dir=tmp_path)
    assert answers[0]["task_id"] == "q1"
    assert answers[0]["submitted_answer"] == "Paris"
```

- [ ] **Step 2: Run full test suite**

Run: `pytest -v`
Expected: all tests passing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_smoke.py
git commit -m "test: end-to-end smoke test with mocked models"
```

---

## Task 24: Dev run against Ollama (manual)

**Files:**
- Create: `scripts/dev_run.py`

- [ ] **Step 1: Create a dev script that runs the real graph against 1 question**

Create `scripts/dev_run.py`:

```python
"""Run the GAIA agent against a few questions using the dev model tier.

Usage: python scripts/dev_run.py [n]
  n: number of questions to try (default 1)
"""
from __future__ import annotations
import sys
from pathlib import Path
from gaia_agent.config import Config
from gaia_agent.models import get_cheap_model, get_strong_model
from gaia_agent.api_client import GaiaApiClient
from gaia_agent.tools import build_tools
from gaia_agent.nodes.perception import make_perception_node
from gaia_agent.graph import build_graph
from gaia_agent.state import new_state

def main(n: int = 1) -> None:
    cfg = Config.from_env()
    client = GaiaApiClient(cfg.api_url)
    questions = client.get_questions()[:n]
    file_dir = Path(cfg.checkpoint_dir) / "files"
    file_dir.mkdir(parents=True, exist_ok=True)

    cheap = get_cheap_model(cfg)
    strong = get_strong_model(cfg)
    tools = build_tools(cfg)
    perception = make_perception_node(client, file_dir)
    graph = build_graph(
        perception_node=perception,
        planner_model=strong,
        executor_model_s1=cheap,
        executor_model_s2=strong,
        verifier_model=strong,
        tools=tools,
    )

    for q in questions:
        print(f"\n=== {q['task_id']}: {q['question'][:100]} ===")
        state = new_state(q["task_id"], q["question"])
        result = graph.invoke(state)
        print(f"ANSWER: {result.get('final_answer')}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(n)
```

- [ ] **Step 2: Manual smoke test**

Run: `python scripts/dev_run.py 1`
Expected: produces an answer (may be wrong; we're validating wiring, not quality).

- [ ] **Step 3: Commit**

```bash
git add scripts/dev_run.py
git commit -m "chore(scripts): dev_run script for manual real-model testing"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Every node from the architecture diagram has a task: Perception (14), Planner (15), Router (16), Executor S1/S2 (17), Verifier (18), Formatter (19), Working Memory state (3), Graph wiring (20). Every tool is covered: search (6), web (7), python (8), files (9), audio (10), youtube (11). Infrastructure: config (2), models (4), api client (5), runner (21), app.py integration (22), smoke test (23), manual dev run (24).
- [x] **Placeholder scan:** No "TODO", "fill in", "similar to task N", or missing code blocks. Every test has assertions against concrete values. Every implementation step includes the full file contents or the exact block to change.
- [x] **Type consistency:** `AgentState` fields (`plan`, `step_idx`, `observations`, `draft_answer`, `critique`, `retries`, `final_answer`) are defined in Task 3 and used consistently in Tasks 14-22. `PlanStep.tier` is `"S1" | "S2"` throughout. Router target strings (`"exec_s1"`, `"exec_s2"`, `"verifier"`) match node names in Task 20. `verifier_decision` returns `"formatter"` or `"planner"`, both of which are registered node names in `build_graph`.
- [x] **Out-of-scope drops confirmed:** No Playwright/BrowserUse, no bash tool, no dedicated Wikipedia tool, no image tool (handled via multimodal model directly). These were explicitly dropped for Level 1 scope.
