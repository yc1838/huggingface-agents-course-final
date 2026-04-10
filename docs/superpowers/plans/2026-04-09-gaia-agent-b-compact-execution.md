# GAIA Agent Compact Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` with `test-driven-development` for implementation. This document is intentionally concise and should be treated as a control plan, not a line-by-line script.

**Goal:** Replace the stub `BasicAgent` in `app.py` with a resumable LangGraph-based GAIA agent that can answer a useful subset of Level 1 questions and submit results through the existing Gradio UI.

**Architecture:** Keep the intended `Perception -> Planner -> Router -> Executor -> Verifier -> Formatter` graph, but build it as a thin vertical slice first. Start with text questions, lightweight file support, and a small tool surface. Add heavier tools only when real failures justify them.

**Tech Stack:** Python 3.11+, LangGraph, LangChain core, Gradio, requests/httpx, pytest, optional provider adapters via env config.

---

## Why This Replaces The 2.5k-Line Plan

The original plan is useful as a reference spec, but too rigid for this repo's current state. This version keeps:

- The target graph shape
- Clear module boundaries
- TDD-driven checkpoints
- End-to-end proof before breadth

This version drops:

- Full code listings for every step
- Commit-after-every-micro-step ceremony
- Large up-front dependency expansion
- Heavy tools before the thin slice works

---

## First Slice Scope

**In scope for initial implementation**

- Package scaffold under `src/gaia_agent/`
- Env-driven config and model selection hooks
- GAIA API client for `/questions`, `/files/{task_id}`, and `/submit`
- Typed graph state and JSON checkpointing
- Minimal LangGraph wiring
- Text-first tools:
  - web search
  - web fetch/extract
  - safe Python evaluation
  - local file read for fetched attachments
- Gradio integration through `app.py`
- Tests that prove the graph can run end-to-end with mocked models

**Explicitly deferred until after first end-to-end success**

- Audio transcription
- YouTube transcript support
- Broad office-document coverage beyond formats we actually hit
- Multi-provider feature completeness beyond what the first slice needs
- Extra abstractions that do not yet pay for themselves

---

## File Layout

```text
src/gaia_agent/
  __init__.py
  config.py
  state.py
  api_client.py
  models.py
  prompts.py
  graph.py
  runner.py
  tools/
    __init__.py
    search.py
    web.py
    python_exec.py
    files.py
  nodes/
    __init__.py
    perception.py
    planner.py
    router.py
    executor.py
    verifier.py
    formatter.py

tests/
  test_config.py
  test_api_client.py
  test_graph_smoke.py
  test_runner.py
  tools/
  nodes/
```

**Boundary rules**

- `config.py` owns runtime configuration only.
- `models.py` is the only provider-instantiation layer.
- `api_client.py` is the only GAIA HTTP client.
- `tools/` contains no LLM calls.
- `nodes/` contains graph behavior only.
- `app.py` should stay a thin UI shell.

---

## Execution Plan

### Task 1: Establish a runnable package baseline

**Intent:** Create the smallest package + test scaffold needed to build the agent outside `app.py`.

**Files**

- Modify: `requirements.txt`
- Create: `pyproject.toml`
- Create: `src/gaia_agent/__init__.py`
- Create: `tests/__init__.py`
- Create: `.env.example`

**TDD target**

- Add a minimal import smoke test for `gaia_agent`
- Verify it fails before scaffold exists
- Add only the dependencies required for the first slice

**Done when**

- `pytest` can collect tests without import errors
- Package imports from `src/`

### Task 2: Build the non-LLM foundation

**Intent:** Create stable primitives the graph will depend on.

**Files**

- Create: `src/gaia_agent/config.py`
- Create: `src/gaia_agent/state.py`
- Create: `src/gaia_agent/api_client.py`
- Create: `src/gaia_agent/runner.py`
- Create: `tests/test_config.py`
- Create: `tests/test_api_client.py`
- Create: `tests/test_runner.py`

**TDD target**

- `Config.from_env()` reads defaults and overrides correctly
- `GaiaApiClient` handles question fetch and file download boundaries
- `runner` checkpoints answers and can resume safely

**Done when**

- Core runtime behavior is covered without any real model calls

### Task 3: Add the minimal tool belt

**Intent:** Support the common Level 1 information-gathering loop without prematurely building every tool in the original plan.

**Files**

- Create: `src/gaia_agent/tools/search.py`
- Create: `src/gaia_agent/tools/web.py`
- Create: `src/gaia_agent/tools/python_exec.py`
- Create: `src/gaia_agent/tools/files.py`
- Create: `src/gaia_agent/tools/__init__.py`
- Create matching tests under `tests/tools/`

**TDD target**

- Search wrapper returns normalized text results
- Web fetch returns extracted text, not raw HTML noise
- Python tool can evaluate bounded expressions safely enough for this app
- File reader handles the attachment formats we actually decide to support in slice one

**Done when**

- The executor can call tools through one registry
- Tool tests pass without needing live provider calls except where explicitly mocked

### Task 4: Implement the first usable graph

**Intent:** Make one complete question-answer loop work before broadening capability.

**Files**

- Create: `src/gaia_agent/prompts.py`
- Create: `src/gaia_agent/models.py`
- Create: `src/gaia_agent/nodes/perception.py`
- Create: `src/gaia_agent/nodes/planner.py`
- Create: `src/gaia_agent/nodes/router.py`
- Create: `src/gaia_agent/nodes/executor.py`
- Create: `src/gaia_agent/nodes/verifier.py`
- Create: `src/gaia_agent/nodes/formatter.py`
- Create: `src/gaia_agent/graph.py`
- Create matching tests under `tests/nodes/`
- Create: `tests/test_graph_smoke.py`

**TDD target**

- Planner emits a compact plan structure
- Router chooses `S1` vs `S2` deterministically
- Executor records observations into state
- Verifier can approve or reject a draft
- Formatter strips final answers into grader-friendly shape
- Graph smoke test passes with mocked models and tools

**Done when**

- One fully mocked end-to-end graph run succeeds
- The graph can loop through verify-and-replan without corrupting state

### Task 5: Wire the graph into the app

**Intent:** Replace the stub agent without letting UI code absorb orchestration logic.

**Files**

- Modify: `app.py`
- Possibly add: `src/gaia_agent/agent.py` if the constructor logic becomes too large
- Add or expand smoke tests around app import and runner integration

**TDD target**

- App imports cleanly
- Agent instantiation is isolated from the UI
- Question execution uses the new runner and returns submission payloads in the existing format

**Done when**

- `app.py` still acts as the existing Gradio entrypoint
- The stub answer path is gone

### Task 6: Expand only after evidence

**Intent:** Use real failures, not guesswork, to decide what comes next.

**Candidates**

- Better file coverage for PDF/XLSX/DOCX
- Audio/YouTube support
- Stronger verifier prompts
- Better checkpoint inspection and retries
- Provider-specific hardening

**Rule**

- No new tool or abstraction gets added until the current slice has been run and its failure mode is understood

---

## Recommended Implementation Order

1. Task 1
2. Task 2
3. Task 4 with mocked tools and models as early as possible
4. Task 3 only to the degree needed by Task 4
5. Task 5
6. Real run, then Task 6 selectively

This order is deliberate: prove the graph contract early, then fill in the tool surface that the graph actually needs.

---

## Acceptance Criteria

- `BasicAgent` is replaced by a real GAIA agent entrypoint.
- The graph state is typed, serializable, and checkpointable.
- At least one end-to-end smoke test passes with mocked models.
- The app still fetches questions and submits answers through the existing UI flow.
- The implementation remains narrow enough that new failures point to real missing capability rather than uncontrolled complexity.

---

## Notes For Execution

- Use `subagent-driven-development` per task, not one giant implementation dispatch.
- Each implementation task must follow `test-driven-development`: write the test, watch it fail, then write the minimum code.
- Do not implement deferred tools preemptively.
- Do not let `app.py` become the agent framework.
