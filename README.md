title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480



# GAIA Agent

An AI agent that tackles the [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) — a suite of real-world, multi-step information-gathering tasks. The agent uses a **Plan → Execute → Verify** loop with two-tier model execution and runs either as a CLI tool or as a Hugging Face Space.


## Remove checkpoint
`rm -rf .checkpoints/*`
test run: `python scripts/dev_run_gaia.py --limit 1 --level 1`

## Architecture

```
Question + File
      │
  Perception        ← detects file modality, downloads attachments
      │
  Planner           ← strong model: produces 3–5 step JSON plan
      │
  Router            ← rule-based: dispatches each step by tier
   ┌──┴──┐
  S1    S2          ← cheap model (S1) or strong model (S2)
   └──┬──┘
  (tool calls, observations)
      │
  (plan complete?)
      │
  Verifier          ← strong model: APPROVED → Formatter, REJECTED → Planner
      │
  Formatter         ← normalizes answer for exact-match grading
      │
  Answer
```

| Node | Role |
|------|------|
| **Perception** | Detects modality (text, PDF, audio, image, YouTube, web); downloads files |
| **Planner** | Calls strong model once; outputs `{plan: [{description, tier}]}` |
| **Router** | Reads `tier` field; routes S1 (cheap) or S2 (strong); detects plan completion |
| **Executor S1/S2** | Calls tools, records observations, emits draft answer |
| **Verifier** | Approves or rejects; rejected answers loop back to Planner (max 2 retries) |
| **Formatter** | Strips prose prefixes and trailing punctuation for GAIA exact-match scoring |

### Tools

| Tool | Purpose |
|------|---------|
| `tavily_search` | Web search via Tavily API |
| `fetch_url` | Fetch and extract main content from a URL |
| `run_python` | Execute Python in a sandboxed subprocess |
| `read_file` | Read txt, md, csv, xlsx, pdf, docx, json |
| `transcribe_audio` | Transcribe audio with faster-whisper |
| `youtube_transcript` | Fetch YouTube video transcripts |

## Requirements

- Python 3.11+
- API keys for your chosen LLM providers (see [Configuration](#configuration))

## Installation

```bash
git clone <repo-url>
cd huggingface-agents-final

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -e .
pip install -e ".[test]"        # include test dependencies

cp .env.example .env
# edit .env with your API keys and model choices
```

## Running

### CLI (local testing)

```bash
# Run 3 Level-1 questions from the GAIA validation split
python scripts/dev_run_gaia.py --limit 3 --level 1

# Run a single question by task ID
python scripts/dev_run_gaia.py --task-id c61d22de-5f6c-4958-a7f6-5e9707bd3466

# Verbose logging
python scripts/dev_run_gaia.py --limit 5 --verbose
```

| Flag | Default | Description |
|------|---------|-------------|
| `--limit N` | 3 | Number of questions to run |
| `--level 1\|2\|3` | 1 | Difficulty level filter |
| `--split` | validation | `validation` or `test` |
| `--config` | 2023_all | Dataset config (e.g. `2024`) |
| `--task-id UUID` | — | Run a single question; ignores `--limit`/`--level` |
| `--verbose` | off | Enable debug logging |

### Gradio web UI

```bash
python app.py
# Opens at http://localhost:7860
```

1. Click **Login with Hugging Face**
2. Click **Run Evaluation & Submit All Answers**
3. Watch the agent work; results appear in the table

The app auto-detects `SPACE_ID`/`SPACE_HOST` when deployed as a Hugging Face Space and submits answers to the official GAIA scoring endpoint.

## Configuration

Copy `.env.example` to `.env` and set values:

```bash
# GAIA endpoint and checkpoints
GAIA_API_URL=https://agents-course-unit4-scoring.hf.space
GAIA_CHECKPOINT_DIR=.checkpoints

# Cheap model (S1 — routine tool calls)
GAIA_CHEAP_PROVIDER=ollama          # ollama | anthropic | google | huggingface | lmstudio | openai
GAIA_CHEAP_MODEL=gemma3:4b

# Strong model (S2 — planning, reasoning, verification)
GAIA_STRONG_PROVIDER=anthropic
GAIA_STRONG_MODEL=claude-sonnet-4-6

# API keys (omit keys for providers you don't use)
GAIA_ANTHROPIC_API_KEY=sk-...
GAIA_GOOGLE_API_KEY=...
GAIA_HUGGINGFACE_API_KEY=hf_...
GAIA_TAVILY_API_KEY=tvly-...

# Optional
GAIA_WHISPER_MODEL=base             # tiny | base | small | medium | large
GAIA_LMSTUDIO_BASE_URL=http://localhost:1234/v1

# Hugging Face Space (auto-set by HF runtime)
SPACE_ID=
SPACE_HOST=
HF_TOKEN=
```

**Provider pairings that work well:**

| Use case | Cheap (S1) | Strong (S2) |
|----------|------------|-------------|
| Fully local | `ollama/gemma3:4b` | `ollama/qwen3:14b` |
| Low cost | `google/gemini-1.5-flash` | `anthropic/claude-sonnet-4-6` |
| Best accuracy | `anthropic/claude-haiku-4-5` | `anthropic/claude-opus-4-6` |

## Testing

```bash
pytest                      # all tests
pytest tests/nodes/         # node unit tests
pytest tests/tools/         # tool unit tests
pytest -x                   # stop on first failure
pytest -v                   # verbose output
```

## Project Structure

```
src/gaia_agent/
├── config.py           # Config dataclass — loads all env vars
├── state.py            # AgentState TypedDict + supporting types
├── models.py           # LLM provider factory (Anthropic, OpenAI, Google, Ollama, HF, LM Studio)
├── prompts.py          # System prompts for Planner, Executor, Verifier
├── graph.py            # LangGraph definition (nodes + edges)
├── runner.py           # run_agent_on_questions() — checkpointing + submission
├── api_client.py       # GAIA scoring endpoint client
├── gaia_dataset.py     # HuggingFace dataset loader (mirrors api_client surface)
├── nodes/
│   ├── perception.py
│   ├── planner.py
│   ├── router.py
│   ├── executor.py
│   └── verifier.py
│   └── formatter.py
└── tools/
    ├── __init__.py     # build_tools() registry
    ├── search.py
    ├── web.py
    ├── python_exec.py
    ├── files.py
    ├── audio.py
    └── youtube.py
```

## Key Design Decisions

- **One Planner call per question** — the strong model plans once; the Executor loops within that plan, keeping costs predictable.
- **Rule-based Router** — reads the `tier` field from the plan; no LLM call, no latency, no hallucinated routing.
- **Verifier loops to Planner, not Executor** — a bad answer usually means a bad plan; re-planning is more effective than re-executing the same steps.
- **Separate Formatter node** — GAIA uses exact-match grading; normalizing the answer format provides a measurable score boost.
- **Per-question JSON checkpoints** — the run is resumable after interruptions, quota limits, or crashes.
- **Providers via config, not code** — swap `GAIA_CHEAP_PROVIDER`/`GAIA_STRONG_PROVIDER` at runtime with no code changes.

## Adding a Tool

1. Create `src/gaia_agent/tools/my_tool.py` with a `@tool`-decorated function.
2. Import it in `src/gaia_agent/tools/__init__.py` and add it to the list returned by `build_tools()`.
3. The Executor picks it up automatically via `llm.bind_tools()`.

## Deployment on Hugging Face Spaces

Set the following Secrets in your Space settings:

```
GAIA_STRONG_PROVIDER, GAIA_STRONG_MODEL, GAIA_CHEAP_PROVIDER, GAIA_CHEAP_MODEL
GAIA_ANTHROPIC_API_KEY (or equivalent for your provider)
GAIA_TAVILY_API_KEY
HF_TOKEN
```

`SPACE_ID` and `SPACE_HOST` are injected automatically by the HF runtime.
