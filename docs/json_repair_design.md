# JSON Repair Layer — Design Doc

**Status**: Draft · **Owner**: GAIA Agent · **Last updated**: 2026-04-11

## Problem

Phase 3 of the debugging journey claimed "zero JSON parsing failures" after switching to `parse_json_markdown`, but reality disagrees:

1. **Empty / refusal responses** still crash the pipeline — the model returns `""`, `[]`, or a plain-prose refusal ("I cannot help with that..."), and the downstream node blows up with `'NoneType' has no attribute 'get'` or a `KeyError`.
2. **Format errors** (missing commas, stray markdown fences, single quotes) are handled inconsistently — some nodes have local try/except, some don't, and the recovery path differs per node.
3. **No centralized telemetry**: we don't know which prompts/nodes are actually producing the most broken output, so prompt-level bugs get silently patched over instead of fixed at the source.

## Goals

- **Centralize** all JSON recovery into one reusable function that every LLM node can call.
- **Triage** failure modes before acting — treat empty/refusal differently from malformed JSON.
- **Fail-fast with signal** — unrecoverable failures should raise typed exceptions the state machine can route on, not silently return `None`.
- **Generate telemetry** so we can fix root-cause prompt bugs rather than papering over them forever.

## Non-goals

- Replacing `parse_json_markdown` — we keep it as the first-line parser.
- Making the fixer agent do any semantic reasoning. It only touches syntax.
- Handling streaming responses (all GAIA agent LLM calls are currently non-streaming).

## Design

### Layered recovery

Four layers, each only triggered when the previous one fails:

1. **Native structured output** (if the provider supports it): `with_structured_output(Schema)` / JSON mode. Most parse failures die here.
2. **`parse_json_markdown` + Pydantic validation**: handles single quotes, trailing commas, markdown fences, etc. — this is what we already do, plus schema validation on top.
3. **Triage**: classify the failure as Empty/Refusal vs Malformed.
4. **Fixer Agent** (cheap model): only invoked for Malformed. Strict "syntax only, don't change meaning" prompt. Retries with feedback; bounded budget.

### Triage rules

| Symptom | Classification | Action |
|---|---|---|
| Empty string or whitespace-only | **Empty** | Raise `EmptyResponseError` → state machine retries upstream node with "you produced nothing — rephrase" hint |
| No `{` and no `[` anywhere in text | **Empty** (prose refusal) | Same as above — **never** send to fixer (it will hallucinate fields) |
| Contains `{` / `[` but `json.loads` + Pydantic fails | **Malformed** | Send to Fixer Agent |
| LLM returns `[]` but schema expects `{"plan": ...}` | **Schema Mismatch (List)** | **Coerce**: wrap list in common field name (e.g. `plan`, `steps`, `result`) and re-validate locally |
| Fixer produces prose (no `{` / `[`) | **Malformed** (fixer degenerate) | Retry with error fed back; after budget → `UnsalvageableJsonError` |
| Fixer produces valid JSON but fails Pydantic | **Schema drift** | Retry with validation error fed back; after budget → `UnsalvageableJsonError` |

### Fixer Agent

- **Model tier**: cheap (Gemini Flash / Claude Haiku).
- **Prompt discipline**: system message separates instructions from data. Instruction is strict ("syntax only, don't change values, strip conversational text, return only the JSON block").
- **Schema passed as Pydantic `model_json_schema()`**, but trimmed (drop `$defs`/`title` if they bloat the prompt).
- **Retry with feedback**: each retry **must** include the previous failure's error message and previous bad output — otherwise the fixer gets identical input and produces identical failure.
- **Budget**: `max_local_repairs = 2`. After that, fail-fast.

### Extraction heuristic

`extract_raw_json_string(text)` — pulls the JSON substring out of prose wrappers:

- Find the **earliest** of `{` or `[` (not whichever `find` returns first — pick `min` of the non-`-1` indices). The current implementation has a bug here and always prefers `{`.
- Do **balanced-bracket extraction** from that index forward (depth counter on `{`/`}` and `[`/`]`, respecting string literals). Don't just return `text[start_idx:]` — you'll feed trailing prose into the parser.
- Return `None` if no opening bracket exists.

### First-attempt parser — **keep `parse_json_markdown`**

**Regression warning**: the current prototype replaces `parse_json_markdown` with naked `json.loads`. This is a downgrade — `parse_json_markdown` already handles single quotes, trailing commas, and markdown fences. Dropping it will push many previously-recoverable cases into the fixer path, burning cheap-model budget unnecessarily.

Order of operations inside `safe_structured_call`:

```
1. model.invoke(prompt) → raw_text
2. extract_raw_json_string(raw_text) → None?  → raise EmptyResponseError
3. try parse_json_markdown(raw_text) → dict
4. try target_schema.model_validate(dict) → T     ✅ done
5. on failure: invoke fixer (with retry+feedback loop)
6. on exhausted budget: raise UnsalvageableJsonError
```

### State machine integration

Two new typed exceptions:

- `EmptyResponseError` — state machine routes back to the **upstream node** (Planner/Executor/etc.) with an augmented system prompt: *"Your previous output was empty or a refusal. Rephrase and return strictly the requested JSON."*
- `UnsalvageableJsonError` — state machine treats as a hard failure for that step. Increments `json_repair_retries` in `AgentState`; if that exceeds `MAX_JSON_RETRIES` (default 2), force a strategy change (fail-fast to verifier, or try a different executor domain).

Add to `AgentState`:

```python
json_repair_retries: int   # incremented on every UnsalvageableJsonError
```

### Telemetry

On every fixer invocation (success OR failure), log a single JSONL entry to `logs/json_repairs.jsonl`:

```json
{
  "timestamp": "2026-04-11T12:34:56Z",
  "node": "planner",
  "schema": "PlanStepSchema",
  "error": "Expecting ',' delimiter: line 3 column 5",
  "original_text": "...",
  "fixed_text": "...",
  "outcome": "success|failure",
  "attempts": 1
}
```

**Telemetry must never crash the repair path**:
- `os.makedirs("logs", exist_ok=True)` at import time.
- Wrap the write in `try/except Exception` with `log.exception(...)` — a telemetry failure is a warning, never an exception that escapes to the caller.
- Concurrent writes: GAIA nodes run sequentially in LangGraph, so single-file append is safe. If we ever parallelize executors, switch to per-process files or a queue.

**Review cadence**: after every batch GAIA run, scan `logs/json_repairs.jsonl` and group by `(node, schema)`. The top offenders are prompt bugs — fix the prompt, don't let the fixer keep patching them.

## API sketch

```python
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class EmptyResponseError(Exception):
    """Model returned nothing parseable — no JSON brackets at all."""

class UnsalvageableJsonError(Exception):
    """Fixer exhausted its retry budget. Upstream must handle."""

def safe_structured_call(
    model,
    messages: list,              # list of BaseMessage — not a raw string
    target_schema: Type[T],
    cheap_fixer_model=None,
    max_local_repairs: int = 2,
    node_name: str = "unknown",  # for telemetry
) -> T:
    ...
```

Usage inside a node:

```python
try:
    plan = safe_structured_call(
        model=planner_model,
        messages=[SystemMessage(...), HumanMessage(...)],
        target_schema=PlanSchema,
        cheap_fixer_model=cheap_model,
        node_name="planner",
    )
except EmptyResponseError:
    # State machine: bounce back to upstream with rephrase hint
    state["critique"] = "Your last output was empty. Rephrase and return valid JSON."
    return state
except UnsalvageableJsonError:
    state["json_repair_retries"] += 1
    if state["json_repair_retries"] >= MAX_JSON_RETRIES:
        state["critique"] = "Repeated JSON failures — switching strategy."
        # route to verifier or different executor
    return state
```

## Known bugs in the current prototype (to fix before merging)

1. **Lost `parse_json_markdown`** — first-layer parser must be `parse_json_markdown`, not naked `json.loads`. (Regression vs. Phase 3.)
2. **`extract_raw_json_string` bracket selection** — currently always prefers `{` over `[`. Must use `min` of non-`-1` indices. Also needs balanced-bracket extraction, not `text[start_idx:]`.
3. **Fixer retry prompt is static** — the `for attempt in range(...)` loop uses the same prompt every iteration. Must append the previous error and previous bad output on each retry so the fixer has new information.
4. **Telemetry can crash the success path** — `open("logs/json_repairs.jsonl", "a")` throws `FileNotFoundError` if `logs/` doesn't exist. Add `os.makedirs(..., exist_ok=True)` AND wrap the write in `try/except`.
5. **Fixer output not re-triaged** — if the cheap model also returns prose, the current code only catches it via exception in `json.loads`. Should run `extract_raw_json_string` on the fixer's output first, same as the entry triage.

## Rollout plan

1. **Phase A** — Implement `safe_structured_call` + exceptions + telemetry. No node changes yet. Unit tests: empty, prose, malformed, fixer-success, fixer-fail, schema-drift.
2. **Phase B** — Wire into **Planner** only. Run a GAIA batch, read `logs/json_repairs.jsonl`, fix whichever prompts are offenders.
3. **Phase C** — Roll out to State Manager, Reflector, Verifier. Each node defines its own Pydantic schema.
4. **Phase D** — Add `json_repair_retries` to `AgentState` + state-machine routing for `EmptyResponseError` and `UnsalvageableJsonError`.
5. **Phase E** — After a week of telemetry, review prompt offenders and close out the top 3 at the source.

## Open questions → Resolved

| Question | Decision |
|---|---|
| Fixer model config | Use `cheap_model.bind(temperature=0)` at call time. Do NOT mutate the shared instance. |
| EmptyResponseError retry | Change the **prompt** (append a short rephrase hint), NOT temperature. Managing temperature state adds complexity; empty responses are prompt/filter failures, not temperature failures. |
| Global repair budget | Yes. Add `json_repair_retries: int` to `AgentState`. Cap at `cfg.max_json_repairs = 5`. **Must NOT be reset by Planner restarts.** |

See [`json_repair_impl_plan.md`](./json_repair_impl_plan.md) for the full phased implementation plan.
