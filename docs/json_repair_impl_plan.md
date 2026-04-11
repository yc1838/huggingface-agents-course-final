# JSON Repair Layer — Implementation Plan

**Status**: Approved · **Owner**: GAIA Agent · **Last updated**: 2026-04-11

> This document records the finalized design decisions from the `json_repair_design.md` discussion and provides a concrete, ordered implementation plan. See the design doc for problem statement, goals, and architecture rationale.

---

## Finalized Design Decisions

### Q1: Fixer Model Configuration
**Decision**: Use the `cheap_model` (e.g. Gemini Flash) as the fixer, but instantiate it with `temperature=0` **at call time** — do NOT mutate the shared `cheap_model` instance.

```python
# Correct pattern — bind a stricter config without modifying the original model
fixer_model = cheap_model.bind(temperature=0)
```

**Rationale**: Fixer's job is pure syntax correction; zero creativity needed. Sharing the model object avoids a new network connection / model init cost.

---

### Q2: EmptyResponseError Retry Strategy
**Decision**: On retry, **change the prompt, not the temperature**.

The two-line retry hint:
```
"Your previous output was empty or a refusal. Return ONLY a valid JSON object matching the schema. No other text."
```

**Rationale**:
- Empty responses are almost always caused by a prompt misunderstanding or content filter — not by temperature being too low.
- Raising temperature introduces randomness that degrades subsequent token quality.
- Managing temperature state (raising and "putting it back") adds state complexity with no measurable benefit.
- A targeted prompt injection is free to produce and trivially reversible.

---

### Q3: Global Repair Budget
**Decision**: Track `json_repair_retries: int` in `AgentState`. Cap at `max_json_repairs` (default `5`, configurable via `Config`).

**Critical rule**: The Planner node **MUST NOT reset** `json_repair_retries` when it restarts. This field accumulates globally across the whole task lifetime.

Other fields that **are** reset by Planner on a new attempt:
- `observations` (only if no prior critique)
- `working_memory` (only if no prior critique)
- `draft_answer` → `None`
- `critique` → `None`

Fields that **must survive** Planner restarts:
- `json_repair_retries` ← **NEW**
- `task_chronicle`

---

## Implementation Plan (Phase A → D)

### Phase A — Core Utilities (no node changes yet)

**Files to create/modify:**

#### `src/gaia_agent/json_repair.py` (NEW)
```python
"""
Layered JSON extraction and repair for GAIA agent LLM nodes.
"""
import json
import logging
import os
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)

class EmptyResponseError(Exception):
    """Model returned nothing parseable — no JSON brackets at all."""

class UnsalvageableJsonError(Exception):
    """Fixer exhausted its retry budget. Upstream must handle."""


def extract_raw_json_string(text: str) -> str | None:
    """
    Extract the outermost balanced JSON structure from prose.

    Uses min(first_curly, first_bracket) as the entry point,
    then performs a balanced-bracket walk to find the matching close.
    Returns None if no opening bracket exists.
    """
    # Find earliest opening bracket
    i_curly = text.find("{")
    i_square = text.find("[")
    candidates = [i for i in (i_curly, i_square) if i != -1]
    if not candidates:
        return None
    start = min(candidates)
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"

    # Balanced bracket walk respecting string literals
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # Unbalanced


def safe_structured_call(
    model,
    messages: list,
    target_schema: Type[T],
    cheap_fixer_model=None,
    max_local_repairs: int = 2,
    node_name: str = "unknown",
) -> T:
    """
    Invoke a model and return a validated Pydantic object.

    Pipeline:
      1. Invoke model → raw text
      2. extract_raw_json_string → None → EmptyResponseError
      3. parse_json_markdown → JSON dict
      4. target_schema.model_validate → T  ✅
      5. On failure → invoke fixer (retry with error feedback)
      6. On budget exhausted → UnsalvageableJsonError
    """
    from langchain_core.utils.json import parse_json_markdown

    response = model.invoke(messages)
    raw_text = getattr(response, "content", str(response))
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    # Step 2: triage — is there any JSON at all?
    raw_json = extract_raw_json_string(raw_text)
    if raw_json is None:
        log.warning("[%s] EmptyResponseError — no JSON brackets found", node_name)
        _log_repair(node_name, target_schema.__name__, raw_text, None, "empty", 0)
        raise EmptyResponseError(f"No JSON found in {node_name} response")

    # Step 3+4: try parse + validate
    try:
        parsed = parse_json_markdown(raw_json)
        if parsed and isinstance(parsed, dict):
            return target_schema.model_validate(parsed)
        elif parsed and isinstance(parsed, list):
            # Wrap if schema expects a container
            return target_schema.model_validate({"items": parsed})
    except Exception as e:
        first_error = str(e)

    # Step 5: Fixer loop
    if cheap_fixer_model is None:
        raise UnsalvageableJsonError(f"[{node_name}] no fixer model provided")

    fixer = cheap_fixer_model.bind(temperature=0)
    schema_str = json.dumps(target_schema.model_json_schema(), indent=2)
    bad_output = raw_json
    prev_error = first_error

    for attempt in range(max_local_repairs):
        fixer_prompt = (
            f"You are a JSON syntax fixer. Fix ONLY syntax errors. Do NOT change values.\n"
            f"Return ONLY the valid JSON. No other text.\n\n"
            f"Target schema:\n{schema_str}\n\n"
            f"Previous error: {prev_error}\n"
            f"Bad JSON:\n{bad_output}\n"
        )
        try:
            fix_response = fixer.invoke(fixer_prompt)
            fix_text = getattr(fix_response, "content", str(fix_response))
            fix_json = extract_raw_json_string(fix_text)
            if fix_json is None:
                prev_error = "Fixer produced no JSON brackets"
                bad_output = fix_text
                continue
            parsed = parse_json_markdown(fix_json)
            result = target_schema.model_validate(parsed)
            _log_repair(node_name, target_schema.__name__, raw_text, fix_json, "success", attempt + 1)
            return result
        except Exception as e:
            prev_error = str(e)
            bad_output = fix_json or fix_text

    _log_repair(node_name, target_schema.__name__, raw_text, bad_output, "failure", max_local_repairs)
    raise UnsalvageableJsonError(f"[{node_name}] fixer exhausted after {max_local_repairs} attempts")


def _log_repair(node, schema, original, fixed, outcome, attempts):
    """Write a single JSONL entry to logs/json_repairs.jsonl. Never crashes."""
    import json as _json
    from datetime import datetime, timezone
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": node,
            "schema": schema,
            "original_text": (original or "")[:500],
            "fixed_text": (fixed or "")[:500],
            "outcome": outcome,
            "attempts": attempts,
        }
        with open("logs/json_repairs.jsonl", "a") as f:
            f.write(_json.dumps(entry) + "\n")
    except Exception:
        log.exception("[json_repair] telemetry write failed (non-fatal)")
```

#### `src/gaia_agent/state.py` — Add `json_repair_retries`
```python
class AgentState(TypedDict):
    # ... existing fields ...
    json_repair_retries: int   # global across task; NOT reset by Planner
```

#### `src/gaia_agent/state.py` — Update `new_state`
```python
def new_state(...) -> AgentState:
    return AgentState(
        # ... existing fields ...
        json_repair_retries=0,
    )
```

---

### Phase B — Wire into Planner

**File**: `src/gaia_agent/nodes/planner.py`

Replace the raw `extract_json(raw)` call with `safe_structured_call`. Planner is the highest-value target because its `[]` response is the crash we saw in production.

```python
from gaia_agent.json_repair import EmptyResponseError, UnsalvageableJsonError, safe_structured_call
from pydantic import BaseModel

class PlanStepSchema(BaseModel):
    thought: str
    description: str
    tier: str

class PlanSchema(BaseModel):
    plan: list[PlanStepSchema]


# Inside make_planner_node, before the return:
try:
    plan_obj = safe_structured_call(
        model=model,
        messages=[SystemMessage(content=planner_prompt), HumanMessage(content="\n".join(human_lines))],
        target_schema=PlanSchema,
        cheap_fixer_model=cheap_model,  # needs to be passed into make_planner_node
        node_name="planner",
    )
    plan = [s.model_dump() for s in plan_obj.plan]
except EmptyResponseError:
    log.warning("[planner] empty response — returning empty plan")
    plan = []
except UnsalvageableJsonError as e:
    log.error("[planner] unsalvageable: %s", e)
    plan = []
    return {
        "plan": [],
        "step_idx": 0,
        "todo_list": [],
        "json_repair_retries": state["json_repair_retries"] + 1,
        "draft_answer": None,
        "critique": None,
    }
```

> **Note**: `make_planner_node` signature needs `cheap_model` added, and `build_graph` needs to pass it through.

---

### Phase C — State Manager, Reflector, Verifier

Each node defines its own Pydantic schema and migrates to `safe_structured_call`. Order of priority:

1. `state_manager` (StateManagerSchema: `has_answer`, `draft_answer`, `domain`, `strategy`)
2. `verifier` (VerifierSchema: `decision`, `critique`)
3. `reflector` (unstructured output — lower priority, may not need Pydantic)

---

### Phase D — State Machine Routing

Add `EmptyResponseError` + `UnsalvageableJsonError` routing in `graph.py`:
- `EmptyResponseError` in planner → bounce right back to planner with augmented critique
- `UnsalvageableJsonError` + `json_repair_retries >= cfg.max_json_repairs` → force route to verifier

---

## Unit Tests (Phase A)

File: `tests/test_json_repair.py`

| Test | Input | Expected |
|---|---|---|
| `test_happy_path` | Valid JSON wrapped in prose | Returns validated schema |
| `test_markdown_fenced` | ` ```json { } ``` ` | Returns validated schema |
| `test_empty_string` | `""` | Raises `EmptyResponseError` |
| `test_prose_refusal` | `"I cannot help with that."` | Raises `EmptyResponseError` |
| `test_malformed_fixable` | `"{'key': 'val'}"` | Fixer succeeds, returns schema |
| `test_fixer_fails` | Garbage that fixer can't fix | Raises `UnsalvageableJsonError` after N retries |
| `test_array_top_level` | `[{"thought": ..., "description": ...}]` | Wrapped and validated |
| `test_balanced_bracket` | `"prefix {json} suffix {noise}"` | Extracts first balanced block only |
| `test_global_budget` | Simulated; `json_repair_retries` not reset by planner | Counter preserved across calls |

---

## Config

Add to `Config`:

```python
max_json_repairs: int = 5   # global per-task cap across all nodes
```

---

## Review Cadence

After every GAIA batch run, check:

```bash
cat logs/json_repairs.jsonl | python3 -c "
import sys, json, collections
rows = [json.loads(l) for l in sys.stdin]
for (node, schema), count in collections.Counter((r['node'], r['schema']) for r in rows).most_common(10):
    print(f'{count:3d}  {node}/{schema}')
"
```

Top offenders = prompt bugs. Fix the prompt, don't keep running the fixer.
