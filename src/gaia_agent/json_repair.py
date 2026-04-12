"""
Layered JSON extraction and repair for GAIA agent LLM nodes.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

from gaia_agent.llm_utils import extract_text

# Ensure logs directory exists for telemetry
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
    # Ensure we are working with a string
    if not isinstance(text, str):
        text = extract_text(text)
    
    if not text:
        return None
        
    # Find earliest opening bracket
    i_curly = text.find("{")
    i_square = text.find("[")
    
    candidates = []
    if i_curly != -1:
        candidates.append(i_curly)
    if i_square != -1:
        candidates.append(i_square)
        
    if not candidates:
        return None
        
    start = min(candidates)
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"

    # Balanced bracket walk respecting string literals
    depth = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        ch = text[i]
        
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
    # CRITICAL FIX: Always extract text to handle list-type contents
    raw_text = extract_text(getattr(response, "content", response))

    log.debug("[%s] trying to parse response (len=%d)", node_name, len(raw_text))

    # Step 2: triage — is there any JSON at all?
    raw_json = extract_raw_json_string(raw_text)
    if raw_json is None:
        log.warning("[%s] EmptyResponseError — no JSON brackets found", node_name)
        _log_repair(node_name, target_schema.__name__, raw_text, None, "empty", 0)
        raise EmptyResponseError(f"No JSON found in {node_name} response")

    # Step 3+4: try parse + validate
    first_error = ""
    try:
        parsed = parse_json_markdown(raw_json)
        if isinstance(parsed, dict):
            return target_schema.model_validate(parsed)
        elif isinstance(parsed, list):
            # Step 4a: Handle empty list - usually a soft failure sign
            if len(parsed) == 0:
                raise Exception("Model returned empty list []. Content is required.")

            # Try to validate directly; Pydantic handles list models too.
            try:
                return target_schema.model_validate(parsed)
            except Exception:
                # If direct validation fails, assume it's a list that needs wrapping.
                # Heuristic: wrap in 'plan', 'steps', or 'result' based on common GAIA schema fields.
                fields = list(target_schema.model_fields.keys())
                if len(fields) == 1:
                    return target_schema.model_validate({fields[0]: parsed})
                # Fallback to specific common field names if multi-field
                for common_field in ["plan", "steps", "result"]:
                    if common_field in fields:
                        return target_schema.model_validate({common_field: parsed})
                raise # Re-raise if no obvious wrapper field exists
    except Exception as e:
        first_error = str(e)
        log.warning("[%s] initial parse/validate failed: %s", node_name, first_error)

    # Step 5: Fixer loop
    if cheap_fixer_model is None:
        log.error("[%s] parse failed and no fixer available", node_name)
        raise UnsalvageableJsonError(f"[{node_name}] parse failed and no fixer available: {first_error}")

    fixer = cheap_fixer_model.bind(temperature=0)
    schema_str = json.dumps(target_schema.model_json_schema(), indent=2)
    bad_output = raw_json
    prev_error = first_error

    for attempt in range(max_local_repairs):
        log.info("[%s] invoking JSON fixer (attempt %d/%d)", node_name, attempt + 1, max_local_repairs)
        fixer_prompt = (
            f"You are a STRICT JSON syntax fixer. Your task is to fix syntax errors in the provided JSON string "
            f"so it strictly follows the target schema. Do NOT change values, facts, or reasoning.\n"
            f"CRITICAL: Return ONLY the valid JSON block. NO PROSE. NO EXPLANATIONS. NO REFUSALS.\n\n"
            f"Target schema:\n{schema_str}\n\n"
            f"Previous error: {prev_error}\n"
            f"Bad JSON:\n{bad_output}\n"
        )
        try:
            fix_response = fixer.invoke(fixer_prompt)
            fix_text = extract_text(getattr(fix_response, "content", fix_response))
            
            # GUARD: Detect if the fixer is "yapping" instead of fixing
            if any(indicator in fix_text.lower() for indicator in ["i cannot", "i apologize", "input should be", "validation error", "is not a valid", "expected a"]):
                log.warning("[%s] fixer returned prose/refusal instead of JSON. Skipping this attempt.", node_name)
                prev_error = "Fixer returned prose/explanation instead of raw JSON"
                continue

            fix_json = extract_raw_json_string(fix_text)
            
            if fix_json is None:
                prev_error = "Fixer produced no JSON brackets"
                bad_output = fix_text
                continue
                
            parsed = parse_json_markdown(fix_json)
            result = target_schema.model_validate(parsed)
            
            log.info("[%s] JSON repair successful on attempt %d", node_name, attempt + 1)
            _log_repair(node_name, target_schema.__name__, raw_text, fix_json, "success", attempt + 1)
            return result
        except Exception as e:
            prev_error = str(e)
            bad_output = fix_json if 'fix_json' in locals() and fix_json else fix_text
            log.warning("[%s] repair attempt %d failed: %s", node_name, attempt + 1, prev_error)

    # Step 6: Fail
    _log_repair(node_name, target_schema.__name__, raw_text, bad_output, "failure", max_local_repairs)
    raise UnsalvageableJsonError(f"[{node_name}] fixer exhausted after {max_local_repairs} attempts. Last error: {prev_error}")


def _log_repair(node, schema, original, fixed, outcome, attempts):
    """Write a single JSONL entry to logs/json_repairs.jsonl. Never crashes."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": node,
            "schema": schema,
            "original_text_snippet": (original or "")[:1000],
            "fixed_text_snippet": (fixed or "")[:1000],
            "outcome": outcome,
            "attempts": attempts,
        }
        with open("logs/json_repairs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        log.exception("[json_repair] telemetry write failed (non-fatal)")
