# GAIA Agent Code Review

**Date:** 2026-04-12
**Scope:** Full codebase review of the GAIA benchmark agent (~2,900 lines Python)

---

## Summary

The agent is well-built for the GAIA benchmark with thoughtful error handling, prompt engineering, and multi-provider support. 13 issues were identified and fixed across 4 categories.

---

## Issues Found & Fixes Applied

### Bugs

#### Issue 1: Duplicate return in reflector.py
- **File:** `src/gaia_agent/nodes/reflector.py`
- **Problem:** Line 85 had a second `return reflector` that was dead code after the first return at line 83.
- **Fix:** Removed the duplicate line. Also removed unused imports (`re`, `extract_text`).

#### Issue 2: Variable scope issue in vision.py
- **File:** `src/gaia_agent/tools/vision.py`
- **Problem:** `_call_vision()` referenced `b64_data`, `mime_type`, and `content` from the outer scope, but these weren't defined until later. Worked by accident since `_call_vision` was only called after the variables were set, but fragile.
- **Fix:** Added `b64_data`, `mime_type`, `content` as explicit parameters to `_call_vision()` and updated all 4 call sites.

#### Issue 3: `content` variable unused for fal provider
- **File:** `src/gaia_agent/tools/vision.py`
- **Problem:** The fal code path constructed its own `image_data_url` from `b64_data`/`mime_type` but the `content` list was only used by the Google path.
- **Fix:** Resolved as part of Issue 2 by making all dependencies explicit parameters.

#### Issue 4: `_norm` function redefined in a loop
- **File:** `scripts/dev_run_gaia.py`
- **Problem:** `_norm()` was defined inside the `for q in questions` loop, recreating it every iteration.
- **Fix:** Moved the definition before the loop.

### Design Concerns

#### Issue 5: verifier_decision routes to formatter with no answer on max retries
- **File:** `src/gaia_agent/nodes/verifier.py`
- **Problem:** When retries exceeded MAX_RETRIES (6), both `draft_answer` and `final_answer` were None (set to None by the last rejection). The formatter would produce an empty string, submitting a blank answer.
- **Fix:** In the REJECTED branch, when `state["retries"] + 1 > MAX_RETRIES`, the last `draft_answer` is preserved instead of being cleared. Applied to both the normal rejection path and the exception handler. A rejected answer is still better than no answer for GAIA exact-match scoring.

#### Issue 6: Redundant re-initialization in app.py
- **File:** `app.py`
- **Problem:** Lines 39-42 re-created `self.strong` when `GAIA_STRONG_PROVIDER == "lmstudio"`, but `self.strong` was already set identically at line 36. This was a no-op.
- **Fix:** Removed the redundant block.

#### Issue 7: `orchestrator_model` parameter unused in build_graph()
- **File:** `src/gaia_agent/graph.py`, `app.py`
- **Problem:** `build_graph()` accepted `orchestrator_model` but never used it. The state manager uses `cheap_model` directly.
- **Fix:** Removed the parameter from the signature and the call site. Updated the comment in `router.py` ("Orchestrator" -> "State Manager").

#### Issue 8: Multiple `import dataclasses` in dev_run_gaia.py
- **File:** `scripts/dev_run_gaia.py`
- **Problem:** `import dataclasses` appeared 3 times inline in different branches.
- **Fix:** Consolidated to a single import at the top of the file.

#### Issue 9: Prompt contradiction between SCRAPING PROHIBITION and NETWORK PATIENCE RULE
- **File:** `src/gaia_agent/prompts.py`
- **Problem:** BASE_EXECUTOR forbids `requests`/`httpx` in `run_python`, but RESEARCH_SPECIALIST and GENERAL_EXECUTOR told the executor to set `timeout=60` for web requests in `run_python` -- appearing contradictory.
- **Clarification:** The NETWORK PATIENCE RULE is about academic API calls (Wayback CDX, Arxiv OAI-PMH, CrossRef REST) that are an allowed exception to the scraping prohibition. These also need `User-Agent` headers to avoid being blocked.
- **Fix:** Reworded the NETWORK PATIENCE RULE in both specialists to explicitly frame it as an "API EXCEPTION" requiring both `timeout=60` and a `User-Agent` header.

#### Issue 10: No test coverage for reflector and state_manager nodes
- **Files:** `tests/nodes/`
- **Problem:** Tests existed for executor, formatter, perception, planner, router, and verifier, but not for reflector or state_manager.
- **Fix:** Created `test_reflector.py` (5 tests) and `test_state_manager.py` (5 tests) following the existing mock-model pattern.

### Cleanup

#### Issue 11: Log files in root directory
- **File:** `.gitignore`
- **Problem:** `.gitignore` had `gaia*.log` which only matched GAIA-specific logs.
- **Fix:** Broadened to `*.log` for generic coverage.

#### Issue 12: Dead orchestrator.py and ORCHESTRATOR_SYSTEM alias
- **Files:** `src/gaia_agent/nodes/orchestrator.py`, `src/gaia_agent/prompts.py`, `src/gaia_agent/nodes/state_manager.py`
- **Problem:** `orchestrator.py` was never imported by `graph.py` or any live module -- fully superseded by `state_manager.py`. The `ORCHESTRATOR_SYSTEM` alias and an unused `extract_json` import were its only remaining references.
- **Fix:** Deleted `orchestrator.py`, removed the `ORCHESTRATOR_SYSTEM` alias, and removed the unused imports in `state_manager.py`.

#### Issue 13: json_utils.py with zero live consumers
- **File:** `src/gaia_agent/json_utils.py`
- **Problem:** After removing `orchestrator.py` (Issue 12), `json_utils.extract_json()` had zero consumers. `json_repair.py` provides the actively-used `extract_raw_json_string()` and `safe_structured_call()`.
- **Fix:** Deleted `json_utils.py`.

---

## Files Modified

| File | Action | Issues |
|------|--------|--------|
| `src/gaia_agent/nodes/orchestrator.py` | DELETED | 12 |
| `src/gaia_agent/json_utils.py` | DELETED | 13 |
| `src/gaia_agent/prompts.py` | Edited | 9, 12 |
| `src/gaia_agent/nodes/state_manager.py` | Edited | 12 |
| `src/gaia_agent/graph.py` | Edited | 7 |
| `app.py` | Edited | 6, 7 |
| `src/gaia_agent/nodes/reflector.py` | Edited | 1 |
| `src/gaia_agent/nodes/router.py` | Edited | 7 |
| `src/gaia_agent/tools/vision.py` | Edited | 2, 3 |
| `src/gaia_agent/nodes/verifier.py` | Edited | 5 |
| `scripts/dev_run_gaia.py` | Edited | 4, 8 |
| `.gitignore` | Edited | 11 |
| `tests/nodes/test_reflector.py` | CREATED | 10 |
| `tests/nodes/test_state_manager.py` | CREATED | 10 |
| `tests/nodes/test_verifier.py` | Edited | 5 |

## Verification Results

- All 15 new/modified tests pass
- 75 existing tests pass (27 pre-existing failures unrelated to these changes)
- Zero stale references to `orchestrator` or `json_utils` in `src/`
- Import smoke test passes: `from gaia_agent.graph import build_graph`

## Pre-existing Issues Not Addressed

- `test_formatter.py` imports a bare `formatter` function that doesn't exist (should use `make_formatter_node`)
- `test_verifier_approves_draft` sends `"critique": None` but `VerifierSchema.critique` requires a `str`
- Multiple test files construct `Config()` without the newer required fields (`extra_strong_provider`, `vision_provider`, etc.)
