from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.json_utils import extract_json
from gaia_agent.llm_utils import extract_text
from gaia_agent.state import AgentState

log = logging.getLogger(__name__)

_MAX_OBS_CHARS = 2000

from gaia_agent.prompts import STATE_MANAGER_SYSTEM, apply_caveman

def make_state_manager_node(model, caveman: bool = False, caveman_mode: str = "full"):
    def state_manager(state: AgentState) -> dict:
        # If we already have a draft answer, skip
        if state["draft_answer"]:
            log.info("[state_manager] draft already exists, passing through")
            return {}

        # Build context
        lines = [f"Question: {state['question']}"]
        if state["file_path"]:
            lines.append(f"Available File: {state['file_path']} ({state['modality']})")

        if state["task_chronicle"]:
            lines.append("")
            lines.append("Task Chronicle (High-Level Facts):")
            lines.append(state["task_chronicle"])

        if state["todo_list"]:
            lines.append("")
            lines.append("Dynamic Todo List:")
            for i, todo in enumerate(state["todo_list"]):
                lines.append(f"{i}. {todo}")

        if state["working_memory"]:
            lines.append("")
            lines.append("Working Memory (Synthesized Facts):")
            lines.append(state["working_memory"])
        elif state["observations"]:
            lines.append("")
            lines.append("Observations so far:")
            for obs in state["observations"]:
                r = obs["result"]
                if len(r) > _MAX_OBS_CHARS:
                    r = r[:_MAX_OBS_CHARS] + "...[truncated]"
                lines.append(f"- step {obs['step_idx']} [{obs['tool']}]: {r}")

        log.info("[state_manager] evaluating state...")
        
        sm_prompt = apply_caveman(STATE_MANAGER_SYSTEM, caveman, caveman_mode)
        
        response = model.invoke(
            [
                SystemMessage(content=sm_prompt),
                HumanMessage(content="\n".join(lines)),
            ]
        )

        raw = extract_text(response.content)
        log.info("[state_manager] raw response (len=%d): %r", len(raw), raw[:500] if raw else '')
        try:
            payload = extract_json(raw)
        except Exception:
            log.warning("[state_manager] failed to parse JSON, defaulting to general")
            payload = {"has_answer": False, "domain": "general", "strategy": "continue"}

        if not isinstance(payload, dict):
            payload = {"has_answer": False, "domain": "general", "strategy": "Analyze observations."}

        has_answer = payload.get("has_answer", False)
        domain = payload.get("domain", "general")
        strategy = payload.get("strategy", "")

        if has_answer and payload.get("draft_answer"):
            draft = str(payload["draft_answer"]).strip()
            log.info("[state_manager] EARLY EXIT — draft_answer=%r", draft)
            return {
                "draft_answer": draft,
                "current_domain": domain,
                "current_strategy": strategy,
            }

        log.info("[state_manager] domain=%s strategy=%s", domain, strategy)
        return {
            "current_domain": domain,
            "current_strategy": strategy,
        }

    return state_manager
