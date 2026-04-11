from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.json_utils import extract_json
from gaia_agent.llm_utils import extract_text
from gaia_agent.state import AgentState

log = logging.getLogger(__name__)

_MAX_OBS_CHARS = 2000

from gaia_agent.prompts import ORCHESTRATOR_SYSTEM, apply_caveman

def make_orchestrator_node(model, caveman: bool = False, caveman_mode: str = "full"):
    def orchestrator(state: AgentState) -> dict:
        # If we already have a draft answer, skip orchestration
        if state["draft_answer"]:
            log.info("[orchestrator] draft already exists, passing through")
            return {}

        # Build context with question, plan progress, and observations
        lines = [f"Question: {state['question']}"]
        if state["file_path"]:
            lines.append(f"Available File: {state['file_path']} ({state['modality']})")

        if state.get("modality") == "audio":
            lines.append("CRITICAL: This is an audio task. Prioritize 'audio' domain.")

        if state["task_chronicle"]:
            lines.append("")
            lines.append("Task Chronicle (High-Level Facts):")
            lines.append(state["task_chronicle"])
        lines.append("Plan progress:")
        for i, step in enumerate(state["plan"]):
            done = "DONE" if i < state["step_idx"] else "TODO"
            marker = ">>" if i == state["step_idx"] else "  "
            lines.append(f"{marker} {i}. [{done}] {step['description']}")

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

        if state["step_idx"] >= len(state["plan"]):
            lines.append("")
            lines.append("ALL PLAN STEPS ARE DONE. You MUST set has_answer=true and provide the draft_answer.")

        log.info("[orchestrator] evaluating after step %d", state["step_idx"])
        
        orch_prompt = apply_caveman(ORCHESTRATOR_SYSTEM, caveman, caveman_mode)
        
        response = model.invoke(
            [
                SystemMessage(content=orch_prompt),
                HumanMessage(content="\n".join(lines)),
            ]
        )

        raw = extract_text(response.content)
        log.info("[orchestrator] raw response (len=%d): %r", len(raw), raw[:500] if raw else '')
        try:
            payload = extract_json(raw)
        except Exception:
            log.warning("[orchestrator] failed to parse JSON, defaulting to general")
            payload = {"has_answer": False, "domain": "general", "strategy": "continue"}

        if not isinstance(payload, dict):
            log.warning("[orchestrator] payload is not a dict: %r", payload)
            payload = {"has_answer": False, "domain": "general", "strategy": "Analyze observations."}

        has_answer = payload.get("has_answer", False)
        domain = payload.get("domain", "general")
        strategy = payload.get("strategy", "")

        if has_answer and payload.get("draft_answer"):
            draft = str(payload["draft_answer"]).strip()
            log.info("[orchestrator] EARLY EXIT — draft_answer=%r", draft)
            return {
                "draft_answer": draft,
                "current_domain": domain,
                "current_strategy": strategy,
            }

        log.info("[orchestrator] domain=%s strategy=%s", domain, strategy)
        return {
            "current_domain": domain,
            "current_strategy": strategy,
        }

    return orchestrator
