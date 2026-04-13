from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.state import AgentState

log = logging.getLogger(__name__)

_MAX_OBS_CHARS = 2000

from typing import Literal, Optional
from pydantic import BaseModel, Field
from gaia_agent.json_repair import EmptyResponseError, UnsalvageableJsonError, safe_structured_call

class StateManagerSchema(BaseModel):
    has_answer: bool = Field(description="True if we have a definitive answer ready.")
    draft_answer: Optional[str] = Field(default=None, description="The draft answer if has_answer is True.")
    domain: Literal["math", "research", "vision", "audio", "file", "general"] = Field(default="general")
    strategy: str = Field(description="A brief description of the next step or strategy.")

from gaia_agent.prompts import STATE_MANAGER_SYSTEM, apply_caveman

def make_state_manager_node(model, cheap_model=None, caveman: bool = False, caveman_mode: str = "full"):
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
        
        try:
            payload = safe_structured_call(
                model=model,
                messages=[
                    SystemMessage(content=sm_prompt),
                    HumanMessage(content="\n".join(lines)),
                ],
                target_schema=StateManagerSchema,
                cheap_fixer_model=cheap_model,
                node_name="state_manager",
            )
            
            has_answer = payload.has_answer
            domain = payload.domain
            strategy = payload.strategy
            draft_answer = payload.draft_answer
            replan_count = state["replan_count"]

            # SAFETY CHECK: If no answer and no todos, we are stuck
            if not has_answer and not state["todo_list"]:
                replan_count += 1
                log.warning("[state_manager] no plan steps and no answer — forcing a re-plan (attempt %d/3)", replan_count)
                if replan_count > 3:
                    log.error("[state_manager] CRITICAL: Re-plan limit reached. Forcing failure state.")
                    has_answer = True
                    draft_answer = "AGENT ERROR: Task could not be planned after multiple attempts. Possible data retrieval failure or ambiguous request."
                else:
                    domain = "research"
                    strategy = (
                        f"All previous plan steps exhausted without a result. This is plan attempt #{replan_count + 1}. "
                        f"Previous Critique: {state['critique'] or 'None'}. "
                        "We MUST re-plan with more granular search steps or alternative sources. Do NOT repeat the failed plan."
                    )

            if has_answer and draft_answer:
                draft = str(draft_answer).strip()
                log.info("[state_manager] EARLY EXIT — draft_answer=%r", draft)
                return {
                    "draft_answer": draft,
                    "current_domain": domain,
                    "current_strategy": strategy,
                    "json_repair_retries": state["json_repair_retries"],
                    "replan_count": replan_count,
                }

            log.info("[state_manager] domain=%s strategy=%s", domain, strategy)
            return {
                "current_domain": domain,
                "current_strategy": strategy,
                "json_repair_retries": state["json_repair_retries"],
                "replan_count": replan_count,
            }
            
        except (EmptyResponseError, UnsalvageableJsonError) as e:
            log.warning("[state_manager] structured call failed: %s. Defaulting to general/continue", e)
            return {
                "current_domain": "general",
                "current_strategy": "Continue with current plan.",
                "json_repair_retries": state["json_repair_retries"] + 1,
            }

    return state_manager
