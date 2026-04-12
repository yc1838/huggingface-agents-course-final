from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.llm_utils import extract_text
from gaia_agent.prompts import REFLECTOR_SYSTEM, apply_caveman
from gaia_agent.state import AgentState

log = logging.getLogger(__name__)


from pydantic import BaseModel, Field
from typing import Optional
from gaia_agent.json_repair import EmptyResponseError, UnsalvageableJsonError, safe_structured_call

class ReflectorSchema(BaseModel):
    reasoning: str = Field(description="Your brief thought process.")
    updated_working_memory: str = Field(description="The complete, revised working memory.")
    chronicle_update: Optional[str] = Field(default=None, description="A single, concise sentence summarizing the NEW fact found.")
    match_found: Optional[str] = Field(default=None, description="The final answer string if the question is fully answered.")


def make_reflector_node(model, cheap_model=None, caveman: bool = False, caveman_mode: str = "full"):
    def reflector(state: AgentState) -> dict:
        # Check if we have observations to reflect on
        if not state["observations"]:
            return {}

        last_obs = state["observations"][-1]
        
        # Prepare context for the reflector
        context = f"Original Question: {state['question']}\n\n"
        context += f"Current Working Memory: {state['working_memory'] or 'Empty'}\n\n"
        context += f"Last Tool: {last_obs['tool']}\n"
        context += f"Last Tool Result:\n{last_obs['result'][:2000]}\n" # Cap at 2k chars

        log.info("[reflector] reflecting on step %d", state["step_idx"] - 1)
        
        reflector_prompt = apply_caveman(REFLECTOR_SYSTEM, caveman, caveman_mode)
        
        try:
            payload = safe_structured_call(
                model=model,
                messages=[
                    SystemMessage(content=reflector_prompt),
                    HumanMessage(content=context),
                ],
                target_schema=ReflectorSchema,
                cheap_fixer_model=cheap_model,
                node_name="reflector",
            )
            
            new_memory = payload.updated_working_memory
            
            # Extract Task Chronicle update
            new_chronicle = state["task_chronicle"]
            update = payload.chronicle_update
            if update and update.strip() and update not in new_chronicle:
                prefix = f" - Step {state['step_idx']}: "
                new_chronicle = (new_chronicle + "\n" + prefix + update.strip()).strip()
            
            # Check for Early Exit
            draft_answer = payload.match_found
            if draft_answer:
                log.info("[reflector] EARLY EXIT TRIGGERED: %s", draft_answer)

            return {
                "working_memory": new_memory,
                "task_chronicle": new_chronicle,
                "draft_answer": draft_answer or state["draft_answer"],
                "json_repair_retries": state["json_repair_retries"],
            }
            
        except (EmptyResponseError, UnsalvageableJsonError) as e:
            log.warning("[reflector] structured call failed: %s. Preserving state.", e)
            return {
                "json_repair_retries": state["json_repair_retries"] + 1,
            }

    return reflector

    return reflector
