from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.llm_utils import extract_text
from gaia_agent.prompts import REFLECTOR_SYSTEM
from gaia_agent.state import AgentState

log = logging.getLogger(__name__)


def make_reflector_node(model):
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
        response = model.invoke(
            [
                SystemMessage(content=REFLECTOR_SYSTEM),
                HumanMessage(content=context),
            ]
        )
        
        raw_content = extract_text(response.content)
        
        # Extract the updated working memory
        new_memory = state["working_memory"]
        if "UPDATED WORKING MEMORY:" in raw_content:
            parts = raw_content.split("UPDATED WORKING MEMORY:")
            new_memory = parts[1].split("MATCH FOUND:")[0].strip()
        
        # Check for Early Exit — but ONLY if we're at or past the last plan step.
        # If there are remaining plan steps (e.g., cross-verification), do NOT exit early.
        draft_answer = None
        remaining_steps = len(state["plan"]) - state["step_idx"]
        if "MATCH FOUND:" in raw_content and remaining_steps <= 1:
            match = re.search(r"MATCH FOUND:\s*(.*)", raw_content)
            if match:
                draft_answer = match.group(1).strip()
                log.info("[reflector] EARLY EXIT TRIGGERED: %s", draft_answer)
        elif "MATCH FOUND:" in raw_content:
            log.info("[reflector] MATCH FOUND detected but %d plan steps remain — suppressing early exit", remaining_steps)

        return {
            "working_memory": new_memory,
            "draft_answer": draft_answer or state["draft_answer"]
        }

    return reflector
