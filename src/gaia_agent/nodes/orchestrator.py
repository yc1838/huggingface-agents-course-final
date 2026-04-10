from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.json_utils import extract_json
from gaia_agent.llm_utils import extract_text
from gaia_agent.prompts import ORCHESTRATOR_SYSTEM
from gaia_agent.state import AgentState

log = logging.getLogger(__name__)


def make_orchestrator_node(model):
    def orchestrator(state: AgentState) -> dict:
        # If we already have a draft answer, skip orchestration
        if state["draft_answer"] or state["step_idx"] >= len(state["plan"]):
            return {}

        current_step = state["plan"][state["step_idx"]]
        
        context = f"Question: {state['question']}\n\n"
        context += f"CURRENT STEP: {current_step['description']}\n"
        if state["file_path"]:
            context += f"Available File: {state['file_path']} ({state['modality']})\n"
        
        # Explicit priority check for audio modality
        if state.get("modality") == "audio":
            context += "CRITICAL: This is an audio task. Prioritize 'audio' domain.\n"
        
        if state["observations"]:
            last_obs = state["observations"][-1]
            context += f"Last Tool Used: {last_obs['tool']}\n"
            context += f"Last Result: {last_obs['result'][:200]}...\n"

        log.info("[orchestrator] classifying step %d", state["step_idx"])
        response = model.invoke(
            [
                SystemMessage(content=ORCHESTRATOR_SYSTEM),
                HumanMessage(content=context),
            ]
        )
        
        raw = extract_text(response.content)
        payload = extract_json(raw)
        
        domain = payload.get("domain", "general")
        strategy = payload.get("strategy", "")
        
        log.info("[orchestrator] domain=%s strategy=%s", domain, strategy)
        
        return {
            "current_domain": domain,
            "current_strategy": strategy,
        }

    return orchestrator
