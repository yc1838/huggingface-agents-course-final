from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.llm_utils import extract_text
from gaia_agent.prompts import VERIFIER_SYSTEM, apply_caveman

log = logging.getLogger(__name__)
MAX_RETRIES = 6
_MAX_OBS_CHARS = 8000


from pydantic import BaseModel, Field
from typing import Literal
from gaia_agent.json_repair import EmptyResponseError, UnsalvageableJsonError, safe_structured_call

class VerifierSchema(BaseModel):
    decision: Literal["APPROVED", "REJECTED"] = Field(description="APPROVED if the draft answer is correct and complete, REJECTED otherwise.")
    critique: str = Field(description="Detailed explanation of why the answer was rejected, or why it was approved.")


def make_verifier_node(model, cheap_model=None, caveman: bool = False, caveman_mode: str = "full"):
    def verifier(state) -> dict:
        lines = [
            f"Question: {state['question']}",
            f"Draft answer: {state['draft_answer']}",
            "",
            "Plan:",
        ]
        for step in state["plan"]:
            lines.append(f"- [{step['tier']}] {step['description']}")
        if state["observations"]:
            lines.append("")
            lines.append("Observations:")
            for observation in state["observations"]:
                result = observation["result"]
                if len(result) > _MAX_OBS_CHARS:
                    result = result[:_MAX_OBS_CHARS] + "...[truncated]"
                lines.append(f"- [{observation['tool']}] {result}")

        log.info("[verifier] draft_answer=%r  retries=%d", state["draft_answer"], state["retries"])
        
        verifier_prompt = apply_caveman(VERIFIER_SYSTEM, caveman, caveman_mode)
        
        try:
            payload = safe_structured_call(
                model=model,
                messages=[
                    SystemMessage(content=verifier_prompt),
                    HumanMessage(content="\n".join(lines)),
                ],
                target_schema=VerifierSchema,
                cheap_fixer_model=cheap_model,
                node_name="verifier",
            )
            
            decision = payload.decision
            critique = payload.critique

            log.info("[verifier] decision=%s  critique=%s", decision, critique)
            if decision == "APPROVED":
                return {
                    "final_answer": state["draft_answer"], 
                    "critique": None,
                    "json_repair_retries": state["json_repair_retries"],
                }

            next_retries = state["retries"] + 1
            preserve_draft = next_retries > MAX_RETRIES
            return {
                "critique": critique or "Answer rejected without explanation.",
                "draft_answer": state["draft_answer"] if preserve_draft else None,
                "final_answer": None,
                "retries": next_retries,
                "json_repair_retries": state["json_repair_retries"],
            }

        except (EmptyResponseError, UnsalvageableJsonError) as e:
            log.warning("[verifier] structured call failed: %s. Defaulting to REJECTED", e)
            next_retries = state["retries"] + 1
            preserve_draft = next_retries > MAX_RETRIES
            return {
                "critique": f"Verifier crash: {str(e)}",
                "draft_answer": state["draft_answer"] if preserve_draft else None,
                "final_answer": None,
                "retries": next_retries,
                "json_repair_retries": state["json_repair_retries"] + 1,
            }

    return verifier


def verifier_decision(state) -> str:
    if state["final_answer"]:
        return "formatter"
    if state["retries"] > MAX_RETRIES:
        return "formatter"
    return "planner"
