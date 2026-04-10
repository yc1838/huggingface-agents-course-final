from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.json_utils import extract_json
from gaia_agent.llm_utils import extract_text
from gaia_agent.prompts import VERIFIER_SYSTEM

log = logging.getLogger(__name__)
MAX_RETRIES = 5
_MAX_OBS_CHARS = 8000


def make_verifier_node(model):
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
        response = model.invoke(
            [
                SystemMessage(content=VERIFIER_SYSTEM),
                HumanMessage(content="\n".join(lines)),
            ]
        )
        raw = extract_text(response.content)
        log.debug("[verifier] raw response:\n%s", raw)
        try:
            payload = extract_json(raw)
        except Exception:
            log.warning("[verifier] failed to parse JSON, rejecting draft")
            payload = {"decision": "REJECTED", "critique": "Verifier response was not valid JSON; re-plan."}
        log.info("[verifier] decision=%s  critique=%s", payload.get("decision"), payload.get("critique"))
        if payload.get("decision") == "APPROVED":
            return {"final_answer": state["draft_answer"], "critique": None}

        return {
            "critique": payload.get("critique") or "Answer rejected without explanation.",
            "draft_answer": None,
            "final_answer": None,
            "retries": state["retries"] + 1,
        }

    return verifier


def verifier_decision(state) -> str:
    if state["final_answer"]:
        return "formatter"
    if state["retries"] > MAX_RETRIES:
        return "formatter"
    return "planner"
