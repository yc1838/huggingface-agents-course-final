from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.prompts import VERIFIER_SYSTEM

MAX_RETRIES = 2
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    match = _JSON_BLOCK.search(text)
    if match is None:
        return {"decision": "APPROVED", "critique": None}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"decision": "APPROVED", "critique": None}


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
                lines.append(f"- [{observation['tool']}] {observation['result']}")

        response = model.invoke(
            [
                SystemMessage(content=VERIFIER_SYSTEM),
                HumanMessage(content="\n".join(lines)),
            ]
        )
        payload = _extract_json(response.content if isinstance(response.content, str) else str(response.content))
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
