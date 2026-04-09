from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.prompts import PLANNER_SYSTEM
from gaia_agent.state import AgentState, PlanStep

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    match = _JSON_BLOCK.search(text)
    if match is None:
        raise ValueError(f"No JSON object found in planner response: {text[:200]}")
    return json.loads(match.group(0))


def make_planner_node(model):
    def planner(state: AgentState) -> dict:
        human_lines = [f"Question: {state['question']}"]
        if state["file_path"]:
            human_lines.append(f"Associated file: {state['file_path']} ({state['modality']})")
        if state["critique"]:
            human_lines.append(f"Prior critique: {state['critique']}")

        response = model.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content="\n".join(human_lines)),
            ]
        )
        payload = _extract_json(response.content if isinstance(response.content, str) else str(response.content))
        plan: list[PlanStep] = [
            {
                "description": step["description"],
                "tier": step.get("tier", "S1"),
            }
            for step in payload.get("plan", [])
        ]
        return {
            "plan": plan,
            "step_idx": 0,
            "observations": [],
            "draft_answer": None,
            "critique": None,
        }

    return planner
