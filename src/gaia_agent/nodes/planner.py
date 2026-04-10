from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.json_utils import extract_json
from gaia_agent.llm_utils import extract_text
from gaia_agent.prompts import PLANNER_SYSTEM
from gaia_agent.state import AgentState, PlanStep

log = logging.getLogger(__name__)


def make_planner_node(model):
    def planner(state: AgentState) -> dict:
        human_lines = [f"Question: {state['question']}"]
        if state["file_path"]:
            human_lines.append(f"Associated file: {state['file_path']} ({state['modality']})")
        if state["critique"]:
            human_lines.append(f"Prior critique: {state['critique']}")

        log.info("[planner] invoking model for task=%s", state["task_id"])
        response = model.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content="\n".join(human_lines)),
            ]
        )
        raw = extract_text(response.content)
        log.debug("[planner] raw response:\n%s", raw)
        payload = extract_json(raw)
        log.info("[planner] plan steps: %s", [s["description"] for s in payload.get("plan", [])])
        plan = []
        for step in payload.get("plan", []):
            if isinstance(step, dict):
                description = step.get("description", str(step))
                tier = step.get("tier", "S1")
            else:
                description = str(step)
                tier = "S1"
            plan.append({"description": description, "tier": tier})
        return {
            "plan": plan,
            "step_idx": 0,
            "observations": [],
            "draft_answer": None,
            "critique": None,
        }

    return planner
