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
        if not isinstance(payload, dict):
            log.warning("[planner] payload is not a dict: %r", payload)
            payload = {"plan": []}

        plan_raw = payload.get("plan", [])
        if not isinstance(plan_raw, list):
            plan_raw = []

        log.info("[planner] plan steps: %s", [s.get("description", str(s)) if isinstance(s, dict) else str(s) for s in plan_raw])
        plan = []
        for step in plan_raw:
            if isinstance(step, dict):
                thought = step.get("thought", "No rationale provided.")
                description = step.get("description", str(step))
                tier = step.get("tier", "S1")
            else:
                thought = "No rationale provided."
                description = str(step)
                tier = "S1"
            plan.append({"thought": thought, "description": description, "tier": tier})

        return {
            "plan": plan,
            "step_idx": 0,
            "observations": [],
            "working_memory": "",
            "draft_answer": None,
            "critique": None,
        }

    return planner
