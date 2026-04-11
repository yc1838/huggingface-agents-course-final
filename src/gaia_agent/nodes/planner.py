from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.json_utils import extract_json
from gaia_agent.llm_utils import extract_text
from gaia_agent.prompts import PLANNER_SYSTEM, apply_caveman
from gaia_agent.state import AgentState, PlanStep

log = logging.getLogger(__name__)


def make_planner_node(model, caveman: bool = False, caveman_mode: str = "full"):
    def planner(state: AgentState) -> dict:
        human_lines = [f"Question: {state['question']}"]
        if state["file_path"]:
            human_lines.append(f"Associated file: {state['file_path']} ({state['modality']})")
        if state["task_chronicle"]:
            human_lines.append(f"Task Chronicle (facts found so far):\n{state['task_chronicle']}")
            
        if state["critique"]:
            human_lines.append(f"Prior critique: {state['critique']}")
            if state["draft_answer"]:
                human_lines.append(f"Prior draft answer: {state['draft_answer']}")
            if state["working_memory"]:
                # Provide a snippet of the latest working memory to help the planner
                mem_snippet = state["working_memory"][-3000:]
                human_lines.append(f"Working memory context (last 3000 chars): {mem_snippet}")

        log.info("[planner] invoking model (type=%s) for task=%s", type(model).__name__, state["task_id"])
        
        planner_prompt = apply_caveman(PLANNER_SYSTEM, caveman, caveman_mode)
        
        response = model.invoke(
            [
                SystemMessage(content=planner_prompt),
                HumanMessage(content="\n".join(human_lines)),
            ]
        )
        # EXTREME LOGGING: Capture everything about the response
        try:
            log.info("[planner] response metadata: %r", getattr(response, "response_metadata", {}))
            log.info("[planner] usage metadata: %r", getattr(response, "usage_metadata", {}))
            log.info("[planner] additional_kwargs: %r", getattr(response, "additional_kwargs", {}))
        except Exception as e:
            log.warning("[planner] failed to log metadata: %s", e)

        raw = extract_text(response.content)
        log.info("[planner] raw response (len=%d):\n%r", len(raw), raw)
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
            thought = "No rationale provided."
            description = str(step)
            tier = "S1"

            if isinstance(step, dict):
                thought = step.get("thought", thought)
                description = step.get("description", description)
                tier = step.get("tier", tier)
            elif isinstance(step, str):
                # Aggressively try to find JSON if it's a string
                trimmed = step.strip()
                start_idx = trimmed.find("{")
                end_idx = trimmed.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_candidate = trimmed[start_idx : end_idx + 1]
                    try:
                        nested = extract_json(json_candidate)
                        if isinstance(nested, dict):
                            thought = nested.get("thought", thought)
                            description = nested.get("description", description)
                            tier = nested.get("tier", tier)
                    except Exception:
                        pass

            plan.append({"thought": thought, "description": description, "tier": tier})

        return {
            "plan": plan,
            "step_idx": 0,
            "observations": state["observations"] if state["critique"] else [],
            "working_memory": state["working_memory"] if state["critique"] else "",
            "draft_answer": None,
            "critique": None,
        }

    return planner
