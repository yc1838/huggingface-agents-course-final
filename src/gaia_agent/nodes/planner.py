from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.llm_utils import extract_text
from gaia_agent.prompts import PLANNER_SYSTEM, apply_caveman
from gaia_agent.state import AgentState

log = logging.getLogger(__name__)


from pydantic import BaseModel, Field
from gaia_agent.json_repair import EmptyResponseError, UnsalvageableJsonError, safe_structured_call

class PlanStepSchema(BaseModel):
    thought: str = Field(description="Rationale for this step")
    description: str = Field(description="Detailed action description")
    tier: str = Field(default="S1", description="Model tier (S1/S2)")

class PlanSchema(BaseModel):
    plan: list[PlanStepSchema]


def make_planner_node(model, cheap_model=None, caveman: bool = False, caveman_mode: str = "full"):
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
        
        try:
            plan_obj = safe_structured_call(
                model=model,
                messages=[
                    SystemMessage(content=planner_prompt),
                    HumanMessage(content="\n".join(human_lines)),
                ],
                target_schema=PlanSchema,
                cheap_fixer_model=cheap_model,
                node_name="planner",
            )
            plan = [step.model_dump() for step in plan_obj.plan]
        except EmptyResponseError:
            log.warning("[planner] empty response — default to empty plan")
            plan = []
        except UnsalvageableJsonError as e:
            log.error("[planner] JSON unsalvageable: %s", e)
            return {
                "plan": [],
                "step_idx": 0,
                "todo_list": [],
                "json_repair_retries": state["json_repair_retries"] + 1,
                "draft_answer": None,
                "critique": None,
            }

        log.info("[planner] final plan steps: %s", [s["description"] for s in plan])
        
        return {
            "plan": plan,
            "step_idx": 0,
            "observations": state["observations"] if state["critique"] else [],
            "working_memory": state["working_memory"] if state["critique"] else "",
            "todo_list": [s["description"] for s in plan],
            "draft_answer": None,
            "critique": None,
            "json_repair_retries": state["json_repair_retries"],
        }

    return planner
