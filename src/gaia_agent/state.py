from __future__ import annotations

from typing import Any, Literal, TypedDict


Tier = Literal["S1", "S2"]
Modality = Literal["text", "web", "youtube", "pdf", "excel", "csv", "docx", "audio", "image"]


class PlanStep(TypedDict):
    thought: str
    description: str
    tier: Tier


class Observation(TypedDict):
    step_idx: int
    tool: str
    args: dict[str, Any]
    result: str


class AgentState(TypedDict):
    task_id: str
    question: str
    file_path: str | None
    modality: Modality
    plan: list[PlanStep]
    step_idx: int
    observations: list[Observation]
    working_memory: str
    draft_answer: str | None
    critique: str | None
    current_domain: str | None
    current_strategy: str | None
    retries: int
    task_chronicle: str
    todo_list: list[str]
    final_answer: str | None
    json_repair_retries: int


def new_state(task_id: str, question: str) -> AgentState:
    return AgentState(
        task_id=task_id,
        question=question,
        file_path=None,
        modality="text",
        plan=[],
        step_idx=0,
        observations=[],
        working_memory="",
        draft_answer=None,
        critique=None,
        current_domain=None,
        current_strategy=None,
        retries=0,
        task_chronicle="",
        todo_list=[],
        final_answer=None,
        json_repair_retries=0,
    )
