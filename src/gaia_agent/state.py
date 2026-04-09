from __future__ import annotations

from typing import Any, Literal, TypedDict


Tier = Literal["S1", "S2"]
Modality = Literal["text", "web", "youtube", "pdf", "excel", "csv", "docx", "audio", "image"]


class PlanStep(TypedDict):
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
    draft_answer: str | None
    critique: str | None
    retries: int
    final_answer: str | None


def new_state(task_id: str, question: str) -> AgentState:
    return AgentState(
        task_id=task_id,
        question=question,
        file_path=None,
        modality="text",
        plan=[],
        step_idx=0,
        observations=[],
        draft_answer=None,
        critique=None,
        retries=0,
        final_answer=None,
    )
