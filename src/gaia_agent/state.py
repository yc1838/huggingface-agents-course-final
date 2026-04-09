from __future__ import annotations

from typing import TypedDict


class PlanStep(TypedDict):
    description: str
    status: str


class Observation(TypedDict):
    source: str
    content: str


class AgentState(TypedDict):
    task_id: str
    question: str
    file_path: str | None
    modality: str | None
    plan: list[PlanStep]
    step_idx: int
    observations: list[Observation]
    draft_answer: str
    critique: str
    retries: int
    final_answer: str


def new_state(task_id: str, question: str) -> AgentState:
    return AgentState(
        task_id=task_id,
        question=question,
        file_path=None,
        modality=None,
        plan=[],
        step_idx=0,
        observations=[],
        draft_answer="",
        critique="",
        retries=0,
        final_answer="",
    )
