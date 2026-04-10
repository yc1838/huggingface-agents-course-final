from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gaia_agent.state import new_state


def run_agent_on_questions(graph: Any, questions: list[dict], checkpoint_dir: str | Path) -> list[dict]:
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    answers: list[dict] = []

    for question in questions:
        task_id = question.get("task_id")
        prompt = question.get("question")
        if not task_id or not prompt:
            continue

        checkpoint_path = checkpoint_root / f"{task_id}.json"
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            answers.append(
                {
                    "task_id": task_id,
                    "submitted_answer": _submitted_answer_from_checkpoint(checkpoint),
                }
            )
            continue

        state = new_state(task_id=task_id, question=prompt)
        try:
            result = graph.invoke(state, {"recursion_limit": 50})
        except Exception as exc:  # pragma: no cover - exercised in tests
            answers.append(
                {
                    "task_id": task_id,
                    "submitted_answer": f"AGENT ERROR: {exc}",
                }
            )
            continue

        checkpoint = {**state, **result}
        submitted_answer = checkpoint.get("final_answer") or ""
        checkpoint["submitted_answer"] = submitted_answer
        # Only cache if we got a real, non-empty answer
        if submitted_answer and not submitted_answer.startswith("AGENT ERROR"):
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True))
        answers.append({"task_id": task_id, "submitted_answer": submitted_answer})

    return answers


def _submitted_answer_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    return checkpoint.get("submitted_answer") or checkpoint.get("final_answer", "")
