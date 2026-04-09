from __future__ import annotations


def route_next(state) -> str:
    if state["draft_answer"]:
        return "verifier"
    if state["step_idx"] >= len(state["plan"]):
        return "verifier"
    current = state["plan"][state["step_idx"]]
    return "exec_s1" if current["tier"] == "S1" else "exec_s2"
