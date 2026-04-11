from __future__ import annotations


def route_next(state) -> str:
    if state.get("draft_answer"):
        return "verifier"
    if state["step_idx"] >= len(state["plan"]):
        return "verifier"
    
    # Route based on the domain determined by the Orchestrator
    domain = state.get("current_domain") or "general"
    valid_domains = ["math", "research", "vision", "audio", "file", "general"]
    if domain not in valid_domains:
        domain = "general"
        
    return f"exec_{domain}"
