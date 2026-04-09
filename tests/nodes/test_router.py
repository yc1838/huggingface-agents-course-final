from gaia_agent.nodes.router import route_next
from gaia_agent.state import new_state


def _state_with_plan(plan, step_idx=0, draft=None):
    state = new_state("task-1", "q")
    state["plan"] = plan
    state["step_idx"] = step_idx
    state["draft_answer"] = draft
    return state


def test_route_to_s1():
    assert route_next(_state_with_plan([{"description": "search", "tier": "S1"}])) == "exec_s1"


def test_route_to_s2():
    assert route_next(_state_with_plan([{"description": "reason", "tier": "S2"}])) == "exec_s2"


def test_route_to_verifier_when_plan_exhausted():
    state = _state_with_plan([{"description": "a", "tier": "S1"}], step_idx=1)
    assert route_next(state) == "verifier"


def test_route_to_verifier_when_draft_exists():
    state = _state_with_plan(
        [
            {"description": "a", "tier": "S1"},
            {"description": "b", "tier": "S2"},
        ],
        step_idx=1,
        draft="ready",
    )
    assert route_next(state) == "verifier"
