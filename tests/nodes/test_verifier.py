import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from gaia_agent.nodes.verifier import make_verifier_node, verifier_decision
from gaia_agent.state import new_state


def _model(text: str):
    model = MagicMock()
    model.invoke.return_value = AIMessage(content=text)
    return model


def _draft_state(draft="42"):
    state = new_state("task-1", "what is 6 times 7?")
    state["plan"] = [{"description": "compute", "tier": "S1"}]
    state["step_idx"] = 1
    state["draft_answer"] = draft
    return state


def test_verifier_approves_draft():
    node = make_verifier_node(_model(json.dumps({"decision": "APPROVED", "critique": ""})))

    out = node(_draft_state())

    assert out["critique"] == ""
    assert out["final_answer"] == "42"


def test_verifier_rejects_and_requests_replan():
    node = make_verifier_node(
        _model(json.dumps({"decision": "REJECTED", "critique": "should be 42"}))
    )

    out = node(_draft_state("41"))

    assert out["critique"] == "should be 42"
    assert out["draft_answer"] is None
    assert out["final_answer"] is None
    assert out["retries"] == 1


def test_verifier_decision_routes_to_formatter_for_final_answer():
    state = _draft_state()
    state["final_answer"] = "42"

    assert verifier_decision(state) == "formatter"


def test_verifier_decision_routes_back_to_planner_when_retrying():
    state = _draft_state()
    state["critique"] = "nope"
    state["retries"] = 1

    assert verifier_decision(state) == "planner"


def test_verifier_preserves_draft_on_max_retry():
    """When retries hit MAX_RETRIES, the rejected draft should be preserved."""
    from gaia_agent.nodes.verifier import MAX_RETRIES

    node = make_verifier_node(
        _model(json.dumps({"decision": "REJECTED", "critique": "still wrong"}))
    )
    state = _draft_state("close-answer")
    state["retries"] = MAX_RETRIES  # Next rejection will exceed limit

    out = node(state)
    assert out["draft_answer"] == "close-answer"
    assert out["retries"] == MAX_RETRIES + 1


def test_verifier_decision_routes_formatter_on_max_retries():
    """When retries exceed MAX_RETRIES and final_answer is None, route to formatter."""
    from gaia_agent.nodes.verifier import MAX_RETRIES

    state = _draft_state()
    state["final_answer"] = None
    state["retries"] = MAX_RETRIES + 1

    assert verifier_decision(state) == "formatter"
