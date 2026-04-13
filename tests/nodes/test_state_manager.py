import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from gaia_agent.nodes.state_manager import make_state_manager_node
from gaia_agent.state import new_state


def _model(text: str):
    model = MagicMock()
    model.invoke.return_value = AIMessage(content=text)
    return model


def _base_state():
    state = new_state("task-1", "what is 6 times 7?")
    state["plan"] = [{"description": "compute", "tier": "S1", "thought": "multiply"}]
    state["todo_list"] = ["compute 6*7"]
    return state


def test_passthrough_when_draft_exists():
    node = make_state_manager_node(_model("unused"))
    state = _base_state()
    state["draft_answer"] = "42"
    out = node(state)
    assert out == {}


def test_detects_answer():
    payload = {
        "has_answer": True,
        "draft_answer": "42",
        "domain": "math",
        "strategy": "done",
    }
    node = make_state_manager_node(_model(json.dumps(payload)))
    state = _base_state()
    out = node(state)
    assert out["draft_answer"] == "42"


def test_dispatches_domain():
    payload = {
        "has_answer": False,
        "draft_answer": None,
        "domain": "math",
        "strategy": "run calculation",
    }
    node = make_state_manager_node(_model(json.dumps(payload)))
    state = _base_state()
    out = node(state)
    assert out["current_domain"] == "math"


def test_forces_replan_when_stuck():
    payload = {
        "has_answer": False,
        "draft_answer": None,
        "domain": "general",
        "strategy": "continue",
    }
    node = make_state_manager_node(_model(json.dumps(payload)))
    state = _base_state()
    state["todo_list"] = []  # Empty todos triggers stuck detection
    out = node(state)
    assert out["replan_count"] == 1


def test_handles_json_failure():
    node = make_state_manager_node(_model("this is not json at all"))
    state = _base_state()
    state["json_repair_retries"] = 0
    out = node(state)
    assert out["json_repair_retries"] == 1
