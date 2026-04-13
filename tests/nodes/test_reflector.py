import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from gaia_agent.nodes.reflector import make_reflector_node
from gaia_agent.state import new_state


def _model(text: str):
    model = MagicMock()
    model.invoke.return_value = AIMessage(content=text)
    return model


def _obs_state():
    state = new_state("task-1", "what is 6 times 7?")
    state["observations"] = [
        {"step_idx": 0, "tool": "run_python", "args": {}, "result": "42"}
    ]
    state["step_idx"] = 1
    return state


def test_reflector_returns_empty_when_no_observations():
    node = make_reflector_node(_model("unused"))
    state = new_state("task-1", "question")
    out = node(state)
    assert out == {}


def test_reflector_updates_working_memory():
    payload = {
        "reasoning": "computed product",
        "updated_working_memory": "6*7 = 42",
        "chronicle_update": None,
        "match_found": None,
    }
    node = make_reflector_node(_model(json.dumps(payload)))
    state = _obs_state()
    out = node(state)
    assert out["working_memory"] == "6*7 = 42"


def test_reflector_early_exit_on_match_found():
    payload = {
        "reasoning": "answer confirmed",
        "updated_working_memory": "6*7 = 42",
        "chronicle_update": None,
        "match_found": "42",
    }
    node = make_reflector_node(_model(json.dumps(payload)))
    state = _obs_state()
    out = node(state)
    assert out["draft_answer"] == "42"


def test_reflector_appends_chronicle():
    payload = {
        "reasoning": "found fact",
        "updated_working_memory": "memory",
        "chronicle_update": "Discovered that 6*7=42",
        "match_found": None,
    }
    node = make_reflector_node(_model(json.dumps(payload)))
    state = _obs_state()
    state["task_chronicle"] = ""
    out = node(state)
    assert "Discovered that 6*7=42" in out["task_chronicle"]


def test_reflector_handles_json_failure():
    node = make_reflector_node(_model("this is not json at all"))
    state = _obs_state()
    state["json_repair_retries"] = 0
    out = node(state)
    assert out["json_repair_retries"] == 1
