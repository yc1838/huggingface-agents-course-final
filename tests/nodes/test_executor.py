from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from gaia_agent.nodes.executor import make_executor_node
from gaia_agent.state import new_state


@tool
def fake_tool(query: str) -> str:
    """Return a fake result for the supplied query."""
    return f"result for {query}"


def _state():
    state = new_state("task-1", "Who is X?")
    state["plan"] = [
        {"description": "search for X", "tier": "S1"},
        {"description": "write answer", "tier": "S2"},
    ]
    return state


def test_executor_runs_tool_call_and_records_observation():
    model = MagicMock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "fake_tool", "args": {"query": "X"}, "id": "call_1"}],
    )
    node = make_executor_node(model, [fake_tool])

    out = node(_state())

    assert out["step_idx"] == 1
    assert out["draft_answer"] is None
    assert out["observations"] == [
        {
            "step_idx": 0,
            "tool": "fake_tool",
            "args": {"query": "X"},
            "result": "result for X",
        }
    ]


def test_executor_promotes_draft_answer_from_prefixed_text():
    model = MagicMock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(content="DRAFT: 42")
    node = make_executor_node(model, [fake_tool])
    state = _state()
    state["step_idx"] = 1

    out = node(state)

    assert out["step_idx"] == 2
    assert out["draft_answer"] == "42"


def test_executor_records_plain_text_reasoning():
    model = MagicMock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(content="some reasoning result")
    node = make_executor_node(model, [fake_tool])

    out = node(_state())

    assert out["step_idx"] == 1
    assert out["observations"] == [
        {
            "step_idx": 0,
            "tool": "reasoning",
            "args": {},
            "result": "some reasoning result",
        }
    ]
