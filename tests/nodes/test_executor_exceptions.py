import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from gaia_agent.nodes.executor import make_executor_node
from gaia_agent.state import new_state

@tool
def failing_tool(query: str) -> str:
    """A tool that always raises an exception."""
    raise ValueError(f"Simulated failure for {query}")

@tool
def error_returning_tool(query: str) -> str:
    """A tool that returns an error string instead of raising."""
    return f"Error: something went wrong with {query}"

@tool
def happy_tool(query: str) -> str:
    """A tool that works fine."""
    return f"Success for {query}"

def _state():
    state = new_state("task-1", "Who is X?")
    state["plan"] = [
        {"description": "use tool", "tier": "S1"},
    ]
    return state

def test_executor_catches_tool_exception():
    model = MagicMock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "failing_tool", "args": {"query": "X"}, "id": "call_1"}],
    )
    node = make_executor_node(model, [failing_tool])

    state = _state()
    out = node(state)

    obs = out["observations"][0]
    error_msg = obs["result"]
    assert "The tool executed but returned the following error: Simulated failure for X" in error_msg
    assert "Please consider this error as an observation and adjust your next action accordingly." in error_msg
    # Ensure it's recorded under the correct tool name
    assert obs["tool"] == "failing_tool"

def test_executor_wraps_error_string():
    model = MagicMock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "error_returning_tool", "args": {"query": "X"}, "id": "call_1"}],
    )
    node = make_executor_node(model, [error_returning_tool])

    state = _state()
    out = node(state)

    obs = out["observations"][0]
    error_msg = obs["result"]
    assert "The tool executed but returned the following error: Error: something went wrong with X" in error_msg
    assert "Please consider this error as an observation and adjust your next action accordingly." in error_msg

def test_executor_happy_path_remains_untouched():
    model = MagicMock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "happy_tool", "args": {"query": "X"}, "id": "call_1"}],
    )
    node = make_executor_node(model, [happy_tool])

    state = _state()
    out = node(state)

    obs = out["observations"][0]
    assert obs["result"] == "Success for X"
