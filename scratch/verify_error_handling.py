import sys
import os
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

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

def _state():
    state = new_state("task-1", "Who is X?")
    state["plan"] = [
        {"description": "use tool", "tier": "S1"},
    ]
    return state

def test_exception():
    print("Testing exception handling...")
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
    result = obs["result"]
    print(f"Result: {result}")
    assert "The tool executed but returned the following error: Simulated failure for X" in result
    print("Exception test passed!")

def test_error_string():
    print("\nTesting error string wrapping...")
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
    result = obs["result"]
    print(f"Result: {result}")
    assert "The tool executed but returned the following error: Error: something went wrong with X" in result
    print("Error string test passed!")

if __name__ == "__main__":
    try:
        test_exception()
        test_error_string()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        sys.exit(1)
