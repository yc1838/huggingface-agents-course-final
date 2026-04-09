import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from gaia_agent.graph import build_graph
from gaia_agent.state import new_state


@tool
def fake_search(query: str) -> str:
    """Return a canned search result."""
    return "Paris is the capital of France."


def _planner_model():
    model = MagicMock()
    model.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "plan": [
                    {"description": "search for capital of France", "tier": "S1"},
                    {"description": "answer from search result", "tier": "S2"},
                ]
            }
        )
    )
    return model


def _executor_model_s1():
    model = MagicMock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "fake_search", "args": {"query": "capital of France"}, "id": "c1"}],
    )
    return model


def _executor_model_s2():
    model = MagicMock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(content="DRAFT: Paris")
    return model


def _verifier_model():
    model = MagicMock()
    model.invoke.return_value = AIMessage(
        content=json.dumps({"decision": "APPROVED", "critique": None})
    )
    return model


def _perception(_state):
    return {"file_path": None, "modality": "text"}


def test_graph_happy_path():
    graph = build_graph(
        perception_node=_perception,
        planner_model=_planner_model(),
        executor_model_s1=_executor_model_s1(),
        executor_model_s2=_executor_model_s2(),
        verifier_model=_verifier_model(),
        tools=[fake_search],
    )

    final = graph.invoke(new_state("task-1", "What is the capital of France?"))

    assert final["final_answer"] == "Paris"
