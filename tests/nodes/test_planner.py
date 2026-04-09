import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from gaia_agent.nodes.planner import make_planner_node
from gaia_agent.state import new_state


def _fake_model(response_text: str):
    model = MagicMock()
    model.invoke.return_value = AIMessage(content=response_text)
    return model


def test_planner_parses_json_plan():
    model = _fake_model(
        json.dumps(
            {
                "plan": [
                    {"description": "search for X", "tier": "S1"},
                    {"description": "synthesize answer", "tier": "S2"},
                ]
            }
        )
    )
    node = make_planner_node(model)

    out = node(new_state("task-1", "Who is X?"))

    assert out["plan"] == [
        {"description": "search for X", "tier": "S1"},
        {"description": "synthesize answer", "tier": "S2"},
    ]
    assert out["step_idx"] == 0
    assert out["observations"] == []
    assert out["draft_answer"] is None
    assert out["critique"] is None


def test_planner_extracts_json_from_fenced_response():
    model = _fake_model(
        'Here is the plan:\n```json\n{"plan":[{"description":"a","tier":"S2"}]}\n```'
    )
    node = make_planner_node(model)

    out = node(new_state("task-2", "q"))

    assert out["plan"] == [{"description": "a", "tier": "S2"}]


def test_planner_includes_critique_in_retry_prompt():
    model = _fake_model(json.dumps({"plan": [{"description": "retry", "tier": "S1"}]}))
    node = make_planner_node(model)
    state = new_state("task-3", "q")
    state["critique"] = "previous answer was unsupported"
    state["retries"] = 1

    node(state)

    args, _kwargs = model.invoke.call_args
    messages = args[0]
    text = " ".join(str(getattr(message, "content", "")) for message in messages)
    assert "previous answer was unsupported" in text
