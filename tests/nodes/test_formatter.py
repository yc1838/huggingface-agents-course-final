from gaia_agent.nodes.formatter import make_formatter_node
from gaia_agent.state import new_state


def _state(final="The answer is 42."):
    state = new_state("task-1", "q")
    state["final_answer"] = final
    state["draft_answer"] = final
    return state


def test_formatter_strips_answer_phrase():
    formatter = make_formatter_node(model=None)
    out = formatter(_state("The answer is 42."))
    assert out["final_answer"] == "42"


def test_formatter_strips_final_answer_prefix():
    formatter = make_formatter_node(model=None)
    out = formatter(_state("FINAL ANSWER: Paris"))
    assert out["final_answer"] == "Paris"


def test_formatter_uses_draft_when_final_missing():
    formatter = make_formatter_node(model=None)
    state = new_state("task-2", "q")
    state["draft_answer"] = "50"
    out = formatter(state)
    assert out["final_answer"] == "50"
