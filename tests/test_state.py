from gaia_agent.state import AgentState, Observation, PlanStep, new_state


def test_new_state_returns_expected_defaults():
    state = new_state(task_id="task-1", question="What is 2+2?")

    assert state == AgentState(
        task_id="task-1",
        question="What is 2+2?",
        file_path=None,
        modality=None,
        plan=[],
        step_idx=0,
        observations=[],
        draft_answer="",
        critique="",
        retries=0,
        final_answer="",
    )


def test_state_types_are_serializable_dicts():
    step = PlanStep(description="Inspect prompt", status="pending")
    observation = Observation(source="tool", content="No file attached")
    state = new_state(task_id="task-2", question="Where is the moon?")
    state["plan"].append(step)
    state["observations"].append(observation)

    assert step == {"description": "Inspect prompt", "status": "pending"}
    assert observation == {"source": "tool", "content": "No file attached"}
    assert state["plan"][0]["description"] == "Inspect prompt"
    assert state["observations"][0]["source"] == "tool"
