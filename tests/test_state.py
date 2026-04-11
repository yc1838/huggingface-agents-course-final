from gaia_agent.state import AgentState, Observation, PlanStep, new_state


def test_new_state_returns_expected_defaults():
    state = new_state(task_id="task-1", question="What is 2+2?")

    assert state == AgentState(
        task_id="task-1",
        question="What is 2+2?",
        file_path=None,
        modality="text",
        plan=[],
        step_idx=0,
        observations=[],
        working_memory="",
        draft_answer=None,
        critique=None,
        current_domain=None,
        current_strategy=None,
        retries=0,
        task_chronicle="",
        todo_list=[],
        final_answer=None,
    )


def test_state_types_are_serializable_dicts():
    step = PlanStep(description="Inspect prompt", tier="S1")
    observation = Observation(
        step_idx=0,
        tool="reasoning",
        args={},
        result="No file attached",
    )
    state = new_state(task_id="task-2", question="Where is the moon?")
    state["plan"].append(step)
    state["observations"].append(observation)

    assert step == {"description": "Inspect prompt", "tier": "S1"}
    assert observation == {
        "step_idx": 0,
        "tool": "reasoning",
        "args": {},
        "result": "No file attached",
    }
    assert state["plan"][0]["description"] == "Inspect prompt"
    assert state["observations"][0]["tool"] == "reasoning"
