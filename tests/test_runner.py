import json

from gaia_agent.runner import run_agent_on_questions


class RecordingGraph:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, state):
        self.calls.append(state)
        return self.result


class ExplodingGraph:
    def invoke(self, state):
        raise RuntimeError("boom")


def test_runner_invokes_graph_and_writes_checkpoint(tmp_path):
    graph = RecordingGraph({"final_answer": "4"})
    questions = [{"task_id": "task-1", "question": "What is 2+2?"}]

    answers = run_agent_on_questions(graph, questions, tmp_path)

    assert answers == [{"task_id": "task-1", "submitted_answer": "4"}]
    assert graph.calls == [
        {
            "task_id": "task-1",
            "question": "What is 2+2?",
            "file_path": None,
            "modality": None,
            "plan": [],
            "step_idx": 0,
            "observations": [],
            "draft_answer": "",
            "critique": "",
            "retries": 0,
            "final_answer": "",
        }
    ]
    checkpoint = json.loads((tmp_path / "task-1.json").read_text())
    assert checkpoint["task_id"] == "task-1"
    assert checkpoint["final_answer"] == "4"


def test_runner_uses_checkpoint_without_invoking_graph(tmp_path):
    checkpoint_path = tmp_path / "task-2.json"
    checkpoint_path.write_text(
        json.dumps({"task_id": "task-2", "final_answer": "cached"})
    )
    graph = RecordingGraph({"final_answer": "fresh"})
    questions = [{"task_id": "task-2", "question": "Ignored"}]

    answers = run_agent_on_questions(graph, questions, tmp_path)

    assert answers == [{"task_id": "task-2", "submitted_answer": "cached"}]
    assert graph.calls == []


def test_runner_uses_submitted_answer_from_checkpoint_when_present(tmp_path):
    checkpoint_path = tmp_path / "task-4.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "task_id": "task-4",
                "final_answer": "draft",
                "submitted_answer": "cached",
            }
        )
    )
    graph = RecordingGraph({"final_answer": "fresh"})
    questions = [{"task_id": "task-4", "question": "Ignored"}]

    answers = run_agent_on_questions(graph, questions, tmp_path)

    assert answers == [{"task_id": "task-4", "submitted_answer": "cached"}]
    assert graph.calls == []


def test_runner_records_agent_errors(tmp_path):
    graph = ExplodingGraph()
    questions = [{"task_id": "task-3", "question": "Will fail"}]

    answers = run_agent_on_questions(graph, questions, tmp_path)

    assert len(answers) == 1
    assert answers[0]["task_id"] == "task-3"
    assert answers[0]["submitted_answer"].startswith("AGENT ERROR: boom")
